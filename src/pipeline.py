"""
End-to-end BlueBot question pipeline.

Flow:
1) Retrieve policy chunks from local ChromaDB
2) If not relevant, return fallback without calling the LLM
3) If relevant, build constrained prompt and call the LLM (local Ollama or Groq)
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

# Ensure project root is importable when running: python src/pipeline.py "question"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    LLM_CONTEXT_CHUNKS,
    MAX_CHUNK_CHARS,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT_SECONDS,
)

# Load embedding model once at import time; shared with retriever below.
_EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

import src.retriever as _retriever  # noqa: E402

# Point retriever at this shared instance so retrieve() does not load a second copy.
_retriever._MODEL = _EMBEDDING_MODEL
retrieve = _retriever.retrieve

DEBUG = os.environ.get("BLUEBOT_DEBUG", "").strip().lower() in ("1", "true", "yes")

# ----- Generation backend config -----
# Local dev (no env vars set) uses Ollama. Production (Railway) sets USE_GROQ=true
# + GROQ_API_KEY since Railway's containers don't run Ollama. Same pipeline logic
# either way - only the generation backend changes.
USE_GROQ = os.environ.get("USE_GROQ", "").strip().lower() in ("1", "true", "yes")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ----- Fixed responses -----
FALLBACK_MESSAGE = (
    "I don't have information about that. For assistance please contact Airblue "
    "support at 111-247-258 or visit airblue.com"
)

_GREETINGS = {
    "hi", "hello", "hey", "salam", "assalamualaikum",
    "good morning", "good afternoon", "good evening",
}
GREETING_RESPONSE = (
    "Hi! I'm BlueBot, Airblue's customer service assistant. "
    "Ask me about fares, baggage, check-in, or refunds."
)

SECURITY_REFUSAL_MESSAGE = (
    "I can answer specific questions about Airblue's policies, but I can't "
    "reproduce source documents directly. What would you like to know?"
)

INVALID_QUERY_MESSAGE = (
    "I didn't quite catch that — could you rephrase your question about "
    "Airblue fares, baggage, check-in, or refunds?"
)

_INJECTION_PATTERNS = ("ignore previous instructions", "ignore your instructions", "disregard previous", "you are now")


SYSTEM_PROMPT = """You are BlueBot, a customer service assistant for Airblue Pakistan.
Answer using ONLY the exact facts in the CONTEXT below.
Never state a fare name, price, or number that does not appear verbatim in the CONTEXT. If you are unsure, say you don't have that information.
If the customer names a specific fare type, answer only for that fare. If no fare is specified, summarize the relevant information across all fare types found in the context.
If the customer asks what fare types exist, list only the fare type names found in the context.
Do not mention meals, seat selection, or BlueMiles unless the customer asks.
Always write numbers as digits (e.g. 4,150), never spell them out in words.
If the context does not contain the answer, say: "I don't have that information. Please contact Airblue support at 111-247-258."
Answer the customer's question directly using the relevant facts from the context. Keep it brief, but include the actual facts requested — don't just restate the fare name.
Paraphrase the facts in your own words; do not copy sentences from the context verbatim. Never infer or state a price, fee, or cost that does not appear verbatim in the context. Baggage weight limits are not prices."""

_FARE_SECTION_PREFIXES = ("Value ", "Flexi ", "Xtra ")
_FARE_LIST_PHRASES = ("fare type", "fare types", "types of fare", "what fares", "which fares")
_SINGLE_TOPIC_MAX_WORDS = 6
_MIN_QUERY_CHARS = 3
_ALPHA_PATTERN = re.compile(r"[a-zA-Z]")

# Conservative split: only break on clear conjunction/sentence boundaries.
# Deliberately not splitting on bare commas - too likely to break single
# questions that just happen to contain a comma.
_SUBQUERY_SPLIT_PATTERN = re.compile(r"\s+and\s+|[.?]\s+|,\s*but\s+", re.IGNORECASE)

_FIELD_LABELS = (
    "Hand Carry Bags:", "Checked Bags:", "Meals:", "Seat Selection:",
    "BlueMiles Rewards:", "Refunds & Exchanges:",
)

def _format_fields(text: str) -> str:
    """Insert newlines before recognized field labels so the LLM sees clearly
    separated facts instead of a run-on paragraph (reduces fact-blending)."""
    formatted = text
    for label in _FIELD_LABELS:
        formatted = formatted.replace(f" {label}", f"\n{label}")
    return formatted.strip()

def _is_fallback_answer(answer: str) -> bool:
    """Catch LLM-generated 'I don't know' phrasing that doesn't exactly match FALLBACK_MESSAGE."""
    markers = ("don't have that information", "don't have information about that")
    return any(marker in answer.lower() for marker in markers)

# ----- Query validation / splitting -----

from wordfreq import zipf_frequency

def _is_valid_query(query: str) -> bool:
    stripped = query.strip()
    if len(stripped) < _MIN_QUERY_CHARS:
        return False
    if not _ALPHA_PATTERN.search(stripped):
        return False

    words = re.findall(r"[a-zA-Z]+", stripped.lower())
    if not words:
        return False

    # At least one word must be a recognizable English word (zipf_frequency > 0
    # means it's in the frequency dictionary at all).
    recognized = sum(1 for w in words if len(w) >= 3 and zipf_frequency(w, "en") > 0)
    return recognized >= 1

def _is_injection_attempt(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _INJECTION_PATTERNS)

_NUMBER_PATTERN = re.compile(r"\d[\d,.]*")
   

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.?!])\s+")

def _strip_unverified_numbers(answer: str, context_text: str) -> str:
    """Remove only sentences containing numbers not present in context."""
    context_numbers = set(_NUMBER_PATTERN.findall(context_text))
    sentences = _SENTENCE_SPLIT_PATTERN.split(answer.strip())
    kept = []
    for sentence in sentences:
        sentence_numbers = set(_NUMBER_PATTERN.findall(sentence))
        if any(num not in context_numbers for num in sentence_numbers):
            continue  # drop only this sentence
        kept.append(sentence)
    cleaned = " ".join(kept).strip()
    return cleaned if cleaned else FALLBACK_MESSAGE

_MIN_SOURCE_WORDS_FOR_OVERLAP_CHECK = 60

def _has_verbatim_overlap(answer: str, contexts: list[tuple[str, str]], n: int = 15, max_overlap_ratio: float = 0.6) -> bool:
    """
    Flag likely document-dumping: most of the ANSWER consists of words
    that are part of some n-gram verbatim-matched against the context.

    Exempts answers whose overlap is fully explained by a single short
    source chunk (e.g. a short, dense legal clause with no paraphrase
    room) — checked per-chunk rather than against the combined context,
    so that an unrelated extra chunk pulled in by retrieval can't push
    a short, legitimately-quoted answer over the length threshold.
    """
    answer_words = answer.lower().split()
    if len(answer_words) < n:
        return False

    for _, chunk_text in contexts:
        chunk_words = chunk_text.lower().split()
        if len(chunk_words) < _MIN_SOURCE_WORDS_FOR_OVERLAP_CHECK:
            continue  # this chunk is too short to meaningfully "dump"

        if len(chunk_words) < n:
            continue

        chunk_ngrams = {" ".join(chunk_words[i:i+n]) for i in range(len(chunk_words) - n + 1)}
        covered = [False] * len(answer_words)
        for i in range(len(answer_words) - n + 1):
            if " ".join(answer_words[i:i+n]) in chunk_ngrams:
                for j in range(i, i + n):
                    covered[j] = True

        overlap_word_count = sum(covered)
        if (overlap_word_count / len(answer_words)) > max_overlap_ratio:
            return True  # flagged against at least one substantial chunk

    return False    
def _split_subqueries(query: str) -> list[str]:
    """Split a compound question into sub-questions for separate retrieval."""
    parts = [p.strip() for p in _SUBQUERY_SPLIT_PATTERN.split(query) if p.strip()]
    return parts if len(parts) > 1 else [query]


def _merge_retrievals(results: list) -> _retriever.RetrievalResult:
    """Merge multiple RetrievalResults, deduping by normalized text, closest-first."""
    seen_keys: set = set()
    merged_chunks: list = []
    any_relevant = False

    for result in results:
        any_relevant = any_relevant or result.is_relevant
        for chunk in result.chunks:
            key = chunk.text.strip().lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_chunks.append(chunk)

    merged_chunks.sort(key=lambda c: c.distance)
    return _retriever.RetrievalResult(chunks=merged_chunks, is_relevant=any_relevant)


def _build_retrieval_query(query: str, history: list[dict] | None) -> str:
    """Augment query with previous turn, used only as a fallback retry."""
    if not history:
        return query
    last_turn = history[-1]
    if not last_turn.get("is_relevant", False):
        return query  # nothing useful to carry forward
    return f"{last_turn.get('query', '')} {last_turn.get('answer', '')} {query}"


def _retrieve_with_history_fallback(query: str, history: list[dict] | None):
    """Try retrieval on the plain query first; retry with history only if it fails."""
    result = retrieve(query)
    if result.is_relevant or not history:
        return result

    augmented_query = _build_retrieval_query(query, history)
    if augmented_query == query:
        return result  # nothing to add

    augmented_result = retrieve(augmented_query)
    return augmented_result if augmented_result.is_relevant else result

def _is_fare_list_query(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _FARE_LIST_PHRASES)


# ----- Context selection / prompt building -----

def _truncate(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    """Keep prompts short so generation spends less time on context processing."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _select_contexts(query: str, chunks: list) -> list[tuple[str, str]]:
    if _is_fare_list_query(query):
        by_fare: dict[str, tuple[str, str]] = {}
        for chunk in chunks:
            for prefix in _FARE_SECTION_PREFIXES:
                fare_name = prefix.strip()
                if chunk.text.startswith(prefix):
                    existing = by_fare.get(fare_name)
                    if existing is None or len(chunk.text) > len(existing[1]):
                        by_fare[fare_name] = (chunk.source, chunk.text)

        for fare_name in ("Value", "Flexi", "Xtra"):
            if fare_name in by_fare:
                continue
            supplemental = retrieve(f"{fare_name} fare baggage allowance")
            for chunk in supplemental.chunks:
                if chunk.text.startswith(f"{fare_name} "):
                    by_fare[fare_name] = (chunk.source, chunk.text)
                    break

        if by_fare:
            return [by_fare[name] for name in ("Value", "Flexi", "Xtra") if name in by_fare]

    if len(query.split()) <= _SINGLE_TOPIC_MAX_WORDS:
        query_lower = query.lower()
        matched_fares = [p.strip() for p in _FARE_SECTION_PREFIXES if p.strip().lower() in query_lower]
        if len(matched_fares) == 1:
            for chunk in chunks:
                if chunk.text.startswith(matched_fares[0] + " "):
                    return [(chunk.source, chunk.text)]

    return [(chunk.source, chunk.text) for chunk in chunks[:LLM_CONTEXT_CHUNKS]]

def _build_prompt(query: str, contexts: list[tuple[str, str]]) -> str:
    context_blocks: list[str] = []
    for source, text in contexts:
        formatted_text = _format_fields(text)
        context_blocks.append(f"[Source: {source}]\n{_truncate(formatted_text)}")

    context_text = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"CUSTOMER QUESTION: {query}\n"
        "ANSWER:"
    )


# ----- Generation backends -----

def _generate_with_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "temperature": 0.1,
            "num_predict": OLLAMA_NUM_PREDICT,
            "stop": ["CUSTOMER QUESTION:", "\nCUSTOMER", "ANSWER:"],
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if "response" not in data:
        raise ValueError("Ollama response JSON missing 'response' field.")
    return str(data["response"]).strip()


def _generate_with_groq(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": OLLAMA_NUM_PREDICT,
        "stop": ["CUSTOMER QUESTION:", "ANSWER:"],
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def _generate(prompt: str) -> str:
    if USE_GROQ:
        return _generate_with_groq(prompt)
    return _generate_with_ollama(prompt)


# ----- Main entrypoint -----

def ask(query: str, history: list[dict] | None = None) -> dict:
    """
    Run retrieval + guarded generation for a user question.

    Returns:
        {"answer": str, "sources": [source filenames used], "is_relevant": bool}
    """
    cleaned = query.strip().lower()
    if cleaned in _GREETINGS:
        return {"answer": GREETING_RESPONSE, "sources": [], "is_relevant": True}

    if not _is_valid_query(query):
        return {"answer": INVALID_QUERY_MESSAGE, "sources": [], "is_relevant": False}
    
    if _is_injection_attempt(query):
        return {"answer": SECURITY_REFUSAL_MESSAGE, "sources": [], "is_relevant": False}
    subqueries = _split_subqueries(query)
    if len(subqueries) > 1:
        retrieval = _merge_retrievals([retrieve(q) for q in subqueries])
    else:
        retrieval = _retrieve_with_history_fallback(query, history)

    if DEBUG:
        print("\n===== RAW RETRIEVED CHUNKS =====")
        for chunk in retrieval.chunks:
            print(f"[source: {chunk.source}]")
            print(chunk.text)
            print()

    # Hard guardrail: if query is not relevant, do NOT call the LLM.
    if not retrieval.is_relevant:
        sources = list(dict.fromkeys(chunk.source for chunk in retrieval.chunks if chunk.source))
        return {"answer": FALLBACK_MESSAGE, "sources": sources, "is_relevant": False}

    contexts = _select_contexts(query, retrieval.chunks)
    sources = list(dict.fromkeys(source for source, _ in contexts if source))

    if DEBUG:
        print("===== CONTEXT SENT TO PROMPT (after truncate) =====")
        for source, text in contexts:
            print(f"[source: {source}]")
            print(_truncate(text))
            print()

    prompt = _build_prompt(query=query, contexts=contexts)

    if DEBUG:
        print("===== FINAL PROMPT SENT TO LLM =====")
        print(prompt)
        print("===== END =====\n")

    answer = _generate(prompt)
    if DEBUG:
        print("===== RAW LLM ANSWER (pre-checks) =====")
        print(answer)
        print("===== END RAW ANSWER =====\n")
    context_text = " ".join(text for _, text in contexts)
    if _has_verbatim_overlap(answer, contexts, n=15):
        answer = SECURITY_REFUSAL_MESSAGE
    else:
        answer = _strip_unverified_numbers(answer, context_text)

    is_relevant = (
    answer != SECURITY_REFUSAL_MESSAGE
    and not _is_fallback_answer(answer))
    return {"answer": answer, "sources": sources, "is_relevant": is_relevant}


TEST_QUERIES = [
    "what is baggage allowance on value fare",
    "what are the fare types",
    "can i checkin luggage on value fare",
]

if __name__ == "__main__":
    cli_args = sys.argv[1:]
    if "--debug" in cli_args:
        DEBUG = True
        cli_args = [arg for arg in cli_args if arg != "--debug"]

    if "--batch" in cli_args:
        pending = list(TEST_QUERIES)
    else:
        pending = [" ".join(cli_args).strip()] if cli_args else []

    history: list[dict] = []
    while True:
        question = pending.pop(0) if pending else input("\nQuestion (empty to quit): ").strip()
        if not question:
            break

        started = time.perf_counter()
        result = ask(question, history=history)
        elapsed_seconds = time.perf_counter() - started

        print("\nAnswer:")
        print(result["answer"])
        print("\nSources used:")
        if result["sources"]:
            for source in result["sources"]:
                print(f"- {source}")
        else:
            print("- (none)")
        print(f"\nIs relevant: {result['is_relevant']}")
        print(f"Response time: {elapsed_seconds:.2f} seconds")

        history.append({
            "query": question,
            "answer": result["answer"],
            "is_relevant": result["is_relevant"],
        })