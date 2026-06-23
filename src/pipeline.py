"""
End-to-end BlueBot question pipeline.

Flow:
1) Retrieve policy chunks from local ChromaDB
2) If not relevant, return fallback without calling the LLM
3) If relevant, build constrained prompt and call local Ollama HTTP API
"""

from __future__ import annotations

import os
import os

USE_GROQ = os.environ.get("USE_GROQ", "").strip().lower() in ("1", "true", "yes")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _generate_with_groq(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": OLLAMA_NUM_PREDICT,
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def _generate(prompt: str) -> str:
    if USE_GROQ:
        return _generate_with_groq(prompt)
    return _generate_with_ollama(prompt)
from pathlib import Path
import sys
import time

import requests
from sentence_transformers import SentenceTransformer

import re

# Conservative split: only break on clear conjunction/sentence boundaries.
# Deliberately not splitting on bare commas - too likely to break single
# questions that just happen to contain a comma.
_SUBQUERY_SPLIT_PATTERN = re.compile(r"\s+and\s+|[.?]\s+|,\s*but\s+", re.IGNORECASE)


def _split_subqueries(query: str) -> list[str]:
    """Split a compound question into sub-questions for separate retrieval."""
    parts = [p.strip() for p in _SUBQUERY_SPLIT_PATTERN.split(query) if p.strip()]
    return parts if len(parts) > 1 else [query]


def _merge_retrievals(results: list) -> _retriever.RetrievalResult:
    """Merge multiple RetrievalResults, deduping by chunk_id, keeping closest-first order."""
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

_GREETINGS = {"hi", "hello", "hey", "salam", "assalamualaikum", "good morning", "good afternoon", "good evening"}
GREETING_RESPONSE = "Hi! I'm BlueBot, Airblue's customer service assistant. Ask me about fares, baggage, check-in, or refunds."
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


FALLBACK_MESSAGE = (
    "I don't have information about that. For assistance please contact Airblue "
    "support at 111-247-258 or visit airblue.com"
)

SYSTEM_PROMPT = """You are BlueBot, a customer service assistant for Airblue Pakistan.
Answer using ONLY the exact facts in the CONTEXT below.Never state a fare name, price, or number that does not appear verbatim in the CONTEXT. If you are unsure, say you don't have that information.
If the customer asks about one specific fare type, answer only for that fare.
If the customer asks what fare types exist, list only the fare type names found in the context.
Do not mention meals, seat selection, or BlueMiles unless the customer asks.
If the context does not contain the answer, say: "I don't have that information. Please contact Airblue support at 111-247-258."
Be concise. One or two sentences maximum.If the CONTEXT discusses a related but different scenario (e.g. delay liability when asked about cancellation, or vice versa), say you don't have that specific information rather than answering with the related scenario's facts."""

OLLAMA_URL = "http://localhost:11434/api/generate"

_FARE_SECTION_PREFIXES = ("Value ", "Flexi ", "Xtra ")
_FARE_LIST_PHRASES = ("fare type", "fare types", "types of fare", "what fares", "which fares")


def _truncate(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    """Keep prompts short so Ollama spends less time on context processing."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _is_fare_list_query(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _FARE_LIST_PHRASES)


_SINGLE_TOPIC_MAX_WORDS = 6

def _select_contexts(query: str, chunks: list) -> list[tuple[str, str]]:
    """Pick prompt context chunks based on query intent."""
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
            supplemental = retrieve(f"{fare_name} fare baggage allowance", k=3)
            for chunk in supplemental.chunks:
                if chunk.text.startswith(f"{fare_name} "):
                    by_fare[fare_name] = (chunk.source, chunk.text)
                    break

        if by_fare:
            return [by_fare[name] for name in ("Value", "Flexi", "Xtra") if name in by_fare]

    # Only take the single-fare shortcut for short, single-topic queries.
    # Longer/compound questions (e.g. "I have a flexi fare and my flight got
    # cancelled...") shouldn't be narrowed to just the fare chunk, since they
    # likely need other context too.
    if len(query.split()) <= _SINGLE_TOPIC_MAX_WORDS:
        for chunk in chunks:
            for prefix in _FARE_SECTION_PREFIXES:
                if prefix.strip().lower() in query.lower() and chunk.text.startswith(prefix):
                    return [(chunk.source, chunk.text)]

    return [(chunk.source, chunk.text) for chunk in chunks[:LLM_CONTEXT_CHUNKS]]

def _build_prompt(query: str, contexts: list[tuple[str, str]]) -> str:
    context_blocks: list[str] = []
    for source, text in contexts:
        context_blocks.append(f"[Source: {source}]\n{_truncate(text)}")

    context_text = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"CUSTOMER QUESTION: {query}\n"
        "ANSWER:"
    )


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
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    response.raise_for_status()

    data = response.json()
    if "response" not in data:
        raise ValueError("Ollama response JSON missing 'response' field.")
    return str(data["response"]).strip()


def ask(query: str) -> dict:
    """
    Run retrieval + guarded generation for a user question.

    Args:
        query: User's question text.

    Returns:
        {
            "answer": str,
            "sources": [list of source filenames used],
            "is_relevant": bool
        }
    """
    cleaned = query.strip().lower()
    if cleaned in _GREETINGS:
        return {"answer": GREETING_RESPONSE, "sources": [], "is_relevant": True}
    subqueries = _split_subqueries(query)
    if len(subqueries) > 1:
        retrieval = _merge_retrievals([retrieve(q) for q in subqueries])
    else:
        retrieval = retrieve(query)

    if DEBUG:
        print("\n===== RAW RETRIEVED CHUNKS =====")
        for chunk in retrieval.chunks:
            print(f"[source: {chunk.source}]")
            print(chunk.text)
            print()

    # Hard guardrail: if query is not relevant, do NOT call the LLM.
    if not retrieval.is_relevant:
        # No LLM call happened, so "sources used" should reflect what was
        # retrieved and rejected, not what was sent to a prompt.
        sources = list(dict.fromkeys(chunk.source for chunk in retrieval.chunks if chunk.source))
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": sources,
            "is_relevant": False,
        }

    # Send only the best chunk(s) to Ollama to keep generation fast.
    contexts = _select_contexts(query, retrieval.chunks)

    # sources now reflects only what actually went into the prompt
    sources = list(dict.fromkeys(source for source, _ in contexts if source))

    if DEBUG:
        print("===== CONTEXT SENT TO PROMPT (after truncate) =====")
        for source, text in contexts:
            print(f"[source: {source}]")
            print(_truncate(text))
            print()

    prompt = _build_prompt(query=query, contexts=contexts)

    if DEBUG:
        print("===== FINAL PROMPT SENT TO OLLAMA =====")
        print(prompt)
        print("===== END =====\n")

    answer = _generate(prompt)

    return {
        "answer": answer,
        "sources": sources,
        "is_relevant": True,
    }
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

    while True:
        question = pending.pop(0) if pending else input("\nQuestion (empty to quit): ").strip()
        if not question:
            break

        started = time.perf_counter()
        result = ask(question)
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
