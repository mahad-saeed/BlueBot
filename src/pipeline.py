"""
End-to-end BlueBot question pipeline.

Flow:
1) Retrieve policy chunks from local ChromaDB
2) If not relevant, return fallback without calling the LLM
3) If relevant, build constrained prompt and call local Ollama HTTP API
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import requests
from sentence_transformers import SentenceTransformer


# Ensure project root is importable when running: python src/pipeline.py "question"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMBEDDING_MODEL_NAME  # noqa: E402

# Load embedding model once at import time; shared with retriever below.
_EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

import src.retriever as _retriever  # noqa: E402

# Point retriever at this shared instance so retrieve() does not load a second copy.
_retriever._MODEL = _EMBEDDING_MODEL
retrieve = _retriever.retrieve


FALLBACK_MESSAGE = (
    "I don't have information about that. For assistance please contact Airblue "
    "support at 111-247-258 or visit airblue.com"
)

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are BlueBot, a customer service assistant for Airblue Pakistan.
Answer using ONLY the exact facts in the CONTEXT below.
For baggage questions, state ONLY the specific fare type asked about.
Do not list or summarize other fare types.
Do not mention meals, seat selection, or BlueMiles unless the customer asks.
If the context does not contain the answer, say: "I don't have that information. Please contact Airblue support at 111-247-258."
Be concise. One or two sentences maximum."""

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

def _extract_relevant_lines(query: str, text: str) -> str:
    query_lower = query.lower()
    fare_keywords = {"value": "Value", "flexi": "Flexi", "xtra": "Xtra"}
    detected_fare = None
    for keyword, label in fare_keywords.items():
        if keyword in query_lower:
            detected_fare = label
            break
    if not detected_fare:
        return text
    lines = text.split("\n")
    capture = False
    result = []
    for line in lines:
        if line.strip().startswith(detected_fare):
            capture = True
        elif any(line.strip().startswith(label) for label in fare_keywords.values()) and capture:
            break
        if capture:
            result.append(line)
    return "\n".join(result) if result else text

def _build_prompt(query: str, contexts: list[tuple[str, str]]) -> str:
    context_blocks: list[str] = []
    for source, text in contexts:
        filtered = _extract_relevant_lines(query, text)
        context_blocks.append(f"[Source: {source}]\n{filtered}")

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
            "temperature": 0.1,
            "num_predict": 150,
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
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
    retrieval = retrieve(query)

    # Collect unique sources while preserving first-seen order.
    sources = list(dict.fromkeys(chunk.source for chunk in retrieval.chunks if chunk.source))

    # Hard guardrail: if query is not relevant, do NOT call the LLM.
    if not retrieval.is_relevant:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": sources,
            "is_relevant": False,
        }

    contexts = [(chunk.source, chunk.text) for chunk in retrieval.chunks]
    prompt = _build_prompt(query=query, contexts=contexts)
    answer = _generate_with_ollama(prompt)

    return {
        "answer": answer,
        "sources": sources,
        "is_relevant": True,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/pipeline.py "your question here"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:]).strip()

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
