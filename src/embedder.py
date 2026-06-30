"""
Embedding pipeline for BlueBot RAG.

This script:
1) Builds chunks from src.chunker.create_chunks()
2) Generates embeddings with sentence-transformers/all-MiniLM-L6-v2
3) Persists documents + embeddings + metadata in ChromaDB
"""

from __future__ import annotations

from pathlib import Path
import sys

import chromadb
from sentence_transformers import SentenceTransformer


# Ensure project root is importable when running: python src/embedder.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import create_chunks  # noqa: E402
from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME  # noqa: E402

def _dedup_key(text: str) -> str:
    """
    Identify near-duplicate chunks that share the same fare-name prefix
    (Value/Flexi/Xtra) even when trailing content diverges due to different
    sections following in different source files.
    """
    stripped = text.strip()
    for prefix in ("Value ", "Flexi ", "Xtra "):
        if stripped.startswith(prefix):
            return prefix.strip().lower()
    return stripped.lower()

def embed_and_store() -> tuple[int, str]:
    chunks = create_chunks()
    if not chunks:
        return 0, "No chunks found to embed."

    # Dedupe identical chunk text across source files (e.g. Value/Xtra fare
    # details are repeated verbatim in both fares.txt and extra_baggage_service.txt).
    # Keep the first occurrence; this also halves embedding work and retrieval
    # slots wasted on literal duplicates.
    seen_text: dict[str, dict] = {}
    duplicate_count = 0
    for chunk in chunks:
        key = _dedup_key(chunk["text"])
        existing = seen_text.get(key)
        if existing is None:
            seen_text[key] = chunk
        elif len(chunk["text"]) > len(existing["text"]):
            seen_text[key] = chunk  # keep the more complete version
            duplicate_count += 1
        else:
            duplicate_count += 1
    chunks = list(seen_text.values())
    if duplicate_count:
        print(f"Skipped {duplicate_count} duplicate chunk(s) found across source files.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        print("Collection already exists, skipping")
        return 0, "Collection already exists, skipping"

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "chunk_id": chunk["chunk_id"]} for chunk in chunks]

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks), "ChromaDB collection created"


if __name__ == "__main__":
    embedded_count, status = embed_and_store()

    print(f"How many chunks were embedded: {embedded_count}")
    print(f"Confirmation that ChromaDB collection was created: {status}")
