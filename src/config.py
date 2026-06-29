"""
Shared configuration constants for BlueBot's local RAG pipeline.
"""

from __future__ import annotations

from pathlib import Path

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ChromaDB persistent directory in project root (not inside src/)
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Collection where policy chunks + embeddings are stored
COLLECTION_NAME = "airblue_policies"

# Embedding model used for both indexing and retrieval
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Distance-margin retrieval: fetch up to K_MAX candidates, keep those within margin of best
RETRIEVAL_K_MAX = 8
DISTANCE_MARGIN = 0.25

# Retrieval / generation tuning (smaller values = faster on low-end hardware)
#RETRIEVAL_TOP_K = 3
#FARE_LIST_RETRIEVAL_K = 10
LLM_CONTEXT_CHUNKS = 6
MAX_CHUNK_CHARS = 450
OLLAMA_MODEL = "qwen2.5:3b-instruct"
OLLAMA_NUM_CTX = 1024
OLLAMA_NUM_PREDICT = 120
OLLAMA_TIMEOUT_SECONDS = 90
