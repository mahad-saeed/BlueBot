"""
FastAPI backend for BlueBot. Wraps src.pipeline.ask() in a REST API.
Run from project root: uvicorn src.api:app --reload
"""

import time

from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import ask

app = FastAPI(title="BlueBot API")


class HistoryTurn(BaseModel):
    query: str
    answer: str
    is_relevant: bool
    
class ChatRequest(BaseModel):
    query: str
    history: list[HistoryTurn] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    is_relevant: bool
    response_time: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    history_dicts = [turn.model_dump() for turn in request.history]
    print(f"[api] history received: {history_dicts}")
    result = ask(request.query, history=history_dicts)
    elapsed = time.perf_counter() - started
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        is_relevant=result["is_relevant"],
        response_time=elapsed,
    )