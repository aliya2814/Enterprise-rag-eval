"""FastAPI endpoint. Run: uvicorn api:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.generate import answer

app = FastAPI(title="enterprise-rag-eval", version="0.1.0")


class Query(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    chunk: int
    score: float


class Response(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Response)
def ask(q: Query):
    a = answer(q.question)
    return Response(
        answer=a.text,
        refused=a.refused,
        sources=[Source(source=h.source, chunk=h.chunk, score=h.score)
                 for h in a.sources],
    )
