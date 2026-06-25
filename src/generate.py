"""Answer generation: retrieve -> guardrail -> prompt LLM -> cited answer.

Usage:
    python -m src.generate --q "What are Apple's main risk factors?"
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass

from . import guardrails
from .llm import complete
from .retriever import Retriever

SYSTEM = (
    "You are a precise financial-document assistant. Answer ONLY from the provided "
    "context. Every claim must cite its source as [source:chunk]. If the context "
    "does not contain the answer, say you don't know. Do not use outside knowledge."
)


def _format_context(hits) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[{h.source}:{h.chunk}]\n{h.text}")
    return "\n\n---\n\n".join(blocks)


@dataclass
class Answer:
    text: str
    sources: list
    refused: bool = False


_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def answer(question: str) -> Answer:
    if guardrails.is_injection(question):
        return Answer(guardrails.refusal_message("possible prompt injection"), [], True)

    hits = _get_retriever().search(question)
    if guardrails.context_is_weak(hits):
        return Answer(guardrails.refusal_message("no strongly relevant context found"),
                     hits, True)

    prompt = (f"Context:\n{_format_context(hits)}\n\n"
              f"Question: {question}\n\n"
              "Answer with inline [source:chunk] citations:")
    text = complete(prompt, system=SYSTEM)
    return Answer(text, hits)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    args = ap.parse_args()
    a = answer(args.q)
    print("\n=== ANSWER ===\n" + a.text)
    print("\n=== SOURCES ===")
    for h in a.sources:
        print(f"  [{h.source}:{h.chunk}] score={h.score:.3f}")
