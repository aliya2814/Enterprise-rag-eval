"""Evaluation harness.

Two layers, both using only FREE resources:

1. Retrieval metrics (no LLM): hit-rate and MRR against a golden set. These check
   whether the expected company's filing is retrieved, so they are a routing proxy.

2. Answer-quality metrics via a self-contained LLM-as-judge (RUN_JUDGE=true). For
   each question the judge LLM (your free Groq/Gemini/Ollama backend) scores:
     - faithfulness:      are the answer's claims supported by the retrieved context?
     - context_relevance: does the retrieved context address the question?
     - answer_relevance:  does the answer address the question?
   This needs no RAGAS / OpenAI / VertexAI, so it does not break on version drift.

Usage (Windows):
    py eval\\run_eval.py                          # retrieval metrics only
    set RUN_JUDGE=true & py eval\\run_eval.py     # + answer-quality judge
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings        # noqa: E402
from src.generate import answer        # noqa: E402
from src.llm import complete           # noqa: E402
from src.retriever import Retriever    # noqa: E402

GOLDEN = Path(__file__).parent / "golden_set.jsonl"


def load_golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------- retrieval metrics (no LLM) ----------------
def lightweight_metrics(rows):
    r = Retriever()
    hits_at_k, recip_ranks = [], []
    for row in rows:
        hits = r.search(row["question"])
        company = row.get("company", "").upper()
        ranks = [i for i, h in enumerate(hits)
                 if company and company in h.source.upper()]
        hits_at_k.append(1.0 if ranks else 0.0)
        recip_ranks.append(1.0 / (ranks[0] + 1) if ranks else 0.0)
    return {
        "n": len(rows),
        "reranker": settings.use_reranker,
        "hit_rate@k": round(sum(hits_at_k) / len(rows), 3),
        "mrr": round(sum(recip_ranks) / len(rows), 3),
    }


# ---------------- LLM-as-judge (free, self-contained) ----------------
_JUDGE_SYS = ("You are a strict evaluation judge for a question-answering system. "
              "Reply with ONLY a single number between 0 and 1, no words.")

# An honest "I don't know" is correct grounded behaviour, not a hallucination.
_ABSTAIN_MARKERS = (
    "i don't know", "i do not know", "does not contain", "not explicitly disclose",
    "does not explicitly", "no information", "cannot answer", "can't answer",
    "does not provide", "not mentioned", "not found", "no relevant",
)


def _is_abstention(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ABSTAIN_MARKERS)


def _score(prompt: str) -> float:
    for attempt in range(3):
        try:
            raw = complete(prompt, system=_JUDGE_SYS, temperature=0.0,
                           model=settings.judge_model)
            m = re.search(r"\d*\.?\d+", raw)        # 0.9, .9, 1, 0 ...
            if not m:
                return 0.0
            return max(0.0, min(1.0, float(m.group())))  # clamp to [0,1]
        except Exception as e:
            if attempt == 2:
                print(f"  [judge call failed: {e}]")
                return 0.0
            time.sleep(3)  # back off on rate limit


def judge_metrics(rows):
    faith, ctx, ans = [], [], []
    abstained = 0
    for i, row in enumerate(rows, 1):
        q = row["question"]
        a = answer(q)
        context = "\n\n".join(h.text for h in a.sources) if a.sources else ""

        # Abstention (hard refusal OR an honest "I don't know"): the system
        # declines instead of guessing. It makes NO claims, so faithfulness is
        # trivially 1.0; but it did not answer, so answer_relevance is 0. This is
        # the correct way to score a grounded system that avoids hallucination.
        if a.refused or not context or _is_abstention(a.text):
            abstained += 1
            c = _score(
                f"QUESTION: {q}\n\nCONTEXT:\n{context}\n\n"
                "How relevant is the CONTEXT for answering the QUESTION? "
                "1 = directly relevant, 0 = irrelevant.") if context else 0.0
            faith.append(1.0)
            ctx.append(c)
            ans.append(0.0)
            print(f"  q{i}: ABSTAINED   faithfulness=1.00 context={c:.2f} answer=0.00 | {q[:46]}")
            time.sleep(1)
            continue

        f = _score(
            f"CONTEXT:\n{context}\n\nANSWER:\n{a.text}\n\n"
            "Estimate the FRACTION of factual claims in the ANSWER that are "
            "directly supported by the CONTEXT. Ignore the [source:chunk] "
            "citation tags. Reply 0 to 1 (e.g. 0.8 means 80% of claims supported).")
        c = _score(
            f"QUESTION: {q}\n\nCONTEXT:\n{context}\n\n"
            "How relevant is the CONTEXT for answering the QUESTION? "
            "1 = directly relevant, 0 = irrelevant.")
        an = _score(
            f"QUESTION: {q}\n\nANSWER:\n{a.text}\n\n"
            "Does the ANSWER address the QUESTION? 1 = fully, 0 = not at all.")
        faith.append(f); ctx.append(c); ans.append(an)
        print(f"  q{i}: ANSWERED    faithfulness={f:.2f} context={c:.2f} answer={an:.2f} | {q[:46]}")
        time.sleep(1)

    n = len(rows)
    return {
        "judge_model": settings.judge_model,
        "reranker": settings.use_reranker,
        "abstention_rate": round(abstained / n, 3),
        "faithfulness": round(sum(faith) / n, 3),
        "context_relevance": round(sum(ctx) / n, 3),
        "answer_relevance": round(sum(ans) / n, 3),
    }


def main():
    rows = load_golden()
    print(f"Loaded {len(rows)} golden questions.  (reranker={settings.use_reranker})\n")

    print("== Retrieval metrics ==")
    light = lightweight_metrics(rows)
    print(json.dumps(light, indent=2))

    quality = None
    if os.getenv("RUN_JUDGE", "false").lower() == "true":
        print("\n== Answer-quality metrics (LLM-as-judge) ==")
        quality = judge_metrics(rows)
        print(json.dumps(quality, indent=2))

    out = {"retrieval": light, "quality": quality}
    Path(__file__).parent.joinpath("last_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> eval/last_results.json")


if __name__ == "__main__":
    main()
