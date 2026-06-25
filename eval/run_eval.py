"""Evaluation harness.

Runs the RAG pipeline over the golden set and reports metrics. Uses RAGAS if
installed/configured; otherwise falls back to lightweight retrieval metrics
(hit-rate, MRR) so the harness ALWAYS produces a number you can put in the README.

Usage:
    python eval/run_eval.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate import answer  # noqa: E402
from src.retriever import Retriever  # noqa: E402

GOLDEN = Path(__file__).parent / "golden_set.jsonl"


def load_golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def lightweight_metrics(rows):
    """Retrieval-quality proxies that need no LLM judge: hit-rate & MRR.

    A 'hit' = a retrieved chunk comes from the expected company's filing.
    Replace/extend with token-overlap to ground_truth for a stronger proxy.
    """
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
        "hit_rate@k": round(sum(hits_at_k) / len(rows), 3),
        "mrr": round(sum(recip_ranks) / len(rows), 3),
    }


def ragas_metrics(rows):
    """Full RAGAS eval (faithfulness, answer relevancy, context precision/recall).
    Requires a judge LLM + embeddings configured for RAGAS. Returns None on failure
    so the harness still completes with lightweight metrics."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                    context_recall, faithfulness)
        records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        for row in rows:
            a = answer(row["question"])
            records["question"].append(row["question"])
            records["answer"].append(a.text)
            records["contexts"].append([h.text for h in a.sources])
            records["ground_truth"].append(row["ground_truth"])
        ds = Dataset.from_dict(records)
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy,
                                       context_precision, context_recall])
        return {k: round(float(v), 3) for k, v in result.items()}
    except Exception as e:
        print(f"[ragas skipped] {e}")
        return None


def main():
    rows = load_golden()
    print(f"Loaded {len(rows)} golden questions.\n")

    print("== Lightweight retrieval metrics ==")
    light = lightweight_metrics(rows)
    print(json.dumps(light, indent=2))

    rag = None
    if os.getenv("RUN_RAGAS", "false").lower() == "true":
        print("\n== RAGAS metrics ==")
        rag = ragas_metrics(rows)
        if rag:
            print(json.dumps(rag, indent=2))

    out = {"lightweight": light, "ragas": rag}
    Path(__file__).parent.joinpath("last_results.json").write_text(
        json.dumps(out, indent=2))
    print("\nSaved -> eval/last_results.json")


if __name__ == "__main__":
    main()
