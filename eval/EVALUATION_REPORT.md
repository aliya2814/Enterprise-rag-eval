# Evaluation Report

> Fill this in after running `python eval/run_eval.py`. This file is the single
> most important page in the repo for a recruiter — it proves you measure, not guess.

## Setup
- Corpus: SEC 10-K filings for **AAPL, MSFT** (latest annual reports).
- Embedding model: `BAAI/bge-small-en-v1.5`
- Reranker: `BAAI/bge-reranker-base`
- LLM generator: **Groq** (`llama-3.1-8b-instant`), free tier
- Golden set: `eval/golden_set.jsonl` (8 Q&A pairs)

## Day-2 baseline result (hybrid retrieval, reranker off)
- **hit_rate@k = 0.875** — the correct company's filing appeared in the top-k for 7 of 8 questions.
- **MRR = 0.812** — on average the first correct hit sat very near the top of the list.
- Next: install the reranker + run the same 8 questions to measure the lift, and add RAGAS faithfulness.

## Ablation: what each component buys you
Run the harness 3 times, toggling env vars, and record the numbers:

| Config | hit_rate@k | MRR | Faithfulness | Ctx Precision | Ctx Recall | Answer Rel. |
|--------|-----------|-----|--------------|---------------|------------|-------------|
| Dense only (`USE_HYBRID=false USE_RERANKER=false`) | | | | | | |
| + Hybrid (`USE_HYBRID=true USE_RERANKER=false`)    | 0.875 | 0.812 | _ | _ | _ | _ |
| + Hybrid + Reranker (both true)                    | | | | | | |

## Chunking experiment
| chunk_size / overlap | hit_rate@k | Faithfulness | Notes |
|----------------------|-----------|--------------|-------|
| 256 / 32 | | | |
| 512 / 64 | | | |
| 1024 / 128 | | | |

**Decision:** _state which you chose and why, citing the numbers above._

## Failure cases found
1. _question → what went wrong → fix._
2. _..._

## Takeaways
_2–3 sentences: what the evaluation taught you about this pipeline._
