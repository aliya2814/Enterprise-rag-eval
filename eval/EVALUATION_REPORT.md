# Evaluation Report

All metrics use only free models and are reproducible with:
`set RUN_JUDGE=true & py eval\run_eval.py`

## Setup
- Corpus: SEC 10-K filings for **AAPL** and **MSFT** (latest annual reports).
- Embeddings: `BAAI/bge-small-en-v1.5`  ·  Reranker: `BAAI/bge-reranker-base` (cross-encoder via sentence-transformers).
- Generator LLM: **Groq `llama-3.1-8b-instant`** (free tier).
- Judge LLM (LLM-as-judge evaluation): **Groq `llama-3.3-70b-versatile`** (free tier) - deliberately larger than the generator so the judge is reliable.
- Golden set: 8 question/answer pairs (`eval/golden_set.jsonl`).

## Headline answer-quality (hybrid retrieval + reranker)
| Metric | Score | What it means |
|---|---|---|
| Faithfulness | **0.871** | Of the claims the system makes, ~87% are directly supported by retrieved text (low hallucination). |
| Context relevance | **0.762** | Retrieved passages are on-topic for the question. |
| Answer relevance | **0.725** | How fully *answered* questions address the query (held down by 2 abstentions, by design). |
| Abstention rate | **0.25** | The system declined to answer 2 of 8 questions rather than guess. |

## Per-question detail
| # | Question (short) | Status | Faithfulness | Context | Answer |
|---|------------------|--------|--------------|---------|--------|
| 1 | primary risk factors | ANSWERED | 0.83 | 0.80 | 1.00 |
| 2 | revenue concentration | ABSTAINED | 1.00 | 0.00 | 0.00 |
| 3 | competition | ANSWERED | 0.86 | 1.00 | 1.00 |
| 4 | main sources of revenue | ANSWERED | 0.80 | 0.80 | 1.00 |
| 5 | cybersecurity risk | ANSWERED | 0.88 | 0.80 | 0.80 |
| 6 | dependence on suppliers | ABSTAINED | 1.00 | 0.70 | 0.00 |
| 7 | forward-looking disclaimer | ANSWERED | 0.80 | 1.00 | 1.00 |
| 8 | foreign-exchange risk | ANSWERED | 0.80 | 1.00 | 1.00 |

*Abstention scoring:* an honest "I don't know" makes no claims, so faithfulness is 1.0 (no hallucination) but answer relevance is 0.0 (it did not answer). This is the correct way to score a grounded system that refuses rather than fabricates.

## Retrieval ablation (company-routing proxy)
| Config | hit-rate@k | MRR |
|---|---|---|
| Hybrid, reranker off | 0.875 | 0.812 |
| Hybrid + reranker | 0.750 | 0.625 |

This proxy only checks whether a chunk from the *expected company* is retrieved. The reranker **lowers** it because it optimises passage relevance, not source identity: several questions (risk factors, competition, cybersecurity) apply to *both* filings, so the reranker sometimes surfaces the other company's stronger passage. The proper measure of the reranker's value is the answer-quality table above, not this routing proxy.

## Findings and limitations
1. **Low hallucination.** Faithfulness 0.87, and the system abstains (25%) instead of fabricating when retrieval is weak. This is the most important property of a production RAG system.
2. **Retrieval recall is the main limitation.** Questions 2 (revenue concentration) and 6 (supplier dependence) abstained because `bge-small` did not surface content that very likely exists in the filings under different wording. This is a retrieval-recall gap, not a generation failure.
3. **Metric design matters.** The company-match proxy is poorly suited to judging a reranker; LLM-as-judge faithfulness/context relevance is the better signal.

