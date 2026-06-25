# Enterprise RAG + Evaluation

Ask plain-English questions about real company annual reports (SEC 10-K filings) and get answers that are grounded in the source text, with citations to the exact section used. Built to show a retrieval pipeline that is measured and guarded, not just a chatbot wrapper.

**Status:** core pipeline working and evaluated (hit-rate 0.88, MRR 0.81 on a held-out question set). Reranker ablation, RAGAS answer-quality scoring, and a hosted live demo are in progress (see Roadmap).

## What it does

Example, straight from the running app:

> **Q:** What are Apple's main business risk factors?
>
> **A:** New business strategies and ventures are inherently risky and may not succeed `[AAPL_10K:210]`. Tariffs and retaliatory trade measures may materially affect operations `[AAPL_10K:105, 116]`. Supply shortages and price increases can result from these measures `[AAPL_10K:155]`.

Every claim points back to the chunk it came from. If the documents do not contain the answer, the system says so instead of inventing one (for example, it refuses "what is the recipe for pizza?").

## Why SEC filings as the dataset

I deliberately avoided toy datasets like Titanic or Iris. SEC 10-K reports are real, public business documents available with no API key, and they are long and citation-heavy, which is exactly where retrieval quality and faithfulness actually matter. The pipeline is corpus-agnostic, so the same code works on policies, contracts, or product manuals.

## Key features

- **Hybrid retrieval:** combines dense semantic search (embeddings) with BM25 keyword search, merged using Reciprocal Rank Fusion.
- **Calibrated relevance scoring:** results are scored by cosine similarity so a single threshold can decide when context is too weak to answer.
- **Guardrails:** refuses on weak retrieval, blocks basic prompt-injection patterns, and the generation prompt forbids using outside knowledge.
- **Evaluation harness:** scores retrieval against a golden question set (hit-rate, MRR), with optional RAGAS metrics for answer faithfulness.
- **Two interfaces:** a Streamlit app that shows the retrieved evidence behind each answer, and a FastAPI endpoint for programmatic use.
- **Reproducible:** pinned requirements, Dockerfile, and unit tests for the guardrail logic.

## Architecture

```
documents (HTML/PDF)
      |
   ingest:  load -> clean -> chunk -> embed -> Chroma (cosine)
      |
question --> hybrid retrieve (dense + BM25, RRF) --> optional rerank --> top-k
      |
   guardrails --> LLM generate (answer + citations) --> response
      |
   evaluation harness: hit-rate, MRR, (RAGAS: faithfulness, context precision/recall)
```

## Results

Measured on an 8-question golden set (`eval/golden_set.jsonl`). Numbers are reproducible with `python eval/run_eval.py`.

| Configuration | hit-rate@k | MRR | Faithfulness |
|---|---|---|---|
| Hybrid retrieval (reranker off) | 0.875 | 0.812 | pending |
| Hybrid + cross-encoder reranker | in progress | in progress | in progress |

## Engineering notes

A few decisions and one real bug worth calling out, since they shaped the design:

- **Scoring bug I hit and fixed.** My weak-context guardrail was rejecting every answer. The cause: with the reranker disabled, final results carried tiny Rank-Fusion scores (around 0.02) instead of a meaningful relevance value, so they always fell below the refusal threshold. Fix: store the dense cosine similarity per chunk and report that as the relevance score. After the fix, relevant chunks score 0.7 to 0.8 and the guardrail behaves correctly.
- **Cosine over L2.** The vector store uses cosine distance so a single similarity threshold is interpretable across queries.
- **Two layers of safety.** Retrieval-level (refuse on weak context) and generation-level (the prompt forbids answering from outside the provided context). The pizza refusal passes the retrieval layer but is correctly stopped at the generation layer.

## Quickstart

```bash
# 1. install (lite set for first run; full set adds reranker + RAGAS)
pip install -r requirements-min.txt

# 2. download a few real filings (free, public EDGAR)
set EDGAR_UA=Your Name your@email.com
python src/download_filings.py --tickers AAPL MSFT --out data/raw

# 3. ingest into the vector store
python -m src.ingest --input data/raw --persist data/processed

# 4. configure a free LLM backend (Groq shown)
set LLM_BACKEND=groq
set GROQ_API_KEY=your_key
set USE_RERANKER=false

# 5. ask a question, run the eval, or launch the app
python -m src.generate --q "What are Apple's main business risk factors?"
python eval/run_eval.py
streamlit run app.py
```

LLM backend is pluggable and every option has a free path: Ollama (local, offline), Groq (free tier), or Gemini (free tier).

## Tech stack

Python, LangChain, ChromaDB, sentence-transformers (bge-small-en), rank-bm25, bge-reranker, RAGAS, FastAPI, Streamlit, Docker, pytest.

## Repo layout

```
src/    ingest, retriever (hybrid + rerank), generate, guardrails, llm, config, download_filings
eval/   golden_set.jsonl, run_eval.py, EVALUATION_REPORT.md
app.py  Streamlit UI (answer + retrieved evidence)
api.py  FastAPI endpoint
tests/  guardrail unit tests
```

## Roadmap

- [ ] Turn on the cross-encoder reranker and record the measured lift vs the baseline above
- [ ] Add RAGAS faithfulness and context precision/recall to the results table
- [ ] Run a chunk-size ablation (256 / 512 / 1024) and keep the best
- [ ] Deploy a hosted demo (Hugging Face Spaces) and link it at the top
- [ ] Short demo video

## License

MIT
