# Build log — enterprise-rag-eval

A short daily log. Commit one entry per working session — it gives recruiters a
visible, honest story of how the project was built (and helps you resume fast).

## Day 1 — scaffold
- Set up repo structure: `src/`, `eval/`, `notebooks/`, `tests/`, app + api + Docker.
- Implemented: config, pluggable LLM (ollama/groq/gemini), EDGAR downloader,
  ingestion (load→chunk→embed→Chroma), hybrid retriever (dense + BM25 + RRF) with
  optional bge-reranker, generation with citations, guardrails, evaluation harness
  (lightweight retrieval metrics + optional RAGAS), Streamlit UI, FastAPI endpoint.
- TODO next session: install deps, download 2–3 real filings, run first ingest +
  baseline eval, record numbers in `eval/EVALUATION_REPORT.md`.

## Day 2 — first working answer
- Installed lite dependencies in a venv; configured Groq (free) as the LLM backend.
- Downloaded real Apple & Microsoft 10-K filings from public SEC EDGAR.
- Ingested filings into Chroma (switched vector store to cosine similarity).
- Fixed a scoring bug: with the reranker off, final hits now report dense cosine
  similarity instead of tiny RRF fusion scores, so the weak-context guardrail is
  calibrated. First end-to-end answer returned with correct citations.
- Observation: BM25-only hits surface with low semantic score (~0.02); the
  cross-encoder reranker (next) is what filters these out.
- TODO: run the Streamlit app, then install full requirements + turn reranker on.

## Day 3 — _your entry_
