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

## Day 3 — reranker, evaluation, and honest error analysis
- Added a cross-encoder reranker (bge-reranker-base via sentence-transformers, after
  hitting a FlagEmbedding/tokenizer bug and switching libraries).
- Replaced fragile RAGAS with a self-contained LLM-as-judge (free Groq 70B as judge),
  measuring faithfulness, context relevance, answer relevance + abstention rate.
- Final (hybrid + reranker): faithfulness 0.871, context 0.762, answer 0.725,
  abstention 0.25. Retrieval proxy: 0.875/0.812 (reranker off) vs 0.750/0.625 (on).
- Per-question analysis exposed 2 abstentions (revenue concentration, supplier
  dependence) where retrieval missed content that likely exists -> documented as a
  retrieval-recall limitation with concrete next steps.
- Fixed an eval bug: honest "I don't know" was being scored as unfaithful; abstentions
  now score faithfulness 1.0 / answer 0.0 (no claims = no hallucination).
- Next: stronger embeddings + query expansion to lift recall; expand golden set; deploy.
