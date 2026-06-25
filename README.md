# enterprise-rag-eval — Grounded Document Q&A with Retrieval Evaluation & Guardrails

A production-style Retrieval-Augmented Generation (RAG) system over real **SEC 10-K filings**, built to demonstrate not just a working chatbot but a **measurable, evaluated, guarded** retrieval pipeline.

> The RAG pipeline itself is commodity in 2026. What this project shows is the part hiring teams actually care about: a **hybrid retriever + reranker**, a **RAGAS evaluation harness** with a golden Q&A set, **guardrails** against hallucination/prompt-injection, and a **deployed live demo**.

---

## Why this dataset
Public SEC 10-K annual reports are: (1) **real** business documents (not Titanic/Iris), (2) **freely available** via the public EDGAR system with no API key, (3) long, structured, and citation-heavy — exactly the kind of corpus where retrieval quality and faithfulness actually matter. The same pipeline works on any folder of PDFs/text (policies, manuals, contracts), so it generalizes to enterprise use.

## Architecture
```
PDFs/HTML ─► ingest (load ► chunk ► embed) ─► vector store (Chroma)
                                                     │
question ─► hybrid retrieve (BM25 + dense) ─► rerank (bge-reranker) ─► top-k
                                                     │
                              guardrails ◄─ generate (LLM + citations) ─► answer + sources
                                                     │
                                   eval harness (RAGAS): faithfulness, context P/R, answer relevancy
```

## Tech stack
Python · LangChain · ChromaDB · sentence-transformers (`bge-small-en`) · `bge-reranker-base` · `rank_bm25` · RAGAS · FastAPI · Streamlit · Docker · pytest. LLM is pluggable: **Ollama** (local, free), **Groq** (free tier), or **Gemini** (free tier).

## Results (fill in after you run the eval)
| Config | Faithfulness | Context Precision | Context Recall | Answer Relevancy |
|--------|-------------|-------------------|----------------|------------------|
| Hybrid, reranker off | hit_rate@k **0.875** · MRR **0.812** | (run RAGAS next) | — | — |
| + Hybrid + Reranker    | _ | _ | _ | _ |

## Quickstart
```bash
pip install -r requirements.txt

# 1. Download a few real 10-K filings (free, public EDGAR)
python src/download_filings.py --tickers AAPL MSFT --out data/raw

# 2. Ingest into the vector store
python -m src.ingest --input data/raw --persist data/processed

# 3. Ask a question from the CLI
python -m src.generate --q "What are Apple's main business risk factors?"

# 4. Run the evaluation harness
python eval/run_eval.py

# 5. Launch the UI / API
streamlit run app.py
uvicorn api:app --reload
```

## Repo layout
```
src/        ingest, retriever (hybrid+rerank), generate, guardrails, llm, config, download_filings
eval/       golden_set.jsonl  +  run_eval.py  +  EVALUATION_REPORT.md
notebooks/  01_ingestion.ipynb, 02_evaluation.ipynb
app.py      Streamlit UI (shows retrieved chunks + citations)
api.py      FastAPI endpoint
tests/      pytest smoke + guardrail tests
Dockerfile  container for reproducible runs
```

## Design decisions (write these as you go — this is what makes it senior)
- **Chunking:** _why this size/overlap (back it with an eval number)._
- **Hybrid vs dense:** _the measured delta._
- **Reranker:** _latency vs precision trade-off._
- **Guardrails:** _how "I don't know" is triggered when context is weak._

## Limitations & next steps
_Be honest here — stating what doesn't work yet reads as maturity._

## License
MIT
