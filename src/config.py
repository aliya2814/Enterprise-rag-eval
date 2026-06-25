"""Central configuration. Override anything via environment variables (.env)."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Embedding + reranker (both free, run locally on CPU)
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    # Chunking — tune these and back your choice with eval numbers
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 512))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 64))

    # Retrieval
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", 20))   # candidates before rerank
    top_k_final: int = int(os.getenv("TOP_K_FINAL", 5))    # passed to the LLM
    use_hybrid: bool = os.getenv("USE_HYBRID", "true").lower() == "true"
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"

    # Vector store
    persist_dir: str = os.getenv("PERSIST_DIR", "data/processed")
    collection: str = os.getenv("COLLECTION", "filings")

    # LLM backend: "ollama" | "groq" | "gemini"
    llm_backend: str = os.getenv("LLM_BACKEND", "ollama")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    # a STRONGER free model used only for LLM-as-judge evaluation
    judge_model: str = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Guardrail thresholds
    min_context_score: float = float(os.getenv("MIN_CONTEXT_SCORE", 0.20))


settings = Settings()
