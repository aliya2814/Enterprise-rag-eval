"""Ingestion: load docs -> clean -> chunk -> embed -> persist in Chroma.

Usage:
    python -m src.ingest --input data/raw --persist data/processed
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


def _load_text(path: Path) -> str:
    if path.suffix.lower() in {".html", ".htm"}:
        from langchain_community.document_loaders import BSHTMLLoader
        docs = BSHTMLLoader(str(path), open_encoding="utf-8").load()
        text = "\n".join(d.page_content for d in docs)
    elif path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    # collapse whitespace common in filings
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text))


def build(input_dir: str, persist_dir: str) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embed_model)
    # fresh collection each ingest for reproducibility
    try:
        client.delete_collection(settings.collection)
    except Exception:
        pass
    coll = client.create_collection(
        settings.collection, embedding_function=ef,
        metadata={"hnsw:space": "cosine"})  # cosine -> distance in [0,2]

    ids, docs, metas = [], [], []
    for path in sorted(Path(input_dir).glob("*")):
        if path.is_dir():
            continue
        text = _load_text(path)
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{path.stem}-{i}")
            docs.append(chunk)
            metas.append({"source": path.name, "chunk": i})

    # add in batches (Chroma has a max batch size)
    B = 256
    for s in range(0, len(docs), B):
        coll.add(ids=ids[s:s+B], documents=docs[s:s+B], metadatas=metas[s:s+B])
    print(f"Ingested {len(docs)} chunks from {input_dir} -> {persist_dir}")
    return len(docs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--persist", default=settings.persist_dir)
    args = ap.parse_args()
    build(args.input, args.persist)
