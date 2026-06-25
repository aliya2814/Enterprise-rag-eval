"""Retrieval: dense + optional BM25 hybrid + optional cross-encoder reranking.

This module is the heart of the project. The whole point is that you can flip
hybrid/reranker on and off (via config) and MEASURE the difference in eval.
"""
from __future__ import annotations
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from .config import settings


@dataclass
class Hit:
    text: str
    source: str
    chunk: int
    score: float


class Retriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=settings.persist_dir)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embed_model)
        self.coll = client.get_collection(settings.collection, embedding_function=ef)
        self._bm25 = None
        self._corpus = None
        self._reranker = None
        if settings.use_hybrid:
            self._build_bm25()

    # ---- BM25 (lexical) ----
    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        data = self.coll.get(include=["documents", "metadatas"])
        self._corpus = list(zip(data["ids"], data["documents"], data["metadatas"]))
        tokenized = [d.lower().split() for _, d, _ in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _dense(self, query: str, k: int) -> list[Hit]:
        res = self.coll.query(query_texts=[query], n_results=k,
                              include=["documents", "metadatas", "distances"])
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                   res["distances"][0]):
            hits.append(Hit(doc, meta["source"], meta["chunk"], 1.0 - dist))
        return hits

    def _lexical(self, query: str, k: int) -> list[Hit]:
        import numpy as np
        scores = self._bm25.get_scores(query.lower().split())
        top = np.argsort(scores)[::-1][:k]
        out = []
        for i in top:
            _id, doc, meta = self._corpus[i]
            out.append(Hit(doc, meta["source"], meta["chunk"], float(scores[i])))
        return out

    @staticmethod
    def _rrf(lists: list[list[Hit]], k: int = 60) -> list[Hit]:
        """Reciprocal Rank Fusion to merge dense + lexical rankings."""
        scores: dict[tuple, float] = {}
        keep: dict[tuple, Hit] = {}
        for lst in lists:
            for rank, h in enumerate(lst):
                key = (h.source, h.chunk)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                keep[key] = h
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        merged = []
        for key, s in ranked:
            h = keep[key]
            merged.append(Hit(h.text, h.source, h.chunk, s))
        return merged

    def _rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        # Cross-encoder reranker via sentence-transformers (stable across
        # transformers versions; uses the same BAAI/bge-reranker-base weights).
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(settings.reranker_model)
        import numpy as np
        pairs = [[query, h.text] for h in hits]
        logits = np.asarray(self._reranker.predict(pairs), dtype=float)
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid -> 0..1 relevance
        for h, s in zip(hits, scores):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

    def search(self, query: str) -> list[Hit]:
        kd = settings.top_k_dense
        dense = self._dense(query, kd)
        # remember the dense cosine similarity per chunk (a meaningful 0..1-ish
        # relevance number) so the guardrail has a calibrated score to check.
        dense_sim = {(h.source, h.chunk): h.score for h in dense}
        if settings.use_hybrid and self._bm25 is not None:
            lexical = self._lexical(query, kd)
            candidates = self._rrf([dense, lexical])[:kd]
        else:
            candidates = dense
        if settings.use_reranker:
            return self._rerank(query, candidates, settings.top_k_final)
        # no reranker: report each hit's dense cosine similarity (not the tiny
        # RRF fusion number) so scores are interpretable as relevance.
        final = candidates[:settings.top_k_final]
        for h in final:
            h.score = dense_sim.get((h.source, h.chunk), h.score)
        return final
