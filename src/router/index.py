"""FAISS-backed vector index over the instruction library."""
from __future__ import annotations

import faiss
import numpy as np


class InstructionIndex:
    """Exact (flat) inner-product index. At ~500 modules exact search is
    sub-millisecond, so there is no reason to trade determinism for the
    approximate search (HNSW) that only pays off at 10k+ scale."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.module_ids: list[str] = []

    def build(self, embeddings: np.ndarray, module_ids: list[str]) -> None:
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.module_ids = list(module_ids)

    def search(self, query_embedding: np.ndarray, top_n: int = 20) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        query = np.ascontiguousarray(query_embedding.reshape(1, -1).astype("float32"))
        faiss.normalize_L2(query)
        top_n = min(top_n, self.index.ntotal)
        scores, indices = self.index.search(query, top_n)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.module_ids):
                results.append((self.module_ids[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    def load(self, path: str, module_ids: list[str]) -> None:
        self.index = faiss.read_index(path)
        self.module_ids = list(module_ids)
