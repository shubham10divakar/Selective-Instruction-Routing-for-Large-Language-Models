"""Dual-representation encoder for instruction modules and queries.

Each module is embedded twice -- once from its (name, description) as the
"semantic" view, once from its capability keyword list as the "functional"
view -- and the two are combined with weight `alpha`. Raw content embeddings
are noisy for long modules (a 2000-token module compressed into a 384-dim
vector loses most of its signal); name+description captures *topic* while
capabilities capture *function*, and both matter for accurate routing.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from src.instruction.module import InstructionModule


class TextEmbedder(Protocol):
    """Minimal interface both backends below satisfy."""

    def encode(self, text: str) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Downloads weights on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)


class HashingEmbedder:
    """Deterministic, dependency-free embedder for offline dev/tests/CI.

    Hashes word n-grams into a fixed-size vector (the "hashing trick").
    Not semantically strong, but it is fast, requires no model download,
    and is good enough to exercise routing logic (ranking, gap detection,
    dedup) without needing network access.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype="float32")
        tokens = text.lower().split()
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        if not grams:
            return vec
        for gram in grams:
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class DualEncoder:
    """Encodes instruction modules and queries with dual representation."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 0.6,
        backend: str = "sentence-transformer",
    ):
        self.alpha = alpha
        self.backend_name = backend
        if backend == "hashing":
            self._embedder: TextEmbedder = HashingEmbedder()
        else:
            self._embedder = SentenceTransformerEmbedder(model_name)
        self.dim = self._embedder.dim

    def encode_module(self, module: InstructionModule) -> np.ndarray:
        """Combine semantic + functional views into one embedding."""
        sem_emb = self._embedder.encode(module.semantic_text)
        func_emb = self._embedder.encode(module.functional_text)

        combined = self.alpha * sem_emb + (1 - self.alpha) * func_emb
        norm = np.linalg.norm(combined)
        return combined / norm if norm > 0 else combined

    def encode_query(self, query: str) -> np.ndarray:
        return self._embedder.encode(query)

    def encode_library(self, modules: list[InstructionModule]) -> np.ndarray:
        """Batch-encode all modules. Returns an (N, dim) float32 matrix."""
        return np.array([self.encode_module(m) for m in modules], dtype="float32")
