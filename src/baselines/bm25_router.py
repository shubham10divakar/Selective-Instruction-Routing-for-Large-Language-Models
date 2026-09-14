"""Baseline: BM25 keyword routing over (name, description, capabilities).

The non-semantic counterpart to SIR's embedding router -- tests whether
routing needs semantic similarity at all, or whether keyword overlap
already gets most of the way there.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.instruction.module import InstructionModule


class BM25Router:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.modules: list[InstructionModule] = []

    def build(self, library: list[InstructionModule]) -> None:
        self.modules = library
        corpus = [f"{m.name} {m.description} {' '.join(m.capabilities)}" for m in library]
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def route(self, query: str, top_k: int = 5) -> list[tuple[InstructionModule, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25Router.build() must be called before route()")
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.modules[idx], float(score)) for idx, score in ranked[:top_k]]
