"""SIRRouter: ties encoder + index + selector + composer into one call.

This is the object baselines/strategies and experiment scripts import --
`SIRRouter(library, selector=...).route(query)` returns a RoutingResult.
"""
from __future__ import annotations

import time

from src.instruction.library import InstructionLibrary
from .composer import ContextComposer, RoutingResult
from .encoder import DualEncoder
from .index import InstructionIndex
from .selector import BaseSelector, TopKSelector


class SIRRouter:
    def __init__(
        self,
        library: InstructionLibrary,
        encoder: DualEncoder | None = None,
        selector: BaseSelector | None = None,
        composer: ContextComposer | None = None,
        top_n_candidates: int = 20,
    ):
        self.library = library
        self.encoder = encoder or DualEncoder()
        self.selector = selector or TopKSelector(k=3)
        self.composer = composer or ContextComposer()
        self.top_n_candidates = top_n_candidates

        self._modules = library.all()
        embeddings = self.encoder.encode_library(self._modules)
        self.index = InstructionIndex(dim=self.encoder.dim)
        self.index.build(embeddings, [m.module_id for m in self._modules])

    def route(self, query: str) -> RoutingResult:
        t0 = time.perf_counter()

        query_embedding = self.encoder.encode_query(query)
        candidates = self.index.search(query_embedding, top_n=self.top_n_candidates)
        selected = self.selector.select(candidates)

        modules = [self.library.get(mid) for mid, _ in selected]
        scores = [s for _, s in selected]
        modules_scores = [(m, s) for m, s in zip(modules, scores) if m is not None]
        modules = [m for m, _ in modules_scores]
        scores = [s for _, s in modules_scores]

        composed = self.composer.compose(modules, scores)
        latency_ms = (time.perf_counter() - t0) * 1000

        return RoutingResult(
            query=query,
            selected_modules=modules,
            scores=scores,
            strategy=type(self.selector).__name__,
            k_selected=len(modules),
            routing_latency_ms=latency_ms,
            composed_context=composed,
            composed_token_count=self.composer.count_tokens(composed),
        )
