"""The strategy comparison matrix used by scripts/run_benchmark.py and
scripts/run_noise_experiment.py: everything from "no instructions" to
"oracle" implements the same `build_context(task, library)` interface so
the experiment runner can loop over them uniformly.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.baselines.bm25_router import BM25Router
from src.baselines.no_instructions import NoInstructionsLoader
from src.baselines.random_k import RandomKSelector
from src.baselines.static_full import StaticFullLoader
from src.instruction.library import InstructionLibrary
from src.router.composer import ContextComposer
from src.router.encoder import DualEncoder
from src.router.router import SIRRouter
from src.router.selector import AdaptiveKSelector, BaseSelector, ThresholdSelector, TopKSelector
from .task import EvalTask


@dataclass
class StrategyResult:
    context: str
    selected_module_ids: list[str]
    routing_latency_ms: float


class Strategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def build_context(self, task: EvalTask, library: InstructionLibrary) -> StrategyResult: ...


class NoInstructionsStrategy(Strategy):
    name = "no_instructions"

    def __init__(self):
        self._loader = NoInstructionsLoader()

    def build_context(self, task: EvalTask, library: InstructionLibrary) -> StrategyResult:
        return StrategyResult(context="", selected_module_ids=[], routing_latency_ms=0.0)


class StaticFullStrategy(Strategy):
    name = "static_full"

    def __init__(self, budget_tokens: int | None = None, model: str = "gpt-4o"):
        self._loader = StaticFullLoader(model=model)
        self.budget_tokens = budget_tokens

    def build_context(self, task: EvalTask, library: InstructionLibrary) -> StrategyResult:
        modules = library.all()
        t0 = time.perf_counter()
        context = self._loader.load(modules, budget=self.budget_tokens)
        latency = (time.perf_counter() - t0) * 1000
        return StrategyResult(context, [m.module_id for m in modules], latency)


class RandomKStrategy(Strategy):
    name = "random_k"

    def __init__(self, k: int = 4, seed: int | None = 42, composer: ContextComposer | None = None):
        self._selector = RandomKSelector(k=k, seed=seed)
        self._composer = composer or ContextComposer()

    def build_context(self, task: EvalTask, library: InstructionLibrary) -> StrategyResult:
        t0 = time.perf_counter()
        modules = self._selector.select(library.all())
        context = self._composer.compose(modules, [1.0] * len(modules))
        latency = (time.perf_counter() - t0) * 1000
        return StrategyResult(context, [m.module_id for m in modules], latency)


class BM25Strategy(Strategy):
    name = "bm25"

    def __init__(self, top_k: int = 5, composer: ContextComposer | None = None):
        self.top_k = top_k
        self._composer = composer or ContextComposer()
        self._router_cache: dict[frozenset, BM25Router] = {}

    def _router_for(self, library: InstructionLibrary) -> BM25Router:
        key = frozenset(library.all() and [m.module_id for m in library.all()])
        router = self._router_cache.get(key)
        if router is None:
            router = BM25Router()
            router.build(library.all())
            self._router_cache[key] = router
        return router

    def build_context(self, task: EvalTask, library: InstructionLibrary) -> StrategyResult:
        router = self._router_for(library)
        t0 = time.perf_counter()
        ranked = router.route(task.request, top_k=self.top_k)
        modules = [m for m, _ in ranked]
        scores = [s for _, s in ranked]
        context = self._composer.compose(modules, scores)
        latency = (time.perf_counter() - t0) * 1000
        return StrategyResult(context, [m.module_id for m in modules], latency)


class SIRStrategy(Strategy):
    """The SIR router itself, parametrized by a selection strategy
    (top-k / threshold / adaptive-k)."""

    def __init__(
        self,
        selector: BaseSelector,
        name: str = "sir",
        encoder: DualEncoder | None = None,
        composer: ContextComposer | None = None,
        top_n_candidates: int = 20,
    ):
        self.name = name
        self.selector = selector
        self._encoder = encoder or DualEncoder()
        self._composer = composer or ContextComposer()
        self.top_n_candidates = top_n_candidates
        self._router_cache: dict[frozenset, SIRRouter] = {}

    def _router_for(self, library: InstructionLibrary) -> SIRRouter:
        key = frozenset(m.module_id for m in library.all())
        router = self._router_cache.get(key)
        if router is None:
            router = SIRRouter(
                library,
                encoder=self._encoder,
                selector=self.selector,
                composer=self._composer,
                top_n_candidates=self.top_n_candidates,
            )
            self._router_cache[key] = router
        return router

    def build_context(self, task: EvalTask, library: InstructionLibrary) -> StrategyResult:
        router = self._router_for(library)
        result = router.route(task.request)
        return StrategyResult(result.composed_context, result.selected_module_ids, result.routing_latency_ms)


class OracleStrategy(Strategy):
    """Cheats by reading the task's ground-truth relevant_modules directly.
    Upper bound for how well any router could do on this benchmark."""

    name = "oracle"

    def __init__(self, composer: ContextComposer | None = None):
        self._composer = composer or ContextComposer()

    def build_context(self, task: EvalTask, library: InstructionLibrary) -> StrategyResult:
        t0 = time.perf_counter()
        modules = [library.get(mid) for mid in task.relevant_modules]
        modules = [m for m in modules if m is not None]
        context = self._composer.compose(modules, [1.0] * len(modules))
        latency = (time.perf_counter() - t0) * 1000
        return StrategyResult(context, [m.module_id for m in modules], latency)


def build_default_strategies(encoder: DualEncoder | None = None, composer: ContextComposer | None = None) -> dict[str, Strategy]:
    """The full strategy matrix from the design doc's STRATEGIES dict."""
    encoder = encoder or DualEncoder()
    composer = composer or ContextComposer()
    return {
        "no_instructions": NoInstructionsStrategy(),
        "static_full": StaticFullStrategy(),
        "random_k": RandomKStrategy(k=4),
        "bm25": BM25Strategy(top_k=5, composer=composer),
        "sir_top3": SIRStrategy(TopKSelector(k=3), name="sir_top3", encoder=encoder, composer=composer),
        "sir_top5": SIRStrategy(TopKSelector(k=5), name="sir_top5", encoder=encoder, composer=composer),
        "sir_threshold": SIRStrategy(ThresholdSelector(threshold=0.70), name="sir_threshold", encoder=encoder, composer=composer),
        "sir_adaptive": SIRStrategy(AdaptiveKSelector(gamma=1.5), name="sir_adaptive", encoder=encoder, composer=composer),
        "oracle": OracleStrategy(composer=composer),
    }
