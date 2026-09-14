"""End-to-end routing tests: encoder + index + selector + composer wired
together via SIRRouter, plus the routing-quality metrics used to score it.
Uses the offline hashing backend throughout so no network/model download
is required.
"""
from src.evaluation.routing_metrics import routing_f1, routing_precision, routing_recall
from src.instruction.library import InstructionLibrary
from src.router.encoder import DualEncoder
from src.router.router import SIRRouter
from src.router.selector import AdaptiveKSelector, TopKSelector


# -- routing_metrics -----------------------------------------------------

def test_routing_precision_perfect_match():
    assert routing_precision(["a", "b"], ["a", "b"]) == 1.0


def test_routing_precision_partial_match():
    assert routing_precision(["a", "b", "c"], ["a"]) == 1 / 3


def test_routing_precision_empty_selection():
    assert routing_precision([], ["a"]) == 0.0


def test_routing_recall_perfect_match():
    assert routing_recall(["a", "b"], ["a", "b"]) == 1.0


def test_routing_recall_no_relevant_modules_is_vacuously_perfect():
    assert routing_recall(["a", "b"], []) == 1.0


def test_routing_f1_zero_when_disjoint():
    assert routing_f1(["a"], ["b"]) == 0.0


def test_routing_f1_matches_harmonic_mean():
    # selected={a,b,c}, relevant={a,b}: precision=2/3, recall=1.0
    f1 = routing_f1(["a", "b", "c"], ["a", "b"])
    expected = 2 * (2 / 3) * 1.0 / (2 / 3 + 1.0)
    assert abs(f1 - expected) < 1e-9


# -- SIRRouter end-to-end --------------------------------------------------

def test_sir_router_returns_composed_context(sample_modules):
    library = InstructionLibrary(sample_modules)
    router = SIRRouter(
        library,
        encoder=DualEncoder(backend="hashing"),
        selector=TopKSelector(k=2),
    )
    result = router.route("write a hypothesis test in python")
    assert result.k_selected == 2
    assert len(result.selected_modules) == 2
    assert result.composed_context  # non-empty
    assert result.routing_latency_ms >= 0


def test_sir_router_adaptive_selects_at_least_one(sample_modules):
    library = InstructionLibrary(sample_modules)
    router = SIRRouter(
        library,
        encoder=DualEncoder(backend="hashing"),
        selector=AdaptiveKSelector(gamma=1.5),
    )
    result = router.route("how do I manage secrets safely in production?")
    assert 1 <= result.k_selected <= len(sample_modules)


def test_sir_router_oracle_style_precision_on_clear_query(sample_modules):
    """A query specifically about secrets management should route the
    secrets management module among its top picks."""
    library = InstructionLibrary(sample_modules)
    router = SIRRouter(
        library,
        encoder=DualEncoder(backend="hashing"),
        selector=TopKSelector(k=1),
    )
    result = router.route("secrets management credentials encryption security")
    assert result.selected_module_ids == ["security_secrets_management_best_practices"]
