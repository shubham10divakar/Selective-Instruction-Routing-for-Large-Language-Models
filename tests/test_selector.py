from src.router.selector import AdaptiveKSelector, ThresholdSelector, TopKSelector


CANDIDATES = [
    ("a", 0.95),
    ("b", 0.92),
    ("c", 0.88),
    ("d", 0.48),
    ("e", 0.46),
]


def test_topk_selector_returns_exactly_k():
    selector = TopKSelector(k=3)
    result = selector.select(CANDIDATES)
    assert [mid for mid, _ in result] == ["a", "b", "c"]


def test_topk_selector_handles_fewer_candidates_than_k():
    selector = TopKSelector(k=10)
    result = selector.select(CANDIDATES)
    assert len(result) == len(CANDIDATES)


def test_threshold_selector_filters_by_score():
    selector = ThresholdSelector(threshold=0.70)
    result = selector.select(CANDIDATES)
    assert [mid for mid, _ in result] == ["a", "b", "c"]


def test_threshold_selector_can_return_empty():
    selector = ThresholdSelector(threshold=0.99)
    assert selector.select(CANDIDATES) == []


def test_adaptive_k_finds_the_gap():
    """Gaps: [0.03, 0.04, 0.40, 0.02]; median=0.035; threshold(gamma=1.5)=0.0525.
    First gap > 0.0525 is at index 2 (0.40) -> k*=3."""
    selector = AdaptiveKSelector(gamma=1.5, min_k=1, max_k=10)
    result = selector.select(CANDIDATES)
    assert [mid for mid, _ in result] == ["a", "b", "c"]


def test_adaptive_k_respects_min_and_max():
    flat_candidates = [("a", 0.5), ("b", 0.5), ("c", 0.5), ("d", 0.5)]
    selector = AdaptiveKSelector(gamma=1.5, min_k=2, max_k=3)
    result = selector.select(flat_candidates)
    assert 2 <= len(result) <= 3


def test_adaptive_k_single_candidate():
    selector = AdaptiveKSelector()
    assert selector.select([("a", 0.9)]) == [("a", 0.9)]


def test_adaptive_k_empty_candidates():
    selector = AdaptiveKSelector()
    assert selector.select([]) == []


def test_adaptive_k_higher_gamma_selects_more():
    """A larger gamma raises the gap threshold, making it harder for any
    gap to trigger a cut -- so k* should never shrink as gamma grows."""
    strict = AdaptiveKSelector(gamma=1.0, min_k=1, max_k=10)
    permissive = AdaptiveKSelector(gamma=3.0, min_k=1, max_k=10)
    assert len(permissive.select(CANDIDATES)) >= len(strict.select(CANDIDATES))
