"""Selection strategies: given ranked (module_id, score) candidates, decide
how many to actually route into context."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

Candidate = tuple[str, float]


class BaseSelector(ABC):
    @abstractmethod
    def select(self, candidates: list[Candidate]) -> list[Candidate]:
        """`candidates` must already be sorted by descending score."""
        raise NotImplementedError


class TopKSelector(BaseSelector):
    """Always return exactly K (or fewer, if there aren't K candidates)."""

    def __init__(self, k: int = 3):
        self.k = k

    def select(self, candidates: list[Candidate]) -> list[Candidate]:
        return candidates[: self.k]


class ThresholdSelector(BaseSelector):
    """Return every candidate at or above a fixed similarity threshold."""

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def select(self, candidates: list[Candidate]) -> list[Candidate]:
        return [(mid, s) for mid, s in candidates if s >= self.threshold]


class AdaptiveKSelector(BaseSelector):
    """The core SIR contribution: pick K by detecting a natural relevance gap.

    Given ranked scores [0.94, 0.89, 0.74, 0.31, 0.28]:
      gaps            = [0.05, 0.15, 0.43, 0.03]
      median gap      = 0.10
      gap_threshold   = gamma * median_gap   (gamma=1.5 -> 0.15)
      first gap > threshold is at index 2 (0.43) -> k* = 3

    This lets different queries pull in different numbers of modules
    instead of forcing a fixed K that over- or under-selects.
    """

    def __init__(self, gamma: float = 1.5, fallback_k: int = 3, min_k: int = 1, max_k: int = 10):
        self.gamma = gamma
        self.fallback_k = fallback_k
        self.min_k = min_k
        self.max_k = max_k

    def select(self, candidates: list[Candidate]) -> list[Candidate]:
        if len(candidates) <= 1:
            return list(candidates)

        scores = [s for _, s in candidates]
        gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

        if not gaps:
            return candidates[: self.fallback_k]

        median_gap = float(np.median(gaps))
        gap_threshold = self.gamma * median_gap if median_gap > 0 else 0.05

        k_star = len(candidates)  # default: no gap found, take everything
        for j, gap in enumerate(gaps):
            if gap > gap_threshold:
                k_star = j + 1
                break

        k_star = max(self.min_k, min(self.max_k, k_star))
        return candidates[:k_star]
