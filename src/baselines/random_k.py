"""Baseline: select K random modules, ignoring the query entirely.

Used to check that SIR's gains come from relevance-aware selection and not
merely from shrinking the context (a random subset also shrinks context).
"""
from __future__ import annotations

import random

from src.instruction.module import InstructionModule


class RandomKSelector:
    def __init__(self, k: int = 4, seed: int | None = None):
        self.k = k
        self._rng = random.Random(seed)

    def select(self, library: list[InstructionModule]) -> list[InstructionModule]:
        return self._rng.sample(library, min(self.k, len(library)))
