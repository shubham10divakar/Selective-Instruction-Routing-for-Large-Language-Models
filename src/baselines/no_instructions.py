"""Baseline: no instruction context at all -- the floor of the comparison."""
from __future__ import annotations


class NoInstructionsLoader:
    def load(self, *_args, **_kwargs) -> str:
        return ""
