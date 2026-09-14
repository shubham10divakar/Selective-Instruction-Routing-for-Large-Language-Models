"""Instruction adherence scoring.

Two modes:
  - `LLMJudge.score(...)["instruction_adherence"]` is the primary metric
    used in the paper (requires an API key).
  - `AdherenceScorer` below is a lightweight, offline heuristic: fraction of
    a routed module's capability keywords that surface in the response.
    Useful as a fast sanity check / for tests without an LLM call.
"""
from __future__ import annotations

import re

from src.instruction.module import InstructionModule


class AdherenceScorer:
    def score(self, response: str, modules: list[InstructionModule]) -> float:
        """Return the fraction of routed modules' capability keywords that
        appear (as whole words, case-insensitive) anywhere in the response."""
        if not modules:
            return 1.0  # nothing was asked of the response

        response_lower = response.lower()
        total_keywords = 0
        hit_keywords = 0
        for module in modules:
            for keyword in module.capabilities:
                total_keywords += 1
                pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
                if re.search(pattern, response_lower):
                    hit_keywords += 1

        return hit_keywords / total_keywords if total_keywords else 1.0
