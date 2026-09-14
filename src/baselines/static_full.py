"""Baseline: load every instruction module into context, unconditionally.

This is the "give the LLM everything" baseline that SIR's noise-degradation
experiment (H1) is measured against.
"""
from __future__ import annotations

import tiktoken

from src.instruction.module import InstructionModule


class StaticFullLoader:
    def __init__(self, model: str = "gpt-4o"):
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def load(self, library: list[InstructionModule], budget: int | None = None) -> str:
        sections = [f"[{m.name}]\n{m.content}" for m in library]
        context = "\n\n".join(sections)
        if budget:
            tokens = self.tokenizer.encode(context)[:budget]
            context = self.tokenizer.decode(tokens)
        return context

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if text else 0
