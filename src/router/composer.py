"""Assembles selected instruction modules into the final context string,
enforcing a token budget."""
from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken

from src.instruction.module import InstructionModule


@dataclass
class RoutingResult:
    """Full output of a routing pass -- what the router picked and why."""

    query: str
    selected_modules: list[InstructionModule]
    scores: list[float]
    strategy: str
    k_selected: int
    routing_latency_ms: float
    composed_context: str
    composed_token_count: int

    @property
    def selected_module_ids(self) -> list[str]:
        return [m.module_id for m in self.selected_modules]


class ContextComposer:
    """Orders selected modules by relevance and truncates to fit a token budget."""

    def __init__(self, budget_tokens: int = 8000, model: str = "gpt-4o"):
        self.budget = budget_tokens
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def compose(self, modules: list[InstructionModule], scores: list[float]) -> str:
        paired = sorted(zip(modules, scores), key=lambda x: x[1], reverse=True)

        sections: list[str] = []
        total_tokens = 0

        for module, _score in paired:
            content = module.content
            content_tokens = len(self.tokenizer.encode(content))

            if total_tokens + content_tokens > self.budget:
                remaining = self.budget - total_tokens
                if remaining > 50:
                    tokens = self.tokenizer.encode(content)[:remaining]
                    content = self.tokenizer.decode(tokens)
                    sections.append(self._format_section(module.name, content))
                break

            sections.append(self._format_section(module.name, content))
            total_tokens += content_tokens

        return "\n\n".join(sections)

    @staticmethod
    def _format_section(name: str, content: str) -> str:
        return f"[{name}]\n{content}"

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if text else 0
