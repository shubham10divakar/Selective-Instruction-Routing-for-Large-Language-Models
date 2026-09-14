"""Token counting and latency tracking utilities used by the experiment runner."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import tiktoken


@dataclass
class ResourceUsage:
    context_tokens: int = 0
    routing_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return self.routing_latency_ms + self.generation_latency_ms


class ResourceMonitor:
    def __init__(self, model: str = "gpt-4o"):
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._marks: dict[str, float] = {}

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if text else 0

    @contextmanager
    def timer(self):
        """Usage: with monitor.timer() as t: ... ; t['ms'] holds elapsed time after the block."""
        result: dict[str, float] = {}
        t0 = time.perf_counter()
        try:
            yield result
        finally:
            result["ms"] = (time.perf_counter() - t0) * 1000
