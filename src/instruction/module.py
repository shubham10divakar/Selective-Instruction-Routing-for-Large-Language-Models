"""The core unit of the instruction library: a single instructional module."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


def _estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token English average)."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


@dataclass
class InstructionModule:
    """A single instructional module in the library.

    `content` is the actual instruction text that gets injected into an
    LLM's context when this module is routed. `semantic_text` and
    `functional_text` are the two views used by the dual-representation
    encoder (see src/router/encoder.py) — they are deliberately much
    shorter than `content` so routing stays cheap.
    """

    module_id: str
    name: str
    domain: str
    description: str
    capabilities: list[str]
    content: str
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = _estimate_tokens(self.content)

    @property
    def semantic_text(self) -> str:
        """Text used for semantic embedding: what the module is about."""
        return f"{self.name}. {self.description}"

    @property
    def functional_text(self) -> str:
        """Text used for functional embedding: what the module enables."""
        return ", ".join(self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstructionModule":
        return cls(
            module_id=data["module_id"],
            name=data["name"],
            domain=data["domain"],
            description=data["description"],
            capabilities=list(data.get("capabilities", [])),
            content=data["content"],
            token_count=data.get("token_count", 0),
            metadata=dict(data.get("metadata", {})),
        )
