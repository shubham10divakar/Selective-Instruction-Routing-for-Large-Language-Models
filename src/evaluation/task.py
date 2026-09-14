"""Evaluation task data structure and loader.

Tasks live at data/tasks/<domain>_tasks.json (one JSON array per domain) so
more domains or tasks can be dropped in later without touching this code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class EvalTask:
    """A single evaluation task in the SIR benchmark."""

    task_id: str
    domain: str
    domains: list[str]
    complexity: str  # "simple" | "compound"
    request: str
    reference_output: str
    relevant_modules: list[str]
    distractor_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalTask":
        return cls(
            task_id=data["task_id"],
            domain=data["domain"],
            domains=list(data.get("domains", [data["domain"]])),
            complexity=data.get("complexity", "simple"),
            request=data["request"],
            reference_output=data["reference_output"],
            relevant_modules=list(data.get("relevant_modules", [])),
            distractor_modules=list(data.get("distractor_modules", [])),
            metadata=dict(data.get("metadata", {})),
        )


def load_tasks(path: str | Path) -> list[EvalTask]:
    """Load every *_tasks.json file under `path` into a flat list of EvalTask."""
    root = Path(path)
    tasks: list[EvalTask] = []
    for json_path in sorted(root.glob("*_tasks.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks.extend(EvalTask.from_dict(t) for t in data)
    return tasks


def save_tasks(tasks: list[EvalTask], path: str | Path, domain: str) -> None:
    """Write all tasks for one domain to data/tasks/<domain>_tasks.json."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / f"{domain}_tasks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in tasks], f, indent=2, ensure_ascii=False)
