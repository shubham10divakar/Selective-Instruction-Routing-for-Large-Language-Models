"""Load and manage the on-disk instruction library.

Modules live at data/instruction_library/<domain>/<module_id>.json so new
domains or modules can be dropped in later without touching any code here.
"""
from __future__ import annotations

import json
from pathlib import Path

from .module import InstructionModule


class InstructionLibrary:
    """In-memory view over the instruction module JSON files on disk."""

    def __init__(self, modules: list[InstructionModule] | None = None):
        self._modules: dict[str, InstructionModule] = {}
        for m in modules or []:
            self._modules[m.module_id] = m

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "InstructionLibrary":
        """Load every *.json module file under `path` (recursing into domain subfolders)."""
        root = Path(path)
        modules = []
        for json_path in sorted(root.glob("**/*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            modules.append(InstructionModule.from_dict(data))
        return cls(modules)

    def save(self, path: str | Path) -> None:
        """Write every module back out as <domain>/<module_id>.json."""
        root = Path(path)
        for module in self._modules.values():
            domain_dir = root / module.domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            out_path = domain_dir / f"{module.module_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(module.to_dict(), f, indent=2, ensure_ascii=False)

    # -- collection interface -----------------------------------------------

    def add(self, module: InstructionModule) -> None:
        self._modules[module.module_id] = module

    def get(self, module_id: str) -> InstructionModule | None:
        return self._modules.get(module_id)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __contains__(self, module_id: str) -> bool:
        return module_id in self._modules

    def all(self) -> list[InstructionModule]:
        return list(self._modules.values())

    def by_domain(self, domain: str) -> list[InstructionModule]:
        return [m for m in self._modules.values() if m.domain == domain]

    def domains(self) -> list[str]:
        return sorted({m.domain for m in self._modules.values()})

    def stats(self) -> dict:
        domains = self.domains()
        return {
            "n_modules": len(self._modules),
            "n_domains": len(domains),
            "modules_per_domain": {d: len(self.by_domain(d)) for d in domains},
            "avg_token_count": (
                sum(m.token_count for m in self._modules.values()) / len(self._modules)
                if self._modules
                else 0
            ),
        }
