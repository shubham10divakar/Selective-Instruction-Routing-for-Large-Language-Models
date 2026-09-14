#!/usr/bin/env python
"""Build the synthetic evaluation-task benchmark.

Generates 80 tasks (40 simple + 40 compound) per evaluation domain --
480 tasks total -- and writes data/tasks/<domain>_tasks.json.

    python scripts/generate_tasks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.task import save_tasks
from src.evaluation.task_generator import generate_all_tasks
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger

log = get_logger("generate_tasks")


def main() -> None:
    config = load_config()
    out_dir = resolve_path(config["evaluation"]["tasks_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating evaluation task benchmark from offline templates...")
    tasks_by_domain = generate_all_tasks()

    total = 0
    for domain, tasks in tasks_by_domain.items():
        save_tasks(tasks, out_dir, domain)
        n_simple = sum(1 for t in tasks if t.complexity == "simple")
        n_compound = sum(1 for t in tasks if t.complexity == "compound")
        log.info(f"  {domain}: {len(tasks)} tasks ({n_simple} simple, {n_compound} compound)")
        total += len(tasks)

    log.info(f"Wrote {total} tasks across {len(tasks_by_domain)} domains to {out_dir}")


if __name__ == "__main__":
    main()
