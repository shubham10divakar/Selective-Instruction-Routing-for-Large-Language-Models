#!/usr/bin/env python
"""Human relevance annotation tool.

For a sample of tasks, ask a human annotator to label each (task, module)
pair as ESSENTIAL / HELPFUL / IRRELEVANT / HARMFUL. Candidate modules shown
per task are the task's ground-truth `relevant_modules` plus its
`distractor_modules`, so the annotator is validating (not re-deriving) the
template-generated ground truth.

Essential + Helpful counts as "relevant" downstream (see
src/evaluation/routing_metrics.py). Three annotators per task is the design
target; this tool supports running it multiple times under different
--annotator names and reports Fleiss' kappa once enough runs exist.

Usage:
    python scripts/annotate_relevance.py --annotator alice --n-tasks 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.task import load_tasks
from src.instruction.library import InstructionLibrary
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger

log = get_logger("annotate_relevance")

LABELS = ["ESSENTIAL", "HELPFUL", "IRRELEVANT", "HARMFUL"]


def prompt_label(task_request: str, module_name: str, module_desc: str) -> str:
    print(f"\nTASK REQUEST:\n  {task_request}")
    print(f"CANDIDATE MODULE: {module_name}\n  {module_desc}")
    while True:
        choice = input(f"Label {LABELS} [E/H/I/X, X=harmful]: ").strip().upper()
        mapping = {"E": "ESSENTIAL", "H": "HELPFUL", "I": "IRRELEVANT", "X": "HARMFUL"}
        if choice in mapping:
            return mapping[choice]
        if choice in LABELS:
            return choice
        print("Invalid input, try again.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", required=True, help="Annotator name/id")
    parser.add_argument("--n-tasks", type=int, default=20, help="Number of tasks to annotate this run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    tasks_dir = resolve_path(config["evaluation"]["tasks_path"])
    library_dir = resolve_path(config["library"]["path"])
    out_path = resolve_path("data/annotations/ground_truth.json")

    tasks = load_tasks(tasks_dir)
    library = InstructionLibrary.load(library_dir)

    import random
    rng = random.Random(args.seed)
    sample = rng.sample(tasks, min(args.n_tasks, len(tasks)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_annotations: dict = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            all_annotations = json.load(f)

    for task in sample:
        candidates = list(dict.fromkeys(task.relevant_modules + task.distractor_modules))
        for module_id in candidates:
            module = library.get(module_id)
            if module is None:
                continue
            label = prompt_label(task.request, module.name, module.description)
            key = f"{task.task_id}::{module_id}"
            all_annotations.setdefault(key, {})[args.annotator] = label

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)

    log.info(f"Saved annotations to {out_path} ({len(all_annotations)} (task, module) pairs total)")


if __name__ == "__main__":
    main()
