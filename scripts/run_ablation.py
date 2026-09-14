#!/usr/bin/env python
"""Ablation studies over routing quality (precision/recall/F1 against
ground-truth relevant_modules) -- no generation or judging needed, so
these run fast and fully offline by default.

  1. Encoding strategy: semantic-only (alpha=1.0), functional-only
     (alpha=0.0), and dual at alpha in {0.4, 0.5, 0.6, 0.7, 0.8}.
  2. Gap sensitivity gamma for AdaptiveKSelector: {1.0, 1.25, 1.5, 2.0, 2.5}.
  3. Embedding model comparison (all-MiniLM-L6-v2 / all-mpnet-base-v2 /
     e5-small-v2) -- opt-in via --embedding-models, since it downloads
     real sentence-transformers weights.

    python scripts/run_ablation.py --which alpha,gamma --n-tasks 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.routing_metrics import routing_metrics
from src.evaluation.task import load_tasks
from src.instruction.library import InstructionLibrary
from src.router.encoder import DualEncoder
from src.router.router import SIRRouter
from src.router.selector import AdaptiveKSelector, TopKSelector
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger

log = get_logger("run_ablation")


def evaluate_router(router: SIRRouter, tasks) -> dict:
    precisions, recalls, f1s, ks = [], [], [], []
    for task in tasks:
        result = router.route(task.request)
        m = routing_metrics(result.selected_module_ids, task.relevant_modules)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])
        ks.append(result.k_selected)
    n = max(len(tasks), 1)
    return {
        "precision": round(sum(precisions) / n, 4),
        "recall": round(sum(recalls) / n, 4),
        "f1": round(sum(f1s) / n, 4),
        "avg_k": round(sum(ks) / n, 2),
    }


def run_alpha_ablation(library, tasks, backend: str, model_name: str) -> list[dict]:
    log.info("Ablation 1: encoding strategy (alpha sweep)")
    rows = []
    configs = [("functional_only", 0.0), ("alpha_0.4", 0.4), ("alpha_0.5", 0.5),
               ("dual_default_0.6", 0.6), ("alpha_0.7", 0.7), ("alpha_0.8", 0.8),
               ("semantic_only", 1.0)]
    for label, alpha in configs:
        encoder = DualEncoder(model_name=model_name, alpha=alpha, backend=backend)
        router = SIRRouter(library, encoder=encoder, selector=TopKSelector(k=3))
        metrics = evaluate_router(router, tasks)
        log.info(f"  {label:20s} alpha={alpha}  -> {metrics}")
        rows.append({"ablation": "alpha", "config": label, "alpha": alpha, **metrics})
    return rows


def run_gamma_ablation(library, tasks, backend: str, model_name: str) -> list[dict]:
    log.info("Ablation 2: gap sensitivity gamma sweep")
    rows = []
    encoder = DualEncoder(model_name=model_name, alpha=0.6, backend=backend)
    for gamma in [1.0, 1.25, 1.5, 2.0, 2.5]:
        router = SIRRouter(library, encoder=encoder, selector=AdaptiveKSelector(gamma=gamma))
        metrics = evaluate_router(router, tasks)
        log.info(f"  gamma={gamma}  -> {metrics}")
        rows.append({"ablation": "gamma", "config": f"gamma_{gamma}", "gamma": gamma, **metrics})
    return rows


def run_embedding_model_ablation(library, tasks) -> list[dict]:
    log.info("Ablation 3: embedding model comparison (downloads weights)")
    rows = []
    for model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "e5-small-v2"]:
        encoder = DualEncoder(model_name=model_name, alpha=0.6, backend="sentence-transformer")
        router = SIRRouter(library, encoder=encoder, selector=TopKSelector(k=3))
        metrics = evaluate_router(router, tasks)
        log.info(f"  {model_name}  -> {metrics}")
        rows.append({"ablation": "embedding_model", "config": model_name, **metrics})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", default="alpha,gamma", help="Comma list of: alpha,gamma,embedding_model")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--backend", default="hashing", choices=["hashing", "sentence-transformer"],
                         help="Encoder backend (hashing = fast/offline, default)")
    parser.add_argument("--out", default="data/results/ablation_results.jsonl")
    args = parser.parse_args()

    config = load_config()
    library = InstructionLibrary.load(resolve_path(config["library"]["path"]))
    tasks = load_tasks(resolve_path(config["evaluation"]["tasks_path"]))
    if args.n_tasks:
        tasks = tasks[: args.n_tasks]
    log.info(f"Loaded {len(library)} modules, {len(tasks)} tasks, backend={args.backend}")

    model_name = config["router"]["encoder"]["model"]
    which = set(args.which.split(","))
    rows = []
    if "alpha" in which:
        rows += run_alpha_ablation(library, tasks, args.backend, model_name)
    if "gamma" in which:
        rows += run_gamma_ablation(library, tasks, args.backend, model_name)
    if "embedding_model" in which:
        rows += run_embedding_model_ablation(library, tasks)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    log.info(f"Wrote {len(rows)} ablation rows to {out_path}")


if __name__ == "__main__":
    main()
