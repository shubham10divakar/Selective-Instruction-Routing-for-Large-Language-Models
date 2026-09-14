#!/usr/bin/env python
"""Noise degradation experiment (H1 / H2).

For each task and each noise level eta, build a library containing that
task's relevant modules plus enough randomly-sampled irrelevant modules to
reach the target noise ratio, then compare Static-Full against SIR
Adaptive-K on that library. Static-Full is expected to degrade as eta
grows (H1); SIR should stay comparatively flat (H2) since it filters the
noise back out regardless of how much was added.

Offline by default (mock backend + heuristic judge) -- see run_benchmark.py
for the same offline/--live split.

    python scripts/run_noise_experiment.py --n-tasks 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.routing_metrics import routing_metrics
from src.evaluation.strategies import SIRStrategy, StaticFullStrategy
from src.evaluation.task import load_tasks
from src.instruction.library import InstructionLibrary
from src.llm.backend import LLMBackend, MockLLMBackend
from src.router.composer import ContextComposer
from src.router.encoder import DualEncoder
from src.router.selector import AdaptiveKSelector
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger

log = get_logger("run_noise_experiment")


def build_noisy_library(task, full_library: InstructionLibrary, noise_level: float, rng: random.Random) -> InstructionLibrary:
    relevant = [m for m in full_library.all() if m.module_id in task.relevant_modules]
    irrelevant = [m for m in full_library.all() if m.module_id not in task.relevant_modules]

    if not relevant:
        relevant = [full_library.all()[0]]

    if noise_level <= 0:
        selected = relevant
    else:
        n_relevant = len(relevant)
        n_total = max(n_relevant + 1, int(round(n_relevant / (1 - noise_level))))
        n_irrelevant = min(n_total - n_relevant, len(irrelevant))
        selected_irrelevant = rng.sample(irrelevant, n_irrelevant)
        selected = relevant + selected_irrelevant
        rng.shuffle(selected)

    return InstructionLibrary(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--live", dest="offline", action="store_false")
    parser.add_argument("--model", default=None, help="litellm model string (--live only)")
    parser.add_argument("--judge-model", default=None,
                         help="litellm model string for LLMJudge (--live only; default: config evaluation.judge_model, e.g. gpt-4o). "
                              "Point this at a local Ollama model (e.g. ollama/llama3.1:8b) to judge with no API key.")
    parser.add_argument("--n-tasks", type=int, default=30, help="Number of tasks to sample")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per (task, noise_level)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/results/noise_experiment.jsonl")
    args = parser.parse_args()

    config = load_config()
    full_library = InstructionLibrary.load(resolve_path(config["library"]["path"]))
    tasks = load_tasks(resolve_path(config["evaluation"]["tasks_path"]))

    rng = random.Random(args.seed)
    tasks = rng.sample(tasks, min(args.n_tasks, len(tasks)))
    noise_levels = config["noise"]["levels"]

    encoder_backend = "hashing" if args.offline else "sentence-transformer"
    encoder = DualEncoder(model_name=config["router"]["encoder"]["model"],
                          alpha=config["router"]["encoder"]["alpha"], backend=encoder_backend)
    composer = ContextComposer(budget_tokens=config["composer"]["budget_tokens"])

    static_strategy = StaticFullStrategy()
    sir_strategy = SIRStrategy(AdaptiveKSelector(gamma=config["router"]["selector"]["gamma"]),
                                name="sir_adaptive", encoder=encoder, composer=composer)

    if args.offline:
        model_backend = MockLLMBackend()
        from src.evaluation.llm_judge import HeuristicJudge
        judge = HeuristicJudge()
    else:
        model_backend = LLMBackend(args.model or config["models"][0])
        judge_model = args.judge_model or config["evaluation"]["judge_model"]
        from src.evaluation.llm_judge import LLMJudge
        judge = LLMJudge(model=judge_model)
        log.info(f"Model: {model_backend.model} | Judge model: {judge.model}")

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_runs = len(tasks) * len(noise_levels) * args.repeats * 2  # x2 strategies
    log.info(f"Running {n_runs} evaluations across {len(noise_levels)} noise levels...")

    i = 0
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            for noise_level in noise_levels:
                for rep in range(args.repeats):
                    rep_rng = random.Random(f"{task.task_id}_{noise_level}_{rep}_{args.seed}")
                    library = build_noisy_library(task, full_library, noise_level, rep_rng)

                    for strategy in (static_strategy, sir_strategy):
                        t0 = time.perf_counter()
                        result = strategy.build_context(task, library)
                        response = model_backend.generate(result.context, task.request)

                        routed_modules = [library.get(mid) for mid in result.selected_module_ids]
                        routed_modules = [m for m in routed_modules if m is not None]
                        if hasattr(judge, "score") and judge.__class__.__name__ == "HeuristicJudge":
                            score = judge.score(task.request, result.context[:500], response,
                                                 task.reference_output, modules=routed_modules)
                        else:
                            score = judge.score(task.request, result.context[:500], response, task.reference_output)

                        rmetrics = routing_metrics(result.selected_module_ids, task.relevant_modules)
                        row = {
                            "task_id": task.task_id,
                            "noise_level": noise_level,
                            "repeat": rep,
                            "strategy": strategy.name,
                            "performance": score["overall"],
                            "context_tokens": composer.count_tokens(result.context),
                            "n_library_modules": len(library),
                            "routing_precision": rmetrics["precision"],
                            "routing_recall": rmetrics["recall"],
                            "routing_f1": rmetrics["f1"],
                            "latency_ms": (time.perf_counter() - t0) * 1000,
                        }
                        f.write(json.dumps(row) + "\n")
                        rows.append(row)
                        i += 1
                        if i % 100 == 0 or i == n_runs:
                            log.info(f"  {i}/{n_runs}")

    log.info(f"Wrote {len(rows)} rows to {out_path}")

    import pandas as pd
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="noise_level", columns="strategy", values="performance", aggfunc="mean").round(2)
    log.info("\nMean performance by noise level:\n" + pivot.to_string())


if __name__ == "__main__":
    main()
