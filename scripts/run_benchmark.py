#!/usr/bin/env python
"""Main experiment runner: strategies x models x tasks.

By default runs fully offline (--offline, the default): a mock LLM backend
and a heuristic judge stand in for real API calls, so the entire pipeline
-- routing, composition, "generation", scoring, metrics -- can be
exercised and tested with zero API keys and zero cost.

For the real experiment reported in the paper, pass --live and a list of
real litellm model strings (needs the corresponding API keys set, e.g.
OPENAI_API_KEY / ANTHROPIC_API_KEY, or a local Ollama server):

    python scripts/run_benchmark.py --live --models gpt-4o,claude-3-5-sonnet-20241022

Quick offline smoke test:

    python scripts/run_benchmark.py --n-tasks 10 --strategies sir_adaptive,static_full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.routing_metrics import routing_metrics
from src.evaluation.strategies import build_default_strategies
from src.evaluation.task import load_tasks
from src.instruction.library import InstructionLibrary
from src.llm.backend import LLMBackend, MockLLMBackend
from src.router.composer import ContextComposer
from src.router.encoder import DualEncoder
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger

log = get_logger("run_benchmark")


def run_one(strategy, model_backend, judge, task, library):
    strategy_result = strategy.build_context(task, library)

    t0 = time.perf_counter()
    response = model_backend.generate(strategy_result.context, task.request)
    generation_ms = (time.perf_counter() - t0) * 1000

    routed_modules = [library.get(mid) for mid in strategy_result.selected_module_ids]
    routed_modules = [m for m in routed_modules if m is not None]

    if hasattr(judge, "score") and judge.__class__.__name__ == "HeuristicJudge":
        score = judge.score(task.request, strategy_result.context[:500], response, task.reference_output, modules=routed_modules)
    else:
        score = judge.score(task.request, strategy_result.context[:500], response, task.reference_output)

    rmetrics = routing_metrics(strategy_result.selected_module_ids, task.relevant_modules)

    return {
        "strategy": strategy.name,
        "model": model_backend.model,
        "task_id": task.task_id,
        "domain": task.domain,
        "complexity": task.complexity,
        "performance": score["overall"],
        "correctness": score["correctness"],
        "completeness": score["completeness"],
        "adherence": score["instruction_adherence"],
        "context_tokens": ContextComposer().count_tokens(strategy_result.context),
        "routing_latency_ms": strategy_result.routing_latency_ms,
        "generation_latency_ms": generation_ms,
        "modules_selected": len(strategy_result.selected_module_ids),
        "routing_precision": rmetrics["precision"],
        "routing_recall": rmetrics["recall"],
        "routing_f1": rmetrics["f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", default=True, help="Use mock backend + heuristic judge (default)")
    parser.add_argument("--live", dest="offline", action="store_false", help="Use real LLM calls via litellm")
    parser.add_argument("--strategies", default=None, help="Comma-separated subset of strategy names (default: all)")
    parser.add_argument("--models", default=None, help="Comma-separated litellm model strings (--live only)")
    parser.add_argument("--judge-model", default=None,
                         help="litellm model string for LLMJudge (--live only; default: config evaluation.judge_model, e.g. gpt-4o). "
                              "Point this at a local Ollama model (e.g. ollama/llama3.1:8b) to judge with no API key.")
    parser.add_argument("--api-base", default=None,
                         help="Base URL for a self-hosted OpenAI-compatible server (vLLM/LM Studio/llama.cpp server) "
                              "serving the --models under test, e.g. http://localhost:8000/v1. Use with --models openai/<name>.")
    parser.add_argument("--judge-api-base", default=None,
                         help="Base URL for a self-hosted OpenAI-compatible server serving --judge-model, if different "
                              "from --api-base (e.g. judging locally while the model under test is hosted, or vice versa).")
    parser.add_argument("--n-tasks", type=int, default=None, help="Cap number of tasks (default: all)")
    parser.add_argument("--out", default="data/results/benchmark_results.jsonl")
    args = parser.parse_args()

    config = load_config()
    library = InstructionLibrary.load(resolve_path(config["library"]["path"]))
    tasks = load_tasks(resolve_path(config["evaluation"]["tasks_path"]))
    if args.n_tasks:
        tasks = tasks[: args.n_tasks]
    log.info(f"Loaded {len(library)} modules, {len(tasks)} tasks")

    encoder_backend = "hashing" if args.offline else "sentence-transformer"
    encoder = DualEncoder(model_name=config["router"]["encoder"]["model"],
                          alpha=config["router"]["encoder"]["alpha"], backend=encoder_backend)
    composer = ContextComposer(budget_tokens=config["composer"]["budget_tokens"])

    all_strategies = build_default_strategies(encoder=encoder, composer=composer)
    if args.strategies:
        names = args.strategies.split(",")
        strategies = {n: all_strategies[n] for n in names}
    else:
        strategies = all_strategies
    log.info(f"Strategies: {list(strategies.keys())}")

    if args.offline:
        model_backends = [MockLLMBackend()]
        from src.evaluation.llm_judge import HeuristicJudge
        judge = HeuristicJudge()
    else:
        model_names = (args.models or ",".join(config["models"])).split(",")
        model_backends = [LLMBackend(m, api_base=args.api_base) for m in model_names]
        judge_model = args.judge_model or config["evaluation"]["judge_model"]
        from src.evaluation.llm_judge import LLMJudge
        judge = LLMJudge(model=judge_model, api_base=args.judge_api_base)
    log.info(f"Models: {[m.model for m in model_backends]} | mode={'offline' if args.offline else 'live'}")
    if not args.offline:
        log.info(f"Judge model: {judge.model}"
                 + (f" | model api_base={args.api_base}" if args.api_base else "")
                 + (f" | judge api_base={args.judge_api_base}" if args.judge_api_base else ""))

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_runs = len(strategies) * len(model_backends) * len(tasks)
    log.info(f"Running {n_runs} evaluations...")

    results = []
    i = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for strategy in strategies.values():
            for model_backend in model_backends:
                for task in tasks:
                    row = run_one(strategy, model_backend, judge, task, library)
                    f.write(json.dumps(row) + "\n")
                    results.append(row)
                    i += 1
                    if i % 50 == 0 or i == n_runs:
                        log.info(f"  {i}/{n_runs}")

    log.info(f"Wrote {len(results)} rows to {out_path}")

    # Quick summary per strategy
    import pandas as pd
    df = pd.DataFrame(results)
    summary = df.groupby("strategy").agg(
        performance=("performance", "mean"),
        context_tokens=("context_tokens", "mean"),
        routing_f1=("routing_f1", "mean"),
        routing_latency_ms=("routing_latency_ms", "mean"),
    ).round(2)
    log.info("\n" + summary.to_string())


if __name__ == "__main__":
    main()
