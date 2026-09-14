#!/usr/bin/env python
"""Turn a results JSONL file (from run_benchmark.py / run_noise_experiment.py
/ run_ablation.py) into a Markdown + LaTeX summary table.

    python scripts/generate_tables.py data/results/benchmark_results.jsonl \
        --group-by strategy --metrics performance,context_tokens,routing_f1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tabulate import tabulate

from src.utils.config import resolve_path
from src.utils.logging_utils import get_logger

log = get_logger("generate_tables")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", help="Path to a results .jsonl file")
    parser.add_argument("--group-by", default="strategy")
    parser.add_argument("--metrics", default=None, help="Comma-separated columns to aggregate (default: all numeric)")
    parser.add_argument("--out-prefix", default=None, help="Write <prefix>.md and <prefix>.tex (default: alongside input)")
    args = parser.parse_args()

    path = resolve_path(args.results_path)
    df = pd.read_json(path, lines=True)
    if df.empty:
        log.warning(f"{path} has no rows")
        return

    if args.metrics:
        metric_cols = args.metrics.split(",")
    else:
        metric_cols = [c for c in df.select_dtypes(include="number").columns]

    summary = df.groupby(args.group_by)[metric_cols].mean().round(3)
    summary["n"] = df.groupby(args.group_by).size()

    md_table = tabulate(summary, headers="keys", tablefmt="github")
    latex_table = tabulate(summary, headers="keys", tablefmt="latex_booktabs")

    print(md_table)

    out_prefix = resolve_path(args.out_prefix) if args.out_prefix else path.with_suffix("")
    md_path = Path(f"{out_prefix}.md")
    tex_path = Path(f"{out_prefix}.tex")
    md_path.write_text(md_table, encoding="utf-8")
    tex_path.write_text(latex_table, encoding="utf-8")
    log.info(f"Wrote {md_path} and {tex_path}")


if __name__ == "__main__":
    main()
