#!/usr/bin/env python
"""Build the synthetic instruction library.

Default (no flags): generates all 500 modules offline via templates and
writes them to data/instruction_library/<domain>/<module_id>.json.

    python scripts/generate_instructions.py

Use --llm to instead generate via a real LLM (needs an API key configured
for litellm, e.g. OPENAI_API_KEY):

    python scripts/generate_instructions.py --llm --model gpt-4o
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.instruction.generator import TAXONOMY, ASPECTS, generate_library_modules, generate_module_llm
from src.instruction.library import InstructionLibrary
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger

log = get_logger("generate_instructions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="Generate via LLM instead of templates")
    parser.add_argument("--model", default="gpt-4o", help="litellm model string for --llm mode")
    parser.add_argument("--out", default=None, help="Output dir (default: config library.path)")
    args = parser.parse_args()

    config = load_config()
    out_dir = resolve_path(args.out or config["library"]["path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.llm:
        log.info(f"Generating instruction library via LLM ({args.model})...")
        modules = []
        for domain, subjects in TAXONOMY.items():
            for subject in subjects:
                for aspect_key, _title in ASPECTS:
                    topic = f"{subject}_{aspect_key}"
                    log.info(f"  {domain}/{topic}")
                    modules.append(generate_module_llm(domain, topic, model=args.model))
    else:
        log.info("Generating instruction library from offline templates...")
        modules = generate_library_modules()

    library = InstructionLibrary(modules)
    library.save(out_dir)

    stats = library.stats()
    log.info(f"Wrote {stats['n_modules']} modules across {stats['n_domains']} domains to {out_dir}")
    for domain, count in stats["modules_per_domain"].items():
        log.info(f"  {domain}: {count} modules")
    log.info(f"Average token count per module: {stats['avg_token_count']:.0f}")


if __name__ == "__main__":
    main()
