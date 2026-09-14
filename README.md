# SIR — Selective Instruction Routing

Experimental pipeline for **"Give LLMs What They Need, Not Everything: Selective
Instruction Routing for Large Language Models"** (Subham Divakar, Rojalina
Priyadarshini).

SIR dynamically routes task-relevant instructional modules into an LLM's
context instead of statically loading an entire instruction library. This repo
implements the instruction library, the router (dual-representation encoder +
FAISS index + selection strategies + context composer), baselines, evaluation
pipeline, and the noise/ablation experiments described in `sir-design-doc.md`.

## Setup

```bash
pip install -r requirements.txt
```

Everything below runs **fully offline by default** — no API key required. The
router's default encoder backend still uses a real `sentence-transformers`
model when instantiated directly (`DualEncoder()`), but every script defaults
to a fast, dependency-free hashing backend so the full pipeline, including
tests, runs with zero network calls and zero cost. Pass `--live` to any
experiment script to use real LLMs via [litellm](https://github.com/BerriAI/litellm)
(set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / point at a local Ollama server,
as appropriate for the model string you pass).

## Quickstart

```bash
# 1. Generate the synthetic instruction library (500 modules, 10 domains x 50)
python scripts/generate_instructions.py

# 2. Generate the synthetic evaluation benchmark (480 tasks, 6 domains x 80)
python scripts/generate_tasks.py

# 3. Run the tests (fast, offline, no model downloads)
pytest tests/ -v

# 4. Run a quick offline benchmark smoke test
python scripts/run_benchmark.py --n-tasks 20 --strategies sir_adaptive,static_full,oracle

# 5. Run the noise-degradation experiment (H1/H2) offline
python scripts/run_noise_experiment.py --n-tasks 20

# 6. Run routing-quality ablations (encoding alpha, gap gamma)
python scripts/run_ablation.py --n-tasks 60

# 7. Turn a results file into a Markdown/LaTeX table
python scripts/generate_tables.py data/results/benchmark_results.jsonl --group-by strategy
```

## Project layout

```
config/settings.yaml          All hyperparameters (encoder, selector, budgets, models, noise levels)

data/
  instruction_library/<domain>/<module_id>.json   The 500-module library, one file per module
  tasks/<domain>_tasks.json                        The 480-task benchmark, one file per eval domain
  annotations/ground_truth.json                    Human relevance labels (via scripts/annotate_relevance.py)
  results/                                         Experiment output (gitignored except .gitkeep)

src/
  instruction/    InstructionModule, InstructionLibrary, synthetic generator (offline + optional LLM mode)
  router/         DualEncoder, InstructionIndex (FAISS), selectors (TopK/Threshold/AdaptiveK), ContextComposer, SIRRouter
  baselines/      StaticFullLoader, RandomKSelector, BM25Router, NoInstructionsLoader
  evaluation/     routing_metrics (P/R/F1), LLMJudge (+ offline HeuristicJudge), AdherenceScorer,
                  EvalTask + loader, synthetic task_generator, the Strategy comparison matrix
  llm/            LLMBackend (litellm-backed) + MockLLMBackend (offline stand-in)

scripts/          One script per pipeline stage (see Quickstart above)
tests/            pytest suite, entirely offline (hashing encoder backend, no API calls)
```

Both `data/instruction_library/` and `data/tasks/` are organized so more
content can be dropped in later without touching code: add a new domain
folder under `instruction_library/` (or new `*.json` module files inside an
existing one) and it's picked up automatically by `InstructionLibrary.load()`;
add a new `<domain>_tasks.json` under `tasks/` and it's picked up by
`load_tasks()`.

## Synthetic data generation — two modes

**Offline (default).** `src/instruction/generator.py` and
`src/evaluation/task_generator.py` build the library and benchmark
deterministically from a domain -> subject -> aspect taxonomy (10 domains x
10 subjects x 5 aspects = 500 modules; 6 eval domains x 80 tasks = 480 tasks),
assembling instruction content from parametrized phrase banks. No API key,
no network call, fully reproducible (seeded per module/task id).

**LLM-backed (opt-in).** `generate_module_llm()` mirrors the design doc's
`GENERATION_PROMPT` and calls a real LLM via litellm for higher-fidelity
module content:

```bash
python scripts/generate_instructions.py --llm --model gpt-4o
```

## Running the real experiment (--live)

`--live` swaps `MockLLMBackend` → `LLMBackend` (real litellm calls) and
`HeuristicJudge` → `LLMJudge` (real GPT-4o-as-judge scoring per the design
doc's three-dimension rubric: correctness, completeness, instruction
adherence). It also switches the router's encoder backend from the offline
hashing embedder to the real `sentence-transformers` model in
`config/settings.yaml` (`all-MiniLM-L6-v2` by default).

`LLMBackend` is a thin wrapper over `litellm.completion()`, so it is
model-agnostic — any model string litellm understands works, hosted or
local. `config/settings.yaml`'s `models:` list is just the example set used
by the full benchmark sweep; any individual script call can target a
different model via `--models` / `--model`.

### Hosted models (OpenAI, Anthropic, ...)

Set the provider's API key as an environment variable, then pass its
litellm model string.

**Linux / macOS:**
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

python scripts/run_benchmark.py --live --models gpt-4o,claude-3-5-sonnet-20241022
python scripts/run_noise_experiment.py --live --model gpt-4o
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."

python scripts/run_benchmark.py --live --models gpt-4o,claude-3-5-sonnet-20241022
python scripts/run_noise_experiment.py --live --model gpt-4o
```
(`set OPENAI_API_KEY=sk-...` if using `cmd.exe` instead of PowerShell.)

The judge (`LLMJudge`) also needs a key for whichever model
`config/settings.yaml` → `evaluation.judge_model` points at (`gpt-4o` by
default) — set that provider's key even if the model under test is
something else (e.g. a local Ollama model), or repoint `judge_model` at a
model you do have a key for.

### Local models via Ollama (Llama, Qwen, Mistral, ...)

No API key needed — litellm talks to Ollama's local server
(`http://localhost:11434` by default) for any `ollama/<model>` string.

**1. Install Ollama**

Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Windows: download the installer from [ollama.com](https://ollama.com) or
```powershell
winget install Ollama.Ollama
```
It installs as a background service on both platforms — no separate
`ollama serve` step needed unless you stopped it.

**2. Pull a model** (same command on both OSes)
```bash
ollama pull llama3.1:8b     # Llama
ollama pull qwen2.5:7b      # Qwen
ollama pull mistral:7b      # Mistral
```

**3. Verify it's running**
```bash
ollama list
curl http://localhost:11434/api/tags
```

**4. Run a small live smoke test first**

Don't start with the full task set or the `static_full` strategy: with all
500 modules loaded uncapped (`static_full` has no token budget by default)
it composes ~175k tokens of context, which blows past Ollama's default
context window (2k-4k tokens unless you raise `num_ctx`). Start narrow:

```bash
python scripts/run_benchmark.py --live --models ollama/llama3.1:8b \
  --n-tasks 5 --strategies sir_adaptive,sir_top3,oracle
```

(swap `llama3.1:8b` for `qwen2.5:7b` / `mistral:7b` / whatever tag you
pulled — same `ollama/<tag>` prefix either way). If you don't have
`OPENAI_API_KEY` set for the judge, either export it (see above) or edit
`evaluation.judge_model` in `config/settings.yaml` to `ollama/llama3.1:8b`
so judging runs on Ollama too (weaker signal — a model judging its own
output — but needs nothing but Ollama to run end-to-end).

Once that's confirmed working, widen `--n-tasks` and add `static_full`
back in — but raise Ollama's context window first:
```bash
OLLAMA_CONTEXT_LENGTH=32768 ollama serve      # Linux, foreground
```
```powershell
$env:OLLAMA_CONTEXT_LENGTH = "32768"; ollama serve   # Windows, foreground
```
(or set it once via `ollama run <model> --keepalive ...` / the Ollama
desktop app settings, depending on your install, so you don't have to
relaunch the service manually each time).

## Human annotation

```bash
python scripts/annotate_relevance.py --annotator alice --n-tasks 20
```

Labels each (task, candidate module) pair as ESSENTIAL / HELPFUL / IRRELEVANT
/ HARMFUL; run under multiple `--annotator` names to build the 3-annotator
ground-truth set described in the design doc, then compute inter-annotator
agreement over `data/annotations/ground_truth.json`.

## Key design decisions

See `sir-design-doc.md` §8 for the full rationale (zero-shot embedding
routing over classification, dual representation over raw content embeddings,
adaptive-K over fixed-K, flat FAISS over HNSW at this scale). The one
deviation from the doc worth calling out: this implementation adds an
offline hashing-embedder fallback and mock LLM/judge path everywhere the doc
assumed a live API, specifically so the whole pipeline — generation, routing,
noise sweep, ablations, tests — is runnable and testable without any external
service.
