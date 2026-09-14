# SIR — Selective Instruction Routing: Design & Implementation Guide

**Paper**: "Give LLMs What They Need, Not Everything: Selective Instruction Routing for Large Language Models"  
**Authors**: Subham Divakar, Rojalina Priyadarshini  
**Status**: Experiment-ready design doc

---

## 1. Project Overview

### 1.1 What We're Building
A complete experimental pipeline to validate Selective Instruction Routing (SIR) — a framework that dynamically selects task-relevant instructional modules for LLMs instead of loading everything into context.

### 1.2 Central Hypothesis
- **H1**: As irrelevant instructional context increases, LLM task performance and instruction adherence degrade.
- **H2**: Selectively loading task-relevant instructions through an embedding-based router mitigates this degradation.
- **H3**: Selective routing reduces token usage while maintaining or improving task performance.

### 1.3 Key Differentiator
SIR is **not** RAG for documents. It's routing for **instructions** — the system prompt content that governs LLM behaviour. The instruction modules are authored content (skills, policies, workflows), not retrieved knowledge passages.

---

## 2. Project Structure

```
sir/
├── config/
│   └── settings.yaml                # All hyperparameters
├── data/
│   ├── instruction_library/          # The 500 instruction modules
│   │   ├── coding/
│   │   │   ├── python_conventions.json
│   │   │   ├── code_review.json
│   │   │   └── ...
│   │   ├── statistics/
│   │   ├── research/
│   │   ├── writing/
│   │   ├── data_analysis/
│   │   ├── debugging/
│   │   ├── testing/
│   │   ├── security/
│   │   ├── database/
│   │   └── deployment/
│   ├── tasks/                        # Evaluation benchmark
│   │   ├── coding_tasks.json
│   │   ├── statistics_tasks.json
│   │   ├── research_tasks.json
│   │   ├── writing_tasks.json
│   │   ├── data_analysis_tasks.json
│   │   └── debugging_tasks.json
│   ├── annotations/                  # Human relevance labels
│   │   └── ground_truth.json
│   └── results/                      # Experiment outputs
├── src/
│   ├── __init__.py
│   ├── instruction/
│   │   ├── __init__.py
│   │   ├── module.py                 # InstructionModule dataclass
│   │   ├── library.py                # InstructionLibrary manager
│   │   └── generator.py             # Synthetic instruction generator
│   ├── router/
│   │   ├── __init__.py
│   │   ├── encoder.py                # Dual-representation encoder
│   │   ├── index.py                  # FAISS vector index wrapper
│   │   ├── selector.py               # Top-K, threshold, adaptive-K
│   │   └── composer.py               # Context composition
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── static_full.py            # Load-everything baseline
│   │   ├── random_k.py               # Random selection baseline
│   │   └── bm25_router.py            # BM25 keyword routing
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── llm_judge.py              # GPT-4o-as-judge scorer
│   │   ├── adherence.py              # Instruction adherence scorer
│   │   ├── routing_metrics.py        # Precision/Recall/F1 for routing
│   │   └── resource_monitor.py       # Token counting, latency
│   ├── llm/
│   │   ├── __init__.py
│   │   └── backend.py                # Unified LLM interface (litellm)
│   └── utils/
│       ├── __init__.py
│       └── logging_utils.py
├── scripts/
│   ├── generate_instructions.py      # Build the 500-module library
│   ├── generate_tasks.py             # Build the 480-task benchmark
│   ├── annotate_relevance.py         # Tool for human annotation
│   ├── run_benchmark.py              # Main experiment runner
│   ├── run_noise_experiment.py       # Noise degradation sweep
│   ├── run_ablation.py               # Encoding ablation
│   └── generate_tables.py            # LaTeX table generation
├── notebooks/
│   ├── 01_explore_instructions.ipynb
│   ├── 02_routing_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── tests/
│   ├── test_encoder.py
│   ├── test_selector.py
│   ├── test_composer.py
│   └── test_routing_quality.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 3. Core Data Structures

### 3.1 Instruction Module

```python
# src/instruction/module.py
from dataclasses import dataclass, field

@dataclass
class InstructionModule:
    """A single instructional module in the library."""
    module_id: str                          # e.g., "coding_python_conventions"
    name: str                               # e.g., "Python Coding Conventions"
    domain: str                             # e.g., "coding"
    description: str                        # 1-line purpose summary
    capabilities: list[str]                 # Functional keyword list
    content: str                            # Full instruction text (200-2000 tokens)
    token_count: int = 0                    # Cached token count
    metadata: dict = field(default_factory=dict)

    @property
    def semantic_text(self) -> str:
        """Text used for semantic embedding."""
        return f"{self.name}. {self.description}"
    
    @property
    def functional_text(self) -> str:
        """Text used for functional embedding."""
        return ", ".join(self.capabilities)
```

### 3.2 Evaluation Task

```python
@dataclass
class EvalTask:
    """A single evaluation task."""
    task_id: str
    domain: str                             # Primary domain
    domains: list[str]                      # All relevant domains (for compound tasks)
    complexity: str                         # "simple" or "compound"
    request: str                            # The user query/request
    reference_output: str                   # Gold-standard response
    relevant_modules: list[str]             # Ground-truth module IDs
```

### 3.3 Routing Result

```python
@dataclass
class RoutingResult:
    """Output of the SIR router."""
    query: str
    selected_modules: list[InstructionModule]
    scores: list[float]                     # Similarity scores
    strategy: str                           # "top_k", "threshold", "adaptive"
    k_selected: int                         # Number of modules selected
    routing_latency_ms: float               # Router overhead
    composed_context: str                   # Final assembled context
    composed_token_count: int               # Token count of composed context
```

---

## 4. Implementation — Phase by Phase

### Phase 1: Instruction Library Construction (Days 1-3)

Build 500 instruction modules across 10 domains (50 per domain).

```python
# scripts/generate_instructions.py
"""
Generate the instruction library.
Each module has:
  - Clear name and description
  - 5-15 capability keywords
  - 200-2000 tokens of instruction content
  
Strategy: Use an LLM to generate realistic instruction modules,
then manually curate for quality and diversity.
"""

DOMAINS = {
    "coding": [
        "python_conventions", "javascript_style", "code_review",
        "api_design", "error_handling", "type_safety",
        "testing_patterns", "documentation", "refactoring",
        "performance_optimisation", ...  # 50 total
    ],
    "statistics": [
        "hypothesis_testing", "regression_analysis", "anova",
        "bayesian_inference", "time_series", "survival_analysis",
        ...
    ],
    "research": [...],
    "writing": [...],
    "data_analysis": [...],
    "debugging": [...],
    "testing": [...],
    "security": [...],
    "database": [...],
    "deployment": [...]
}

# For each module, generate via LLM:
GENERATION_PROMPT = """
Create a detailed instruction module for an AI assistant.

Domain: {domain}
Topic: {topic}

Generate a JSON object with:
- "name": A clear, descriptive name (3-6 words)
- "description": One sentence describing what this module governs
- "capabilities": A list of 5-15 specific capabilities this module enables
- "content": Detailed instructions (200-2000 tokens) that an AI should follow
  when performing tasks in this area. Include specific guidelines, dos/don'ts,
  formatting requirements, and domain-specific best practices.

The content should be realistic — the kind of instruction set a senior engineer
or domain expert would write for an AI coding assistant.
"""
```

**Module format on disk:**
```json
{
  "module_id": "statistics_hypothesis_testing",
  "name": "Hypothesis Testing Protocol",
  "domain": "statistics",
  "description": "Guidelines for conducting statistical hypothesis tests including test selection, assumption checking, and result interpretation.",
  "capabilities": [
    "hypothesis testing",
    "t-tests",
    "chi-square tests",
    "ANOVA",
    "p-value interpretation",
    "effect size calculation",
    "assumption checking",
    "multiple comparison correction",
    "significance reporting"
  ],
  "content": "When performing hypothesis testing, follow these guidelines:\n\n1. BEFORE TESTING\n- State the null and alternative hypotheses explicitly...\n- Check sample size adequacy using power analysis...\n... (200-2000 tokens of detailed instructions)"
}
```

### Phase 2: Task Benchmark Construction (Days 4-6)

Build 480 evaluation tasks: 80 per evaluation domain × 6 domains.

```python
# scripts/generate_tasks.py
"""
Each task is a natural-language request that requires specific
instruction modules to be answered well.

Task types:
  - Simple (240): Requires 1-2 modules from a single domain
  - Compound (240): Requires 2-4 modules from multiple domains

Example simple task:
  Request: "Perform a two-sample t-test comparing conversion rates
            between groups A and B."
  Relevant modules: [statistics_hypothesis_testing, statistics_inference]

Example compound task:
  Request: "Write a Python function that performs hypothesis testing on
            the given dataset and generates a formatted report."
  Relevant modules: [coding_python_conventions, statistics_hypothesis_testing,
                     writing_technical_reports, data_analysis_exploratory]
"""

TASK_GENERATION_PROMPT = """
Create an evaluation task for testing instruction routing.

Primary domain: {domain}
Complexity: {complexity}  # "simple" or "compound"
Target modules: {target_module_names}

Generate a JSON object with:
- "request": A natural-language user request (1-3 sentences)
- "reference_output": A high-quality response (200-500 tokens)
- "relevant_modules": List of module IDs that are genuinely needed
- "irrelevant_distractors": 3-5 module IDs that are semantically
  similar but NOT needed (for testing routing precision)
"""
```

### Phase 3: Ground-Truth Annotation (Days 7-8)

For 200 tasks (subset), get human annotations:

```python
# scripts/annotate_relevance.py
"""
Human annotation protocol:

For each (task, module) pair, annotate:
  - ESSENTIAL: Module is required for correct task completion
  - HELPFUL: Module provides useful but non-essential guidance
  - IRRELEVANT: Module has no bearing on the task
  - HARMFUL: Module's instructions conflict with task requirements

Three annotators per task. Majority vote for final label.
Compute inter-annotator agreement (Fleiss' kappa).

Essential + Helpful = "relevant" for routing evaluation.
"""
```

### Phase 4: Router Implementation (Days 9-14)

#### 4.1 Dual-Representation Encoder

```python
# src/router/encoder.py
from sentence_transformers import SentenceTransformer
import numpy as np

class DualEncoder:
    """
    Encodes instruction modules with dual representation:
    - Semantic: name + description
    - Functional: capability keywords
    Combined via weighted average.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", alpha: float = 0.6):
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def encode_module(self, module: InstructionModule) -> np.ndarray:
        """Compute dual-representation embedding for an instruction module."""
        sem_emb = self.model.encode(module.semantic_text, normalize_embeddings=True)
        func_emb = self.model.encode(module.functional_text, normalize_embeddings=True)
        
        combined = self.alpha * sem_emb + (1 - self.alpha) * func_emb
        # Re-normalise the combined vector
        combined = combined / np.linalg.norm(combined)
        return combined
    
    def encode_query(self, query: str) -> np.ndarray:
        """Compute embedding for a user query."""
        return self.model.encode(query, normalize_embeddings=True)
    
    def encode_library(self, library: list[InstructionModule]) -> np.ndarray:
        """Batch-encode all modules. Returns (N, dim) matrix."""
        embeddings = []
        for module in library:
            embeddings.append(self.encode_module(module))
        return np.array(embeddings, dtype='float32')
```

#### 4.2 Vector Index

```python
# src/router/index.py
import faiss
import numpy as np

class InstructionIndex:
    """FAISS-backed vector index for instruction retrieval."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after L2 norm)
        self.module_ids: list[str] = []
    
    def build(self, embeddings: np.ndarray, module_ids: list[str]):
        """Build index from pre-computed embeddings."""
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.module_ids = module_ids
    
    def search(self, query_embedding: np.ndarray, top_n: int = 20) -> list[tuple[str, float]]:
        """Retrieve top-N candidates with similarity scores."""
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_n)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.module_ids) and idx >= 0:
                results.append((self.module_ids[idx], float(score)))
        return results
    
    def save(self, path: str):
        faiss.write_index(self.index, path)
    
    def load(self, path: str):
        self.index = faiss.read_index(path)
```

#### 4.3 Selection Strategies

```python
# src/router/selector.py
from abc import ABC, abstractmethod
import numpy as np

class BaseSelector(ABC):
    @abstractmethod
    def select(self, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        pass

class TopKSelector(BaseSelector):
    def __init__(self, k: int = 3):
        self.k = k
    
    def select(self, candidates):
        return candidates[:self.k]

class ThresholdSelector(BaseSelector):
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
    
    def select(self, candidates):
        return [(mid, s) for mid, s in candidates if s >= self.threshold]

class AdaptiveKSelector(BaseSelector):
    """
    The core SIR contribution: detect natural relevance gaps.
    
    Given ranked scores [0.94, 0.89, 0.74, 0.31, 0.28]:
    Gaps:              [0.05, 0.15, 0.43, 0.03]
    Median gap: 0.10
    With gamma=1.5, threshold = 0.15
    First gap > 0.15 is at position 2 (gap=0.43)
    → Select indices 0, 1, 2 → K* = 3
    """
    
    def __init__(self, gamma: float = 1.5, fallback_k: int = 3, min_k: int = 1, max_k: int = 10):
        self.gamma = gamma
        self.fallback_k = fallback_k
        self.min_k = min_k
        self.max_k = max_k
    
    def select(self, candidates):
        if len(candidates) <= 1:
            return candidates
        
        scores = [s for _, s in candidates]
        
        # Compute gaps between consecutive scores
        gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        
        if not gaps:
            return candidates[:self.fallback_k]
        
        median_gap = float(np.median(gaps))
        gap_threshold = self.gamma * median_gap if median_gap > 0 else 0.05
        
        # Find first position where gap exceeds threshold
        k_star = len(candidates)  # default: take all
        for j, gap in enumerate(gaps):
            if gap > gap_threshold:
                k_star = j + 1  # Include the item before the gap
                break
        
        # Clamp to [min_k, max_k]
        k_star = max(self.min_k, min(self.max_k, k_star))
        
        return candidates[:k_star]
```

#### 4.4 Context Composer

```python
# src/router/composer.py
import tiktoken

class ContextComposer:
    """
    Assembles selected instruction modules into final context.
    Handles deduplication, ordering, and budget enforcement.
    """
    
    def __init__(self, budget_tokens: int = 8000, model: str = "gpt-4o"):
        self.budget = budget_tokens
        self.tokenizer = tiktoken.encoding_for_model(model)
    
    def compose(self, modules: list[InstructionModule], 
                scores: list[float]) -> str:
        """Build the final instruction context from selected modules."""
        
        # Step 1: Order by descending relevance (already sorted from selector)
        paired = list(zip(modules, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        # Step 2: Build context with budget enforcement
        sections = []
        total_tokens = 0
        
        for module, score in paired:
            content = module.content
            content_tokens = len(self.tokenizer.encode(content))
            
            if total_tokens + content_tokens > self.budget:
                # Truncate this module to fit remaining budget
                remaining = self.budget - total_tokens
                if remaining > 50:  # Only include if meaningful content fits
                    tokens = self.tokenizer.encode(content)[:remaining]
                    content = self.tokenizer.decode(tokens)
                    sections.append(self._format_section(module.name, content))
                break
            
            sections.append(self._format_section(module.name, content))
            total_tokens += content_tokens
        
        return "\n\n".join(sections)
    
    def _format_section(self, name: str, content: str) -> str:
        return f"[{name}]\n{content}"
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
```

### Phase 5: Baselines (Days 15-16)

```python
# src/baselines/static_full.py
class StaticFullLoader:
    """Baseline: load ALL instruction modules into context."""
    def load(self, library: list[InstructionModule], budget: int = None) -> str:
        sections = [f"[{m.name}]\n{m.content}" for m in library]
        context = "\n\n".join(sections)
        if budget:
            # Truncate to budget
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
            tokens = tokenizer.encode(context)[:budget]
            context = tokenizer.decode(tokens)
        return context

# src/baselines/random_k.py
import random

class RandomKSelector:
    """Baseline: select K random modules."""
    def select(self, library: list[InstructionModule], k: int) -> list[InstructionModule]:
        return random.sample(library, min(k, len(library)))

# src/baselines/bm25_router.py
from rank_bm25 import BM25Okapi

class BM25Router:
    """Baseline: BM25 keyword matching for instruction selection."""
    def __init__(self):
        self.bm25 = None
        self.modules = []
    
    def build(self, library: list[InstructionModule]):
        self.modules = library
        corpus = [f"{m.name} {m.description} {' '.join(m.capabilities)}" for m in library]
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
    
    def route(self, query: str, top_k: int = 5) -> list[tuple[InstructionModule, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.modules[idx], score) for idx, score in ranked[:top_k]]
```

### Phase 6: Evaluation Pipeline (Days 17-20)

#### 6.1 LLM-as-Judge

```python
# src/evaluation/llm_judge.py
from litellm import completion

class LLMJudge:
    """
    GPT-4o-as-judge evaluation.
    Scores on three dimensions: correctness, completeness, instruction adherence.
    Each scored 0-100, final score is average.
    """
    
    JUDGE_PROMPT = """You are an expert evaluator. Score the following AI response
on three dimensions, each from 0 to 100:

1. CORRECTNESS: Is the response factually correct and logically sound?
2. COMPLETENESS: Does it fully address all aspects of the request?
3. INSTRUCTION_ADHERENCE: Does it follow the domain-specific guidelines
   that were provided in its instructions?

User Request: {request}

Instructions Given to AI: {instructions_summary}

AI Response: {response}

Reference Output: {reference}

Return a JSON object:
{{
  "correctness": <0-100>,
  "completeness": <0-100>,
  "instruction_adherence": <0-100>,
  "overall": <0-100>,
  "reasoning": "<brief explanation>"
}}"""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
    
    def score(self, request, instructions_summary, response, reference) -> dict:
        prompt = self.JUDGE_PROMPT.format(
            request=request,
            instructions_summary=instructions_summary,
            response=response,
            reference=reference
        )
        result = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(result.choices[0].message.content)
```

#### 6.2 Routing Quality Metrics

```python
# src/evaluation/routing_metrics.py

def routing_precision(selected_ids: list[str], relevant_ids: list[str]) -> float:
    """What fraction of selected modules are actually relevant?"""
    if not selected_ids:
        return 0.0
    selected = set(selected_ids)
    relevant = set(relevant_ids)
    return len(selected & relevant) / len(selected)

def routing_recall(selected_ids: list[str], relevant_ids: list[str]) -> float:
    """What fraction of relevant modules were selected?"""
    if not relevant_ids:
        return 1.0
    selected = set(selected_ids)
    relevant = set(relevant_ids)
    return len(selected & relevant) / len(relevant)

def routing_f1(selected_ids: list[str], relevant_ids: list[str]) -> float:
    p = routing_precision(selected_ids, relevant_ids)
    r = routing_recall(selected_ids, relevant_ids)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
```

### Phase 7: Experiment Execution (Days 21-28)

#### 7.1 Main Benchmark Runner

```python
# scripts/run_benchmark.py
"""
Full experiment matrix:
  - 6 strategies × 4 LLMs × 480 tasks = 11,520 evaluations
  - Plus noise sweep, ablation, sensitivity analysis
"""

STRATEGIES = {
    "no_instructions": NoInstructionsStrategy(),
    "static_full": StaticFullStrategy(),
    "random_k": RandomKStrategy(k=4),
    "bm25": BM25Strategy(),
    "sir_top3": SIRStrategy(selector=TopKSelector(k=3)),
    "sir_top5": SIRStrategy(selector=TopKSelector(k=5)),
    "sir_threshold": SIRStrategy(selector=ThresholdSelector(tau=0.70)),
    "sir_adaptive": SIRStrategy(selector=AdaptiveKSelector(gamma=1.5)),
    "oracle": OracleStrategy(),
}

MODELS = ["gpt-4o", "claude-3-5-sonnet-20241022", 
          "ollama/llama3.1:8b", "ollama/mistral:7b"]

def run_experiment(strategy_name, model_name, task, library):
    # 1. Route
    t0 = time.perf_counter()
    context = strategy.build_context(task.request, library)
    routing_time = (time.perf_counter() - t0) * 1000
    
    # 2. Generate response
    t0 = time.perf_counter()
    response = generate(model_name, context, task.request)
    generation_time = (time.perf_counter() - t0) * 1000
    
    # 3. Evaluate
    score = judge.score(task.request, context[:500], response, task.reference_output)
    
    # 4. Record
    return {
        "strategy": strategy_name,
        "model": model_name,
        "task_id": task.task_id,
        "domain": task.domain,
        "complexity": task.complexity,
        "performance": score["overall"],
        "correctness": score["correctness"],
        "completeness": score["completeness"],
        "adherence": score["instruction_adherence"],
        "context_tokens": count_tokens(context),
        "routing_latency_ms": routing_time,
        "generation_latency_ms": generation_time,
        "modules_selected": len(strategy.last_selected),
        "routing_precision": routing_precision(...),
        "routing_recall": routing_recall(...),
        "routing_f1": routing_f1(...),
    }
```

#### 7.2 Noise Degradation Experiment

```python
# scripts/run_noise_experiment.py
"""
Experiment 1: Vary noise ratio from 0% to 95%.

For each noise level:
  - Start with only relevant modules
  - Add irrelevant modules to reach target noise ratio
  - Run static loading and SIR adaptive
  - Record performance
"""

NOISE_LEVELS = [0.0, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95]

def run_noise_experiment(task, full_library, noise_level):
    relevant = [m for m in full_library if m.module_id in task.relevant_modules]
    irrelevant = [m for m in full_library if m.module_id not in task.relevant_modules]
    
    # Calculate how many irrelevant modules to include
    n_relevant = len(relevant)
    if noise_level == 0:
        library = relevant
    else:
        n_total = int(n_relevant / (1 - noise_level))
        n_irrelevant = n_total - n_relevant
        selected_irrelevant = random.sample(irrelevant, min(n_irrelevant, len(irrelevant)))
        library = relevant + selected_irrelevant
        random.shuffle(library)
    
    # Run both strategies
    static_result = run_static_full(task, library)
    sir_result = run_sir_adaptive(task, library)
    
    return static_result, sir_result
```

#### 7.3 Ablation Experiments

```python
# scripts/run_ablation.py
"""
Ablation 1: Encoding strategy
  - Raw content embedding only
  - Semantic only (name + description)
  - Functional only (capabilities)
  - Dual with alpha = 0.4, 0.5, 0.6, 0.7, 0.8

Ablation 2: Gap sensitivity gamma
  - gamma = 1.0, 1.25, 1.5, 2.0, 2.5

Ablation 3: Embedding model
  - all-MiniLM-L6-v2 (22M)
  - all-mpnet-base-v2 (110M)
  - e5-small-v2 (33M)
"""
```

---

## 5. Configuration

```yaml
# config/settings.yaml

# Instruction Library
library:
  path: "data/instruction_library/"
  n_modules: 500
  domains: ["coding", "statistics", "research", "writing", 
            "data_analysis", "debugging", "testing", "security",
            "database", "deployment"]
  modules_per_domain: 50

# Router
router:
  encoder:
    model: "all-MiniLM-L6-v2"
    alpha: 0.6                  # Semantic vs functional weight
    dim: 384
  index:
    type: "faiss_flat_ip"       # FAISS flat inner product
    top_n: 20                   # Candidates before selection
  selector:
    strategy: "adaptive"        # top_k | threshold | adaptive
    top_k: 3                    # For fixed top-k
    threshold: 0.70             # For threshold-based
    gamma: 1.5                  # For adaptive gap detection
    fallback_k: 3
    min_k: 1
    max_k: 10

# Context Composition
composer:
  budget_tokens: 8000
  dedup_ngram_size: 5
  dedup_threshold: 0.8

# Evaluation
evaluation:
  judge_model: "gpt-4o"
  tasks_path: "data/tasks/"
  n_tasks_per_domain: 80
  eval_domains: ["coding", "statistics", "research", 
                 "writing", "data_analysis", "debugging"]

# Target LLMs
models:
  - "gpt-4o"
  - "claude-3-5-sonnet-20241022"
  - "ollama/llama3.1:8b"
  - "ollama/mistral:7b"

# Noise experiment
noise:
  levels: [0.0, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
  tasks_per_level: 100
  repeats: 3
```

---

## 6. Dependencies

```
# requirements.txt

# Core
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
numpy>=1.24.0
tiktoken>=0.5.0

# LLM backends
litellm>=1.30.0
openai>=1.12.0

# BM25 baseline
rank-bm25>=0.2.2

# Evaluation
pandas>=2.0.0
scikit-learn>=1.3.0      # For inter-annotator agreement

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Results
tabulate>=0.9.0          # LaTeX table generation
jsonlines>=3.1.0

# Dev
pytest>=7.4.0
```

---

## 7. Execution Timeline

| Phase | Days | Deliverable | Blocks On |
|-------|------|-------------|-----------|
| 1. Instruction library | 1-3 | 500 modules in JSON | Nothing |
| 2. Task benchmark | 4-6 | 480 tasks in JSON | Phase 1 |
| 3. Annotation | 7-8 | Ground-truth labels for 200 tasks | Phase 2 |
| 4. Router impl. | 9-14 | Encoder, index, selector, composer | Phase 1 |
| 5. Baselines | 15-16 | Static, random, BM25 baselines | Phase 4 |
| 6. Eval pipeline | 17-20 | Judge, metrics, runner | Phase 4-5 |
| 7. Experiments | 21-28 | Full benchmark, noise sweep, ablation | Phase 6 |
| 8. Analysis | 29-31 | Tables, figures, statistical tests | Phase 7 |
| 9. Paper writing | 32-38 | Complete manuscript | Phase 8 |

**Total: ~6 weeks from start to submittable draft**

---

## 8. Key Design Decisions

### 8.1 Why Embedding Routing, Not Classification?
A classifier would require training data mapping queries to module labels. Embedding routing is zero-shot: it works with any instruction library without retraining.

### 8.2 Why Dual Representation?
Raw content embeddings are noisy for long modules (2000 tokens compressed to 384 dims). The name+description captures topic; capabilities capture function. Both signals matter for accurate routing.

### 8.3 Why Adaptive-K Over Fixed-K?
Different queries need different numbers of modules. "Write a Python function" needs 1-2 modules. "Research React performance and write a benchmark" needs 3-4. Fixed-K either over- or under-selects.

### 8.4 Why Not Fine-Tune the Router?
Intentional. A fine-tuned router is model-specific and dataset-specific. A zero-shot embedding router works across any LLM backend and any instruction library, making the approach immediately deployable.

### 8.5 Why FAISS Flat, Not HNSW?
At 500 modules, exact search is <1ms. Approximate search (HNSW) only matters at 10K+ scale. Using flat search keeps results deterministic and eliminates a confound.

---

## 9. Expected Figures

1. **Figure 1**: SIR architecture diagram (pipeline flow)
2. **Figure 2**: Noise degradation curve (η vs. performance, Static vs. SIR)
3. **Figure 3**: Pareto frontier (tokens vs. performance for all strategies)
4. **Figure 4**: Domain-specific radar chart (Static vs. SIR per domain)
5. **Figure 5**: Adaptive-K distribution histogram (how many modules are typically selected)
6. **Figure 6**: Relevance score distribution for sample queries showing the "gap"

---

## 10. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM-judge scores are noisy | Medium | Human annotation subset; report correlation |
| Adaptive-K doesn't beat Top-K | High | Report all strategies honestly; the gap mechanism is the contribution |
| Static Full doesn't degrade with noise | Fatal | Pilot with 50 tasks at extreme noise levels first |
| API rate limits slow experiments | Medium | Use local models for bulk runs; API models for validation |
| Instruction modules are too similar across domains | Medium | Manual curation; test intra-domain vs inter-domain similarity |
