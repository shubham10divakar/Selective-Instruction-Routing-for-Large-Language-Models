"""Synthetic instruction library generator.

Two generation modes:

1. Offline / template mode (default, no API key required). Combines a
   per-domain taxonomy of subjects with a fixed set of "aspects"
   (conventions, best practices, checklist, pitfalls, advanced patterns)
   and assembles realistic-looking instruction content from parametrized
   phrase banks. This is what `scripts/generate_instructions.py` uses by
   default so the whole pipeline is runnable with zero external
   dependencies.

2. LLM mode (`generate_module_llm`), which mirrors the GENERATION_PROMPT
   from the design doc and calls an LLM via litellm. Use this to produce
   higher-fidelity modules once API keys are configured; it is a drop-in
   replacement that returns the same InstructionModule shape.

10 domains x 10 subjects x 5 aspects = 500 modules, matching the target
library size in the design doc (50 modules/domain).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from .module import InstructionModule

# ---------------------------------------------------------------------------
# Taxonomy: 10 domains x 10 subjects each
# ---------------------------------------------------------------------------

TAXONOMY: dict[str, list[str]] = {
    "coding": [
        "python", "javascript", "typescript", "api_design", "error_handling",
        "type_safety", "code_review", "documentation", "refactoring", "performance",
    ],
    "statistics": [
        "hypothesis_testing", "regression_analysis", "anova", "bayesian_inference",
        "time_series", "survival_analysis", "experimental_design", "sampling",
        "nonparametric_methods", "multivariate_analysis",
    ],
    "research": [
        "literature_review", "research_design", "citation_management", "peer_review",
        "grant_writing", "data_collection", "qualitative_methods", "systematic_review",
        "research_ethics", "reproducibility",
    ],
    "writing": [
        "technical_writing", "editing", "tone_and_style", "documentation_writing",
        "executive_summaries", "persuasive_writing", "grammar_and_clarity",
        "content_structuring", "audience_adaptation", "proofreading",
    ],
    "data_analysis": [
        "exploratory_analysis", "data_cleaning", "feature_engineering", "visualization",
        "dashboarding", "cohort_analysis", "ab_testing", "reporting",
        "data_pipelines", "outlier_detection",
    ],
    "debugging": [
        "stack_trace_analysis", "root_cause_analysis", "logging_strategy",
        "reproducing_bugs", "memory_leaks", "concurrency_bugs",
        "performance_profiling", "regression_bisection", "exception_handling",
        "production_incident_response",
    ],
    "testing": [
        "unit_testing", "integration_testing", "test_coverage", "mocking",
        "test_data_management", "regression_testing", "load_testing",
        "end_to_end_testing", "test_driven_development", "flaky_test_management",
    ],
    "security": [
        "input_validation", "authentication", "authorization", "secrets_management",
        "dependency_scanning", "threat_modeling", "encryption", "secure_coding",
        "incident_response", "vulnerability_disclosure",
    ],
    "database": [
        "schema_design", "query_optimization", "indexing", "migrations",
        "transactions", "normalization", "connection_pooling",
        "backup_and_recovery", "replication", "nosql_modeling",
    ],
    "deployment": [
        "ci_cd_pipelines", "containerization", "infrastructure_as_code",
        "rollback_strategy", "blue_green_deployment", "monitoring_and_alerting",
        "environment_configuration", "release_management", "canary_releases",
        "capacity_planning",
    ],
}

ASPECTS: list[tuple[str, str]] = [
    ("conventions", "{subject} Conventions"),
    ("best_practices", "{subject} Best Practices"),
    ("checklist", "{subject} Checklist"),
    ("common_pitfalls", "{subject} Common Pitfalls"),
    ("advanced_patterns", "Advanced {subject} Patterns"),
]

# ---------------------------------------------------------------------------
# Phrase banks used to assemble template-mode content deterministically
# ---------------------------------------------------------------------------

_GUIDELINE_TEMPLATES = [
    "Before starting {subject} work, confirm the scope and constraints so the approach matches the actual requirement.",
    "State assumptions about {subject} explicitly before proceeding, rather than leaving them implicit.",
    "Prefer the simplest {subject} approach that satisfies the requirement over a more general one that isn't needed yet.",
    "When {subject} decisions are ambiguous, favor the option that is easiest to reverse or extend later.",
    "Document any non-obvious {subject} decision with a short rationale so future readers understand the 'why', not just the 'what'.",
    "Validate {subject} outputs against a concrete example before treating the work as complete.",
    "Keep {subject} changes scoped to what was asked; avoid unrelated cleanup in the same pass.",
    "Check that {subject} terminology is used consistently throughout, matching existing conventions in the surrounding context.",
    "When multiple {subject} strategies are viable, name the trade-off explicitly rather than silently picking one.",
    "Surface known limitations of a chosen {subject} approach instead of presenting it as complete when it is not.",
    "Re-check edge cases relevant to {subject} (empty input, boundary values, concurrent access) before calling the task done.",
    "Prefer explicit, readable {subject} solutions over clever ones that are hard to verify.",
    "When {subject} guidance conflicts with a specific user instruction, defer to the user's explicit instruction and note the conflict.",
    "Keep {subject} outputs reproducible: fix seeds, record versions, and avoid hidden non-determinism where it matters.",
    "Measure before optimizing any {subject} bottleneck; do not guess at what is slow.",
]

_PITFALL_TEMPLATES = [
    "Do not apply generic {subject} advice without checking it fits the specific context at hand.",
    "Do not silently drop {subject} edge cases to make an example look cleaner than the real system.",
    "Avoid introducing new {subject} abstractions before there is a second concrete use case that needs them.",
    "Do not treat a single passing example as full verification of {subject} correctness.",
    "Avoid mixing unrelated {subject} concerns into a single change; keep each change focused.",
    "Do not skip documenting {subject} assumptions just because they seem obvious in the moment.",
]

_PURPOSE_TEMPLATES = {
    "conventions": (
        "This module defines the naming, formatting, and structural style conventions to follow "
        "whenever writing, implementing, or reviewing {subject_readable} code or output, so results "
        "stay consistent with the existing {subject_readable} style guide."
    ),
    "best_practices": (
        "This module lists the best, most commonly recommended practices for writing, implementing, "
        "and building {subject_readable} — the default, everyday guidance to apply on typical, "
        "ordinary {subject_readable} tasks and implementations, not specialized or unusual ones."
    ),
    "checklist": (
        "This module is a short pre-completion checklist of specific items to verify before marking "
        "any {subject_readable} task, implementation, or piece of work complete."
    ),
    "common_pitfalls": (
        "This module lists frequently observed mistakes, bugs, and anti-patterns seen in everyday "
        "{subject_readable} work, so they can be recognized and avoided during typical {subject_readable} tasks."
    ),
    "advanced_patterns": (
        "This module documents advanced, specialized {subject_readable} techniques reserved for "
        "unusually complex, large-scale, or non-standard situations — it does not apply to small, "
        "typical, everyday {subject_readable} work where the basic approach is already sufficient."
    ),
}

_CLOSING_TEMPLATES = [
    "When in doubt, prefer clarity and explicit reasoning about {subject_readable} over a shortcut that saves a few lines but hides intent.",
    "These guidelines are a starting point for {subject_readable}, not an exhaustive rulebook — use judgment when a rule doesn't fit the situation.",
    "Revisit these {subject_readable} guidelines if the task's constraints change significantly during the work.",
]

_CAPABILITY_EXTRA_TERMS = {
    "conventions": ["naming conventions", "style guide", "formatting", "syntax rules", "structural consistency"],
    "best_practices": ["best practices", "everyday tasks", "recommended approach", "implementation guidance", "typical work"],
    "checklist": ["pre-completion checklist", "verification steps", "review checklist", "done criteria"],
    "common_pitfalls": ["common mistakes", "anti-patterns", "failure modes", "pitfalls to avoid"],
    "advanced_patterns": ["advanced techniques", "complex scenarios", "large-scale systems", "non-standard cases", "specialized patterns"],
}


def _seeded_rng(module_id: str) -> random.Random:
    seed = int(hashlib.sha256(module_id.encode("utf-8")).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def _readable(subject: str) -> str:
    return subject.replace("_", " ")


def generate_module(domain: str, subject: str, aspect_key: str, title_template: str) -> InstructionModule:
    """Deterministically build one InstructionModule from the taxonomy (no LLM call)."""
    rng = _seeded_rng(f"{domain}_{subject}_{aspect_key}")
    subject_readable = _readable(subject)

    name = title_template.format(subject=subject_readable.title())
    description = _PURPOSE_TEMPLATES[aspect_key].format(subject_readable=subject_readable)

    n_guidelines = rng.randint(7, 12)
    guidelines = rng.sample(_GUIDELINE_TEMPLATES, k=min(n_guidelines, len(_GUIDELINE_TEMPLATES)))
    guidelines = [g.format(subject=subject_readable) for g in guidelines]

    n_pitfalls = rng.randint(3, 5)
    pitfalls = rng.sample(_PITFALL_TEMPLATES, k=min(n_pitfalls, len(_PITFALL_TEMPLATES)))
    pitfalls = [p.format(subject=subject_readable) for p in pitfalls]

    closing = rng.choice(_CLOSING_TEMPLATES).format(subject_readable=subject_readable)

    guideline_block = "\n".join(f"{i + 1}. {g}" for i, g in enumerate(guidelines))
    pitfall_block = "\n".join(f"- {p}" for p in pitfalls)

    content = (
        f"{description}\n\n"
        f"CORE GUIDELINES\n{guideline_block}\n\n"
        f"COMMON PITFALLS TO AVOID\n{pitfall_block}\n\n"
        f"CLOSING NOTE\n{closing}"
    )

    capabilities = sorted(set(
        [subject_readable, domain, aspect_key.replace("_", " ")]
        + _CAPABILITY_EXTRA_TERMS[aspect_key]
        + subject_readable.split()
    ))[:15]
    if len(capabilities) < 5:
        capabilities += [f"{domain} {aspect_key}"]

    module_id = f"{domain}_{subject}_{aspect_key}"
    return InstructionModule(
        module_id=module_id,
        name=name,
        domain=domain,
        description=description,
        capabilities=capabilities,
        content=content,
        metadata={"subject": subject, "aspect": aspect_key, "source": "template"},
    )


def generate_library_modules(taxonomy: dict[str, list[str]] | None = None) -> list[InstructionModule]:
    """Generate the full set of modules for every domain x subject x aspect combo."""
    taxonomy = taxonomy or TAXONOMY
    modules = []
    for domain, subjects in taxonomy.items():
        for subject in subjects:
            for aspect_key, title_template in ASPECTS:
                modules.append(generate_module(domain, subject, aspect_key, title_template))
    return modules


# ---------------------------------------------------------------------------
# Optional LLM-backed generation (mirrors the design doc's GENERATION_PROMPT)
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """Create a detailed instruction module for an AI assistant.

Domain: {domain}
Topic: {topic}

Generate a JSON object with:
- "name": A clear, descriptive name (3-6 words)
- "description": One sentence describing what this module governs
- "capabilities": A list of 5-15 specific capabilities this module enables
- "content": Detailed instructions (200-2000 tokens) that an AI should follow
  when performing tasks in this area. Include specific guidelines, dos/don'ts,
  formatting requirements, and domain-specific best practices.

The content should be realistic -- the kind of instruction set a senior engineer
or domain expert would write for an AI coding assistant.

Respond with ONLY the JSON object, no markdown fences.
"""


def generate_module_llm(domain: str, topic: str, model: str = "gpt-4o") -> InstructionModule:
    """Generate a single module via an LLM (requires litellm + a configured API key)."""
    from litellm import completion  # imported lazily so offline mode has no hard dependency

    prompt = GENERATION_PROMPT.format(domain=domain, topic=topic)
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    data["module_id"] = f"{domain}_{topic}"
    data["domain"] = domain
    data["metadata"] = {"source": "llm", "generation_model": model}
    return InstructionModule.from_dict(data)
