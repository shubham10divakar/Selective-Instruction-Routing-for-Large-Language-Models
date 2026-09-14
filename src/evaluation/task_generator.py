"""Synthetic evaluation-task generator (offline, template-based -- mirrors
`src/instruction/generator.py`'s approach so the whole benchmark is
reproducible with no external API calls).

For each of the 6 evaluation domains this produces 80 tasks: 40 "simple"
tasks (need 1-2 modules from one domain) and 40 "compound" tasks (need 2-4
modules spanning two domains), each carrying ground-truth `relevant_modules`
and a handful of semantically-adjacent `distractor_modules` so routing
precision/recall can be measured.

Module ids referenced here must match `src.instruction.generator`'s
`{domain}_{subject}_{aspect}` scheme.
"""
from __future__ import annotations

import hashlib
import random

from src.instruction.generator import ASPECTS, TAXONOMY
from .task import EvalTask

EVAL_DOMAINS = ["coding", "statistics", "research", "writing", "data_analysis", "debugging"]

_ASPECT_KEYS = [key for key, _ in ASPECTS]

# Domains a given eval domain most naturally pairs with, for compound tasks.
_SECONDARY_DOMAINS: dict[str, list[str]] = {
    "coding": ["testing", "security", "database", "deployment"],
    "statistics": ["coding", "data_analysis", "research", "writing"],
    "research": ["writing", "statistics", "data_analysis", "coding"],
    "writing": ["research", "coding", "data_analysis", "testing"],
    "data_analysis": ["coding", "statistics", "writing", "database"],
    "debugging": ["coding", "testing", "deployment", "security"],
}

_PRIMARY_ACTION = {
    "coding": "Write a small program",
    "statistics": "Run a statistical analysis",
    "research": "Produce a research write-up",
    "writing": "Draft a document",
    "data_analysis": "Build a data analysis report",
    "debugging": "Diagnose and fix the issue",
}

_SIMPLE_TEMPLATES: dict[str, list[str]] = {
    "coding": [
        "Write a {subject} implementation for a small utility function, following our {subject} guidelines.",
        "Review this {subject} snippet and point out anything that violates our {subject} conventions.",
        "Refactor the attached {subject} module to align with best practices for {subject}.",
        "Explain the key {subject} guidelines our AI assistant should follow when generating code.",
    ],
    "statistics": [
        "Perform a {subject} on the provided dataset and interpret the results.",
        "Design an analysis plan using {subject} for our upcoming study.",
        "Explain when {subject} is the appropriate method and what assumptions it requires.",
        "Check whether the assumptions for {subject} are met in this dataset and report any violations.",
    ],
    "research": [
        "Draft a {subject} plan for a new research project on user retention.",
        "Summarize best practices for {subject} that our research team should follow.",
        "Review our current process for {subject} and suggest improvements.",
        "Explain the key steps involved in {subject} for a first-time researcher.",
    ],
    "writing": [
        "Write a short piece using strong {subject} for a technical audience.",
        "Edit this draft to improve its {subject}.",
        "Explain the guidelines for {subject} that our writing team follows.",
        "Rewrite this paragraph applying best practices for {subject}.",
    ],
    "data_analysis": [
        "Perform {subject} on the attached dataset and summarize your findings.",
        "Set up a {subject} workflow for our weekly reporting process.",
        "Explain the best practices for {subject} that apply to this dataset.",
        "Review our current {subject} approach and identify gaps.",
    ],
    "debugging": [
        "Help debug this issue using proper {subject} techniques.",
        "Walk through how to apply {subject} to find the root cause of this failure.",
        "Explain the guidelines for {subject} our engineers should follow during an incident.",
        "Review this incident report and assess whether {subject} was followed correctly.",
    ],
}


def _readable(subject: str) -> str:
    return subject.replace("_", " ")


def _module_id(domain: str, subject: str, aspect: str) -> str:
    return f"{domain}_{subject}_{aspect}"


def _module_name(domain: str, subject: str, aspect: str) -> str:
    title_template = dict(ASPECTS)[aspect]
    return title_template.format(subject=_readable(subject).title())


def _rng_for(task_id: str) -> random.Random:
    seed = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def _distractors(rng: random.Random, exclude_domains: set[str], exclude_subjects: dict[str, set[str]],
                  n: int = 4) -> list[str]:
    """Pick module ids that are semantically adjacent but not relevant: same
    domain(s) as the task but a different subject, or an unrelated domain."""
    pool: list[str] = []
    for domain, subjects in TAXONOMY.items():
        for subject in subjects:
            if domain in exclude_subjects and subject in exclude_subjects[domain]:
                continue
            aspect = rng.choice(_ASPECT_KEYS)
            pool.append(_module_id(domain, subject, aspect))
    return rng.sample(pool, k=min(n, len(pool)))


def _make_simple_task(domain: str, subject: str, variant_idx: int) -> EvalTask:
    task_id = f"{domain}_simple_{subject}_{variant_idx}"
    rng = _rng_for(task_id)
    template = _SIMPLE_TEMPLATES[domain][variant_idx]
    request = template.format(subject=_readable(subject))

    # Alternate between needing 1 and 2 modules from the same subject.
    aspects = ["best_practices"] if variant_idx % 2 == 0 else ["best_practices", "checklist"]
    relevant_modules = [_module_id(domain, subject, a) for a in aspects]
    relevant_names = [_module_name(domain, subject, a) for a in aspects]

    distractors = _distractors(rng, {domain}, {domain: {subject}}, n=4)

    reference_output = (
        f"A strong response follows the guidance in {', '.join(relevant_names)}: "
        f"it addresses the request about {_readable(subject)} directly, states any "
        f"assumptions, checks relevant edge cases, and avoids introducing unrelated changes."
    )

    return EvalTask(
        task_id=task_id,
        domain=domain,
        domains=[domain],
        complexity="simple",
        request=request,
        reference_output=reference_output,
        relevant_modules=relevant_modules,
        distractor_modules=distractors,
        metadata={"subject": subject, "source": "template"},
    )


def _make_compound_task(domain: str, subject: str, subject_idx: int, rep: int) -> EvalTask:
    secondary_domains = _SECONDARY_DOMAINS[domain]
    sec_domain = secondary_domains[rep % len(secondary_domains)]
    sec_subjects = TAXONOMY[sec_domain]
    sec_subject = sec_subjects[(subject_idx + rep) % len(sec_subjects)]

    task_id = f"{domain}_compound_{subject}_{sec_domain}_{sec_subject}"
    rng = _rng_for(task_id)

    request = (
        f"{_PRIMARY_ACTION[domain]} that applies {_readable(subject)} practices, "
        f"and make sure the result also satisfies our {sec_domain.replace('_', ' ')} "
        f"guidelines around {_readable(sec_subject)}."
    )

    relevant_modules = [
        _module_id(domain, subject, "best_practices"),
        _module_id(domain, subject, "checklist"),
        _module_id(sec_domain, sec_subject, "best_practices"),
    ]
    relevant_names = [
        _module_name(domain, subject, "best_practices"),
        _module_name(domain, subject, "checklist"),
        _module_name(sec_domain, sec_subject, "best_practices"),
    ]

    distractors = _distractors(
        rng, {domain, sec_domain}, {domain: {subject}, sec_domain: {sec_subject}}, n=4
    )

    reference_output = (
        f"A strong response satisfies both {relevant_names[0]} / {relevant_names[1]} and "
        f"{relevant_names[2]}: it completes the {_readable(subject)} portion of the request "
        f"correctly, then explicitly checks it against the {_readable(sec_subject)} guidance "
        f"from {sec_domain.replace('_', ' ')} before calling the task done."
    )

    return EvalTask(
        task_id=task_id,
        domain=domain,
        domains=[domain, sec_domain],
        complexity="compound",
        request=request,
        reference_output=reference_output,
        relevant_modules=relevant_modules,
        distractor_modules=distractors,
        metadata={"subject": subject, "secondary_domain": sec_domain,
                  "secondary_subject": sec_subject, "source": "template"},
    )


def generate_tasks_for_domain(domain: str) -> list[EvalTask]:
    """40 simple + 40 compound tasks for one eval domain (80 total)."""
    subjects = TAXONOMY[domain]
    tasks: list[EvalTask] = []

    for variant_idx in range(4):
        for subject in subjects:
            tasks.append(_make_simple_task(domain, subject, variant_idx))

    for subject_idx, subject in enumerate(subjects):
        for rep in range(4):
            # rep cycles through all 4 secondary domains for this subject
            tasks.append(_make_compound_task(domain, subject, subject_idx, rep))

    return tasks


def generate_all_tasks() -> dict[str, list[EvalTask]]:
    return {domain: generate_tasks_for_domain(domain) for domain in EVAL_DOMAINS}
