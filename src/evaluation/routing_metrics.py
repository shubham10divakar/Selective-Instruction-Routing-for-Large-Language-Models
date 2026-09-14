"""Precision / recall / F1 of a routing decision against ground-truth
relevant module IDs for a task."""
from __future__ import annotations


def routing_precision(selected_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of selected modules that are actually relevant."""
    if not selected_ids:
        return 0.0
    selected, relevant = set(selected_ids), set(relevant_ids)
    return len(selected & relevant) / len(selected)


def routing_recall(selected_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of relevant modules that were selected."""
    if not relevant_ids:
        return 1.0
    selected, relevant = set(selected_ids), set(relevant_ids)
    return len(selected & relevant) / len(relevant)


def routing_f1(selected_ids: list[str], relevant_ids: list[str]) -> float:
    p = routing_precision(selected_ids, relevant_ids)
    r = routing_recall(selected_ids, relevant_ids)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def routing_metrics(selected_ids: list[str], relevant_ids: list[str]) -> dict[str, float]:
    return {
        "precision": routing_precision(selected_ids, relevant_ids),
        "recall": routing_recall(selected_ids, relevant_ids),
        "f1": routing_f1(selected_ids, relevant_ids),
    }
