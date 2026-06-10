"""Proof-trace to EXACT-facing explanation helpers."""

from __future__ import annotations

from exact.logic.ir import SolveResult
from exact.logic.kb import KnowledgeBase


def explain_result(result: SolveResult, kb: KnowledgeBase) -> tuple[str, list[str], list[str]]:
    """Return explanation, CoT-style trace, and cited premise labels."""

    premise_labels = [f"P{idx + 1}" for idx in result.supporting_premises]

    if result.label == "Unknown":
        return (
            "The provided premises do not prove the claim or its negation, so the answer is Unknown.",
            ["No symbolic proof was found for the claim or for its negation."],
            [],
        )

    cot = [step.natural_language or f"Derived {step.derived.display()}." for step in result.proof]
    support_text = ", ".join(premise_labels) if premise_labels else "the parsed premises"

    if result.label == "Yes":
        explanation = f"Using {support_text}, the symbolic proof derives {result.claim.display()}."
    else:
        explanation = (
            f"Using {support_text}, the symbolic proof derives the negation of "
            f"{result.claim.display()}."
        )

    return explanation, cot, premise_labels


def kb_to_fol_like_text(kb: KnowledgeBase) -> str:
    """Readable formalization for the optional `fol` field."""

    lines: list[str] = []
    for fact in kb.facts:
        lines.append(f"P{fact.source_idx + 1}: {fact.atom.display()}")
    for rule in kb.rules:
        conditions = " AND ".join(condition.display() for condition in rule.conditions)
        lines.append(f"P{rule.source_idx + 1}: {conditions} -> {rule.conclusion.display()}")
    return "\n".join(lines)
