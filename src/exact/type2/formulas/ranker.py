from __future__ import annotations

from typing import Any

from exact.type2.formulas.classifier import TopicClassification


def ontology_score(summary: dict[str, Any], classification: TopicClassification) -> int:
    if not classification.known_topic:
        return 0
    domain = str(summary.get("domain") or "").lower()
    subfield = str(summary.get("subfield") or "").lower()
    text = " ".join(str(value).lower() for value in summary.values())
    score = 0
    if classification.topic in {domain, subfield} or classification.topic in text:
        score += 8
    if classification.subtopic in {domain, subfield} or classification.subtopic in text:
        score += 12
    score += 2 * sum(1 for term in classification.matched_terms if term in text)
    return score


def narrow_with_ontology(
    summaries: list[dict[str, Any]],
    classification: TopicClassification,
    *,
    rescue_limit: int,
) -> list[dict[str, Any]]:
    if not classification.known_topic:
        return summaries
    matching = [item for item in summaries if ontology_score(item, classification) > 0]
    if not matching:
        return summaries
    rescue = [item for item in summaries if item not in matching][:rescue_limit]
    return matching + rescue

