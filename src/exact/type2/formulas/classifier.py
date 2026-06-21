from __future__ import annotations

from dataclasses import dataclass

from exact.type2.formulas.ontology import ONTOLOGY
from exact.type2.schemas import Extraction


@dataclass(frozen=True)
class TopicClassification:
    topic: str
    subtopic: str
    confidence: float
    matched_terms: tuple[str, ...]
    known_topic: bool
    question_kind: str


def classify_formula_topic(question: str, extraction: Extraction | None = None) -> TopicClassification:
    text = " ".join(
        part
        for part in (
            question.lower(),
            extraction.target.lower() if extraction and extraction.target else "",
            " ".join(extraction.quantities).lower() if extraction else "",
        )
        if part
    )
    best_topic = "physics"
    best_subtopic = "unknown"
    best_terms: tuple[str, ...] = ()
    best_score = 0
    for topic, subtopics in ONTOLOGY.items():
        for subtopic, terms in subtopics.items():
            matched = tuple(term for term in terms if term in text)
            score = len(matched)
            if score > best_score:
                best_topic = topic
                best_subtopic = subtopic
                best_terms = matched
                best_score = score
    confidence = min(0.95, 0.2 + 0.2 * best_score) if best_score else 0.0
    kind = extraction.kind.value if extraction else "unknown"
    return TopicClassification(
        topic=best_topic,
        subtopic=best_subtopic,
        confidence=confidence,
        matched_terms=best_terms,
        known_topic=confidence >= 0.35,
        question_kind=kind,
    )

