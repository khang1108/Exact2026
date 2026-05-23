from __future__ import annotations

from exact.type2.extraction.verifier import (
    EXTRACTION_VERIFICATION_ERROR,
    verify_type2_extraction,
)
from exact.type2.schemas import Extraction, Type2QuestionKind


def test_extraction_verifier_rejects_numerical_without_quantities() -> None:
    extraction = Extraction(
        kind=Type2QuestionKind.NUMERICAL,
        normalized_question="Find the current.",
        target="current",
        quantities={},
    )

    review = verify_type2_extraction(extraction)

    assert review.verification.ok is False
    assert "No numerical quantities" in review.verification.message
    assert EXTRACTION_VERIFICATION_ERROR == "type2_extraction_verification_failed"


def test_extraction_verifier_allows_conceptual_without_quantities() -> None:
    extraction = Extraction(
        kind=Type2QuestionKind.CONCEPTUAL,
        normalized_question="Explain why current increases.",
        target=None,
        quantities={},
    )

    review = verify_type2_extraction(extraction)

    assert review.verification.ok is True
