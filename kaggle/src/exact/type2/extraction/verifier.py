from __future__ import annotations

from dataclasses import dataclass

from exact.type2.schemas import Extraction, Type2QuestionKind, Verification


EXTRACTION_VERIFICATION_ERROR = "type2_extraction_verification_failed"


@dataclass(frozen=True)
class ExtractionReview:
    verification: Verification
    warnings: tuple[str, ...] = ()


class Type2ExtractionVerifier:
    """Checklist-style verifier for extracted Type 2 problem structure."""

    def verify(self, extraction: Extraction) -> ExtractionReview:
        failures: list[str] = []
        warnings: list[str] = []

        if extraction.kind != Type2QuestionKind.CONCEPTUAL and not extraction.quantities:
            failures.append("No numerical quantities were extracted.")

        if extraction.kind != Type2QuestionKind.CONCEPTUAL and extraction.target is None:
            warnings.append("Target quantity is not explicit; formula retrieval will be broader.")

        for key, quantity in extraction.quantities.items():
            if not quantity.evidence.strip():
                warnings.append(f"Quantity `{key}` has no evidence span.")
            try:
                quantity.value.to_base_units()
            except Exception as exc:
                failures.append(f"Quantity `{key}` has an invalid or non-convertible unit: {exc}")

        duplicate_names = _duplicate_quantity_names(extraction)
        for name in duplicate_names:
            warnings.append(
                f"Multiple `{name}` quantities were extracted; object identity must remain clear."
            )

        if failures:
            return ExtractionReview(
                verification=Verification(False, "Extraction checklist failed: " + "; ".join(failures)),
                warnings=tuple(warnings),
            )

        message = "Extraction checklist passed."
        if warnings:
            message = f"{message} Warnings: {'; '.join(warnings)}"
        return ExtractionReview(verification=Verification(True, message), warnings=tuple(warnings))


def verify_type2_extraction(extraction: Extraction) -> ExtractionReview:
    return Type2ExtractionVerifier().verify(extraction)


def _duplicate_quantity_names(extraction: Extraction) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for quantity in extraction.quantities.values():
        counts[quantity.name] = counts.get(quantity.name, 0) + 1
    return tuple(sorted(name for name, count in counts.items() if count > 1))
