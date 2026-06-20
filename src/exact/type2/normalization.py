from __future__ import annotations

from dataclasses import dataclass

from exact.common.schemas import PredictionRequest


@dataclass(frozen=True)
class Type2RequestCompat:
    query_id: str
    problem_text: str


def normalize_request(request: PredictionRequest | Type2RequestCompat) -> Type2RequestCompat:
    if isinstance(request, Type2RequestCompat):
        return request
    return Type2RequestCompat(
        query_id=request.query_id or "",
        problem_text=request.question,
    )


def normalize_units_and_notation(extraction, settings=None):
    """Compatibility no-op for callers using the legacy extraction tests.

    The reference Type 2 extractor already normalizes notation before returning
    either the new extraction object or the compatibility shim object.
    """

    return extraction
