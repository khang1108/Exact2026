"""Shared request and response schemas for the Type 2 EXACT pipeline.

Dataset-specific parsing should live under `exact.datasets`; objects here
describe the normalized boundary between the product surface and Type 2.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class TaskType(str, Enum):
    """Task family currently implemented by the application."""
    TYPE1_LOGIC = "type1_logic"
    TYPE2_PHYSICS = "type2_physics"
    UNKNOWN = "unknown"


class QuestionType(str, Enum):
    """Question shape emitted by the Type 2 pipeline."""

    # Type 1: MCQ, 
    MCQ = "mcq"
    YNU = "ynu" # yes no uncertain

    OPEN_ENDED = "open_ended"
    NUMERICAL = "numerical"
    UNKNOWN = "unknown"


class AppBaseModel(BaseModel):
    """Strict base class for internal responses and controlled schema objects."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class InboundBaseModel(BaseModel):
    """Lenient base class for external organizer/API payloads."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        str_strip_whitespace=True,
    )


class PredictionRequest(InboundBaseModel):
    """Unified Type 1 / Type 2 prediction input.

    The public API uses ``query_id`` and ``query``. Legacy callers may still
    send ``id`` and ``question``.
    """

    query_id: str | None = Field(default=None, validation_alias=AliasChoices("query_id", "id"))
    question: str = Field(validation_alias=AliasChoices("query", "question"))
    type: Literal["type1", "type2"] | None = None
    premises: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("premises", "premises-NL", "premises_nl"),
    )
    options: list[str] | dict[str, str] | None = None

    @property
    def id(self) -> str | None:
        """Compatibility alias used by existing Type 2 pipelines."""

        return self.query_id

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value

    @field_validator("premises")
    @classmethod
    def clean_premises(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [item.strip() for item in value if item and item.strip()]


class UnifiedPredictionRequest(PredictionRequest):
    """Strict public ``/predict`` contract for routing both task types."""

    query_id: str = Field(validation_alias=AliasChoices("query_id", "id"))
    type: Literal["type1", "type2"]


class ParsePremisesRequest(InboundBaseModel):
    """Input for the standalone ``/parser`` endpoint: raw NL premises only."""

    premises: list[str] = Field(
        validation_alias=AliasChoices("premises", "premises-NL", "premises_nl"),
    )

    @field_validator("premises")
    @classmethod
    def premises_must_not_be_empty(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("premises must contain at least one non-empty string")
        return cleaned


class ParsedPremise(AppBaseModel):
    """One natural-language premise translated to FOL."""

    id: str
    original_text: str
    fol: str
    ast: dict[str, Any]


class ParsePremisesResponse(AppBaseModel):
    """FOL translations aligned 1:1 with the input premises."""

    premises: list[ParsedPremise]


class ParserResponse(AppBaseModel):
    """Full parse result from the /parser endpoint."""

    premises: list[ParsedPremise]
    verified: bool
    issues: list[str]
    renames: list[dict[str, Any]]


class QParserRequest(InboundBaseModel):
    """Input for the ``/qparser`` endpoint: a question, its options, and premises."""

    question: str = Field(validation_alias=AliasChoices("question", "query"))
    premises: list[str] = Field(
        validation_alias=AliasChoices("premises", "premises-NL", "premises_nl"),
    )
    options: Any | None = None

    @field_validator("premises")
    @classmethod
    def premises_must_not_be_empty(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("premises must contain at least one non-empty string")
        return cleaned

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


class QParserResponse(AppBaseModel):
    """QuerySpec inspection result from the /qparser endpoint (no solving)."""

    question_format: str
    solver_mode: str
    can_interpretation: str
    main_claim_text: str | None
    main_claim_fol: str | None
    negate_claim: bool
    supported: bool
    issues: list[str]
    marker_style: str | None = None
    role_distribution: dict[str, int] | None = None
    extraction_diagnostics: list[str] = []
    option_claims: list[dict[str, Any]]
    proof_connectivity: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(AppBaseModel):
    """Competition-facing prediction plus local metadata for debugging.

    Field names follow the official EXACT submission spec (section 4.2).
    Legacy callers that construct with ``id=...`` are supported via the
    ``AliasChoices`` on ``query_id``.
    """

    # --- Official submission fields (section 4.2) ----------------------------
    query_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("query_id", "id"),
    )
    answer: str
    explanation: str
    unit: str = ""                          # "" for Type 1; ASCII unit for Type 2
    premises_used: list[int] | None = None  # 0-based indices; [] for Type 2
    reasoning: dict[str, Any] | None = None # structured evidence; null if unused

    # --- Internal / debug fields (not submitted) -----------------------------
    task_type: TaskType | None = None
    question_type: QuestionType = QuestionType.UNKNOWN
    fol: str | None = None
    cot: list[str] | None = None
    premises: list[str] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    error: str | None = None
    routing_diagnostics: dict[str, Any] | None = None


def to_official_response(response: PredictionResponse) -> dict[str, Any]:
    """Convert an internal response into the stable EXACT submission shape.

    Returns only the fields defined in the competition spec (section 4.2).
    """

    return {
        "query_id": response.query_id,
        "answer": response.answer,
        "unit": response.unit,
        "explanation": response.explanation,
        "premises_used": response.premises_used if response.premises_used is not None else [],
        "reasoning": response.reasoning,
    }
