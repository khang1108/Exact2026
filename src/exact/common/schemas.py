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
    """Normalized input sample."""

    query_id: str | None = Field(default=None, validation_alias=AliasChoices("id", "query_id"))
    question: str = Field(validation_alias=AliasChoices("question", "query"))
    type: Literal["type1", "type2"] | None = None
    premises: list[str] | None = None
    options: Any | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


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


class PredictionResponse(AppBaseModel):
    """Competition-facing prediction plus local metadata for debugging."""

    answer: str
    explanation: str
    fol: str | None = None
    cot: list[str] | None = None
    premises: list[str] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    id: str | None = None
    task_type: TaskType | None = None
    question_type: QuestionType = QuestionType.UNKNOWN
    unit: str | None = None
    error: str | None = None
    routing_diagnostics: dict[str, Any] | None = None


def to_official_response(response: PredictionResponse) -> dict[str, Any]:
    """Convert an internal response into the stable EXACT submission shape."""

    return {
        "answer": response.answer,
        "explanation": response.explanation,
        "fol": response.fol,
        "cot": response.cot,
        "premises": response.premises,
        "confidence": response.confidence,
    }
