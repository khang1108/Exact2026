"""Shared request, response, routing, and evidence schemas for EXACT.

This module owns contracts that are used by both Type 1 logic and Type 2
physics pipelines, plus the API router, task router, dataset loader, and batch
prediction runner. Dataset-specific parsing should live under `exact.datasets`;
objects here describe the normalized boundary between the product surface and
all task-specific implementations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(str, Enum):
    """Top-level EXACT task family selected by the task router."""

    TYPE1_LOGIC = "type1_logic"
    TYPE2_PHYSICS = "type2_physics"
    UNKNOWN = "unknown"


class QuestionType(str, Enum):
    """Question shape used to choose answer semantics within a task pipeline."""

    MCQ = "mcq"
    YES_NO_UNCERTAIN = "yes_no_uncertain"
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
    """Normalized input sample for either Type 1 logic or Type 2 physics."""

    id: str | None = None
    question: str
    premises_nl: list[str] | None = Field(default=None, alias="premises-NL")

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value

    @property
    def inferred_task_type(self) -> TaskType:
        if self.premises_nl:
            return TaskType.TYPE1_LOGIC
        return TaskType.TYPE2_PHYSICS


class BatchPredictionRequest(InboundBaseModel):
    """Batch API payload containing one or more normalized prediction requests."""

    instances: list[PredictionRequest]

    @field_validator("instances")
    @classmethod
    def instances_must_not_be_empty(
        cls,
        value: list[PredictionRequest],
    ) -> list[PredictionRequest]:
        if not value:
            raise ValueError("instances must not be empty")
        return value


class ErrorInfo(AppBaseModel):
    """Structured error payload for failures that can be surfaced to callers."""

    code: str
    message: str
    recoverable: bool = False


class ProofStep(AppBaseModel):
    """Task-agnostic explanation step with optional rule and support links."""

    step_id: str
    derived: str
    rule_id: str | None = None
    support: list[str] = Field(default_factory=list)
    natural_language: str | None = None


class Type1Evidence(AppBaseModel):
    """Symbolic proof evidence emitted by the Type 1 logic pipeline."""

    solver: str
    label: Literal["entailed", "contradicted", "unknown"]
    premises_used: list[str] = Field(default_factory=list)
    proof_trace: list[ProofStep] = Field(default_factory=list)


class EquationStep(AppBaseModel):
    """One trace step from a Type 2 numerical or formula-based solution."""

    step_id: str
    expression: str
    result: str | None = None
    unit: str | None = None
    natural_language: str | None = None


class Type2Evidence(AppBaseModel):
    """Equation and unit-conversion evidence emitted by the Type 2 pipeline."""

    executor: str
    formula: str | None = None
    given: dict[str, Any] = Field(default_factory=dict)
    unit_conversions: list[str] = Field(default_factory=list)
    equation_trace: list[EquationStep] = Field(default_factory=list)


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


class OfficialPredictionResponse(AppBaseModel):
    """Official EXACT API response shape.

    The challenge requires `answer` and `explanation`; the remaining fields are
    optional evidence that can improve reasoning-depth review.
    """

    answer: str
    explanation: str
    fol: str | None = None
    cot: list[str] | None = None
    premises: list[str] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class BatchPredictionResponse(AppBaseModel):
    """Batch API response containing one prediction result per input instance."""

    predictions: list[PredictionResponse]


class OfficialBatchPredictionResponse(AppBaseModel):
    """Batch wrapper using the official per-item response shape."""

    predictions: list[OfficialPredictionResponse]


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
