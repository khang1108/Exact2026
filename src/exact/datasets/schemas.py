"""
Schemas for EXACT dataset

Author: Khang P. Nguyen
Date: 2026-05-22

Example:
    payload = {
    "id": "q001",
    "premises-NL": [
        "If a student has high GPA, the student is eligible for scholarship.",
        "An has high GPA."
    ],
    "question": "Is An eligible for scholarship?",
    "difficulty": "easy",
    }

    request = PredictionRequest.model_validate(payload)

    print(request.inferred_task_type)
    print(request.premises_nl)
    """

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum

class Input(BaseModel):
    """Base Input Schema for both of types of documents"""
    id: str | None = Field(default=None, description="Document ID")
    question: str | None = Field(default=None, description="Question text")
    premises_nl : list[str] | None = Field(default=None, description="Premises list in natural language format")
    
    # Cho phép dùng alias lẫn tên field Python đều được.
    model_config = {
        "populate_by_name": True,
        "extra": "allow"
    }

class TaskType(str, Enum):
    TYPE1_LOGIC = "type1_logic"
    TYPE2_PHYSICS = "type2_physics"

    UNKNOWN = "unknown"

class QuestionType(str, Enum):
    MCQ = "mcq"
    YES_NO_UNCERTAIN = "yes_no_uncertain"
    OPEN_ENDED = "open_ended"
    NUMERICAL = "numerical"

    UNKNOWN = "unknown"

class AppBaseModel(BaseModel):
    """
    Strict base model for internal objects and outbound responses.
    Internal data should be controlled, so unknown fields are forbidden.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class InboundBaseModel(BaseModel):
    """
    Lenient base model for external inputs.

    Organizer/dataset payloads may contain metadata fields we do not use yet,
    so extra fields are allowed at the boundary.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        str_strip_whitespace=True,
    )


class PredictionRequest(InboundBaseModel):
    """
    Input sample sent to our API or loaded from dataset.

    Type 1:
        premises-NL + question

    Type 2:
        question only
    """

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
    code: str
    message: str
    recoverable: bool = False


class ProofStep(AppBaseModel):
    step_id: str
    derived: str
    rule_id: str | None = None
    support: list[str] = Field(default_factory=list)
    natural_language: str | None = None


class Type1Evidence(AppBaseModel):
    solver: str
    label: Literal["entailed", "contradicted", "unknown"]
    premises_used: list[str] = Field(default_factory=list)
    proof_trace: list[ProofStep] = Field(default_factory=list)


class EquationStep(AppBaseModel):
    step_id: str
    expression: str
    result: str | None = None
    unit: str | None = None
    natural_language: str | None = None


class Type2Evidence(AppBaseModel):
    executor: str
    formula: str | None = None
    given: dict[str, Any] = Field(default_factory=dict)
    unit_conversions: list[str] = Field(default_factory=list)
    equation_trace: list[EquationStep] = Field(default_factory=list)


class PredictionResponse(AppBaseModel):
    """
    Internal response object.

    For official submission/API, we can later export only:
        answer, explanation, unit
    if the organizer requires a minimal schema.
    """

    id: str | None = None
    task_type: TaskType
    question_type: QuestionType = QuestionType.UNKNOWN
    answer: str
    explanation: str
    unit: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: Type1Evidence | Type2Evidence | None = None
    error: ErrorInfo | None = None


class BatchPredictionResponse(AppBaseModel):
    predictions: list[PredictionResponse]


def to_official_response(response: PredictionResponse) -> dict[str, Any]:
    """
    Convert internal rich response into minimal official response.

    Use this if the competition API only accepts answer/explanation/unit.
    """
    data: dict[str, Any] = {
        "answer": response.answer,
        "explanation": response.explanation,
    }

    if response.unit is not None:
        data["unit"] = response.unit

    return data