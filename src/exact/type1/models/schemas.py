"""Schemas for Type 1 Logic tasks.

This module defines the Pydantic models for the request and response formats of Type 1 Logic tasks,
as well as the schema for First-Order Logic items and related concepts such as ``sorts``, ``constants``, and ``predicates``.
"""

from __future__ import annotations
from enum import Enum

from pydantic import BaseModel
from dataclasses import dataclass
from typing import Literal, Optional

from exact.common.schemas import QuestionType

__all__ = ["FolItem", "Type1Request", "Type1Response"]


class FolItem(BaseModel):
    """Schema for a First-Order Logic item."""

    id: str
    original_text: str
    fol: str
    ast: dict | None = None
    role: Literal["premise", "goal", "option"]
    used: bool = False


class Type1Request(BaseModel):
    """Request schema for Logic tasks."""

    premises_nl: list[str]
    question: str
    options: Optional[dict[str, str]] | None = None


class Type1Response(BaseModel):
    """Response schema for Logic tasks"""

    answer: str
    confidence: float
    cot: str
    fol: list[FolItem]
    used_premises: list[str]


class Sort(BaseModel):
    """Schema for a sort that describes a type/domain of objects."""

    name: str
    description: Optional[str] = None


class Constant(BaseModel):
    """Schema for a constant that represents a named object."""

    name: str
    sort: str
    aliases: list[str]


class Predicate(BaseModel):
    """Schema for a predicate that represents a relation or a true/false property."""

    name: str
    arg_sorts: list[str]
    description: Optional[str] = None
    aliases: list[str]

    @property
    def arity(self) -> int:
        """Returns the arity of the predicate."""
        return len(self.arg_sorts)

class QueryMode(Enum):
    """Enum for query modes."""

    MUST_TRUE = "must_true"
    MUST_FALSE = "must_false"
    POSSIBLY_TRUE = "possibly_true"
    POSSIBLY_FALSE = "possibly_false"

    WHICH_ENTAILED = "w_entailed"
    WHICH_NOT_ENTAILED = "w_not_entailed"
    WHICH_POSSIBLY_TRUE = "w_possibly_true"
    WHICH_POSSIBLY_FALSE = "w_possibly_false"

    MIN_PROOF = "min_proof"
    STRONGEST_SUPPORTED = "strongest_supported"

class SolverOp(Enum):
    CHECK_ENTAILMENT = "check_entailment"
    CHECK_CONTRADICTION = "check_contradiction"
    CHECK_SAT = "check_satisfiability"
    RANK_BY_ENTAILMENT = "rank_by_entailment"
    RANK_BY_MIN_PROOF = "rank_by_min_proof"

@dataclass(frozen=True)
class QuerySpec:
    question_id: str
    raw_question: str

    question_type: QuestionType

    claim_text: str | None

    assumptions: list[str]
    solver_op: SolverOp
