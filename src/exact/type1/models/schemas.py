"""Schemas for Type 1 Logic tasks.

This module defines the Pydantic models for the request and response formats of Type 1 Logic tasks,
as well as the schema for First-Order Logic items and related concepts such as ``sorts``, ``constants``, and ``predicates``.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Literal, Optional

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
