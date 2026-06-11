"""Structured response schemas produced by Type 1 parser operations.

Each parser operation has a narrow output contract. These Pydantic models are
sent to vLLM as JSON schemas and then validate the returned model content.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ParserResult(BaseModel):
    """Base schema that rejects fields not requested by the parser operation."""

    model_config = ConfigDict(extra="forbid")


class RephraseResult(ParserResult):
    """Result of minimally rewriting a sentence for clearer FOL parsing."""

    rephrased: str


class QuantifiedResult(ParserResult):
    """Result of extracting one outer quantifier and its remaining sentence."""

    quantifier: Literal["ForAll", "ThereExists"]
    variable: str
    scope_sentence: str


class LogicalResult(ParserResult):
    """Result of splitting a sentence at its outermost logical operator."""

    operator: Literal["AND", "OR", "IMPLIES", "IFF", "NOT"]
    left_operand: str
    right_operand: str | None


class AtomicResult(ParserResult):
    """Result of extracting a predicate and arguments from an atomic sentence."""

    predicate: str
    arguments: list[str]
    negated: bool


class CoreferenceResult(ParserResult):
    """Result of resolving pronouns in a right-hand clause."""

    resolved_right: str
