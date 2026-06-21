from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import pint


class Type2QuestionKind(str, Enum):
    NUMERICAL = "numerical"
    CONCEPTUAL = "conceptual"
    MIXED = "mixed"


@dataclass(frozen=True)
class Quantity:
    name: str
    value: pint.Quantity
    evidence: str
    confidence: float = 1.0


@dataclass(frozen=True)
class Formula:
    id: str
    domain: str
    subfield: str
    target: str
    required: tuple[str, ...]
    output_unit: str
    expression: str
    explanation_template: str
    keywords: tuple[str, ...]
    solve: Callable[[dict[str, pint.Quantity]], pint.Quantity]
    variables: dict[str, str] = field(default_factory=dict)
    conditions: tuple[str, ...] = ()
    common_mistakes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Extraction:
    kind: Type2QuestionKind
    normalized_question: str
    target: str | None
    quantities: dict[str, Quantity]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verification:
    ok: bool
    message: str


@dataclass(frozen=True)
class Type2SolveResult:
    answer: str
    unit: str | None
    value: pint.Quantity | None
    formula: Formula | None
    extraction: Extraction
    verification: Verification
    cot: list[str] = field(default_factory=list)
    premises: list[str] = field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None
    routing_diagnostics: dict[str, Any] | None = None
    formula_ids_used: tuple[str, ...] = ()
    retrieved_formula_ids: tuple[str, ...] = ()
    generated_code: str | None = None
    code_semantic_summary: str | None = None
    computed_values: dict[str, Any] = field(default_factory=dict)
    verification_warnings: tuple[str, ...] = ()
