from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pint


@dataclass(frozen=True)
class ContractKnown:
    name: str
    value: pint.Quantity | None
    dimension: str
    unit: str | None
    evidence: str


@dataclass(frozen=True)
class PhysicsConstraint:
    kind: str
    variables: tuple[str, ...]
    relation: str
    value: str | float | None = None


@dataclass(frozen=True)
class PhysicsContract:
    target: str
    knowns: dict[str, ContractKnown]
    unknowns: tuple[str, ...]
    topic: str
    subtopic: str
    principle: str
    expected_dimension: str
    expected_unit: str | None
    constraints: tuple[PhysicsConstraint, ...] = ()
    formula_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    source: str = "heuristic"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicsContractValidation:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    confidence_delta: float = 0.0

