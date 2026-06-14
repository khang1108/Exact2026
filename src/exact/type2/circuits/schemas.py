from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Scope = Literal["total", "component", "branch", "source", "primary", "secondary"]


@dataclass(frozen=True)
class CircuitQuantity:
    value: float
    unit: str
    original: str = ""


@dataclass(frozen=True)
class CircuitComponent:
    id: str
    kind: str
    properties: dict[str, CircuitQuantity] = field(default_factory=dict)
    model: str | None = None


@dataclass(frozen=True)
class CircuitTarget:
    quantity: str
    scope: Scope | None
    unit: str | None
    component_id: str | None = None
    branch_id: str | None = None


@dataclass(frozen=True)
class CircuitEvidence:
    text: str
    mapped_to: dict[str, Any]


@dataclass(frozen=True)
class CircuitContract:
    system_type: str
    target: CircuitTarget
    source: dict[str, CircuitQuantity | str | bool] = field(default_factory=dict)
    components: tuple[CircuitComponent, ...] = ()
    topology: dict[str, Any] = field(default_factory=dict)
    knowns: dict[str, CircuitQuantity] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)
    primary: dict[str, CircuitQuantity] = field(default_factory=dict)
    secondary: dict[str, CircuitQuantity] = field(default_factory=dict)
    domain: str = "circuits"
    parse_confidence: float = 0.0
    evidence: tuple[CircuitEvidence, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class CircuitValidationIssue:
    reason: str
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatedCircuitContract:
    contract: CircuitContract

