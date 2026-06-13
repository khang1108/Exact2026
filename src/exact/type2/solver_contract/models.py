from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContractTarget:
    quantity: str
    at: str | None = None
    point: str | None = None
    body: str | None = None
    caused_by: tuple[str, ...] = ()
    output: str = "magnitude"
    unit: str | None = None
    condition: str | None = None


@dataclass(frozen=True)
class ContractBody:
    id: str
    body_type: str  # "charge", "force", "mass", "particle", "field_point", "capacitor", "resistor"
    value: Any | None = None
    unit: str | None = None
    point: str | None = None
    role: str = "given"  # "source", "target", "test", "given", "unknown"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedQuantity:
    raw: str
    quantity: Any | None
    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class ContractPoint:
    id: str
    role: str | None = None


@dataclass(frozen=True)
class GeometryRelation:
    type: str
    points: tuple[str, ...]
    value: ParsedQuantity | None
    raw_value: str | None


@dataclass(frozen=True)
class ContractGeometry:
    family: str
    points: dict[str, ContractPoint] = field(default_factory=dict)
    relations: tuple[GeometryRelation, ...] = ()
    point_order: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SolverContract:
    domain: str
    answer_mode: str
    target: ContractTarget
    bodies: tuple[ContractBody, ...]
    geometry: ContractGeometry
    quantities: dict[str, ParsedQuantity] = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] | None = None
    semantic_contract: Any | None = None

    def has_unresolved(self) -> bool:
        return len(self.unresolved) > 0


@dataclass(frozen=True)
class ContractValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    required_missing: list[str] = field(default_factory=list)

