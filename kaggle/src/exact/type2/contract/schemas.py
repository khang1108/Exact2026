from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pint


BodyRole = Literal["source", "target", "test_charge", "unknown"]
BodyKind = Literal["charge", "point_charge"]
ConstraintType = Literal[
    "distance",
    "angle",
    "midpoint",
    "on_line",
    "perpendicular_bisector",
    "perpendicular",
    "parallel",
    "equilateral",
    "equilateral_triangle",
    "right_triangle",
    "square",
    "coordinate",
    "same_point",
]
TargetQuantity = Literal[
    "electric_field",
    "electric_force",
    "electric_potential",
    "potential_energy",
    "zero_electric_field_location",
    "zero_potential_location",
    "equilibrium_condition",
    "unknown_charge",
    "unknown_position",
    "magnitude",
    "direction",
    "vector",
    "symbolic_expression",
    "numeric_value",
]
TargetOutput = Literal["magnitude", "magnitude_direction", "direction", "vector", "symbolic_expression", "numeric_value"]


@dataclass(frozen=True)
class ContractEvidence:
    text: str
    mapped_to: dict[str, Any]


@dataclass(frozen=True)
class ContractChargeValue:
    magnitude: float | None
    unit: str
    sign: Literal["+", "-", "unknown"] = "unknown"
    original: str = ""

    @property
    def signed_magnitude(self) -> float | None:
        if self.magnitude is None:
            return None
        return -self.magnitude if self.sign == "-" else self.magnitude


@dataclass(frozen=True)
class ContractBody:
    id: str
    kind: BodyKind
    value: ContractChargeValue | None
    point: str | None
    role: BodyRole
    evidence: tuple[ContractEvidence, ...] = ()


@dataclass(frozen=True)
class ContractConstraint:
    type: ConstraintType
    points: tuple[str, ...] = ()
    value: float | None = None
    unit: str | None = None
    original: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[ContractEvidence, ...] = ()


@dataclass(frozen=True)
class ContractUnknown:
    id: str
    kind: Literal["coordinate", "charge"]
    point: str | None = None
    axis: Literal["x", "y"] | None = None
    unit: str | None = None


@dataclass(frozen=True)
class ContractTarget:
    quantity: TargetQuantity
    at: str | None = None
    point: str | None = None
    body: str | None = None
    caused_by: tuple[str, ...] = ()
    output: TargetOutput = "magnitude"
    unit: str | None = None
    condition: str | None = None
    evidence: tuple[ContractEvidence, ...] = ()


@dataclass(frozen=True)
class PhysicsSceneContract:
    bodies: tuple[ContractBody, ...]
    points: tuple[str, ...]
    constraints: tuple[ContractConstraint, ...]
    target: ContractTarget
    domain: str = "electrostatics"
    system_type: str = "multi_charge_vector_field"
    unknowns: tuple[ContractUnknown, ...] = ()
    parse_confidence: float = 0.0
    evidence: tuple[ContractEvidence, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationIssue:
    reason: str
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatedPhysicsScene:
    contract: PhysicsSceneContract
    charge_values: dict[str, pint.Quantity]
    body_points: dict[str, str]
    source_ids: tuple[str, ...]
    target_point: str | None
    target_body: str | None
