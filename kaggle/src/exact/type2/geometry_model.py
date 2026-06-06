from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pint


BodyKind = Literal["charge", "mass", "force_vector"]
BodyRole = Literal["source", "target", "test_charge", "unknown"]
ConstraintKind = Literal[
    "distance",
    "midpoint",
    "on_line",
    "between",
    "extension",
    "equidistant",
    "center",
    "remaining_vertex",
    "equal_length",
    "right_angle",
    "parallel",
    "perpendicular",
    "shape",
]


@dataclass(frozen=True)
class Body:
    id: str
    kind: BodyKind
    role: BodyRole = "unknown"
    value: pint.Quantity | None = None
    point: str | None = None
    sign: Literal["positive", "negative", "neutral"] | None = None
    symbolic_value: str | None = None
    evidence: str = ""

    @property
    def name(self) -> str:
        return self.id


@dataclass(frozen=True)
class GeometryEdge:
    a: str
    b: str
    length: pint.Quantity
    evidence: str


@dataclass(frozen=True)
class GeometryConstraint:
    kind: ConstraintKind
    points: tuple[str, ...] = ()
    value: pint.Quantity | None = None
    edges: tuple[tuple[str, str], ...] = ()
    shape: str | None = None
    evidence: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    body: str | None
    quantity: str | None
    output: Literal["magnitude", "direction", "symbolic", "unknown"] = "magnitude"


@dataclass(frozen=True)
class GeometrySpec:
    bodies: dict[str, Body]
    points: frozenset[str] = frozenset()
    constraints: tuple[GeometryConstraint, ...] = ()
    edges: tuple[GeometryEdge, ...] = ()
    target: TargetSpec = TargetSpec(body=None, quantity=None)
    shape_hints: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    source_text: str = ""
    metadata: dict[str, pint.Quantity] = field(default_factory=dict)

    @property
    def target_body(self) -> str | None:
        return self.target.body

    @property
    def target_quantity(self) -> str | None:
        return self.target.quantity


# Backward-compatible names used by the earlier geometry refactor.
GeometryBody = Body
