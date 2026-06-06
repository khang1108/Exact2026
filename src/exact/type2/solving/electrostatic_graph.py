from __future__ import annotations

from dataclasses import dataclass

import pint

from exact.type2.geometry_coordinates import build_coordinates
from exact.type2.geometry_extractor import extract_geometry_spec
from exact.type2.schemas import Extraction, Type2SolveResult
from exact.type2.solving.vector_solver import solve_geometry_vector_problem


@dataclass(frozen=True)
class ChargeNode:
    name: str
    point: str
    value: pint.Quantity


@dataclass(frozen=True)
class GeometryGraph:
    charges: dict[str, ChargeNode]
    distances: dict[frozenset[str], pint.Quantity]
    coordinates: dict[str, tuple[pint.Quantity, pint.Quantity]]
    target_charge: str
    layout: str


def solve_electrostatic_force_graph(extraction: Extraction) -> Type2SolveResult | None:
    """Compatibility wrapper around the GeometrySpec/vector solver pipeline."""

    return solve_geometry_vector_problem(extraction)


def build_electrostatic_graph(extraction: Extraction) -> GeometryGraph | None:
    """Build the legacy graph shape from the new GeometrySpec representation."""

    spec = extract_geometry_spec(extraction)
    if spec is None or spec.target_body is None:
        return None
    coordinates = build_coordinates(spec)
    if coordinates is None:
        return None
    charges = {
        name: ChargeNode(name=body.name, point=body.point, value=body.value)
        for name, body in spec.bodies.items()
        if body.kind == "charge" and body.value is not None
    }
    if spec.target_body not in charges:
        return None
    distances = {
        frozenset((edge.a, edge.b)): edge.length
        for edge in spec.edges
    }
    return GeometryGraph(
        charges=charges,
        distances=distances,
        coordinates=coordinates.coordinates,
        target_charge=spec.target_body,
        layout=coordinates.layout,
    )
