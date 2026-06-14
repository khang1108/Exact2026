from __future__ import annotations

from exact.type2.geometry_model import (
    Body,
    BodyKind,
    GeometryBody,
    GeometryConstraint,
    GeometryEdge,
    GeometrySpec,
    TargetSpec,
)
from exact.type2.geometry_parser import parse_geometry
from exact.type2.object_parser import normalize_math_text, parse_objects
from exact.type2.schemas import Extraction


def extract_geometry_spec(extraction: Extraction) -> GeometrySpec | None:
    """Layered geometry extraction entrypoint.

    Layer 1 parses physical objects and roles.
    Layer 2 parses point/shape constraints.
    Later layers build coordinates and solve physics.
    """

    text = normalize_math_text(extraction.normalized_question)
    bodies = parse_objects(text)
    if not bodies and not any(key in extraction.quantities for key in ("force", "force_2")):
        return None

    points, constraints, edges, target, shape_hints, metadata = parse_geometry(
        text,
        bodies,
        extraction.target,
    )
    notes: list[str] = []
    if target.body is None and len(bodies) >= 3:
        notes.append("Could not identify target body from object roles or target wording.")
    if len(bodies) >= 3 and not edges and not shape_hints:
        notes.append("Geometry has bodies but no usable edge or shape constraints.")

    return GeometrySpec(
        bodies=bodies,
        points=points,
        constraints=constraints,
        edges=edges,
        target=target,
        shape_hints=shape_hints,
        notes=tuple(notes),
        source_text=text,
        metadata=metadata,
    )


__all__ = [
    "Body",
    "BodyKind",
    "GeometryBody",
    "GeometryConstraint",
    "GeometryEdge",
    "GeometrySpec",
    "TargetSpec",
    "extract_geometry_spec",
]
