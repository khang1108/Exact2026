from __future__ import annotations

import re

from exact.type2.contract.schemas import (
    ContractBody,
    ContractChargeValue,
    ContractConstraint,
    ContractEvidence,
    ContractTarget,
    PhysicsSceneContract,
)
from exact.type2.geometry_extractor import extract_geometry_spec
from exact.type2.geometry_model import GeometrySpec
from exact.type2.schemas import Extraction


def extract_contract(extraction: Extraction) -> PhysicsSceneContract | None:
    """Parser boundary for electrostatics graph solving.

    This function may inspect the natural-language extraction. Downstream
    deterministic solvers must consume only the returned contract.
    """

    spec = extract_geometry_spec(extraction)
    if spec is None:
        return None
    return contract_from_geometry_spec(spec, extraction)


def contract_from_geometry_spec(spec: GeometrySpec, extraction: Extraction) -> PhysicsSceneContract:
    bodies = tuple(_contract_body(body_id, body) for body_id, body in spec.bodies.items())
    referenced_points = set(spec.points)
    referenced_points.update(body.point for body in spec.bodies.values() if body.point)
    for edge in spec.edges:
        referenced_points.update((edge.a, edge.b))
    for constraint in spec.constraints:
        referenced_points.update(constraint.points)
    points = tuple(sorted(referenced_points))
    constraints = tuple(_contract_constraint(constraint) for constraint in spec.constraints)
    distance_constraints = tuple(
        ContractConstraint(
            type="distance",
            points=(edge.a, edge.b),
            value=float(edge.length.to("m").magnitude),
            unit="m",
            original=edge.evidence,
            evidence=(ContractEvidence(edge.evidence, {"type": "distance", "from": edge.a, "to": edge.b}),),
        )
        for edge in spec.edges
    )
    target = _contract_target(spec, extraction, points)
    return PhysicsSceneContract(
        bodies=bodies,
        points=points,
        constraints=(*constraints, *distance_constraints),
        target=target,
        parse_confidence=0.75 if spec.notes else 0.86,
        evidence=tuple(evidence for body in bodies for evidence in body.evidence),
        unresolved=spec.notes,
    )


def _contract_body(body_id: str, body) -> ContractBody:
    value = None
    if body.value is not None:
        magnitude = float(abs(body.value.to("C").magnitude))
        sign = "+" if float(body.value.to("C").magnitude) >= 0 else "-"
        value = ContractChargeValue(
            magnitude=magnitude,
            unit="C",
            sign=sign,
            original=body.evidence,
        )
    mapped = {
        "id": body_id,
        "kind": body.kind,
        "role": body.role,
        "point": body.point,
    }
    return ContractBody(
        id=body_id,
        kind="charge",
        value=value,
        point=body.point,
        role=body.role,
        evidence=(ContractEvidence(body.evidence or body_id, mapped),),
    )


def _contract_constraint(constraint) -> ContractConstraint:
    if constraint.kind == "midpoint":
        mapped = {"type": "midpoint", "point": constraint.points[0] if constraint.points else None}
        return ContractConstraint(
            type="midpoint",
            points=constraint.points,
            original=constraint.evidence,
            data=dict(constraint.data),
            evidence=(ContractEvidence(constraint.evidence, mapped),),
        )
    if constraint.kind == "perpendicular":
        return ContractConstraint(type="perpendicular", points=constraint.points, original=constraint.evidence)
    if constraint.kind in {"on_line", "parallel"}:
        return ContractConstraint(type=constraint.kind, points=constraint.points, original=constraint.evidence)
    if constraint.kind == "shape" and constraint.shape:
        shape = constraint.shape.lower().replace("_", " ")
        if "equilateral" in shape:
            return ContractConstraint(type="equilateral", points=constraint.points, original=constraint.evidence)
        if "right" in shape:
            return ContractConstraint(type="right_triangle", points=constraint.points, original=constraint.evidence)
    if constraint.kind == "distance" and constraint.value is not None and len(constraint.points) >= 2:
        return ContractConstraint(
            type="distance",
            points=constraint.points[:2],
            value=float(constraint.value.to("m").magnitude),
            unit="m",
            original=constraint.evidence,
        )
    return ContractConstraint(type="on_line", points=constraint.points, original=constraint.evidence)


def _contract_target(spec: GeometrySpec, extraction: Extraction, points: tuple[str, ...]) -> ContractTarget:
    quantity = "electric_force" if spec.target_quantity == "force" else spec.target_quantity
    if quantity not in {"electric_field", "electric_force", "electric_potential", "potential_energy"}:
        quantity = "electric_force" if extraction.target == "force" else "electric_field"

    target_body = spec.target_body
    target_point = None
    if quantity == "electric_force" and target_body and target_body in spec.bodies:
        target_point = spec.bodies[target_body].point
    if quantity == "electric_field":
        target_point = _field_target_point(extraction.normalized_question, points, spec)

    sources = tuple(
        body_id
        for body_id, body in spec.bodies.items()
        if body_id != target_body and body.value is not None
    )
    output = "direction" if spec.target.output == "direction" else "magnitude"
    return ContractTarget(
        quantity=quantity,
        at=target_point,
        body=target_body,
        caused_by=sources,
        output=output,
        evidence=(ContractEvidence(extraction.normalized_question, {"quantity": quantity, "at": target_point}),),
    )


def _field_target_point(text: str, points: tuple[str, ...], spec: GeometrySpec) -> str | None:
    patterns = (
        r"(?:electric field(?: strength| intensity)?|field strength).*?\bat\s+(?:point\s+|vertex\s+)?(?P<point>[A-Z])\b",
        r"\bat\s+(?:point\s+|vertex\s+)?(?P<point>[A-Z])\b.*?(?:electric field(?: strength| intensity)?|field strength)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group("point").upper()
    occupied = {body.point for body in spec.bodies.values() if body.point}
    candidates = sorted(point for point in points if point not in occupied)
    if len(candidates) == 1:
        return candidates[0]
    return None
