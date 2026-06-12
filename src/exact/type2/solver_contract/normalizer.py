from __future__ import annotations

import math
from typing import Any

from exact.type2.extraction.llm_structured import SemanticContractSpec
from exact.type2.schemas import Extraction
from exact.type2.solver_contract.models import (
    ContractBody,
    ContractGeometry,
    ContractPoint,
    ContractTarget,
    GeometryRelation,
    ParsedQuantity,
    SolverContract,
)
from exact.type2.solver_contract.unit_parser import safe_parse_quantity

TARGET_MAPPING = {
    "force": "electric_force",
    "electric force": "electric_force",
    "net force": "electric_force",
    "electric field strength": "electric_field",
    "electric field": "electric_field",
    "field intensity": "electric_field",
    "voltage": "voltage",
    "potential difference": "voltage",
    "capacitance": "capacitance",
    "energy": "energy",
}

BODY_TYPE_MAPPING = {
    "q": "charge",
    "charge": "charge",
    "test charge": "charge",
    "source charge": "charge",
    "r": "resistor",
    "resistor": "resistor",
    "c": "capacitor",
    "capacitor": "capacitor",
    "f": "force",
    "force": "force",
    "mass": "mass",
    "particle": "particle",
}

GEOMETRY_MAPPING = {
    "perpendicular bisector": "perpendicular_bisector",
    "on the mediator": "perpendicular_bisector",
    "collinear": "collinear",
    "straight line": "collinear",
    "triangle": "triangle",
    "right triangle": "right_triangle",
    "equilateral triangle": "equilateral_triangle",
}

def canonicalize_target(raw: str) -> str:
    if not raw:
        return "unknown"
    lower = raw.lower().strip()
    return TARGET_MAPPING.get(lower, lower.replace(" ", "_"))

def canonicalize_body_type(raw: str) -> str:
    if not raw:
        return "unknown"
    lower = raw.lower().strip()
    return BODY_TYPE_MAPPING.get(lower, lower.replace(" ", "_"))

def canonicalize_geometry_family(raw: str) -> str:
    if not raw:
        return "none"
    lower = raw.lower().strip()
    return GEOMETRY_MAPPING.get(lower, lower.replace(" ", "_"))


def _quantity_to_float_meters(quantity: Any) -> float | None:
    try:
        return float(quantity.to("meter").magnitude)
    except Exception:
        try:
            return float(quantity.magnitude)
        except Exception:
            return None


def _collinear_order_from_distances(relations: list[GeometryRelation]) -> tuple[str, ...] | None:
    distances: dict[frozenset[str], float] = {}
    for rel in relations:
        if rel.type not in {"distance", "length", "side_length"} or len(rel.points) != 2:
            continue
        if rel.value is None or not rel.value.ok or rel.value.quantity is None:
            continue
        value = _quantity_to_float_meters(rel.value.quantity)
        if value is None:
            continue
        distances[frozenset(rel.points)] = value

    if len(distances) < 3:
        return None
    points = sorted({point for pair in distances for point in pair})
    if len(points) != 3:
        return None

    pairs = []
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            value = distances.get(frozenset((a, b)))
            if value is None:
                return None
            pairs.append((value, a, b))
    pairs.sort(reverse=True)
    longest, end_a, end_b = pairs[0]
    middle = next(point for point in points if point not in {end_a, end_b})
    left = distances[frozenset((end_a, middle))]
    right = distances[frozenset((middle, end_b))]
    if math.isclose(left + right, longest, rel_tol=1e-9, abs_tol=1e-12):
        return (end_a, middle, end_b)
    return None


def _hydrate_missing_target_body(
    *,
    target: ContractTarget,
    bodies: list[ContractBody],
    geom_points: dict[str, ContractPoint],
    quantities: dict[str, ParsedQuantity],
    extraction: Extraction | None,
) -> bool:
    if extraction is None or not target.body:
        return False
    if any(body.id == target.body for body in bodies):
        return False

    try:
        from exact.type2.object_parser import parse_objects
    except Exception:
        return False

    objects = parse_objects(extraction.normalized_question)
    source = objects.get(target.body)
    if source is None:
        return False

    point = source.point or target.point or target.at
    role = source.role if source.role not in {None, "unknown"} else "target"
    value = source.value
    if point and point not in geom_points:
        geom_points[point] = ContractPoint(id=point)
    if value is not None:
        quantities[target.body] = ParsedQuantity(
            raw=str(value),
            quantity=value,
            ok=True,
            error=None,
        )
    bodies.append(
        ContractBody(
            id=source.id,
            body_type=canonicalize_body_type(source.kind),
            value=value,
            unit=str(value.units) if value is not None else None,
            point=point,
            role=role,
            properties={
                "id": source.id,
                "type": source.kind,
                "point": point,
                "role": role,
                "evidence": source.evidence,
                "hydrated_from": "heuristic_exact_id",
            },
        )
    )
    return True


def normalize_contract(spec: SemanticContractSpec, ureg: Any, extraction: Extraction | None = None) -> SolverContract:
    unresolved: list[str] = []
    quantities: dict[str, ParsedQuantity] = {}
    
    # Target
    raw_target = spec.target or {}
    t_quantity = canonicalize_target(raw_target.get("quantity", "unknown"))
    target = ContractTarget(
        quantity=t_quantity,
        at=raw_target.get("point") or raw_target.get("at"),
        body=raw_target.get("body"),
        point=raw_target.get("point") or raw_target.get("at"),
        output=raw_target.get("output", "magnitude"),
        unit=raw_target.get("unit"),
    )

    # Bodies
    bodies: list[ContractBody] = []
    geom_points: dict[str, ContractPoint] = {}
    
    for b_dict in spec.bodies:
        b_id = b_dict.get("id", "unknown_id")
        raw_val = b_dict.get("value")
        val = None
        if raw_val:
            parsed = safe_parse_quantity(str(raw_val), ureg)
            quantities[b_id] = parsed
            if parsed.ok:
                val = parsed.quantity
            else:
                unresolved.append(f"quantity_parse_failed:{b_id}")
        
        b_point = b_dict.get("point")
        if b_point and b_point not in geom_points:
            geom_points[b_point] = ContractPoint(id=b_point)
            
        bodies.append(ContractBody(
            id=b_id,
            body_type=canonicalize_body_type(b_dict.get("type", "unknown")),
            value=val,
            unit=b_dict.get("unit"),
            point=b_point,
            role=b_dict.get("role", "given"),
            properties=b_dict
        ))

    hydrated_target = _hydrate_missing_target_body(
        target=target,
        bodies=bodies,
        geom_points=geom_points,
        quantities=quantities,
        extraction=extraction,
    )

    # Geometry
    raw_geom = spec.geometry or {}
    geom_family = canonicalize_geometry_family(raw_geom.get("family", "none"))
    relations: list[GeometryRelation] = []
    
    for i, rel in enumerate(raw_geom.get("relations", [])):
        if isinstance(rel, (list, tuple)) and len(rel) >= 3:
            rel = {"type": "distance", "points": [rel[0], rel[1]], "value": rel[2]}
        if not isinstance(rel, dict):
            unresolved.append(f"relation_parse_failed:{rel}")
            continue
        rel_type = rel.get("type", "distance").lower().replace(" ", "_")
        points = rel.get("points", [])
        if not isinstance(points, list):
            points = [points]
        
        for p in points:
            if p not in geom_points:
                geom_points[p] = ContractPoint(id=p)
        if rel_type == "right_angle" and len(points) >= 3:
            vertex = points[1]
            geom_points[vertex] = ContractPoint(id=vertex, role="right_angle")
                
        raw_v = rel.get("value")
        parsed = None
        if raw_v:
            parsed = safe_parse_quantity(str(raw_v), ureg)
            quantities[f"rel_{i}"] = parsed
            if not parsed.ok:
                unresolved.append(f"relation_parse_failed:{points}")
                
        relations.append(GeometryRelation(
            type=rel_type,
            points=tuple(points),
            value=parsed,
            raw_value=str(raw_v) if raw_v else None
        ))

    geometry = ContractGeometry(
        family=geom_family,
        points=geom_points,
        relations=tuple(relations),
        point_order=None
    )

    collinear_order = _collinear_order_from_distances(relations)
    if collinear_order is not None:
        geometry = ContractGeometry(
            family="collinear",
            points=geom_points,
            relations=tuple(relations),
            point_order=collinear_order,
        )

    return SolverContract(
        domain="physics", # General domain
        answer_mode="numeric", # This gets overridden by policy logic later
        target=target,
        bodies=tuple(bodies),
        geometry=geometry,
        quantities=quantities,
        unresolved=tuple(unresolved),
        diagnostics={
            "raw_target": raw_target,
            "raw_geom_family": raw_geom.get("family"),
            "hydrated_missing_target_body": hydrated_target,
        },
        semantic_contract=spec
    )
