from __future__ import annotations

from typing import Any

from exact.type2.contract.validate_contract import validate_contract
from exact.type2.geometry.coordinate_builder import build_contract_coordinates
from exact.type2.llm_parser.extract_contract import extract_contract
from exact.type2.schemas import Extraction


def build_fallback_geometry_context(extraction: Extraction) -> dict[str, Any] | None:
    """Build structured electrostatics geometry for PoT/fallback prompts.

    This reuses the deterministic contract stack, but it does not require the
    deterministic solver to accept the problem. PoT can use the resolved bodies,
    target, source list, and coordinates as grounding data.
    """

    contract = extract_contract(extraction)
    if contract is None:
        return None
    scene, issue = validate_contract(contract)
    if scene is None:
        return {
            "available": False,
            "reason": issue.reason if issue else "contract validation failed",
            "missing": list(issue.missing) if issue else [],
            "contract": _contract_payload(contract),
        }
    coordinates = build_contract_coordinates(scene)
    payload = {
        "available": coordinates is not None,
        "reason": None if coordinates is not None else "coordinates could not be resolved",
        "contract": _contract_payload(contract),
        "target": {
            "quantity": contract.target.quantity,
            "point": scene.target_point,
            "body": scene.target_body,
            "output": contract.target.output,
            "unit": contract.target.unit,
        },
        "sources": [
            {
                "id": source_id,
                "charge_C": float(scene.charge_values[source_id].to("C").magnitude),
                "point": scene.body_points[source_id],
            }
            for source_id in scene.source_ids
            if source_id in scene.charge_values and source_id in scene.body_points
        ],
        "excluded_bodies": [
            body.id
            for body in contract.bodies
            if body.id not in scene.source_ids and body.point == scene.target_point
        ],
    }
    if coordinates is not None:
        payload["coordinates_m"] = {
            point: [float(x.to("m").magnitude), float(y.to("m").magnitude)]
            for point, (x, y) in coordinates.coordinates.items()
        }
        payload["layout"] = coordinates.layout
        payload["notes"] = list(coordinates.notes)
    return payload


def _contract_payload(contract) -> dict[str, Any]:
    return {
        "domain": contract.domain,
        "system_type": contract.system_type,
        "points": list(contract.points),
        "bodies": [
            {
                "id": body.id,
                "role": body.role,
                "point": body.point,
                "charge_C": (
                    body.value.signed_magnitude
                    if body.value is not None and body.value.unit == "C"
                    else None
                ),
            }
            for body in contract.bodies
        ],
        "constraints": [
            {
                "type": constraint.type,
                "points": list(constraint.points),
                "value": constraint.value,
                "unit": constraint.unit,
                "data": constraint.data,
            }
            for constraint in contract.constraints
        ],
    }
