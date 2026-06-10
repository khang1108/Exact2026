from __future__ import annotations

from exact.type2.contract.normalize_units import normalize_charge, normalize_distance
from exact.type2.contract.schemas import (
    PhysicsSceneContract,
    ValidatedPhysicsScene,
    ValidationIssue,
)


def validate_contract(contract: PhysicsSceneContract) -> tuple[ValidatedPhysicsScene | None, ValidationIssue | None]:
    if contract.domain != "electrostatics":
        return None, ValidationIssue("domain is not electrostatics", ("domain",))
    points = set(contract.points)
    bodies = {body.id: body for body in contract.bodies}
    if not points:
        return None, ValidationIssue("contract has no explicit points", ("points",))
    if not bodies:
        return None, ValidationIssue("contract has no bodies", ("bodies",))

    body_points: dict[str, str] = {}
    charge_values = {}
    for body in contract.bodies:
        if body.kind not in {"charge", "point_charge"}:
            return None, ValidationIssue(f"unsupported body kind `{body.kind}`")
        if body.point is not None and body.point not in points:
            return None, ValidationIssue(
                f"body `{body.id}` references non-existing point `{body.point}`",
                (f"points({body.point})",),
            )
        if body.point is not None:
            body_points[body.id] = body.point
        if body.value is not None and body.value.signed_magnitude is not None:
            try:
                charge_values[body.id] = normalize_charge(body.value.signed_magnitude, body.value.unit)
            except Exception as exc:
                return None, ValidationIssue(f"charge `{body.id}` has invalid or unnormalized units: {exc}")

    for constraint in contract.constraints:
        for point in constraint.points:
            if point not in points:
                return None, ValidationIssue(
                    f"constraint `{constraint.type}` references non-existing point `{point}`",
                    (f"points({point})",),
                )
        if constraint.type == "distance":
            if len(constraint.points) != 2:
                return None, ValidationIssue("distance constraint must reference exactly two points")
            if constraint.value is None or constraint.unit is None:
                return None, ValidationIssue("distance constraint is missing value or unit")
            try:
                normalize_distance(constraint.value, constraint.unit)
            except Exception as exc:
                return None, ValidationIssue(f"distance constraint has invalid or unnormalized units: {exc}")
        elif constraint.type == "coordinate":
            if len(constraint.points) != 1:
                return None, ValidationIssue("coordinate constraint must reference exactly one point")
            if "x" not in constraint.data or "y" not in constraint.data:
                return None, ValidationIssue("coordinate constraint is missing x or y")
        elif constraint.type == "angle":
            if len(constraint.points) != 3:
                return None, ValidationIssue("angle constraint must reference exactly three points")
            if constraint.value is None or constraint.unit not in {"degree", "radian"}:
                return None, ValidationIssue("angle constraint is missing a normalized angular value")

    target = contract.target
    supported_targets = {
        "electric_field",
        "electric_force",
        "electric_potential",
        "potential_energy",
        "zero_electric_field_location",
        "zero_potential_location",
        "equilibrium_condition",
        "unknown_charge",
        "unknown_position",
    }
    if target.quantity not in supported_targets:
        return None, ValidationIssue("requested target quantity is ambiguous")
    if target.output not in {"magnitude", "magnitude_direction", "direction", "vector", "symbolic_expression", "numeric_value"}:
        return None, ValidationIssue("requested target output is ambiguous")
    unit_issue = _validate_target_unit(target.quantity, target.unit)
    if unit_issue is not None:
        return None, unit_issue

    source_ids = target.caused_by or tuple(body.id for body in contract.bodies if body.role == "source")
    if not source_ids:
        source_ids = tuple(
            body.id
            for body in contract.bodies
            if body.id != target.body and body.point is not None
        )
    for source_id in source_ids:
        source = bodies.get(source_id)
        if source is None:
            return None, ValidationIssue(f"target references unknown source `{source_id}`")
        if source.value is None or source_id not in charge_values:
            return None, ValidationIssue(
                f"source charge `{source_id}` has unknown value",
                (f"value({source_id})",),
            )
        if source.point is None:
            return None, ValidationIssue(
                f"source charge `{source_id}` has unknown point",
                (f"point({source_id})",),
            )

    target_point = target.at
    target_body = target.body
    if target.quantity in {"zero_electric_field_location", "zero_potential_location", "unknown_position"}:
        target_point = target.point or target.at
        if target_point is None:
            unknown_points = [unknown.point for unknown in contract.unknowns if unknown.kind == "coordinate" and unknown.point]
            if len(set(unknown_points)) == 1:
                target_point = unknown_points[0]
        if target_point is None:
            return None, ValidationIssue("target asks for location but no unknown coordinate exists", ("unknowns.coordinate",))
        if not any(unknown.kind == "coordinate" and unknown.point == target_point for unknown in contract.unknowns):
            return None, ValidationIssue("target asks for location but no matching unknown coordinate exists", ("unknowns.coordinate",))

    if target.quantity in {"unknown_charge", "equilibrium_condition"}:
        if not any(unknown.kind == "charge" for unknown in contract.unknowns):
            return None, ValidationIssue("target asks for unknown charge but no unknown charge exists", ("unknowns.charge",))

    if target.quantity == "electric_force":
        if target_body is None:
            target_bodies = [body.id for body in contract.bodies if body.role in {"target", "test_charge"}]
            if len(target_bodies) == 1:
                target_body = target_bodies[0]
        if target_body is None or target_body not in bodies:
            return None, ValidationIssue("electric force target body cannot be resolved", ("target.body",))
        if target_body not in charge_values:
            return None, ValidationIssue(f"target charge `{target_body}` has unknown value", (f"value({target_body})",))
        target_point = bodies[target_body].point

    if target_point is None:
        target_point = target.point
    if target_point is None and target.quantity in {"electric_potential", "zero_electric_field_location", "zero_potential_location"}:
        target_point = target.point or target.at
    if target_point is None:
        return None, ValidationIssue("target point cannot be resolved", ("target.at",))
    if target_point not in points:
        return None, ValidationIssue(f"target point `{target_point}` does not exist", (f"points({target_point})",))

    return (
        ValidatedPhysicsScene(
            contract=contract,
            charge_values=charge_values,
            body_points=body_points,
            source_ids=tuple(source_ids),
            target_point=target_point,
            target_body=target_body,
        ),
        None,
    )


def _validate_target_unit(quantity: str, unit: str | None) -> ValidationIssue | None:
    if unit is None:
        return None
    compatible = {
        "electric_field": {"N/C", "V/m"},
        "electric_force": {"N"},
        "electric_potential": {"V"},
        "zero_electric_field_location": {"m"},
        "zero_potential_location": {"m"},
        "unknown_position": {"m"},
        "unknown_charge": {"C"},
        "equilibrium_condition": {None, "C", "N"},
    }
    allowed = compatible.get(quantity)
    if allowed is not None and unit not in allowed:
        return ValidationIssue(f"target unit `{unit}` is incompatible with `{quantity}`", ("target.unit",))
    return None
