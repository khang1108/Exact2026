from __future__ import annotations

from exact.type2.solver_contract.models import SolverContract, ContractValidationResult


def _distance_pairs(contract: SolverContract) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for rel in contract.geometry.relations:
        if rel.type in {"distance", "length", "side_length"} and len(rel.points) == 2:
            pairs.add(frozenset(rel.points))
    return pairs


def _geometry_points(contract: SolverContract) -> set[str]:
    points = set(contract.geometry.points)
    for body in contract.bodies:
        if body.point:
            points.add(body.point)
    return points


def validate_contract(contract: SolverContract) -> ContractValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    required_missing: list[str] = []

    # 1. Target Validation
    if not contract.target.quantity or contract.target.quantity == "unknown":
        errors.append("missing_target_quantity")
        required_missing.append("target.quantity")

    if contract.target.quantity == "electric_force":
        if not contract.target.body:
            errors.append("target_body_missing")
            required_missing.append("target.body")
        else:
            body_by_id = {body.id: body for body in contract.bodies}
            target_body = body_by_id.get(contract.target.body)
            if target_body is None:
                errors.append(f"missing_target_body:{contract.target.body}")
                required_missing.append(f"bodies.{contract.target.body}")
            else:
                if target_body.value is None:
                    errors.append(f"target_body_value_missing:{contract.target.body}")
                    required_missing.append(f"bodies.{contract.target.body}.value")
                if not target_body.point:
                    errors.append(f"target_body_point_missing:{contract.target.body}")
                    required_missing.append(f"bodies.{contract.target.body}.point")
            source_bodies = [
                body
                for body in contract.bodies
                if body.id != contract.target.body and body.body_type == "charge"
            ]
            if not source_bodies:
                errors.append("source_bodies_missing")
                required_missing.append("bodies.source")
            for source in source_bodies:
                if source.value is None:
                    errors.append(f"source_body_value_missing:{source.id}")
                    required_missing.append(f"bodies.{source.id}.value")
                if not source.point:
                    errors.append(f"source_body_point_missing:{source.id}")
                    required_missing.append(f"bodies.{source.id}.point")
        if contract.target.unit not in {None, "N", "newton"}:
            errors.append("electric_force target must output N")

    if contract.target.quantity == "electric_field":
        if contract.target.unit not in {None, "V/m", "N/C", "volt/meter", "newton/coulomb"}:
            errors.append("electric_field target must output V/m or N/C")

    # 2. Parsing Unresolved Errors
    if contract.has_unresolved():
        for unres in contract.unresolved:
            errors.append(unres)

    # 3. Geometry Validation
    geom = contract.geometry
    if geom.family and geom.family != "none":
        if not geom.relations and not geom.points:
            errors.append("geometry_family requires relations or explicit points")

    if geom.family == "collinear":
        has_abs_coords = all(b.point and b.point in geom.points for b in contract.bodies)
        if not geom.point_order and not has_abs_coords:
            errors.append("collinear_order_missing")

    if geom.family == "three_side_triangle":
        points = _geometry_points(contract)
        if len(points) == 3:
            required_pairs: set[frozenset[str]] = set()
            point_list = sorted(points)
            for i, left in enumerate(point_list):
                for right in point_list[i + 1:]:
                    required_pairs.add(frozenset((left, right)))
            missing = sorted("-".join(sorted(pair)) for pair in required_pairs - _distance_pairs(contract))
            if missing:
                errors.append(f"three_side_triangle_missing_side_lengths:{','.join(missing)}")
                required_missing.extend(f"geometry.distance.{pair}" for pair in missing)
        else:
            errors.append(f"three_side_triangle_needs_3_points_got_{len(points)}")

    if geom.family == "right_triangle":
        points = _geometry_points(contract)
        if len(points) != 3:
            errors.append(f"right_triangle_needs_3_points_got_{len(points)}")
        right_angle_points = {
            point.id for point in geom.points.values() if point.role == "right_angle"
        }
        right_angle_points.update(
            rel.points[1] for rel in geom.relations if rel.type == "right_angle" and len(rel.points) >= 3
        )
        if not right_angle_points:
            errors.append("right_triangle_missing_right_angle")
            required_missing.append("geometry.right_angle")
        distance_pairs = _distance_pairs(contract)
        has_two_legs = any(
            sum(1 for pair in distance_pairs if vertex in pair) >= 2
            for vertex in right_angle_points
        )
        has_hypotenuse_and_leg = len(distance_pairs) >= 2
        if not (has_two_legs or has_hypotenuse_and_leg):
            errors.append("right_triangle_missing_sufficient_lengths")
            required_missing.append("geometry.distance")

    target_point = contract.target.point or contract.target.at
    if target_point and target_point not in geom.points:
        errors.append("target_point_unresolved")

    for rel in geom.relations:
        if rel.type in {"distance", "length", "radius", "side_length"} and (not rel.value or not rel.value.ok):
            errors.append(f"distance_parse_failed:{rel.raw_value}")

    # 4. Basic Body Validation
    if not contract.bodies:
        warnings.append("no_bodies_extracted")
    else:
        for b in contract.bodies:
            if b.body_type == "unknown":
                warnings.append(f"unknown_body_type:{b.id}")

    ok = len(errors) == 0

    return ContractValidationResult(
        ok=ok,
        errors=errors,
        warnings=warnings,
        required_missing=required_missing,
    )
