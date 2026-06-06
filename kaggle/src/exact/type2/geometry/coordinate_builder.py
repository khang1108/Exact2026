from __future__ import annotations

import math
from dataclasses import dataclass

import pint

from exact.type2.contract.normalize_units import normalize_distance
from exact.type2.contract.schemas import ValidatedPhysicsScene
from exact.type2.solving.units import ureg


@dataclass(frozen=True)
class ContractCoordinateResult:
    coordinates: dict[str, tuple[pint.Quantity, pint.Quantity]]
    layout: str
    notes: tuple[str, ...] = ()


def build_contract_coordinates(scene: ValidatedPhysicsScene) -> ContractCoordinateResult | None:
    contract = scene.contract
    coordinates: dict[str, tuple[pint.Quantity, pint.Quantity]] = {}
    aliases = _same_point_aliases(contract.constraints)

    for constraint in contract.constraints:
        if constraint.type == "coordinate" and len(constraint.points) == 1:
            x = constraint.data.get("x")
            y = constraint.data.get("y")
            unit = constraint.unit or constraint.data.get("unit") or "m"
            if x is None or y is None:
                return None
            coordinates[constraint.points[0]] = (normalize_distance(float(x), unit), normalize_distance(float(y), unit))

    if coordinates:
        _apply_aliases(coordinates, aliases)
        if _resolve_midpoints(contract.constraints, coordinates) and _resolve_perpendicular_bisectors(
            contract.constraints, coordinates
        ):
            _apply_aliases(coordinates, aliases)
            return ContractCoordinateResult(coordinates, "explicit_coordinates")
        return None

    distances = _distance_map(scene)
    points = set(contract.points)
    target_point = scene.target_point
    if target_point is None:
        return None

    midpoint_result = _build_from_midpoint(contract.constraints, distances)
    if midpoint_result is not None:
        _apply_aliases(midpoint_result.coordinates, aliases)
        return midpoint_result

    perpendicular_result = _build_perpendicular_bisector(contract.constraints, distances)
    if perpendicular_result is not None:
        _apply_aliases(perpendicular_result.coordinates, aliases)
        return perpendicular_result

    square_result = _build_square(contract.constraints)
    if square_result is not None:
        _resolve_midpoints(contract.constraints, square_result.coordinates)
        _apply_aliases(square_result.coordinates, aliases)
        if target_point in square_result.coordinates:
            return square_result

    equilateral_result = _build_equilateral(contract.constraints, distances, points)
    if equilateral_result is not None:
        _resolve_midpoints(contract.constraints, equilateral_result.coordinates)
        _apply_aliases(equilateral_result.coordinates, aliases)
        if target_point in equilateral_result.coordinates:
            return equilateral_result

    triangle_result = _build_three_side(points, target_point, distances)
    if triangle_result is not None:
        _resolve_midpoints(contract.constraints, triangle_result.coordinates)
        _apply_aliases(triangle_result.coordinates, aliases)
        return triangle_result

    line_result = _build_collinear(points, distances)
    if line_result is not None and target_point in line_result.coordinates:
        _resolve_midpoints(contract.constraints, line_result.coordinates)
        _apply_aliases(line_result.coordinates, aliases)
        return line_result

    unknown_line_result = _build_known_line_for_unknown(contract.constraints, distances)
    if unknown_line_result is not None:
        _apply_aliases(unknown_line_result.coordinates, aliases)
        return unknown_line_result

    return None


def _distance_map(scene: ValidatedPhysicsScene) -> dict[frozenset[str], pint.Quantity]:
    distances = {}
    for constraint in scene.contract.constraints:
        if constraint.type != "distance" or constraint.value is None or constraint.unit is None:
            continue
        a, b = constraint.points
        distances[frozenset((a, b))] = normalize_distance(constraint.value, constraint.unit)
    return distances


def _resolve_midpoints(constraints, coordinates: dict[str, tuple[pint.Quantity, pint.Quantity]]) -> bool:
    changed = True
    while changed:
        changed = False
        for constraint in constraints:
            if constraint.type != "midpoint" or len(constraint.points) != 3:
                continue
            point, a, b = constraint.points
            if point in coordinates:
                continue
            if a in coordinates and b in coordinates:
                ax, ay = coordinates[a]
                bx, by = coordinates[b]
                coordinates[point] = ((ax + bx) / 2, (ay + by) / 2)
                changed = True
    unresolved = [
        constraint.points[0]
        for constraint in constraints
        if constraint.type == "midpoint" and len(constraint.points) == 3 and constraint.points[0] not in coordinates
    ]
    return not unresolved


def _resolve_perpendicular_bisectors(constraints, coordinates: dict[str, tuple[pint.Quantity, pint.Quantity]]) -> bool:
    changed = True
    while changed:
        changed = False
        for constraint in constraints:
            if constraint.type != "perpendicular_bisector" or len(constraint.points) < 3:
                continue
            point, a, b = constraint.points[:3]
            if point in coordinates or a not in coordinates or b not in coordinates:
                continue
            distance = _constraint_distance_from_midpoint(constraint)
            if distance is None:
                continue
            ax, ay = coordinates[a]
            bx, by = coordinates[b]
            mid_x = (ax + bx) / 2
            mid_y = (ay + by) / 2
            dx = bx - ax
            dy = by - ay
            length = (dx**2 + dy**2) ** 0.5
            if float(length.to("m").magnitude) == 0:
                return False
            ux = -dy / length
            uy = dx / length
            coordinates[point] = ((mid_x + ux * distance).to("m"), (mid_y + uy * distance).to("m"))
            changed = True
    unresolved = [
        constraint.points[0]
        for constraint in constraints
        if constraint.type == "perpendicular_bisector"
        and len(constraint.points) >= 3
        and constraint.points[0] not in coordinates
    ]
    return not unresolved


def _build_from_midpoint(constraints, distances) -> ContractCoordinateResult | None:
    for constraint in constraints:
        if constraint.type != "midpoint" or len(constraint.points) != 3:
            continue
        point, a, b = constraint.points
        ab = distances.get(frozenset((a, b)))
        if ab is None:
            continue
        coords = {
            a: (0 * ureg.meter, 0 * ureg.meter),
            b: (ab.to("m"), 0 * ureg.meter),
            point: ((ab / 2).to("m"), 0 * ureg.meter),
        }
        return ContractCoordinateResult(coords, "midpoint")
    return None


def _build_perpendicular_bisector(constraints, distances) -> ContractCoordinateResult | None:
    for constraint in constraints:
        if constraint.type != "perpendicular_bisector" or len(constraint.points) < 3:
            continue
        point, a, b = constraint.points[:3]
        ab = distances.get(frozenset((a, b)))
        height = _constraint_distance_from_midpoint(constraint)
        if ab is None or height is None:
            continue
        return ContractCoordinateResult(
            {
                a: (0 * ureg.meter, 0 * ureg.meter),
                b: (ab.to("m"), 0 * ureg.meter),
                point: ((ab / 2).to("m"), height.to("m")),
            },
            "perpendicular_bisector",
        )
    return None


def _constraint_distance_from_midpoint(constraint) -> pint.Quantity | None:
    data = constraint.data.get("distance_from_midpoint")
    if isinstance(data, dict) and data.get("value") is not None and data.get("unit"):
        return normalize_distance(float(data["value"]), str(data["unit"]))
    if constraint.value is not None and constraint.unit is not None:
        return normalize_distance(float(constraint.value), constraint.unit)
    return None


def _build_square(constraints) -> ContractCoordinateResult | None:
    for constraint in constraints:
        if constraint.type != "square" or len(constraint.points) < 4:
            continue
        side_data = constraint.data.get("side")
        if isinstance(side_data, dict) and side_data.get("value") is not None and side_data.get("unit"):
            side = normalize_distance(float(side_data["value"]), str(side_data["unit"]))
        elif constraint.value is not None and constraint.unit is not None:
            side = normalize_distance(float(constraint.value), constraint.unit)
        else:
            continue
        a, b, c, d = constraint.points[:4]
        return ContractCoordinateResult(
            {
                a: (0 * ureg.meter, 0 * ureg.meter),
                b: (side.to("m"), 0 * ureg.meter),
                c: (side.to("m"), side.to("m")),
                d: (0 * ureg.meter, side.to("m")),
            },
            "square",
        )
    return None


def _build_equilateral(constraints, distances, points: set[str]) -> ContractCoordinateResult | None:
    equilateral_points = None
    for constraint in constraints:
        if constraint.type in {"equilateral", "equilateral_triangle"} and len(constraint.points) >= 3:
            equilateral_points = tuple(constraint.points[:3])
            if constraint.value is not None and constraint.unit is not None:
                distances[frozenset((equilateral_points[0], equilateral_points[1]))] = normalize_distance(
                    constraint.value, constraint.unit
                )
            side_data = constraint.data.get("side")
            if isinstance(side_data, dict) and side_data.get("value") is not None and side_data.get("unit"):
                distances[frozenset((equilateral_points[0], equilateral_points[1]))] = normalize_distance(
                    float(side_data["value"]), str(side_data["unit"])
                )
            break
    if equilateral_points is None and len(points) == 3:
        a, b, c = sorted(points)
        if all(frozenset(pair) in distances for pair in ((a, b), (a, c), (b, c))):
            vals = [float(distances[frozenset(pair)].to("m").magnitude) for pair in ((a, b), (a, c), (b, c))]
            if max(vals) - min(vals) <= max(vals) * 1e-6:
                equilateral_points = (a, b, c)
    if equilateral_points is None:
        return None
    a, b, c = equilateral_points
    side = distances.get(frozenset((a, b))) or distances.get(frozenset((a, c))) or distances.get(frozenset((b, c)))
    if side is None:
        return None
    return ContractCoordinateResult(
        {
            a: (0 * ureg.meter, 0 * ureg.meter),
            b: (side.to("m"), 0 * ureg.meter),
            c: ((side / 2).to("m"), (math.sqrt(3) * side / 2).to("m")),
        },
        "equilateral",
    )


def _build_three_side(points: set[str], target: str, distances) -> ContractCoordinateResult | None:
    if len(points) < 3 or target not in points:
        return None
    others = sorted(point for point in points if point != target)
    for i, left in enumerate(others):
        for right in others[i + 1 :]:
            base = distances.get(frozenset((left, right)))
            d_left = distances.get(frozenset((left, target)))
            d_right = distances.get(frozenset((right, target)))
            if base is None or d_left is None or d_right is None:
                continue
            if float(base.to("m").magnitude) == 0:
                continue
            x = (d_left**2 + base**2 - d_right**2) / (2 * base)
            y_sq = d_left**2 - x**2
            y_sq_m = float(y_sq.to("m^2").magnitude)
            if y_sq_m < -1e-12:
                continue
            y = max(0.0, y_sq_m) ** 0.5 * ureg.meter
            return ContractCoordinateResult(
                {
                    left: (0 * ureg.meter, 0 * ureg.meter),
                    right: (base.to("m"), 0 * ureg.meter),
                    target: (x.to("m"), y.to("m")),
                },
                "three_side",
            )
    return None


def _same_point_aliases(constraints) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for constraint in constraints:
        if constraint.type != "same_point" or len(constraint.points) < 2:
            continue
        canonical = constraint.points[0]
        for point in constraint.points[1:]:
            aliases[point] = canonical
    return aliases


def _apply_aliases(
    coordinates: dict[str, tuple[pint.Quantity, pint.Quantity]],
    aliases: dict[str, str],
) -> None:
    changed = True
    while changed:
        changed = False
        for alias, canonical in aliases.items():
            if alias not in coordinates and canonical in coordinates:
                coordinates[alias] = coordinates[canonical]
                changed = True
            if canonical not in coordinates and alias in coordinates:
                coordinates[canonical] = coordinates[alias]
                changed = True


def _build_collinear(points: set[str], distances) -> ContractCoordinateResult | None:
    if len(points) != 3:
        return None
    a, b, c = sorted(points)
    pairs = ((a, b), (a, c), (b, c))
    if not all(frozenset(pair) in distances for pair in pairs):
        return None
    values = {pair: distances[frozenset(pair)] for pair in pairs}
    sorted_pairs = sorted(values.items(), key=lambda item: float(item[1].to("m").magnitude), reverse=True)
    (end1, end2), longest = sorted_pairs[0]
    middle = next(point for point in points if point not in {end1, end2})
    d_end1_mid = distances[frozenset((end1, middle))]
    d_end2_mid = distances[frozenset((end2, middle))]
    if abs(float((d_end1_mid + d_end2_mid - longest).to("m").magnitude)) > 1e-9:
        return None
    return ContractCoordinateResult(
        {
            end1: (0 * ureg.meter, 0 * ureg.meter),
            end2: (longest.to("m"), 0 * ureg.meter),
            middle: (d_end1_mid.to("m"), 0 * ureg.meter),
        },
        "collinear_three_point",
    )


def _build_known_line_for_unknown(constraints, distances) -> ContractCoordinateResult | None:
    for constraint in constraints:
        if constraint.type != "on_line" or len(constraint.points) < 3:
            continue
        point, a, b = constraint.points[:3]
        ab = distances.get(frozenset((a, b)))
        if ab is None:
            continue
        return ContractCoordinateResult(
            {
                a: (0 * ureg.meter, 0 * ureg.meter),
                b: (ab.to("m"), 0 * ureg.meter),
            },
            f"line_with_unknown({point})",
            notes=(f"coordinates({point}) intentionally left symbolic on line {a}-{b}",),
        )
    return None
