from __future__ import annotations

import math
from dataclasses import dataclass

import pint

from exact.type2.geometry_model import GeometryConstraint, GeometrySpec
from exact.type2.solving.units import ureg


@dataclass(frozen=True)
class CoordinateBuildResult:
    coordinates: dict[str, tuple[pint.Quantity, pint.Quantity]]
    layout: str
    notes: tuple[str, ...] = ()


def build_coordinates(spec: GeometrySpec) -> CoordinateBuildResult | None:
    """Build 2D point coordinates from a GeometrySpec or reject it.

    The builder is deliberately conservative. It only returns coordinates when
    the geometry is sufficiently constrained by explicit edges or supported shape
    hints; otherwise callers should fall back to LLM/other solvers.
    """

    if _has_line_segment_metadata(spec):
        result = _line_segment_with_internal_point(spec)
        if result is not None:
            return result

    if _has_shape(spec, "isosceles right triangle") and "right_triangle_leg" in spec.metadata:
        result = _isosceles_right_triangle(spec)
        if result is not None:
            return result

    if _has_shape(spec, "perpendicular_bisector"):
        result = _perpendicular_bisector(spec)
        if result is not None:
            return result

    if _has_shape(spec, "equilateral triangle") or _has_shape(spec, "equilateral"):
        result = _equilateral(spec)
        if result is not None:
            return result

    if _has_shape(spec, "right-angled at a"):
        result = _right_angle_at_a(spec)
        if result is not None:
            return result

    if _has_shape(spec, "rectangle") or _has_shape(spec, "square"):
        result = _rectangle_or_square(spec)
        if result is not None:
            return result

    return _three_side_geometry(spec)


def _has_line_segment_metadata(spec: GeometrySpec) -> bool:
    return "line_segment_length" in spec.metadata and "distance_from_q1" in spec.metadata


def _line_segment_with_internal_point(spec: GeometrySpec) -> CoordinateBuildResult | None:
    if not {"q1", "q2", "q3"}.issubset(spec.bodies):
        return None
    segment = spec.metadata["line_segment_length"]
    from_q1 = spec.metadata["distance_from_q1"]
    if float((segment - from_q1).to("m").magnitude) <= 0:
        return None
    return CoordinateBuildResult(
        coordinates={
            spec.bodies["q1"].point: (0 * ureg.meter, 0 * ureg.meter),
            spec.bodies["q2"].point: (segment.to("m"), 0 * ureg.meter),
            spec.bodies["q3"].point: (from_q1.to("m"), 0 * ureg.meter),
        },
        layout="line_segment_internal_point",
    )


def _isosceles_right_triangle(spec: GeometrySpec) -> CoordinateBuildResult | None:
    if not {"q1", "q2", "q3"}.issubset(spec.bodies):
        return None
    leg = spec.metadata["right_triangle_leg"].to("m")
    target = spec.target_body or "q3"
    if target == "q3":
        coords = {
            spec.bodies["q3"].point: (0 * ureg.meter, 0 * ureg.meter),
            spec.bodies["q1"].point: (leg, 0 * ureg.meter),
            spec.bodies["q2"].point: (0 * ureg.meter, leg),
        }
    else:
        coords = {
            spec.bodies["q1"].point: (0 * ureg.meter, 0 * ureg.meter),
            spec.bodies["q2"].point: (leg, 0 * ureg.meter),
            spec.bodies["q3"].point: (0 * ureg.meter, leg),
        }
    return CoordinateBuildResult(coords, "isosceles_right_triangle")


def _perpendicular_bisector(spec: GeometrySpec) -> CoordinateBuildResult | None:
    ab = _distance(spec, "A", "B")
    if ab is None or spec.target_body is None:
        return None
    target_point = spec.bodies[spec.target_body].point
    target_distance = _distance(spec, "A", target_point) or _distance(spec, "B", target_point)
    if target_distance is None:
        return None
    height_sq = target_distance**2 - (ab / 2) ** 2
    if float(height_sq.to("m^2").magnitude) < -1e-12:
        return None
    height = max(0.0, float(height_sq.to("m^2").magnitude)) ** 0.5 * ureg.meter
    return CoordinateBuildResult(
        {
            "A": ((-ab / 2).to("m"), 0 * ureg.meter),
            "B": ((ab / 2).to("m"), 0 * ureg.meter),
            target_point: (0 * ureg.meter, height.to("m")),
        },
        "perpendicular_bisector",
    )


def _equilateral(spec: GeometrySpec) -> CoordinateBuildResult | None:
    side = _distance(spec, "A", "B") or _distance(spec, "A", "C") or _distance(spec, "B", "C")
    if side is None:
        return None
    return CoordinateBuildResult(
        {
            "A": (0 * ureg.meter, 0 * ureg.meter),
            "B": (side.to("m"), 0 * ureg.meter),
            "C": ((side / 2).to("m"), (math.sqrt(3) * side / 2).to("m")),
        },
        "equilateral_triangle",
    )


def _right_angle_at_a(spec: GeometrySpec) -> CoordinateBuildResult | None:
    ab = _distance(spec, "A", "B")
    ac = _distance(spec, "A", "C")
    bc = _distance(spec, "B", "C")
    if ab is None or (ac is None and bc is None):
        return None
    if ac is None:
        if bc is None:
            return None
        value = bc**2 - ab**2
        if float(value.to("m^2").magnitude) < -1e-12:
            return None
        ac = max(0.0, float(value.to("m^2").magnitude)) ** 0.5 * ureg.meter
    return CoordinateBuildResult(
        {
            "A": (0 * ureg.meter, 0 * ureg.meter),
            "B": (ab.to("m"), 0 * ureg.meter),
            "C": (0 * ureg.meter, ac.to("m")),
        },
        "right_angle_at_a",
    )


def _rectangle_or_square(spec: GeometrySpec) -> CoordinateBuildResult | None:
    ab = _distance(spec, "A", "B")
    bc = _distance(spec, "B", "C")
    cd = _distance(spec, "C", "D")
    da = _distance(spec, "D", "A")
    if _has_shape(spec, "square") and ab is not None:
        bc = bc or ab
    width = ab or cd
    height = bc or da
    if width is None or height is None:
        return None
    return CoordinateBuildResult(
        {
            "A": (0 * ureg.meter, 0 * ureg.meter),
            "B": (width.to("m"), 0 * ureg.meter),
            "C": (width.to("m"), height.to("m")),
            "D": (0 * ureg.meter, height.to("m")),
        },
        "rectangle",
    )


def _three_side_geometry(spec: GeometrySpec) -> CoordinateBuildResult | None:
    points = sorted({edge.a for edge in spec.edges} | {edge.b for edge in spec.edges})
    if len(points) < 3 or spec.target_body is None:
        return None
    target = spec.bodies[spec.target_body].point
    if target not in points:
        return None
    others = [point for point in points if point != target]
    if len(others) < 2:
        return None
    left, right = others[0], others[1]
    base = _distance(spec, left, right)
    d_left = _distance(spec, left, target)
    d_right = _distance(spec, right, target)
    if base is None or d_left is None or d_right is None:
        return None
    if float(base.to("m").magnitude) == 0:
        return None
    x = (d_left**2 + base**2 - d_right**2) / (2 * base)
    y_sq = d_left**2 - x**2
    if float(y_sq.to("m^2").magnitude) < -1e-12:
        return None
    y = max(0.0, float(y_sq.to("m^2").magnitude)) ** 0.5 * ureg.meter
    return CoordinateBuildResult(
        {
            left: (0 * ureg.meter, 0 * ureg.meter),
            right: (base.to("m"), 0 * ureg.meter),
            target: (x.to("m"), y.to("m")),
        },
        "three_side_geometry",
    )


def _distance(spec: GeometrySpec, a: str, b: str) -> pint.Quantity | None:
    key = frozenset((a.upper(), b.upper()))
    for edge in spec.edges:
        if frozenset((edge.a, edge.b)) == key:
            return edge.length
    for constraint in spec.constraints:
        if constraint.kind == "distance" and constraint.value is not None:
            if frozenset(constraint.points[:2]) == key:
                return constraint.value
    return None


def _has_shape(spec: GeometrySpec, shape: str) -> bool:
    normalized = shape.lower().replace("_", " ")
    for hint in spec.shape_hints:
        if hint.lower().replace("_", " ") == normalized:
            return True
    for constraint in spec.constraints:
        if constraint.kind == "shape" and constraint.shape:
            if constraint.shape.lower().replace("_", " ") == normalized:
                return True
    return False


def constraints_by_kind(spec: GeometrySpec, kind: str) -> tuple[GeometryConstraint, ...]:
    return tuple(constraint for constraint in spec.constraints if constraint.kind == kind)
