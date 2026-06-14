from __future__ import annotations

import math
from typing import Any

from exact.type2.solver_contract.models import SolverContract


def _get_distance(contract: SolverContract, p1: str, p2: str) -> Any | None:
    for rel in contract.geometry.relations:
        if rel.type in {"distance", "side_length", "length"} and rel.points:
            # Check if this relation matches p1 and p2 (order independent)
            if set(rel.points) == {p1, p2}:
                return rel.value.quantity if rel.value and rel.value.ok else None
    return None


def _get_common_side_length(contract: SolverContract) -> Any | None:
    # Fallback for equilateral where "side = a" is given without explicit points
    for rel in contract.geometry.relations:
        if rel.type in {"side_length", "side"} and rel.value and rel.value.ok:
            return rel.value.quantity
    return None


def build_right_triangle_coordinates(contract: SolverContract) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    points = list(contract.geometry.points.keys())
    if len(points) != 3:
        raise ValueError(f"right_triangle_needs_3_points_got_{len(points)}")
        return coords

    # Find the right angle vertex. Assume it's explicitly identified by role.
    right_angle_pt = None
    for pid, pt in contract.geometry.points.items():
        if pt.role == "right_angle":
            right_angle_pt = pid
            break
            
    if not right_angle_pt:
        # Fallback heuristic: Try to find which point is shared by the two given lengths
        for pid in points:
            other_pts = [p for p in points if p != pid]
            d1 = _get_distance(contract, pid, other_pts[0])
            d2 = _get_distance(contract, pid, other_pts[1])
            if d1 is not None and d2 is not None:
                right_angle_pt = pid
                break

    if not right_angle_pt:
        raise ValueError("right_triangle_missing_right_angle")
        return coords

    other_pts = [p for p in points if p != right_angle_pt]
    d1 = _get_distance(contract, right_angle_pt, other_pts[0])
    d2 = _get_distance(contract, right_angle_pt, other_pts[1])

    if d1 is None or d2 is None:
        hypotenuse = _get_distance(contract, other_pts[0], other_pts[1])
        known_leg = d1 or d2
        if hypotenuse is not None and known_leg is not None:
            try:
                missing_sq = hypotenuse.to("meter") ** 2 - known_leg.to("meter") ** 2
                if float(missing_sq.magnitude) < -1e-12:
                    raise ValueError("right_triangle_invalid_hypotenuse_leg_lengths")
                missing_leg = max(0.0, float(missing_sq.magnitude)) ** 0.5 * hypotenuse.to("meter").units
            except AttributeError:
                missing_sq = hypotenuse.magnitude**2 - known_leg.magnitude**2
                if float(missing_sq) < -1e-12:
                    raise ValueError("right_triangle_invalid_hypotenuse_leg_lengths")
                missing_leg = max(0.0, float(missing_sq)) ** 0.5 * hypotenuse.units
            if d1 is None:
                d1 = missing_leg
            else:
                d2 = missing_leg
        else:
        # Check if they just said "side length = a" (isosceles right triangle)
            common_len = _get_common_side_length(contract)
            if common_len is not None:
                d1 = d2 = common_len
            else:
                raise ValueError("right_triangle_missing_leg_lengths")
                return coords

    coords[right_angle_pt] = (0.0, 0.0)
    # Convert quantities to standard units (meters) to extract float
    try:
        val1 = float(d1.to("meter").magnitude)
        val2 = float(d2.to("meter").magnitude)
    except Exception:
        val1 = float(d1.magnitude)
        val2 = float(d2.magnitude)

    coords[other_pts[0]] = (val1, 0.0)
    coords[other_pts[1]] = (0.0, val2)

    return coords


def build_collinear_coordinates(contract: SolverContract) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    order = contract.geometry.point_order
    
    if not order:
        # If absolute coordinates are already somehow in properties, we could use them.
        # But we defer to validation error if they aren't.
        return coords

    x = 0.0
    coords[order[0]] = (0.0, 0.0)

    for i in range(len(order) - 1):
        left = order[i]
        right = order[i+1]
        d = _get_distance(contract, left, right)
        
        if d is None:
            raise ValueError(f"missing_distance:{left}-{right}")
            return coords
            
        try:
            val = float(d.to("meter").magnitude)
        except Exception:
            val = float(d.magnitude)
            
        x += val
        coords[right] = (x, 0.0)

    return coords


def build_equilateral_triangle_coordinates(contract: SolverContract) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    points = list(contract.geometry.points.keys())
    if len(points) != 3:
        raise ValueError(f"equilateral_needs_3_points_got_{len(points)}")
        return coords
        
    side = _get_common_side_length(contract)
    if side is None:
        # Try getting any pairwise distance
        side = _get_distance(contract, points[0], points[1])
        
    if side is None:
        raise ValueError("equilateral_missing_side_length")
        return coords
        
    try:
        a = float(side.to("meter").magnitude)
    except Exception:
        a = float(side.magnitude)

    coords[points[0]] = (0.0, 0.0)
    coords[points[1]] = (a, 0.0)
    coords[points[2]] = (a / 2.0, a * math.sqrt(3) / 2.0)

    return coords


def build_perpendicular_bisector_coordinates(contract: SolverContract) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    
    midpoint_rel = None
    for rel in contract.geometry.relations:
        if rel.type == "midpoint" and len(rel.points) >= 3:
            midpoint_rel = rel
            break
            
    if not midpoint_rel:
        raise ValueError("perpendicular_bisector_missing_midpoint_relation")
        
    m, a, b = midpoint_rel.points[0], midpoint_rel.points[1], midpoint_rel.points[2]
    d_ab = _get_distance(contract, a, b)
    if not d_ab:
        raise ValueError(f"perpendicular_bisector_missing_distance_for_base_{a}_{b}")
        
    try:
        val_ab = float(d_ab.to("meter").magnitude)
    except Exception:
        val_ab = float(d_ab.magnitude)
        
    coords[m] = (0.0, 0.0)
    coords[a] = (-val_ab / 2.0, 0.0)
    coords[b] = (val_ab / 2.0, 0.0)
    
    for point in contract.geometry.points:
        if point not in coords:
            d_mp = _get_distance(contract, m, point)
            if d_mp:
                try:
                    val_mp = float(d_mp.to("meter").magnitude)
                except Exception:
                    val_mp = float(d_mp.magnitude)
                coords[point] = (0.0, val_mp)
            
    return coords


def build_three_side_triangle_coordinates(contract: SolverContract) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    points = list(contract.geometry.points.keys())
    
    if len(points) != 3:
        raise ValueError(f"three_side_triangle_needs_3_points_got_{len(points)}")
        return coords
        
    p_a, p_b, p_c = points[0], points[1], points[2]
    
    dist_ab = _get_distance(contract, p_a, p_b)
    dist_ac = _get_distance(contract, p_a, p_c)
    dist_bc = _get_distance(contract, p_b, p_c)
    
    if dist_ab is None or dist_ac is None or dist_bc is None:
        raise ValueError("three_side_triangle_missing_side_lengths")
        return coords
        
    try:
        d_ab = float(dist_ab.to("meter").magnitude)
        d_ac = float(dist_ac.to("meter").magnitude)
        d_bc = float(dist_bc.to("meter").magnitude)
    except Exception:
        d_ab = float(dist_ab.magnitude)
        d_ac = float(dist_ac.magnitude)
        d_bc = float(dist_bc.magnitude)
        
    if d_ab <= 0 or d_ac <= 0 or d_bc <= 0:
        raise ValueError("invalid_triangle_distances_must_be_positive")
        return coords
        
    # Triangle inequality validation
    if not (d_ab + d_ac > d_bc and d_ab + d_bc > d_ac and d_ac + d_bc > d_ab):
        raise ValueError("invalid_triangle_distances")
        return coords
        
    coords[p_a] = (0.0, 0.0)
    coords[p_b] = (d_ab, 0.0)
    
    x = (d_ac**2 + d_ab**2 - d_bc**2) / (2 * d_ab)
    # Floating point safety
    val_inside_sqrt = d_ac**2 - x**2
    if val_inside_sqrt < 0 and val_inside_sqrt > -1e-12:
        val_inside_sqrt = 0.0
    y = math.sqrt(val_inside_sqrt)
    
    coords[p_c] = (x, y)
    
    return coords


def build_coordinates(contract: SolverContract) -> dict[str, tuple[float, float]]:
    fam = contract.geometry.family
    if fam == "right_triangle":
        return build_right_triangle_coordinates(contract)
    elif fam == "collinear":
        return build_collinear_coordinates(contract)
    elif fam == "equilateral_triangle" or fam == "equilateral":
        return build_equilateral_triangle_coordinates(contract)
    elif fam == "three_side_triangle":
        return build_three_side_triangle_coordinates(contract)
    elif fam == "perpendicular_bisector":
        return build_perpendicular_bisector_coordinates(contract)
    
    return {}
