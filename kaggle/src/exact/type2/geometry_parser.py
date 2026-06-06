from __future__ import annotations

import re

import pint

from exact.type2.geometry_model import Body, GeometryConstraint, GeometryEdge, TargetSpec
from exact.type2.solving.units import parse_quantity


def parse_geometry(
    text: str,
    bodies: dict[str, Body],
    target_quantity: str | None,
) -> tuple[frozenset[str], tuple[GeometryConstraint, ...], tuple[GeometryEdge, ...], TargetSpec, tuple[str, ...], dict[str, pint.Quantity]]:
    points = _collect_points(text, bodies)
    edges = tuple(_extract_edges(text))
    constraints = [
        *(_distance_constraints(edges)),
        *(_position_constraints(text, bodies)),
        *(_shape_constraints(text)),
    ]
    metadata = _extract_metadata(text)
    shape_hints = tuple(
        constraint.shape or constraint.kind
        for constraint in constraints
        if constraint.kind == "shape"
    )
    target = TargetSpec(
        body=_target_body(text, bodies),
        quantity=target_quantity,
        output=_target_output(text),
    )
    return points, tuple(constraints), edges, target, shape_hints, metadata


def _collect_points(text: str, bodies: dict[str, Body]) -> frozenset[str]:
    points = {body.point for body in bodies.values() if body.point}
    points.update(re.findall(r"\b[A-Z]\b", text))
    midpoint = _midpoint_match(text)
    if midpoint:
        points.update(point.upper() for point in midpoint)
    if "center" in text.lower() or "centre" in text.lower():
        points.add("O")
    return frozenset(points)


def _extract_edges(text: str) -> list[GeometryEdge]:
    edges: list[GeometryEdge] = []
    seen: set[frozenset[str]] = set()

    for match in re.finditer(
        r"\b(?P<edge1>[A-Z]{2})\s*=\s*(?P<edge2>[A-Z]{2})\s*=\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)\b",
        text,
    ):
        length = parse_quantity(float(match.group("value")), match.group("unit")).to("m")
        for edge_name in (match.group("edge1"), match.group("edge2")):
            edge = _edge(edge_name, length, match.group(0))
            edges.append(edge)
            seen.add(frozenset((edge.a, edge.b)))

    for match in re.finditer(
        r"\b(?P<edge>[A-Z]{2})\s*=\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)\b",
        text,
    ):
        key = frozenset(match.group("edge").upper())
        if key in seen:
            continue
        length = parse_quantity(float(match.group("value")), match.group("unit")).to("m")
        edge = _edge(match.group("edge"), length, match.group(0))
        edges.append(edge)
        seen.add(key)

    for match in re.finditer(
        r"\bdistance\s+from\s+(?P<a>[A-Z])\s+to\s+(?P<b>[A-Z])\s+is\s+"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)\b",
        text,
        flags=re.IGNORECASE,
    ):
        key = frozenset((match.group("a").upper(), match.group("b").upper()))
        if key in seen:
            continue
        edges.append(
            GeometryEdge(
                match.group("a").upper(),
                match.group("b").upper(),
                parse_quantity(float(match.group("value")), match.group("unit")).to("m"),
                match.group(0),
            )
        )
        seen.add(key)

    apart = re.search(
        r"(?:points?\s+(?P<a>[A-Z])\s+and\s+(?P<b>[A-Z]).*?|(?P<c>[A-Z])\s+and\s+(?P<d>[A-Z]).*?|(?P<edge>[A-Z]{2})\s+is|separated by)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)\s+(?:apart|long)",
        text,
        flags=re.IGNORECASE,
    )
    if apart:
        if apart.group("edge"):
            a, b = apart.group("edge").upper()
        else:
            a = (apart.group("a") or apart.group("c") or "A").upper()
            b = (apart.group("b") or apart.group("d") or "B").upper()
        key = frozenset((a, b))
    if apart and key not in seen:
        edges.append(
            GeometryEdge(
                a,
                b,
                parse_quantity(float(apart.group("value")), apart.group("unit")).to("m"),
                apart.group(0),
            )
        )

    side = re.search(
        r"side(?:\s+length)?(?:\s+of)?(?:\s+a)?\s*(?:=|is|of)?\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)",
        text,
        flags=re.IGNORECASE,
    )
    if side and "equilateral" in text.lower():
        length = parse_quantity(float(side.group("value")), side.group("unit")).to("m")
        for edge_name in ("AB", "AC", "BC"):
            if frozenset(edge_name) not in seen:
                edges.append(_edge(edge_name, length, side.group(0)))

    return edges


def _distance_constraints(edges: tuple[GeometryEdge, ...]) -> list[GeometryConstraint]:
    return [
        GeometryConstraint(
            kind="distance",
            points=(edge.a, edge.b),
            value=edge.length,
            evidence=edge.evidence,
        )
        for edge in edges
    ]


def _position_constraints(text: str, bodies: dict[str, Body]) -> list[GeometryConstraint]:
    lower = text.lower()
    constraints: list[GeometryConstraint] = []
    midpoint = _midpoint_match(text)
    if midpoint:
        point, a, b = midpoint
        constraints.append(GeometryConstraint(kind="midpoint", points=(point.upper(), a.upper(), b.upper()), evidence="midpoint"))
    if "line segment" in lower or "along the line" in lower or "line connecting" in lower:
        point = "M"
        for body in bodies.values():
            if body.role in {"target", "test_charge"} and body.point:
                point = body.point
                break
        constraints.append(GeometryConstraint(kind="on_line", points=(point, "A", "B"), evidence="line placement"))
        if "away from q1" in lower or "line segment" in lower:
            constraints.append(GeometryConstraint(kind="between", points=(point, "A", "B"), evidence="line segment"))
    if "extension of line" in lower or "extension of" in lower:
        constraints.append(GeometryConstraint(kind="extension", points=("M", "A", "B"), evidence="extension"))
    if "equidistant" in lower:
        constraints.append(GeometryConstraint(kind="equidistant", points=("M", "A", "B"), evidence="equidistant"))
    if "remaining vertex" in lower:
        constraints.append(GeometryConstraint(kind="remaining_vertex", points=("C", "A", "B"), evidence="remaining vertex"))
    if "center of" in lower or "centre of" in lower:
        constraints.append(GeometryConstraint(kind="center", points=("O",), evidence="center"))
    return constraints


def _midpoint_match(text: str) -> tuple[str, str, str] | None:
    patterns = (
        r"\b(?P<point>[A-Z])\s+(?:is\s+)?(?:the\s+)?midpoint\s+of\s+(?P<a>[A-Z])(?P<b>[A-Z])\b",
        r"\b(?P<point>[A-Z])\s+lies\s+halfway\s+between\s+(?:points?\s+)?(?P<a>[A-Z])\s+and\s+(?P<b>[A-Z])\b",
        r"\b(?P<point>[A-Z])\s+is\s+located\s+at\s+the\s+midpoint\s+of\s+(?P<a>[A-Z])(?P<b>[A-Z])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group("point"), match.group("a"), match.group("b")
    if "midpoint" in text.lower():
        return ("M", "A", "B")
    return None


def _shape_constraints(text: str) -> list[GeometryConstraint]:
    lower = text.lower()
    constraints: list[GeometryConstraint] = []
    if "right-angled at a" in lower or "right angle at a" in lower:
        constraints.append(GeometryConstraint(kind="shape", shape="right-angled at a", evidence="right angle at A"))
    for shape in (
        "isosceles right triangle",
        "equilateral triangle",
        "right triangle",
        "triangle",
        "rhombus",
        "rectangle",
        "square",
    ):
        if shape in lower:
            constraints.append(GeometryConstraint(kind="shape", shape=shape, evidence=shape))
    if "same straight line" in lower:
        constraints.append(GeometryConstraint(kind="shape", shape="line", evidence="same straight line"))
    if "opposite sides" in lower:
        constraints.append(GeometryConstraint(kind="shape", shape="opposite_sides", evidence="opposite sides"))
    if "perpendicular bisector" in lower:
        constraints.append(GeometryConstraint(kind="shape", shape="perpendicular_bisector", evidence="perpendicular bisector"))
    return constraints


def _extract_metadata(text: str) -> dict[str, pint.Quantity]:
    metadata: dict[str, pint.Quantity] = {}
    segment = re.search(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)\s+long line segment",
        text,
        flags=re.IGNORECASE,
    )
    if segment:
        metadata["line_segment_length"] = parse_quantity(float(segment.group("value")), segment.group("unit")).to("m")

    away_q1 = re.search(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)\s+away from q1",
        text,
        flags=re.IGNORECASE,
    )
    if away_q1:
        metadata["distance_from_q1"] = parse_quantity(float(away_q1.group("value")), away_q1.group("unit")).to("m")

    legs = re.search(
        r"legs?\s+of\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)",
        text,
        flags=re.IGNORECASE,
    )
    if legs:
        metadata["right_triangle_leg"] = parse_quantity(float(legs.group("value")), legs.group("unit")).to("m")

    return metadata


def _target_body(text: str, bodies: dict[str, Body]) -> str | None:
    for body_id, body in bodies.items():
        if body.role == "target":
            return body_id
    lower = text.lower()
    match = re.search(
        r"(?:force|net force|electric force|resultant force).*?(?:acting on|exerted on|on)\s+(?:charge\s+)?(?P<name>q0|q1|q2|q3|q)",
        lower,
        flags=re.IGNORECASE,
    )
    if match and match.group("name") in bodies:
        return match.group("name")
    for fallback in ("q0", "q3", "q"):
        if fallback in bodies:
            return fallback
    return None


def _target_output(text: str) -> str:
    lower = text.lower()
    if "direction" in lower:
        return "direction"
    if "symbolic" in lower or "in terms of" in lower or "f0" in lower:
        return "symbolic"
    return "magnitude"


def _edge(name: str, length: pint.Quantity, evidence: str) -> GeometryEdge:
    edge_name = name.upper()
    return GeometryEdge(edge_name[0], edge_name[1], length, evidence)
