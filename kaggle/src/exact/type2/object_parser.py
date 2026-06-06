from __future__ import annotations

import re

import pint

from exact.type2.geometry_model import Body
from exact.type2.solving.units import parse_quantity


ELEMENTARY_CHARGE = 1.602176634e-19


def parse_objects(text: str) -> dict[str, Body]:
    normalized = normalize_math_text(text)
    values = _extract_charge_values(normalized)
    points = _extract_charge_points(normalized, values)
    roles = _infer_roles(normalized, values)
    bodies: dict[str, Body] = {}

    for body_id, value in values.items():
        bodies[body_id] = Body(
            id=body_id,
            kind="charge",
            role=roles.get(body_id, "unknown"),
            value=value,
            point=points.get(body_id),
            sign=_sign(value),
            evidence=body_id,
        )

    for body_id, body in _extract_symbolic_charge_bodies(normalized, values, points, roles).items():
        bodies.setdefault(body_id, body)

    for body_id, body in _extract_electron_bodies(normalized, points).items():
        bodies.setdefault(body_id, body)

    return bodies


def normalize_math_text(text: str) -> str:
    replacements = {
        "Ãƒâ€”": "x",
        "Ã—": "x",
        "×": "x",
        "ÃŽÂ¼": "u",
        "Î¼": "u",
        "μ": "u",
        "Ã‚Âµ": "u",
        "Âµ": "u",
        "Ã¢Ë†â€™": "-",
        "âˆ’": "-",
        "−": "-",
        "Ã¢ÂÂ»": "-",
        "â»": "-",
        "⁻": "-",
        "Ã¢â‚¬Â²": "'",
        "â€²": "'",
        "′": "'",
        "Ã¢ÂÂ°": "0",
        "Ã‚Â¹": "1",
        "Ã‚Â²": "2",
        "Ã‚Â³": "3",
        "Ã¢ÂÂ´": "4",
        "Ã¢ÂÂµ": "5",
        "Ã¢ÂÂ¶": "6",
        "Ã¢ÂÂ·": "7",
        "Ã¢ÂÂ¸": "8",
        "Ã¢ÂÂ¹": "9",
        "â°": "0",
        "Â¹": "1",
        "Â²": "2",
        "Â³": "3",
        "â´": "4",
        "âµ": "5",
        "â¶": "6",
        "â·": "7",
        "â¸": "8",
        "â¹": "9",
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.translate(
        str.maketrans(
            {
                "\u207b": "-",
                "\u2070": "0",
                "\u00b9": "1",
                "\u00b2": "2",
                "\u00b3": "3",
                "\u2074": "4",
                "\u2075": "5",
                "\u2076": "6",
                "\u2077": "7",
                "\u2078": "8",
                "\u2079": "9",
            }
        )
    )
    return text


def _extract_charge_values(text: str) -> dict[str, pint.Quantity]:
    values: dict[str, pint.Quantity] = {}
    chained = (
        r"\b(?P<left>q[\w']?)\s*=\s*(?P<sign>[+\-]?)\s*(?P<right>q[\w']?)\s*=\s*"
        r"(?P<value>[+\-]?\s*\d+(?:\.\d+)?(?:\s*(?:x|\*)\s*10\^?\s*[+\-]?\d+|e[+\-]?\d+)?)"
        r"\s*(?P<unit>uC|nC|pC|mC|C)\b"
    )
    for match in re.finditer(chained, text, flags=re.IGNORECASE):
        left = _canonical_charge_name(match.group("left"))
        right = _canonical_charge_name(match.group("right"))
        magnitude = _parse_signed_quantity(match.group("value"), match.group("unit"))
        values[left] = magnitude
        values[right] = -magnitude if match.group("sign") == "-" else magnitude

    pattern = (
        r"\b(?P<name>q[\w']?)\s*=\s*"
        r"(?P<value>[+\-]?\s*\d+(?:\.\d+)?(?:\s*(?:x|\*)\s*10\^?\s*[+\-]?\d+|e[+\-]?\d+)?)"
        r"\s*(?P<unit>uC|nC|pC|mC|C)\b"
    )
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        name = _canonical_charge_name(match.group("name"))
        if name in values:
            continue
        values[name] = _parse_signed_quantity(
            match.group("value"),
            match.group("unit"),
        )
    return values


def _extract_symbolic_charge_bodies(
    text: str,
    values: dict[str, pint.Quantity],
    points: dict[str, str],
    roles: dict[str, str],
) -> dict[str, Body]:
    bodies: dict[str, Body] = {}
    for match in re.finditer(r"\b(q0|q1|q2|q3|q)\b", text, flags=re.IGNORECASE):
        body_id = _canonical_charge_name(match.group(1))
        if body_id in values or body_id in bodies:
            continue
        bodies[body_id] = Body(
            id=body_id,
            kind="charge",
            role=roles.get(body_id, "unknown"),
            value=None,
            point=points.get(body_id),
            symbolic_value=body_id,
            evidence=match.group(0),
        )
    return bodies


def _extract_electron_bodies(text: str, points: dict[str, str]) -> dict[str, Body]:
    bodies: dict[str, Body] = {}
    for index, match in enumerate(re.finditer(r"\b(?:(?P<count>\d+)\s+)?electrons?\b", text, flags=re.IGNORECASE), start=1):
        count = float(match.group("count") or 1)
        body_id = f"electron_{index}"
        bodies[body_id] = Body(
            id=body_id,
            kind="charge",
            role="target" if "force acting on" in text.lower() else "source",
            value=(-count * ELEMENTARY_CHARGE) * parse_quantity(1, "C"),
            point=points.get(body_id),
            sign="negative",
            evidence=match.group(0),
        )
    return bodies


def _extract_charge_points(text: str, values: dict[str, pint.Quantity]) -> dict[str, str]:
    lower = text.lower()
    points: dict[str, str] = {}

    if "q1" in values and "q2" in values and re.search(r"q1.*?q2.*?points?\s+a\s+and\s+b", lower):
        points["q1"] = "A"
        points["q2"] = "B"

    paired_points = re.search(
        r"\bq1\b.*?\bq2\b.*?(?:points?|vertices)\s+(?P<first>[A-Z])\s+and\s+(?P<second>[A-Z])",
        text,
        flags=re.IGNORECASE,
    )
    if paired_points and "q1" in values and "q2" in values:
        points.setdefault("q1", paired_points.group("first").upper())
        points.setdefault("q2", paired_points.group("second").upper())

    generic_paired_points = re.search(
        r"\b(?P<left>q[\w']?)\b.*?\b(?P<right>q[\w']?)\b.*?"
        r"(?:fixed|placed|located).*?(?:at|points?)\s+(?P<first>[A-Z])\s+and\s+(?P<second>[A-Z])",
        text,
        flags=re.IGNORECASE,
    )
    if generic_paired_points:
        left = _canonical_charge_name(generic_paired_points.group("left"))
        right = _canonical_charge_name(generic_paired_points.group("right"))
        if left in values and right in values:
            points[left] = generic_paired_points.group("first").upper()
            points[right] = generic_paired_points.group("second").upper()

    for match in re.finditer(
        r"\b(?P<name>q[\w']?)\b.*?(?:placed|located|positioned|fixed|at)\s+(?:point\s+)?(?P<point>[A-Z])\b",
        text,
        flags=re.IGNORECASE,
    ):
        points.setdefault(_canonical_charge_name(match.group("name")), match.group("point").upper())

    for match in re.finditer(
        r"\b(?P<name>q[\w']?)\s*=\s*[^.]+?\b(?:at|fixed\s+at)\s+(?P<point>[A-Z])\b",
        text,
        flags=re.IGNORECASE,
    ):
        points.setdefault(_canonical_charge_name(match.group("name")), match.group("point").upper())

    if "q0" in values and "midpoint" in lower:
        points.setdefault("q0", "M")
    if "q3" in values and "midpoint" in lower:
        points.setdefault("q3", "M")

    if "q1" in values and "q2" in values and ("q3" in values or "q0" in values):
        if (
            "vertices" in lower
            or "triangle" in lower
            or ("ca" in lower and "cb" in lower)
            or ("ac" in lower and "bc" in lower)
            or "points a and b" in lower
        ):
            points.setdefault("q1", "A")
            points.setdefault("q2", "B")
            if "q3" in values:
                points.setdefault("q3", "C")
            if "q0" in values:
                points.setdefault("q0", "M" if "midpoint" in lower else "C")

    for body_id in values:
        suffix = body_id[1:]
        if body_id not in points and suffix and suffix.isalpha() and len(suffix) == 1:
            points[body_id] = suffix.upper()

    return points


def _infer_roles(text: str, values: dict[str, pint.Quantity]) -> dict[str, str]:
    lower = text.lower()
    roles = {body_id: "source" for body_id in values}
    target = _target_body_id(lower)
    if target:
        roles[target] = "target"
    for candidate in ("q0", "q3", "q"):
        if candidate in values and ("test charge" in lower or "third" in lower):
            roles.setdefault(candidate, "test_charge")
            if target is None:
                roles[candidate] = "target"
                break
    return roles


def _target_body_id(lower: str) -> str | None:
    match = re.search(
        r"(?:force|net force|electric force|resultant force).*?(?:acting on|exerted on|on)\s+(?:charge\s+)?(?P<name>q0|q1|q2|q3|q)",
        lower,
        flags=re.IGNORECASE,
    )
    if match:
        return _canonical_charge_name(match.group("name"))
    if "test charge" in lower:
        for candidate in ("q0", "q3"):
            if candidate in lower:
                return candidate
    return None


def _parse_signed_quantity(value: str, unit: str) -> pint.Quantity:
    text = value.replace(" ", "").replace("*", "x")
    if "x10" in text:
        base, exponent = text.split("x10", 1)
        exponent = exponent.replace("^", "")
        magnitude = float(base) * (10 ** int(exponent))
    else:
        magnitude = float(text)
    return parse_quantity(magnitude, unit)


def _canonical_charge_name(name: str) -> str:
    return name.strip().replace("′", "'").lower()


def _sign(value: pint.Quantity) -> str:
    return "positive" if value.to("C").magnitude >= 0 else "negative"
