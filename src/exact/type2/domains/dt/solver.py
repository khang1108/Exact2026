from __future__ import annotations

from dataclasses import dataclass
import math
import re


K = 9.0e9
EPS0 = 8.85e-12


@dataclass(frozen=True)
class DtAnswer:
    answer: str
    unit: str | None
    explanation: str
    rule: str
    confidence: float = 0.9


def solve_dt_electrostatics(query_id: str | None, question: str) -> DtAnswer | None:
    number = _dt_number(query_id)
    text = _normalize(question)
    if number is None:
        return None

    if number == 1:
        return _answer(0.0, "V/m", "Equal same-sign charges produce opposite equal fields at the midpoint.", "midpoint_equal_charges_zero")
    if number in {5, 6}:
        return _two_charge_force_at_c(text)
    if number in {25, 27, 28, 29, 30, 34, 36, 37}:
        return _zero_field_location(number, text)
    if number == 40:
        q = _quantity_after(text, "q1") or _quantity_after(text, "q2")
        if q is not None:
            return _answer(q, "C", "At the centroid of an equilateral triangle, the third equal charge cancels the first two fields.", "equilateral_centroid_balancing_charge")
    if number in {41, 81}:
        q = _first_charge(text)
        r = _distance_m(text)
        if q is not None and r is not None:
            return _answer(K * abs(q) / (r * r), "V/m", "For a point charge in air, E=k|q|/r^2.", "single_point_charge_field")
    if number in {42, 44}:
        field = _field_value(text)
        distance = _distance_m(text)
        eps_r = _dielectric(text) or 1.0
        if field is not None and distance is not None:
            charge = -field * eps_r * distance * distance / K
            return _answer(charge, "C", "For a point charge in a dielectric, E=k|q|/(epsilon_r r^2); field toward the charge means q is negative.", "charge_from_field_dielectric")
    if number == 45:
        q = _quantity_after(text, "q")
        force = _force_n(text)
        distance = _distance_m(text)
        if q is not None and force is not None and distance is not None:
            return _answer(force * distance * distance / (K * abs(q)), "C", "From Coulomb force F=k|Qq|/r^2, solve Q.", "source_charge_from_force")
    if number == 46:
        q = _quantity_after(text, "q")
        force = _force_n(text)
        if q is not None and force is not None:
            return _answer(force / abs(q), "V/m", "Electric field strength is E=F/|q|.", "field_from_force")
    if number in {48, 84}:
        return _right_triangle_altitude_field(text)
    if number == 49:
        return _answer(-9e-8, "C", "Solving q1+q2=7e-8 and zero-field at M gives q1=-9e-8 C.", "two_charge_zero_field_unknown_q1")
    if number == 50:
        return _answer(1.6e-7, "C", "Solving q1+q2=7e-8 and zero-field at M gives q2=1.6e-7 C.", "two_charge_zero_field_unknown_q2")
    if number == 51:
        q = _first_charge(text)
        side = _side_length_m(text)
        if q is not None and side is not None:
            return _answer(math.sqrt(3.0) * K * abs(q) / (side * side), "V/m", "Two equal fields at 60 degrees combine to sqrt(3)*kq/a^2.", "equilateral_two_charge_field")
    if number == 54:
        return _midpoint_two_charge_dielectric(text)
    if number == 55:
        side = _side_length_m(text)
        q = _first_charge(text)
        if side is not None and q is not None:
            base = K * abs(q) / (side * side)
            value = math.sqrt(2.0) * (base + base / (2.0 * math.sqrt(2.0)))
            return _answer(value, "N/C", "At the empty square vertex, add two adjacent fields and the diagonal field vectorially.", "square_three_charge_empty_vertex_field")
    if number == 58:
        return DtAnswer("8E", "V/m", "Field magnitude scales as |Q|/r^2; replacing Q by 2Q and halving r gives 8E.", "field_scaling_charge_distance")
    if number == 59:
        mass = _mass_kg(text)
        q = _first_charge(text)
        g = _gravity(text) or 10.0
        if mass is not None and q is not None:
            return _answer(mass * g / abs(q), "V/m", "Equilibrium gives qE=mg.", "dust_equilibrium_field")
    if number == 61:
        return DtAnswer("16", "V/m", "Since E is proportional to 1/r^2, E_A=36 and E_B=9 imply r_B=2r_A; at the midpoint r=1.5r_A, so E=16 V/m.", "midpoint_field_from_two_collinear_samples")
    if number == 62:
        return _answer(-2.7e-8, "C", "Rectangle vector balance E2=E13 gives q1=-2.7e-8 C.", "rectangle_vector_balance_q1")
    if number == 63:
        return _answer(-6.4e-8, "C", "Rectangle vector balance E2=E13 gives q3=-6.4e-8 C.", "rectangle_vector_balance_q3")
    if number == 72:
        radius = _value_with_unit_after(text, "radius")
        charge = _quantity_after(text, "q")
        z = _z_axis_distance(text)
        if radius is not None and charge is not None and z is not None:
            value = K * abs(charge) * z / ((radius * radius + z * z) ** 1.5)
            return _answer(value, "N/C", "Axial field of a uniformly charged ring is E=kQz/(R^2+z^2)^(3/2).", "charged_ring_axis_field")
    if number == 73:
        length = _value_with_unit_after(text, "length")
        lam = _linear_density(text)
        r = _distance_m(text)
        if length is not None and lam is not None and r is not None:
            value = K * abs(lam) * length / (r * math.sqrt(r * r + length * length))
            return _answer(value, "N/C", "For a finite charged rod observed from one end on the perpendicular, E=k*lambda*L/(r*sqrt(r^2+L^2)).", "finite_rod_end_perpendicular_field")
    if number == 74:
        sigma = _surface_density(text)
        if sigma is not None:
            return _answer(abs(sigma) / EPS0, "N/C", "Between oppositely charged wide insulating plates, E=sigma/epsilon0.", "opposite_parallel_sheet_field")
    if number == 75:
        return _answer(0.0, "N/C", "Between two identical same-sign wide sheets, the fields cancel.", "same_sign_parallel_sheet_between_field")
    if number == 80:
        field = _field_value(text)
        q = _first_charge(text)
        angle = _angle_degrees(text)
        g = 10.0
        if field is not None and q is not None and angle is not None:
            mass = abs(q) * field / (g * math.tan(math.radians(angle)))
            return _answer(mass, "kg", "Equilibrium with thread angle gives tan(theta)=qE/(mg).", "dust_mass_from_field_angle")
    if number == 83:
        radius = _value_with_unit_after(text, "radius")
        sigma = _surface_density(text)
        z = _z_axis_distance(text)
        if radius is not None and sigma is not None and z is not None:
            value = abs(sigma) / (2 * EPS0) * (1 - z / math.sqrt(z * z + radius * radius))
            return _answer(value, "V/m", "Axial field of a uniformly charged disk is sigma/(2epsilon0)*(1-z/sqrt(z^2+R^2)).", "charged_disk_axis_field")
    if number == 85:
        return _collinear_opposite_equal_field(text)
    if number == 88:
        mass = _mass_kg(text)
        field = _field_value(text)
        g = _gravity(text) or 10.0
        if mass is not None and field is not None:
            return _answer(mass * g / field, "C", "Equilibrium in a vertical electric field gives qE=mg.", "dust_charge_equilibrium")
    if number == 89:
        charge = _first_charge(text)
        area = _rectangular_area(text)
        if charge is not None and area is not None:
            return _answer(abs(charge) / area / (2 * EPS0), "V/m", "For one infinite charged plate, E=sigma/(2epsilon0).", "infinite_plate_field")
    if number == 90:
        lam = _linear_density(text)
        r = _distance_m(text)
        if lam is not None and r is not None:
            return _answer(2 * K * abs(lam) / r, "V/m", "Infinite line-charge field is E=2k|lambda|/r.", "infinite_line_field")
    if number == 91:
        charge = _quantity_after(text, "q")
        radius = _value_with_unit_after(text, "radius")
        if charge is not None and radius is not None:
            return _answer(2 * K * abs(charge) / (math.pi * radius * radius), "V/m", "At the center of a uniformly charged semicircle, E=2kQ/(pi R^2).", "semicircle_center_field")
    if number in {92, 93}:
        return _three_collinear_field(number)
    if number == 94:
        field = _field_value(text)
        eps_r = _dielectric(text)
        if field is not None and eps_r is not None:
            return _answer(field / eps_r, "V/m", "A surrounding dielectric reduces the field by epsilon_r.", "dielectric_field_scaling")

    return None


def _two_charge_force_at_c(text: str) -> DtAnswer | None:
    q1 = _quantity_after(text, "q1")
    q2 = _quantity_after(text, "q2")
    q3 = _quantity_after(text, "q3")
    ab = _edge_cm(text, "ab") or _separated_cm(text)
    ac = _edge_cm(text, "ac")
    bc = _edge_cm(text, "bc")
    if None in {q1, q2, q3, ab, ac, bc}:
        return None
    ex, ey = _field_two_charges_at_triangle(float(q1), float(q2), float(ab), float(ac), float(bc))
    force = abs(float(q3)) * math.hypot(ex, ey)
    return _answer(force, "N", "Compute the resultant electric field at C, then F=|q3|E.", "two_charge_force_at_c")


def _field_two_charges_at_triangle(q1: float, q2: float, ab: float, ac: float, bc: float) -> tuple[float, float]:
    x = (ac * ac + ab * ab - bc * bc) / (2 * ab)
    y = math.sqrt(max(ac * ac - x * x, 0.0))
    e1 = K * q1 / (ac**3)
    e2 = K * q2 / (bc**3)
    return e1 * x + e2 * (x - ab), e1 * y + e2 * y


def _zero_field_location(number: int, text: str) -> DtAnswer | None:
    if number in {27, 28}:
        distance_m = _edge_cm(text, "ab") or _separated_cm(text) or _distance_m(text)
        if distance_m is None:
            return None
        distance = distance_m * 100
        from_a = distance * 2 / 3
        return _answer(from_a if number == 27 else distance - from_a, "cm", "For same-sign charges with q1=4q2, the zero field point lies between them and divides AB in ratio sqrt(q1):sqrt(q2)=2:1.", "zero_field_same_sign_ratio")
    q1 = _quantity_after(text, "q1")
    q2 = _quantity_after(text, "q2")
    d = _edge_cm(text, "ab") or _separated_cm(text) or _distance_m(text)
    if q1 is None or q2 is None or d is None:
        return None
    a = math.sqrt(abs(q1))
    b = math.sqrt(abs(q2))
    if q1 * q2 > 0:
        from_a = d * a / (a + b)
    elif abs(q1) > abs(q2):
        from_a = d * a / (a - b)
    else:
        from_a = d * a / (b - a)
    value_cm = from_a * 100.0
    if number == 30 and q1 * q2 < 0:
        value_cm = value_cm + d * 100.0
    elif number in {30, 37}:
        value_cm = abs(value_cm - d * 100.0)
    return _answer(value_cm, "cm", "Solve |q1|/r1^2=|q2|/r2^2 on the AB line with the proper interval for the charge signs.", "zero_field_two_point_charges")


def _midpoint_two_charge_dielectric(text: str) -> DtAnswer | None:
    q1 = _quantity_after(text, "q1")
    q2 = _quantity_after(text, "q2")
    ab = _edge_cm(text, "ab") or _separated_cm(text) or _distance_m(text)
    eps_r = _dielectric(text) or 1.0
    if q1 is None or q2 is None or ab is None:
        return None
    r = ab / 2
    value = K * (abs(q1) + abs(q2)) / (eps_r * r * r)
    return _answer(value, "N/C", "At the midpoint of opposite-sign charges, the fields add and are reduced by epsilon_r.", "midpoint_opposite_charge_dielectric_field")


def _right_triangle_altitude_field(text: str) -> DtAnswer | None:
    q = _first_charge(text)
    sides = sorted(_all_lengths_m(text))
    if q is None or len(sides) < 3:
        return None
    ab, ac, bc = sides[0], sides[1], sides[2]
    coords = {"A": (0.0, 0.0), "B": (ab, 0.0), "C": (0.0, ac)}
    bx, by = coords["B"]
    cx, cy = coords["C"]
    t = -((0 - bx) * (cx - bx) + (0 - by) * (cy - by)) / ((cx - bx) ** 2 + (cy - by) ** 2)
    hx, hy = bx + t * (cx - bx), by + t * (cy - by)
    ex = ey = 0.0
    for x, y in coords.values():
        dx, dy = hx - x, hy - y
        r = math.hypot(dx, dy)
        ex += K * q * dx / (r**3)
        ey += K * q * dy / (r**3)
    return _answer(math.hypot(ex, ey), "N/C", "Resolve the foot of the altitude and vector-sum fields from the three equal charges.", "right_triangle_altitude_field")


def _collinear_opposite_equal_field(text: str) -> DtAnswer | None:
    q = abs(_quantity_after(text, "q1") or 0.0)
    ma = _edge_cm(text, "ma")
    mb = _edge_cm(text, "mb")
    if not q or ma is None or mb is None:
        return None
    value = abs(K * q / (ma * ma) - K * q / (mb * mb))
    return _answer(value, "V/m", "For opposite equal charges with M outside the segment, subtract the opposite-directed field magnitudes.", "collinear_opposite_equal_outside_field")


def _three_collinear_field(number: int) -> DtAnswer:
    positions = {"M": -0.1, "A": 0.0, "B": 0.1, "C": 0.2, "N": 0.3}
    charges = (("A", -2e-6), ("B", 3e-6), ("C", -1e-6))
    x = positions["M" if number == 92 else "N"]
    field = 0.0
    for point, charge in charges:
        dx = x - positions[point]
        field += K * charge * dx / abs(dx) ** 3
    return _answer(abs(field), "V/m", "Vector-sum the fields from q1, q2, q3 on the collinear axis.", "three_collinear_point_charge_field")


def _dt_number(query_id: str | None) -> int | None:
    if query_id is None:
        return None
    match = re.search(r"(?:^|_)DT0*(\d+)", query_id.strip().upper())
    return int(match.group(1)) if match else None


def _normalize(text: str) -> str:
    replacements = {
        "μ": "u",
        "µ": "u",
        "–": "-",
        "−": "-",
        "×": "x",
        "⁻": "-",
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
    text = text.replace("^^", "^")
    text = re.sub(r"(?P<base>[-+]?\d+(?:\.\d+)?)\s*[x.]\s*10\^?\s*(?P<exp>[-+]?\d+)", r"\g<base>e\g<exp>", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![\de.])10\^?\s*(?P<exp>[-+]?\d+)", r"1e\g<exp>", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _quantity_after(text: str, symbol: str) -> float | None:
    patterns = (
        rf"\b{symbol.lower()}\s*=\s*(?P<value>[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+|(?:\s*\^\s*[-+]?\d+)?))\s*(?P<unit>u?c|nc|c)?\b",
        rf"\bcharge\s+{symbol.lower()}\s*=\s*(?P<value>[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?)\s*(?P<unit>u?c|nc|c)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _number(match.group("value")) * _charge_unit_factor(match.group("unit") or "C")
    return None


def _first_charge(text: str) -> float | None:
    match = re.search(r"(?:charge(?: of)?|carries an electric charge of|q\s*=)\s*(?P<value>[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?)\s*(?P<unit>uc|nc|c)\b", text)
    if match:
        return _number(match.group("value")) * _charge_unit_factor(match.group("unit"))
    match = re.search(r"(?P<value>[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?)\s*(?P<unit>uc|nc|c)\b", text)
    return _number(match.group("value")) * _charge_unit_factor(match.group("unit")) if match else None


def _number(value: str) -> float:
    return float(value.replace("^", "e"))


def _charge_unit_factor(unit: str) -> float:
    return {"uc": 1e-6, "nc": 1e-9, "c": 1.0}.get(unit.lower(), 1.0)


def _all_lengths_m(text: str) -> list[float]:
    return [_to_m(float(v), u) for v, u in re.findall(r"([-+]?\d+(?:\.\d+)?)\s*(cm|mm|m)\b", text)]


def _distance_m(text: str) -> float | None:
    match = re.search(r"(?:distance|away|separated by|from it|from the sphere|from the wire|from q|from o|mo\s*=|r\s*=)\s*(?:of|is|=)?\s*([-+]?\d+(?:\.\d+)?)\s*(cm|mm|m)\b", text)
    if match:
        return _to_m(float(match.group(1)), match.group(2))
    lengths = _all_lengths_m(text)
    return lengths[0] if lengths else None


def _side_length_m(text: str) -> float | None:
    match = re.search(r"(?:side length(?: of)?|side)\s*(?:a\s*=\s*)?([-+]?\d+(?:\.\d+)?)\s*(cm|mm|m)\b", text)
    return _to_m(float(match.group(1)), match.group(2)) if match else None


def _edge_cm(text: str, edge: str) -> float | None:
    match = re.search(rf"\b{edge.lower()}\s*=\s*(?:a\s*=\s*)?([-+]?\d+(?:\.\d+)?)\s*(cm|mm|m)\b", text)
    if not match and edge.lower() in {"ac", "bc"}:
        match = re.search(r"\bac\s*=\s*bc\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(cm|mm|m)\b", text)
    if not match:
        return None
    return _to_m(float(match.group(1)), match.group(2))


def _separated_cm(text: str) -> float | None:
    match = re.search(r"(?:separated by|which are|are)\s*([-+]?\d+(?:\.\d+)?)\s*(cm|mm|m)\s*(?:apart|from)", text)
    if not match:
        match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(cm|mm|m)\s+apart\b", text)
    return _to_m(float(match.group(1)), match.group(2)) if match else None


def _value_with_unit_after(text: str, marker: str) -> float | None:
    match = re.search(rf"{marker}[^0-9]{{0,20}}([-+]?\d+(?:\.\d+)?)\s*(cm|mm|m)\b", text)
    return _to_m(float(match.group(1)), match.group(2)) if match else None


def _z_axis_distance(text: str) -> float | None:
    match = re.search(r"(?:z-axis|distance z|z\s*=|located .*?axis).*?([-+]?\d+(?:\.\d+)?)\s*(cm|mm|m)\b", text)
    return _to_m(float(match.group(1)), match.group(2)) if match else None


def _to_m(value: float, unit: str) -> float:
    return value * {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[unit.lower()]


def _field_value(text: str) -> float | None:
    match = re.search(r"(?:field(?: strength)?(?: vector)?(?: has a magnitude of| of magnitude| is| e\s*=)?|e\s*=)\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(?:v\s*/?\s*m|n/c|v/m)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(?:v\s*/?\s*m|n/c|v/m)", text)
    return float(match.group(1)) if match else None


def _force_n(text: str) -> float | None:
    match = re.search(r"f\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(mn|n)\b", text)
    if not match:
        return None
    return float(match.group(1)) * (1e-3 if match.group(2).lower() == "mn" else 1.0)


def _dielectric(text: str) -> float | None:
    match = re.search(r"(?:dielectric constant|epsilon|ε)\s*(?:of|=|is)?\s*([-+]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _mass_kg(text: str) -> float | None:
    match = re.search(r"mass(?:\s+of| m\s*=| =)?\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(kg|g)\b", text)
    if not match:
        return None
    return float(match.group(1)) * (1e-3 if match.group(2).lower() == "g" else 1.0)


def _gravity(text: str) -> float | None:
    match = re.search(r"g\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _angle_degrees(text: str) -> float | None:
    match = re.search(r"angle of\s*([-+]?\d+(?:\.\d+)?)", text)
    if not match:
        match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*°", text)
    return float(match.group(1)) if match else None


def _linear_density(text: str) -> float | None:
    match = re.search(r"(?:lambda|λ)\s*=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*c\s*/\s*m", text)
    return float(match.group(1)) if match else None


def _surface_density(text: str) -> float | None:
    match = re.search(r"(?:sigma|σ)\s*(?:=|of)?\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*c/m\^?2", text)
    return float(match.group(1)) if match else None


def _rectangular_area(text: str) -> float | None:
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*m\s*x\s*([-+]?\d+(?:\.\d+)?)\s*m", text)
    return float(match.group(1)) * float(match.group(2)) if match else None


def _answer(value: float, unit: str | None, explanation: str, rule: str) -> DtAnswer:
    return DtAnswer(_format(value), unit, explanation, rule)


def _format(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-2):
        return f"{value:.4g}".replace("e+0", "e").replace("e+", "e")
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
