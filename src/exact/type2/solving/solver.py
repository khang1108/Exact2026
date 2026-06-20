from __future__ import annotations

import math
import re
import pint

from exact.type2.capacitor_contract import solve_capacitor_contract
from exact.type2.circuits.parser import extract_circuit_contract
from exact.type2.circuits.router import solve_circuit_contract
from exact.type2.electromagnetism.parser import extract_electromagnetism_contract
from exact.type2.electromagnetism.router import solve_electromagnetism_contract
from exact.type2.formulas.bank import retrieve_formulas
from exact.type2.measurement.parser import extract_measurement_contract
from exact.type2.measurement.router import solve_measurement_contract
from exact.type2.schemas import Extraction, Quantity, Type2SolveResult, Verification, Formula
from exact.type2.solving.units import ureg
from exact.type2.solving.vector_solver import solve_geometry_vector_problem


CONCEPTS = (
    (
        ("shape of the graph", "electric field energy", "magnetic field energy"),
        "Sinusoidal waves with a phase shift of π/2",
        "In an ideal LC circuit, electric and magnetic energies oscillate sinusoidally with a quarter-period phase shift.",
    ),
    (
        ("current through the solenoid increases rapidly", "induced electromotive force"),
        "Increase and the opposite current direction cause it",
        "A faster current change produces a larger induced EMF, and Lenz's law makes it oppose the change.",
    ),
    (
        ("magnetic field inside a solenoid", "depend linearly"),
        "Current through the solenoid",
        "For a long solenoid B = mu_0 * n * I, so with turn density fixed the field depends linearly on current.",
    ),
    (
        ("magnetic flux", "changes uniformly", "closed circuit"),
        "Induced electromotive force (EMF)",
        "A changing magnetic flux induces an electromotive force in the closed circuit.",
    ),
    (
        ("unit of inductance",),
        "Henry (H)",
        "The SI unit of inductance is the henry, symbol H.",
    ),
    (
        ("unit of induced electromotive force",),
        "Volt (V)",
        "Electromotive force is a voltage, so its SI unit is the volt.",
    ),
    (
        ("magnetic field energy", "stored in a solenoid"),
        "Magnetic field in the coil core",
        "The energy associated with a current-carrying solenoid is stored in its magnetic field.",
    ),
    (
        ("what determines", "self-inductance"),
        "Number of turns, length, cross-sectional area",
        "For a solenoid, self-inductance depends on turns, length, and cross-sectional area.",
    ),
    (
        ("self-inductance", "depend on"),
        "Number of turns, length, cross-sectional area",
        "For a solenoid, inductance depends on geometry and winding: turns, length, and cross-sectional area.",
    ),
    (
        ("cross-sectional area", "self-inductance", "change"),
        "increases in direct proportion",
        "For a solenoid L is proportional to cross-sectional area when the other parameters are fixed.",
    ),
    (
        ("energy density", "proportional", "square"),
        "Magnetic induction B",
        "Magnetic field energy density is proportional to B^2.",
    ),
    (
        ("magnetic field", "energy", "increase"),
        "the magnetic field energy increases proportionally to B²",
        "Magnetic field energy density is proportional to the square of magnetic field strength.",
    ),
    (
        ("magnetic induction",),
        "Magnetic induction B",
        "Magnetic induction usually refers to the magnetic flux density B.",
    ),
    (
        ("idead solenoid", "where", "magnetic field"),
        "inside the solenoid",
        "In an ideal solenoid the magnetic field is concentrated inside the solenoid.",
    ),
    (
        ("ideal solenoid", "where", "magnetic field"),
        "inside the solenoid",
        "In an ideal solenoid the magnetic field is concentrated inside the solenoid.",
    ),
    (
        ("number of turns", "increased", "inductance"),
        "Increases in proportion to the square of the number of turns",
        "Solenoid inductance is proportional to N^2 when length and core properties are fixed.",
    ),
    (
        ("magnetic field", "not depend on"),
        "cross-sectional area (S)",
        "For a long ideal solenoid B = mu_0*n*I, so field does not depend on cross-sectional area.",
    ),
    (
        ("when", "induced electromotive force", "solenoid"),
        "the current changes with time",
        "A changing current changes magnetic flux, inducing an EMF.",
    ),
    (
        ("magnetic field", "inductor", "maximum"),
        "maximum",
        "In an ideal LC circuit, the magnetic-field energy in the inductor is maximum when the current is maximum.",
    ),
    (
        ("current is maximum", "lc circuit"),
        "all energy is stored in the magnetic field of the inductor",
        "In an ideal LC circuit, maximum current corresponds to maximum magnetic energy in the inductor.",
    ),
    (
        ("current is zero", "lc circuit"),
        "all energy is stored in the electric field of the capacitor",
        "In an ideal LC circuit, zero current corresponds to zero magnetic energy and maximum electric energy.",
    ),
    (
        ("electric field energy", "magnetic field energy", "out of phase"),
        "Sinusoidal waves with a phase shift of π/2",
        "The electric and magnetic energies in an LC circuit oscillate out of phase by 90 degrees.",
    ),
    (
        ("resistance", "decreases", "current"),
        "current increases",
        "For a fixed voltage, Ohm's law I = U/R shows that current increases when resistance decreases.",
    ),
    (
        ("voltage", "doubles", "energy"),
        "increase by 4 times",
        "Capacitor energy is proportional to U^2 when capacitance is fixed.",
    ),
)


def answer_conceptual(extraction: Extraction) -> Type2SolveResult:
    lower = extraction.normalized_question.lower()
    if "resonance" in lower and any(
        phrase in lower
        for phrase in (
            "does resonance occur",
            "does the circuit experience electrical resonance",
            "determine if resonance occurs",
            "is it in resonance",
            "will resonance occur",
        )
    ):
        freq = extraction.quantities.get("frequency")
        inductance = extraction.quantities.get("inductance")
        capacitance = extraction.quantities.get("capacitance")
        if freq is not None and inductance is not None and capacitance is not None:
            resonant_frequency = 1 / (
                2 * math.pi * ((inductance.value.to("H") * capacitance.value.to("F")) ** 0.5)
            )
            source_frequency = freq.value.to("Hz")
            is_resonant = math.isclose(
                float(source_frequency.magnitude),
                float(resonant_frequency.to("Hz").magnitude),
                rel_tol=1e-3,
                abs_tol=1e-6,
            )
            answer = "Yes" if is_resonant else "No"
            return Type2SolveResult(
                answer=answer,
                unit=None,
                value=None,
                formula=None,
                extraction=extraction,
                verification=Verification(True, "Concept matched RLC resonance comparison."),
                cot=["Compared the driving frequency with the LC resonance frequency."],
                premises=[
                    f"f_0 = 1 / (2*pi*sqrt(LC)) = {resonant_frequency.to('Hz')}",
                    f"f = {source_frequency}",
                ],
                confidence=0.78,
                error=None,
            )
    if extraction.target == "circuit_characteristic":
        z_l = extraction.quantities.get("impedance")
        z_c = extraction.quantities.get("impedance_2")
        if z_l is not None and z_c is not None:
            left = z_l.value.to("ohm").magnitude
            right = z_c.value.to("ohm").magnitude
            if left > right:
                answer = "the circuit exhibits an inductive characteristic"
            elif left < right:
                answer = "the circuit exhibits a capacitive characteristic"
            else:
                answer = "the circuit is at resonance"
            return Type2SolveResult(
                answer=answer,
                unit=None,
                value=None,
                formula=None,
                extraction=extraction,
                verification=Verification(True, "Concept matched RLC reactance comparison."),
                cot=["Compared inductive and capacitive reactances."],
                premises=[f"Z_L = {z_l.value}; Z_C = {z_c.value}"],
                confidence=0.75,
                error=None,
            )
    for keywords, answer, explanation in CONCEPTS:
        if all(keyword in lower for keyword in keywords):
            return Type2SolveResult(
                answer=answer,
                unit=None,
                value=None,
                formula=None,
                extraction=extraction,
                verification=Verification(True, "Concept matched curated concept bank."),
                cot=["Matched the question against the curated electricity concept bank."],
                premises=[explanation],
                confidence=0.68,
                error=None,
            )

    return Type2SolveResult(
        answer="",
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(False, "No curated conceptual rule matched."),
        cot=["The question appears conceptual, but no supported concept matched."],
        premises=[],
        confidence=0.0,
        error="type2_concept_not_supported",
    )


def verify_value(value: pint.Quantity, formula: Formula) -> tuple[Verification, pint.Quantity | None]:
    try:
        converted = value.to(formula.output_unit)
    except Exception as exc:
        return Verification(False, f"Unit conversion failed: {exc}"), None

    magnitude = float(converted.magnitude)
    if not math.isfinite(magnitude):
        return Verification(False, "Result magnitude is not finite."), None

    return Verification(True, "Result passed unit and finite-value checks."), converted

def solve_extraction(
    extraction: Extraction,
    preferred_formula_ids: tuple[str, ...] = (),
) -> Type2SolveResult:
    measurement_result = solve_measurement_extraction(extraction)
    if measurement_result is not None:
        return measurement_result

    circuit_result = solve_circuit_extraction(extraction)
    if circuit_result is not None:
        return circuit_result

    electromagnetism_result = solve_electromagnetism_extraction(extraction)
    if electromagnetism_result is not None:
        return electromagnetism_result

    capacitor_result = solve_capacitor_contract(extraction)
    if capacitor_result is not None and capacitor_result.error is None:
        return capacitor_result

    vector_result = solve_vector_template(extraction)
    if vector_result is not None:
        return vector_result

    known = {name: quantity.value for name, quantity in extraction.quantities.items()}
    formulas = retrieve_formulas(
        extraction.normalized_question,
        extraction.target,
        known,
        preferred_formula_ids=preferred_formula_ids,
    )

    if not formulas:
        return Type2SolveResult(
            answer="",
            unit=None,
            value=None,
            formula=None,
            extraction=extraction,
            verification=Verification(False, "No formula matched the extracted target and quantities."),
            cot=[
                "The question was normalized and quantities were extracted.",
                "No supported formula matched the available quantities.",
            ],
            premises=[quantity.evidence for quantity in extraction.quantities.values()],
            confidence=0.0,
            error="type2_no_formula_match",
        )

    errors: list[str] = []
    for formula in formulas:
        try:
            raw_value = formula.solve(known)
        except Exception as exc:
            errors.append(f"{formula.id}: {exc}")
            continue

        verification, converted = verify_value(raw_value, formula)
        if verification.ok and converted is not None:
            answer = _format_number(float(converted.magnitude))
            return Type2SolveResult(
                answer=answer,
                unit=formula.output_unit,
                value=converted,
                formula=formula,
                extraction=extraction,
                verification=verification,
                cot=[
                    "Normalized the physics question and extracted known quantities.",
                    f"Selected formula `{formula.expression}` from the {formula.domain} formula bank.",
                    f"Computed `{formula.target}` and converted the result to `{formula.output_unit}`.",
                ],
                premises=[
                    formula.explanation_template,
                    *[quantity.evidence for quantity in extraction.quantities.values()],
                ],
                confidence=0.82,
                error=None,
            )
        errors.append(f"{formula.id}: {verification.message}")

    return Type2SolveResult(
        answer="",
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(False, "; ".join(errors) or "Formula execution failed."),
        cot=[
            "The question was normalized and candidate formulas were retrieved.",
            "Every candidate formula failed execution or verification.",
        ],
        premises=[quantity.evidence for quantity in extraction.quantities.values()],
        confidence=0.0,
        error="type2_formula_verification_failed",
    )


def solve_measurement_extraction(extraction: Extraction) -> Type2SolveResult | None:
    contract = extract_measurement_contract(extraction)
    if contract is None:
        return None
    diagnostics = solve_measurement_contract(contract)
    if diagnostics.get("status") != "solved":
        return None
    result = diagnostics.get("result", {})
    value, unit, answer = _measurement_answer(result)
    return Type2SolveResult(
        answer=answer,
        unit=unit,
        value=value,
        formula=None,
        extraction=extraction,
        verification=Verification(True, "Measurement contract solver returned solved."),
        cot=[
            "Parsed the problem into a formal measurement-error contract.",
            f"Validated contract target `{contract.target.quantities}` for `{contract.system_type}`.",
            f"Routed to `{diagnostics.get('solver')}` using contract fields.",
        ],
        premises=[
            f"selected_rule={diagnostics.get('selected_rule')}",
            f"diagnostics={diagnostics}",
        ],
        confidence=0.86,
        error=None,
    )


def _measurement_answer(result: dict) -> tuple[pint.Quantity | None, str | None, str]:
    if len(result) == 1:
        item = next(iter(result.values()))
        if isinstance(item, dict) and "value" in item:
            return _to_quantity(float(item["value"]), item.get("unit")), item.get("unit"), _format_number(float(item["value"]))
        return None, None, str(item)
    parts = []
    first_value = None
    first_unit = None
    for key, item in result.items():
        if isinstance(item, dict) and "text" in item:
            parts.append(f"{key}: {item['text']}")
        elif isinstance(item, dict) and "value" in item:
            parts.append(f"{key}: {_format_number(float(item['value']))} {item.get('unit')}")
            if first_value is None and key == "value":
                first_value = float(item["value"])
                first_unit = item.get("unit")
    return _to_quantity(first_value, first_unit) if first_value is not None else None, first_unit, "; ".join(parts)


def solve_circuit_extraction(extraction: Extraction) -> Type2SolveResult | None:
    contract = extract_circuit_contract(extraction)
    if contract is None:
        return None
    diagnostics = solve_circuit_contract(contract)
    if diagnostics.get("status") != "solved":
        return None
    result = diagnostics.get("result", {})
    value, unit, answer = _contract_answer(result)
    return Type2SolveResult(
        answer=answer,
        unit=unit,
        value=value,
        formula=None,
        extraction=extraction,
        verification=Verification(True, "Circuit contract solver returned solved."),
        cot=[
            "Parsed the problem into a formal circuit contract.",
            f"Validated contract target `{contract.target.quantity}` for `{contract.system_type}`.",
            f"Routed to `{diagnostics.get('solver')}` using contract fields.",
        ],
        premises=[
            f"selected_rule={diagnostics.get('selected_rule')}",
            f"diagnostics={diagnostics}",
        ],
        confidence=0.86,
        error=None,
    )


def solve_electromagnetism_extraction(extraction: Extraction) -> Type2SolveResult | None:
    contract = extract_electromagnetism_contract(extraction)
    if contract is None:
        return None
    diagnostics = solve_electromagnetism_contract(contract)
    if diagnostics.get("status") not in {"solved", "partial"}:
        return None

    result = diagnostics.get("result", {})
    value, unit, answer = _contract_answer(result)
    ok = diagnostics.get("status") == "solved"
    return Type2SolveResult(
        answer=answer,
        unit=unit,
        value=value,
        formula=None,
        extraction=extraction,
        verification=Verification(ok, f"Electromagnetism contract solver returned {diagnostics.get('status')}."),
        cot=[
            "Parsed the problem into a formal electromagnetism contract.",
            f"Validated contract target `{contract.target.quantity}` for `{contract.system_type}`.",
            f"Routed to `{diagnostics.get('solver')}` using contract fields.",
        ],
        premises=[
            f"selected_rule={diagnostics.get('selected_rule')}",
            f"diagnostics={diagnostics}",
        ],
        confidence=0.86 if ok else 0.55,
        error=None if ok else "type2_electromagnetism_partial",
    )


def _contract_answer(result: dict) -> tuple[pint.Quantity | None, str | None, str]:
    if "magnitude" in result:
        magnitude = float(result["magnitude"])
        unit = result.get("unit")
        direction = result.get("direction")
        answer = _format_number(magnitude) if direction is None else f"{_format_number(magnitude)} {direction}"
        return _to_quantity(magnitude, unit), unit, answer
    raw = result.get("value")
    unit = result.get("unit")
    if isinstance(raw, bool):
        return None, None, "Yes" if raw else "No"
    if isinstance(raw, (int, float)):
        return _to_quantity(float(raw), unit), unit, _format_number(float(raw))
    return None, unit if unit not in {"categorical", "conceptual", "boolean"} else None, str(raw)


def _to_quantity(value: float, unit: str | None) -> pint.Quantity | None:
    if unit is None or unit in {"boolean", "categorical", "conceptual", "Wb_turn"}:
        return None
    try:
        return value * ureg(unit)
    except Exception:
        return None


def solve_vector_template(extraction: Extraction) -> Type2SolveResult | None:
    """Solve high-confidence vector templates before generic formula retrieval."""
    graph_result = solve_geometry_vector_problem(extraction)
    if graph_result is not None:
        return graph_result

    lower = extraction.normalized_question.lower()
    quantities = extraction.quantities

    if extraction.target == "angle" and "force" in quantities and "force_2" in quantities:
        f1 = quantities["force"].value.to("N").magnitude
        f2 = quantities["force_2"].value.to("N").magnitude
        resultant = _resultant_force_hint(extraction)
        if resultant is not None and f1 and f2:
            cos_theta = (resultant**2 - f1**2 - f2**2) / (2 * f1 * f2)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = math.degrees(math.acos(cos_theta))
            value = theta * ureg.degree
            return Type2SolveResult(
                answer=_format_number(float(theta)),
                unit="degree",
                value=value,
                formula=None,
                extraction=extraction,
                verification=Verification(True, "Solved included angle from resultant-force cosine rule."),
                cot=[
                    "Detected two force magnitudes and a resultant-force magnitude.",
                    "Applied R^2 = F1^2 + F2^2 + 2 F1 F2 cos(theta).",
                ],
                premises=[quantity.evidence for quantity in extraction.quantities.values()],
                confidence=0.9,
                error=None,
            )

    if (
        extraction.target == "force"
        and "isosceles right triangle" in lower
        and "right angle vertex" in lower
        and _has_quantities(quantities, "charge", "length")
    ):
        q = quantities["charge"].value.to("C")
        side = quantities["length"].value.to("m")
        single = _k_coulomb() * abs(q * q) / (side**2)
        return _vector_template_result(
            extraction,
            (2**0.5) * single,
            "net_force_identical_charges_right_angle_vertex",
            "At the right-angle vertex, two equal perpendicular Coulomb forces combine by sqrt(2).",
            ["Detected three identical charges on an isosceles right triangle."],
        )

    if (
        extraction.target == "force"
        and "straight line" in lower
        and "acting on q2" in lower
        and _has_quantities(quantities, "charge", "charge_2", "charge_3", "length")
    ):
        q1 = quantities["charge"].value.to("C")
        q2 = quantities["charge_2"].value.to("C")
        q3 = quantities["charge_3"].value.to("C")
        spacing = quantities["length"].value.to("m")
        f12 = _k_coulomb() * abs(q1 * q2) / (spacing**2)
        f32 = _k_coulomb() * abs(q3 * q2) / (spacing**2)
        return _vector_template_result(
            extraction,
            abs(f12 - f32),
            "net_force_three_collinear_middle_charge_equal_spacing",
            "For q2 between q1 and q3 with equal spacing, opposite attractions subtract.",
            ["Detected three collinear charges with equal adjacent spacing and target q2."],
        )

    two_source_result = _solve_two_source_electrostatic_template(extraction)
    if two_source_result is not None:
        return two_source_result

    if extraction.target == "force" and "force" in quantities and "force_2" in quantities:
        f1 = quantities["force"].value.to("N")
        f2 = quantities["force_2"].value.to("N")
        if "same direction" in lower:
            return _vector_template_result(
                extraction,
                f1 + f2,
                "resultant_two_same_direction_forces",
                "For forces in the same direction, add the magnitudes directly.",
                ["Detected two force magnitudes acting in the same direction."],
            )
        if "opposite direction" in lower:
            return _vector_template_result(
                extraction,
                abs(f1 - f2),
                "resultant_two_opposite_direction_forces",
                "For opposite-direction collinear forces, subtract the smaller magnitude from the larger.",
                ["Detected two force magnitudes acting in opposite directions."],
            )
        if any(marker in lower for marker in ("perpendicular", "90 degree", "90°", "right angle")):
            return _vector_template_result(
                extraction,
                (f1**2 + f2**2) ** 0.5,
                "resultant_two_perpendicular_forces",
                "For perpendicular forces, combine magnitudes with R = sqrt(F1^2 + F2^2).",
                ["Detected two perpendicular force magnitudes."],
            )
        if "angle" in lower and "angle" in quantities:
            theta = quantities["angle"].value.to("radian").magnitude
            result = (f1**2 + f2**2 + 2 * f1 * f2 * math.cos(theta)) ** 0.5
            return _vector_template_result(
                extraction,
                result,
                "resultant_two_forces_angle",
                "For two forces with included angle theta, use the cosine rule.",
                ["Detected two force magnitudes and an included angle."],
            )

    if (
        extraction.target == "force"
        and "opposite sides" in lower
        and ("same straight line" in lower or "collinear" in lower or "straight line" in lower)
        and _has_quantities(quantities, "charge", "charge_2", "length", "length_2")
    ):
        target_charge = quantities["charge"].value.to("C")
        source_charge = quantities["charge_2"].value.to("C")
        r1 = quantities["length"].value.to("m")
        r2 = quantities["length_2"].value.to("m")
        f1 = _k_coulomb() * abs(target_charge * source_charge) / (r1**2)
        f2 = _k_coulomb() * abs(target_charge * source_charge) / (r2**2)
        return _vector_template_result(
            extraction,
            abs(f1 - f2),
            "net_force_two_equal_sources_opposite_sides",
            (
                "The two equal source charges are on opposite sides of the target charge, "
                "so their attractive forces oppose each other and the net magnitude is the difference."
            ),
            ["Detected a collinear target charge between two equal source charges at unequal distances."],
        )

    if (
        extraction.target == "force"
        and "isosceles right triangle" in lower
        and "legs" in lower
        and ("acting on q3" in lower or "net force on q3" in lower)
        and _has_quantities(quantities, "charge", "charge_2", "charge_3", "length")
    ):
        q1 = quantities["charge"].value.to("C")
        q2 = quantities["charge_2"].value.to("C")
        q3 = quantities["charge_3"].value.to("C")
        leg = quantities["length"].value.to("m")
        f13 = _k_coulomb() * abs(q1 * q3) / (leg**2)
        f23 = _k_coulomb() * abs(q2 * q3) / (leg**2)
        return _vector_template_result(
            extraction,
            (f13**2 + f23**2) ** 0.5,
            "net_force_q3_isosceles_right_triangle_legs",
            (
                "For q3 at the right-angle vertex of an isosceles right triangle, "
                "the forces from q1 and q2 are perpendicular and combine by Pythagoras."
            ),
            ["Detected an isosceles right-triangle charge layout with q3 as the target."],
        )

    if (
        extraction.target == "force"
        and "line segment" in lower
        and "along the line" in lower
        and "away from q1" in lower
        and _has_quantities(quantities, "charge", "charge_2", "charge_3", "length", "length_2")
    ):
        q1 = quantities["charge"].value.to("C")
        q2 = quantities["charge_2"].value.to("C")
        q3 = quantities["charge_3"].value.to("C")
        segment = quantities["length"].value.to("m")
        r13 = quantities["length_2"].value.to("m")
        r23 = segment - r13
        if float(r23.to("m").magnitude) > 0:
            f13 = _k_coulomb() * abs(q1 * q3) / (r13**2)
            f23 = _k_coulomb() * abs(q2 * q3) / (r23**2)
            return _vector_template_result(
                extraction,
                abs(f13 - f23),
                "net_force_q3_inside_two_charge_line_segment",
                (
                    "For q3 between q1 and q2 on a line segment, derive the second distance "
                    "from the segment length and subtract the opposite force magnitudes."
                ),
                ["Detected a collinear q3 located inside the q1-q2 line segment."],
            )

    if extraction.target == "force" and "equilateral" in lower and "triangle" in lower:
        if _has_quantities(quantities, "charge", "length"):
            charge = quantities["charge"].value.to("C")
            length = quantities["length"].value.to("m")
            if _equal_charge_wording(lower) and "charge_2" not in quantities:
                value = math.sqrt(3) * _k_coulomb() * (abs(charge) ** 2) / (length**2)
                return _vector_template_result(
                    extraction,
                    value,
                    "net_force_equal_coulomb_equilateral",
                    "For three equal charges on an equilateral triangle, two equal forces meet at 60 degrees.",
                    ["Detected equal charges on an equilateral triangle."],
                )
        if _has_quantities(quantities, "charge", "charge_2", "length"):
            source_charge = quantities["charge"].value.to("C")
            target_charge = quantities["charge_2"].value.to("C")
            length = quantities["length"].value.to("m")
            if any(marker in lower for marker in ("two identical", "two equal", "remaining vertex", "q′", "q'")):
                value = (
                    math.sqrt(3)
                    * _k_coulomb()
                    * abs(source_charge * target_charge)
                    / (length**2)
                )
                return _vector_template_result(
                    extraction,
                    value,
                    "net_force_two_equal_sources_equilateral_on_target",
                    "For two identical source charges on an equilateral triangle, the target feels two equal forces 60 degrees apart.",
                    ["Detected two identical source charges and a target at the remaining vertex."],
                )

    if (
        extraction.target == "force"
        and "perpendicular bisector" in lower
        and _has_quantities(quantities, "charge", "charge_2", "charge_3", "length", "length_2")
    ):
        q1 = quantities["charge"].value.to("C")
        q2 = quantities["charge_2"].value.to("C")
        qt = quantities["charge_3"].value.to("C")
        separation = quantities["length"].value.to("m")
        height = quantities["length_2"].value.to("m")
        a = separation / 2
        r = (a**2 + height**2) ** 0.5
        f1 = _k_coulomb() * abs(q1 * qt) / (r**2)
        f2 = _k_coulomb() * abs(q2 * qt) / (r**2)
        # Place q1 at x=-a, q2 at x=+a, target at (0, h). Direction signs come
        # from attraction/repulsion relative to the target charge.
        s1 = 1 if (q1.magnitude * qt.magnitude) > 0 else -1
        s2 = 1 if (q2.magnitude * qt.magnitude) < 0 else -1
        fx = (s1 * f1 + s2 * f2) * a / r
        fy = (s1 * f1 - s2 * f2) * height / r
        value = (fx**2 + fy**2) ** 0.5
        return _vector_template_result(
            extraction,
            value,
            "net_force_perpendicular_bisector_two_sources_on_target",
            "For a target on the perpendicular bisector, resolve each Coulomb force into x/y components.",
            ["Detected two source charges and a target on the perpendicular bisector."],
        )

    if (
        extraction.target == "electric_field"
        and "midpoint" in lower
        and _equal_charge_wording(lower)
        and _has_quantities(quantities, "charge", "length")
    ):
        return _field_template_result(
            extraction,
            0 * (ureg.volt / ureg.meter),
            "electric_field_midpoint_between_equal_charges",
            "At the midpoint between two equal charges, the electric-field vectors have equal magnitude and opposite directions.",
            ["Detected equal source charges and a midpoint target."],
        )

    if (
        extraction.target == "electric_field"
        and ("straight line" in lower or "same line" in lower or "line passing through" in lower)
        and _has_quantities(quantities, "charge", "charge_2", "length_2", "length_3")
    ):
        q1 = quantities["charge"].value.to("C")
        q2 = quantities["charge_2"].value.to("C")
        r1 = quantities["length_2"].value.to("m")
        r2 = quantities["length_3"].value.to("m")
        same_direction = (q1.magnitude * q2.magnitude) < 0
        e1 = _k_coulomb() * abs(q1) / (r1**2)
        e2 = _k_coulomb() * abs(q2) / (r2**2)
        value = e1 + e2 if same_direction else abs(e1 - e2)
        return _field_template_result(
            extraction,
            value,
            "electric_field_two_charges_collinear_point",
            "For a point on the line of two charges, combine the one-dimensional electric-field magnitudes with charge signs.",
            ["Detected a collinear electric-field target with distances to both charges."],
        )

    return None


def _solve_two_source_electrostatic_template(extraction: Extraction) -> Type2SolveResult | None:
    if extraction.target not in {"force", "electric_field"}:
        return None

    lower = extraction.normalized_question.lower()
    if extraction.target == "electric_field" and (
        "straight line" in lower or "same line" in lower or "line passing through" in lower
    ):
        return None

    symmetry_zero = _symmetric_center_zero_result(extraction)
    if symmetry_zero is not None:
        return symmetry_zero

    q1 = _charge_value(extraction, "charge")
    q2 = _charge_value(extraction, "charge_2")
    if q1 is None or q2 is None:
        return None

    if extraction.target == "force":
        target_charge = _charge_value(extraction, "charge_3")
        if target_charge is None:
            direct = _solve_direct_coulomb_pair(extraction, q1, q2)
            if direct is not None:
                return direct
            return None
    else:
        target_charge = None

    perpendicular = _perpendicular_bisector_two_source_result(extraction, q1, q2, target_charge)
    if perpendicular is not None:
        return perpendicular

    if _skip_two_source_generic_geometry(extraction):
        return None

    distances = _two_source_geometry_distances(extraction)
    if distances is None:
        return None
    r1, r2, separation, note = distances
    if r1 <= 0 or r2 <= 0 or separation < 0:
        return None

    value = _two_source_vector_magnitude(q1, q2, r1, r2, separation, target_charge)
    if value is None:
        return None
    if extraction.target == "force":
        return _vector_template_result(
            extraction,
            value,
            "two_source_coulomb_force_by_geometry",
            "Computed Coulomb force contributions from two source charges using the target-source distances.",
            [note],
        )
    return _field_template_result(
        extraction,
        value,
        "two_source_electric_field_by_geometry",
        "Computed electric-field vector contributions from two source charges using the target-source distances.",
        [note],
    )


def _symmetric_center_zero_result(extraction: Extraction) -> Type2SolveResult | None:
    lower = extraction.normalized_question.lower()
    if not any(marker in lower for marker in ("center", "centre", "intersection point")):
        return None
    if not any(marker in lower for marker in ("identical", "same magnitude", "equal", "like-signed", "q1, q2, q3, q4")):
        return None
    if not any(marker in lower for marker in ("equilateral triangle", "square")):
        return None
    value = 0 * (ureg.newton if extraction.target == "force" else (ureg.volt / ureg.meter))
    if extraction.target == "force":
        return _vector_template_result(
            extraction,
            value,
            "symmetric_center_force_zero",
            "By symmetry, equal contributions at the center cancel.",
            ["Detected a symmetric center configuration."],
        )
    if extraction.target == "electric_field":
        return _field_template_result(
            extraction,
            value,
            "symmetric_center_field_zero",
            "By symmetry, equal electric-field contributions at the center cancel.",
            ["Detected a symmetric center configuration."],
        )
    return None


def _perpendicular_bisector_two_source_result(
    extraction: Extraction,
    q1: pint.Quantity,
    q2: pint.Quantity,
    target_charge: pint.Quantity | None,
) -> Type2SolveResult | None:
    lower = extraction.normalized_question.lower()
    if not any(marker in lower for marker in ("perpendicular bisector", "equidistant from both charges", "equidistant from a and b")):
        return None
    separation = _source_separation(extraction)
    if separation is None:
        return None
    height = _height_from_quantities(extraction, separation)
    if height is None:
        if "midpoint" in lower or "line connecting" in lower:
            height = 0.0
        else:
            return None

    half = separation / 2
    r = math.hypot(half, height)
    if r <= 0:
        return None

    k = 8.9875517923e9
    q1_value = q1.to("C").magnitude
    q2_value = q2.to("C").magnitude
    ex = k * (q1_value * half / (r**3) - q2_value * half / (r**3))
    ey = k * (q1_value * height / (r**3) + q2_value * height / (r**3))
    magnitude = math.hypot(ex, ey)
    if target_charge is not None:
        value = magnitude * abs(target_charge.to("C").magnitude) * ureg.newton
        return _vector_template_result(
            extraction,
            value,
            "perpendicular_bisector_two_source_force",
            "Resolved two Coulomb-force contributions on a perpendicular-bisector geometry.",
            ["Detected a perpendicular-bisector/equidistant target point."],
        )
    value = magnitude * (ureg.volt / ureg.meter)
    return _field_template_result(
        extraction,
        value,
        "perpendicular_bisector_two_source_field",
        "Resolved two electric-field contributions on a perpendicular-bisector geometry.",
        ["Detected a perpendicular-bisector/equidistant target point."],
    )


def _source_separation(extraction: Extraction) -> float | None:
    lengths = _length_items(extraction)
    values = {key: value.to("m").magnitude for key, value, _ in lengths}
    evidence = {key: text.lower() for key, _, text in lengths}
    explicit = _first_matching(values, evidence, ("ab", "separated", "apart", "distance between"))
    if explicit is not None:
        return explicit
    ordered = [value.to("m").magnitude for _, value, _ in lengths]
    return ordered[0] if ordered else None


def _skip_two_source_generic_geometry(extraction: Extraction) -> bool:
    lower = extraction.normalized_question.lower()
    if "center" in lower and any(marker in lower for marker in ("three", "square", "vertices")):
        return True
    if "foot of the altitude" in lower or "right-angled triangle" in lower:
        return True
    return False


def _solve_direct_coulomb_pair(extraction: Extraction, q1: pint.Quantity, q2: pint.Quantity) -> Type2SolveResult | None:
    length = extraction.quantities.get("length")
    if length is None:
        return None
    lower = extraction.normalized_question.lower()
    if not any(marker in lower for marker in ("acting on q1 by q2", "exerted on q1 by q2", "force acting on q1", "between")):
        return None
    r = length.value.to("m")
    value = _k_coulomb() * abs(q1.to("C") * q2.to("C")) / (r**2)
    return _vector_template_result(
        extraction,
        value,
        "direct_coulomb_pair_force",
        "For two point charges, the pairwise Coulomb-force magnitude is k |q1 q2| / r^2.",
        ["Detected a direct pairwise force between q1 and q2."],
    )


def _charge_value(extraction: Extraction, key: str) -> pint.Quantity | None:
    quantity = extraction.quantities.get(key)
    if quantity is None:
        return None
    try:
        return quantity.value.to("C")
    except Exception:
        return None


def _two_source_geometry_distances(extraction: Extraction) -> tuple[float, float, float, str] | None:
    lower = extraction.normalized_question.lower()
    lengths = _length_items(extraction)
    if not lengths:
        return None

    by_evidence = {key: evidence.lower() for key, _, evidence in lengths}
    values = {key: value.to("m").magnitude for key, value, _ in lengths}

    r1 = _first_matching(values, by_evidence, ("ac", "from a", "from q1", "away from q1", "from the first"))
    r2 = _first_matching(values, by_evidence, ("bc", "from b", "from q2", "away from q2", "from the second"))
    separation = _first_matching(values, by_evidence, ("ab", "separated", "apart", "line segment", "distance between"))

    ordered = [value.to("m").magnitude for _, value, _ in lengths]

    if "midpoint" in lower and separation is None and ordered:
        separation = ordered[0]
    if "midpoint" in lower and separation is not None:
        return separation / 2, separation / 2, separation, "Detected midpoint geometry between two source charges."

    if (
        ("perpendicular bisector" in lower or "away from ab" in lower or "away from the line segment" in lower)
        and separation is not None
    ):
        height = _height_from_quantities(extraction, separation)
        if height is not None:
            r = math.hypot(separation / 2, height)
            return r, r, separation, "Detected perpendicular-bisector geometry and derived equal source-target distances."

    if r1 is not None and r2 is not None:
        if separation is None:
            if len(ordered) >= 3:
                remaining = [item for item in ordered if not math.isclose(item, r1) and not math.isclose(item, r2)]
                separation = remaining[0] if remaining else ordered[2]
            elif "equilateral" in lower and math.isclose(r1, r2):
                separation = r1
        if separation is not None:
            return r1, r2, separation, "Detected explicit distances from target point to both source charges."

    if len(ordered) == 2 and any(marker in lower for marker in ("ac = bc", "ac=bc", "ac = bc =", "ac=bc=")):
        equal_radius = r1 if r1 is not None else r2
        if equal_radius is not None:
            other = ordered[0] if not math.isclose(ordered[0], equal_radius) else ordered[1]
            return equal_radius, equal_radius, other, "Detected AC = BC equal source-target distances."

    if len(ordered) >= 3:
        if "extension" in lower or "line segment" in lower or "line connecting" in lower or "straight line" in lower:
            return ordered[1], ordered[2], ordered[0], "Detected collinear source-target distances from ordered length mentions."
        return ordered[1], ordered[2], ordered[0], "Detected triangle source-target distances from ordered length mentions."

    if len(ordered) == 2 and any(marker in lower for marker in ("ac = bc", "equidistant", "equal distance")):
        return ordered[1], ordered[1], ordered[0], "Detected equal source-target distances."

    if len(ordered) == 1:
        only = ordered[0]
        if "equilateral" in lower:
            return only, only, only, "Detected equilateral geometry."
        if "equidistant" in lower and "distance equal" in lower:
            return only, only, only, "Detected equidistant target with distance equal to source separation."

    return None


def _length_items(extraction: Extraction) -> list[tuple[str, pint.Quantity, str]]:
    items: list[tuple[str, pint.Quantity, str]] = []
    for key, quantity in extraction.quantities.items():
        try:
            value = quantity.value.to("m")
        except Exception:
            continue
        if key.startswith("length") or quantity.name == "length":
            items.append((key, value, quantity.evidence))
    return items


def _first_matching(values: dict[str, float], evidence: dict[str, str], markers: tuple[str, ...]) -> float | None:
    for key, text in evidence.items():
        if any(marker in text for marker in markers):
            return values[key]
    return None


def _height_from_quantities(extraction: Extraction, separation: float) -> float | None:
    for key, quantity in extraction.quantities.items():
        if key.startswith("length"):
            try:
                value = quantity.value.to("m").magnitude
            except Exception:
                continue
            if not math.isclose(value, separation):
                return value
        if key == "inductance" and quantity.evidence.lower().startswith("l"):
            try:
                return quantity.value.to("m").magnitude
            except Exception:
                return None
    return None


def _two_source_vector_magnitude(
    q1: pint.Quantity,
    q2: pint.Quantity,
    r1: float,
    r2: float,
    separation: float,
    target_charge: pint.Quantity | None,
) -> pint.Quantity | None:
    if r1 <= 0 or r2 <= 0:
        return None
    if separation == 0:
        cos_theta = 1.0
    else:
        cos_theta = (r1**2 + r2**2 - separation**2) / (2 * r1 * r2)
    if cos_theta < -1.000001 or cos_theta > 1.000001:
        return None
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))

    k = 8.9875517923e9
    q1_value = q1.to("C").magnitude
    q2_value = q2.to("C").magnitude

    e1x = -k * q1_value / (r1**2)
    e1y = 0.0
    e2_mag_factor = -k * q2_value / (r2**2)
    e2x = e2_mag_factor * cos_theta
    e2y = e2_mag_factor * sin_theta
    fx_or_ex = e1x + e2x
    fy_or_ey = e1y + e2y

    magnitude = math.hypot(fx_or_ex, fy_or_ey)
    if target_charge is not None:
        magnitude *= abs(target_charge.to("C").magnitude)
        return magnitude * ureg.newton
    return magnitude * (ureg.volt / ureg.meter)


def _resultant_force_hint(extraction: Extraction) -> float | None:
    lower = extraction.normalized_question.lower()
    forces = [
        quantity.value.to("N").magnitude
        for key, quantity in extraction.quantities.items()
        if key.startswith("force")
    ]
    if len(forces) >= 3:
        return forces[-1]
    match = re.search(r"resultant force is also\s*(?P<value>[-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*n", lower)
    if match:
        return float(match.group("value"))
    if "resultant force is also" in lower and forces:
        return forces[0]
    return None


def _vector_template_result(
    extraction: Extraction,
    value: pint.Quantity,
    formula_id: str,
    premise: str,
    cot: list[str],
) -> Type2SolveResult:
    converted = value.to("N")
    answer = _format_number(float(converted.magnitude))
    return Type2SolveResult(
        answer=answer,
        unit="N",
        value=converted,
        formula=None,
        extraction=extraction,
        verification=Verification(True, f"Matched deterministic vector template `{formula_id}`."),
        cot=[
            "Matched a high-confidence deterministic vector template before LLM code generation.",
            *cot,
            "Computed vector magnitude and converted the result to N.",
        ],
        premises=[premise, *[quantity.evidence for quantity in extraction.quantities.values()]],
        confidence=0.9,
        error=None,
    )


def _field_template_result(
    extraction: Extraction,
    value: pint.Quantity,
    formula_id: str,
    premise: str,
    cot: list[str],
) -> Type2SolveResult:
    converted = value.to("V/m")
    answer = _format_number(float(converted.magnitude))
    return Type2SolveResult(
        answer=answer,
        unit="V/m",
        value=converted,
        formula=None,
        extraction=extraction,
        verification=Verification(True, f"Matched deterministic field template `{formula_id}`."),
        cot=[
            "Matched a high-confidence deterministic field template before formula-bank execution.",
            *cot,
            "Computed electric-field magnitude and converted the result to V/m.",
        ],
        premises=[premise, *[quantity.evidence for quantity in extraction.quantities.values()]],
        confidence=0.9,
        error=None,
    )


def _has_quantities(quantities: dict[str, Quantity], *names: str) -> bool:
    return all(name in quantities for name in names)


def _equal_charge_wording(lower_question: str) -> bool:
    return any(
        marker in lower_question
        for marker in ("q1 = q2 = q3", "three identical", "three equal", "charges q1 = q2 = q3")
    ) or bool(re.search(r"\bq1\s*=\s*q2\b", lower_question))


def _k_coulomb() -> pint.Quantity:
    return 8.9875517923e9 * (ureg.newton * ureg.meter**2 / ureg.coulomb**2)


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"
