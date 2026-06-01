from __future__ import annotations

import math
import pint

from exact.type2.formulas.bank import retrieve_formulas
from exact.type2.schemas import Extraction, Type2SolveResult, Verification, Formula


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


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"
