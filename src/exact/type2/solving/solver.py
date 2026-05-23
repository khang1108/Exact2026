from __future__ import annotations

import math
import pint

from exact.type2.formulas.bank import retrieve_formulas
from exact.type2.schemas import Extraction, Type2SolveResult, Verification, Formula


CONCEPTS = (
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




def solve_extraction(extraction: Extraction) -> Type2SolveResult:
    known = {name: quantity.value for name, quantity in extraction.quantities.items()}
    formulas = retrieve_formulas(extraction.normalized_question, extraction.target, known)

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
