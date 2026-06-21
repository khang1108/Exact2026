from __future__ import annotations

import math

import pint

from exact.type2.equation_graph.models import EquationGraphResult, EquationStep
from exact.type2.formulas.bank import FORMULAS
from exact.type2.formulas.knowledge import RetrievedFormulaContext
from exact.type2.physics_contract.dimensions import canonical_dimension
from exact.type2.physics_contract.models import PhysicsContract
from exact.type2.schemas import Extraction


def solve_with_equation_graph(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
    contract: PhysicsContract,
) -> EquationGraphResult:
    if extraction.kind.value == "conceptual":
        return EquationGraphResult(False, error="conceptual_question")
    if contract.target in {"force", "electric_field"} and any(
        constraint.kind.startswith("vector") or constraint.kind == "midpoint"
        for constraint in contract.constraints
    ):
        return EquationGraphResult(False, error="vector_or_geometry_target")

    formulas = [formula for formula in FORMULAS if formula.id in set(formula_context.formula_ids)]
    if not formulas:
        return EquationGraphResult(False, error="no_executable_formula_candidates")

    values: dict[str, pint.Quantity] = {
        name: known.value
        for name, known in contract.knowns.items()
        if known.value is not None
    }
    steps: list[EquationStep] = []
    used: list[str] = []
    target = canonical_dimension(contract.target)

    for _ in range(max(1, len(formulas))):
        progressed = False
        for formula in formulas:
            if formula.id in used:
                continue
            if formula.target in values:
                continue
            if not set(formula.required) <= set(values):
                continue
            try:
                result = formula.solve(values)
                canonical = _canonicalize_quantity(result, formula.output_unit)
                magnitude = float(canonical.magnitude)
            except Exception:
                continue
            if not math.isfinite(magnitude):
                continue
            values[formula.target] = canonical
            used.append(formula.id)
            steps.append(
                EquationStep(
                    formula_id=formula.id,
                    output=formula.target,
                    expression=formula.expression,
                    value=canonical,
                )
            )
            progressed = True
            if formula.target == target or canonical_dimension(formula.target) == target:
                return EquationGraphResult(
                    solved=True,
                    answer=_format_number(float(canonical.magnitude)),
                    unit=_normalize_unit(str(canonical.units)),
                    value=canonical,
                    formula_ids_used=tuple(used),
                    steps=tuple(steps),
                )
        if not progressed:
            break
    return EquationGraphResult(
        solved=False,
        formula_ids_used=tuple(used),
        steps=tuple(steps),
        error=f"target_not_reached:{target}",
    )


def _canonicalize_quantity(quantity: pint.Quantity, output_unit: str) -> pint.Quantity:
    if output_unit and output_unit != "dimensionless":
        try:
            return quantity.to(output_unit)
        except Exception:
            return quantity
    return quantity


def _normalize_unit(unit: str) -> str:
    return (
        unit.replace("ampere", "A")
        .replace("volt", "V")
        .replace("newton", "N")
        .replace("coulomb", "C")
        .replace("farad", "F")
        .replace("joule", "J")
        .replace("watt", "W")
        .replace("second", "s")
        .replace("meter", "m")
        .replace("ohm", "ohm")
        .replace("N / C", "N/C")
        .replace("V / m", "V/m")
    )


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"

