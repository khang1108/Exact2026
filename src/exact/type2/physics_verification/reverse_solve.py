from __future__ import annotations

import math
from typing import Any

from exact.type2.formulas.bank import FORMULAS
from exact.type2.physics_contract.models import PhysicsContract
from exact.type2.physics_verification.models import RuleResult


def reverse_solve_checks(
    candidate_value: Any,
    formula_ids_used: list[str],
    contract: PhysicsContract,
    *,
    relative_tolerance: float = 0.02,
) -> list[RuleResult]:
    if candidate_value is None or not formula_ids_used:
        return []

    formula_by_id = {formula.id: formula for formula in FORMULAS}
    known_values = {
        name: known.value
        for name, known in contract.knowns.items()
        if known.value is not None
    }
    results: list[RuleResult] = []
    for formula_id in formula_ids_used:
        formula = formula_by_id.get(formula_id)
        if formula is None:
            continue
        if formula.target != contract.target:
            continue
        if not set(formula.required) <= set(known_values):
            continue
        try:
            reference = formula.solve(known_values)
            converted = candidate_value.to(reference.units)
            candidate_number = float(converted.magnitude)
            reference_number = float(reference.magnitude)
        except Exception as exc:
            results.append(
                RuleResult(
                    rule_id=f"reverse.{formula_id}",
                    passed=False,
                    message=f"reverse_solve_failed:{formula_id}:{exc}",
                    severity="warning",
                    confidence_delta=-0.03,
                )
            )
            continue
        if not (math.isfinite(candidate_number) and math.isfinite(reference_number)):
            results.append(
                RuleResult(
                    rule_id=f"reverse.{formula_id}",
                    passed=False,
                    message=f"reverse_solve_nonfinite:{formula_id}",
                    severity="error",
                    confidence_delta=-0.1,
                )
            )
            continue
        scale = max(abs(reference_number), 1e-12)
        rel_error = abs(candidate_number - reference_number) / scale
        passed = rel_error <= relative_tolerance
        results.append(
            RuleResult(
                rule_id=f"reverse.{formula_id}",
                passed=passed,
                message=(
                    f"reverse_solve_passed:{formula_id}"
                    if passed
                    else f"reverse_solve_mismatch:{formula_id}:relative_error={rel_error:.6g}"
                ),
                severity="error",
                confidence_delta=0.1 if passed else -0.2,
            )
        )
    return results

