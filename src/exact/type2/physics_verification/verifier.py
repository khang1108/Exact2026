from __future__ import annotations

import math
from typing import Any

from exact.type2.physics_contract.dimensions import units_compatible
from exact.type2.physics_contract.models import PhysicsContract
from exact.type2.physics_verification.models import PhysicsSanityReport, RuleResult
from exact.type2.physics_verification.reverse_solve import reverse_solve_checks


def verify_physics_sanity(
    *,
    answer: str,
    unit: str | None,
    value: Any,
    formula_ids_used: list[str],
    contract: PhysicsContract,
    check_dimensions: bool = True,
) -> PhysicsSanityReport:
    rule_results: list[RuleResult] = []
    if check_dimensions:
        rule_results.extend(_dimension_rules(unit, contract))
    rule_results.extend(_invariant_rules(answer, value, contract))
    rule_results.extend(reverse_solve_checks(value, formula_ids_used, contract))

    errors = tuple(result.message for result in rule_results if not result.passed and result.severity == "error")
    warnings = tuple(result.message for result in rule_results if not result.passed and result.severity == "warning")
    confidence_delta = sum(result.confidence_delta for result in rule_results)
    if rule_results and not errors:
        confidence_delta += 0.03
    return PhysicsSanityReport(
        accepted=not errors,
        confidence_delta=confidence_delta,
        errors=errors,
        warnings=warnings,
        rule_results=tuple(rule_results),
    )


def _dimension_rules(unit: str | None, contract: PhysicsContract) -> list[RuleResult]:
    expected = contract.expected_unit
    if expected is None:
        return []
    passed = units_compatible(unit, expected)
    return [
        RuleResult(
            rule_id="dimension.expected_unit",
            passed=passed,
            message=(
                f"dimension_passed:{unit}->{expected}"
                if passed
                else f"dimension_mismatch:actual={unit}:expected={expected}"
            ),
            severity="error",
            confidence_delta=0.05 if passed else -0.25,
        )
    ]


def _invariant_rules(answer: str, value: Any, contract: PhysicsContract) -> list[RuleResult]:
    results: list[RuleResult] = []
    target = contract.target
    magnitude = _numeric_magnitude(answer, value)
    if target in {"capacitance", "resistance", "current", "voltage", "pressure", "volume"}:
        results.append(_positive_rule(target, magnitude))
    if target in {"energy", "stored_energy", "work", "potential_energy"}:
        results.append(
            RuleResult(
                rule_id=f"invariant.{target}.nonnegative",
                passed=magnitude is None or magnitude >= -1e-12,
                message=f"{target}_nonnegative" if magnitude is None or magnitude >= -1e-12 else f"{target}_negative",
                severity="error",
                confidence_delta=0.03 if magnitude is None or magnitude >= -1e-12 else -0.2,
            )
        )
    results.extend(_network_rules(magnitude, contract))
    results.extend(_electrostatic_midpoint_rules(magnitude, contract))
    return results


def _positive_rule(target: str, magnitude: float | None) -> RuleResult:
    passed = magnitude is None or magnitude > 0
    return RuleResult(
        rule_id=f"invariant.{target}.positive",
        passed=passed,
        message=f"{target}_positive" if passed else f"{target}_not_positive",
        severity="error",
        confidence_delta=0.03 if passed else -0.2,
    )


def _network_rules(magnitude: float | None, contract: PhysicsContract) -> list[RuleResult]:
    if magnitude is None or contract.target not in {"resistance", "equivalent_resistance"}:
        return []
    resistances = [
        _quantity_to_float(known.value, "ohm")
        for known in contract.knowns.values()
        if known.dimension == "resistance" and known.value is not None
    ]
    resistances = [item for item in resistances if item is not None and item > 0]
    if len(resistances) < 2:
        return []
    relation = next((constraint.relation for constraint in contract.constraints if constraint.kind == "network"), None)
    if relation == "series":
        passed = magnitude > max(resistances)
        return [
            RuleResult(
                "invariant.resistors.series",
                passed,
                "series_resistance_gt_max" if passed else "series_resistance_not_gt_max",
                confidence_delta=0.04 if passed else -0.25,
            )
        ]
    if relation == "parallel":
        passed = magnitude < min(resistances)
        return [
            RuleResult(
                "invariant.resistors.parallel",
                passed,
                "parallel_resistance_lt_min" if passed else "parallel_resistance_not_lt_min",
                confidence_delta=0.04 if passed else -0.25,
            )
        ]
    return []


def _electrostatic_midpoint_rules(magnitude: float | None, contract: PhysicsContract) -> list[RuleResult]:
    if magnitude is None:
        return []
    has_midpoint = any(constraint.kind == "midpoint" for constraint in contract.constraints)
    has_opposite = any(
        constraint.kind == "charge_signs" and constraint.relation == "opposite"
        for constraint in contract.constraints
    )
    if not (has_midpoint and has_opposite):
        return []
    if contract.target == "electric_field":
        passed = abs(magnitude) > 1e-12
        return [
            RuleResult(
                "electrostatics.midpoint.field_opposite_charges",
                passed,
                "opposite_charge_midpoint_field_nonzero" if passed else "opposite_charge_midpoint_field_zero",
                confidence_delta=0.05 if passed else -0.3,
            )
        ]
    if contract.target in {"potential", "electric_potential", "voltage"}:
        passed = abs(magnitude) <= 1e-9
        return [
            RuleResult(
                "electrostatics.midpoint.potential_opposite_charges",
                passed,
                "opposite_charge_midpoint_potential_zero" if passed else "opposite_charge_midpoint_potential_nonzero",
                confidence_delta=0.05 if passed else -0.3,
            )
        ]
    return []


def _numeric_magnitude(answer: str, value: Any) -> float | None:
    if value is not None:
        try:
            number = float(value.magnitude)
            if math.isfinite(number):
                return number
        except Exception:
            pass
    try:
        number = float(answer)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantity_to_float(quantity: Any, unit: str) -> float | None:
    try:
        number = float(quantity.to(unit).magnitude)
    except Exception:
        return None
    return number if math.isfinite(number) else None
