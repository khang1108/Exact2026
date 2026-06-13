from __future__ import annotations

from exact.type2.measurement.diagnostics import solved
from exact.type2.measurement.formatting_policy import apply_rounding
from exact.type2.measurement.schemas import MeasurementContract
from exact.type2.measurement.unit_normalizer import normalize_quantity, q


def solve(contract: MeasurementContract) -> dict:
    true_value = contract.true_value
    measured_value = contract.measured_value
    assert true_value is not None and measured_value is not None
    measured = normalize_quantity(measured_value, true_value.unit)
    delta = abs(measured.value - true_value.value)
    denom_policy = contract.error_model.get("denominator_policy")
    denominator = abs(true_value.value) if denom_policy == "true_value" else abs(measured.value)
    relative = delta / denominator
    computed = {
        "absolute_error": q(delta, true_value.unit),
        "relative_error": q(relative, "dimensionless"),
        "percentage_error": q(relative * 100, "percent"),
    }
    result = {key: computed[key] for key in contract.target.quantities if key in computed}
    return solved(
        "true_vs_measured_solver",
        "true_vs_measured_error",
        apply_rounding(result, contract.rounding_policy),
        computed=computed,
    )

