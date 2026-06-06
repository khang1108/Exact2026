from __future__ import annotations

from exact.type2.measurement.diagnostics import solved
from exact.type2.measurement.formatting_policy import apply_rounding
from exact.type2.measurement.schemas import MeasurementContract
from exact.type2.measurement.uncertainty_policy import absolute_uncertainty
from exact.type2.measurement.unit_normalizer import q


def solve(contract: MeasurementContract) -> dict:
    quantity = next(iter(contract.measured_quantities.values()))
    delta = absolute_uncertainty(quantity)
    relative = delta / abs(quantity.value)
    computed = {
        "value": q(quantity.value, quantity.unit),
        "absolute_error": q(delta, quantity.unit),
        "relative_error": q(relative, "dimensionless"),
        "percentage_error": q(relative * 100, "percent"),
    }
    result = {key: computed[key] for key in contract.target.quantities if key in computed}
    return solved("direct_error_solver", "direct_uncertainty", apply_rounding(result, contract.rounding_policy), computed=computed)

