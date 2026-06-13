from __future__ import annotations

from exact.type2.measurement.diagnostics import solved
from exact.type2.measurement.formatting_policy import apply_rounding
from exact.type2.measurement.schemas import MeasurementContract
from exact.type2.measurement.statistics_solver import repeated_measurement_stats
from exact.type2.measurement.unit_normalizer import q


def solve(contract: MeasurementContract) -> dict:
    stats = repeated_measurement_stats(
        contract.measurements,
        contract.error_model["mean_error_definition"],
        contract.target.unit,
    )
    relative = stats["mean_absolute_error"] / abs(stats["mean"])
    computed = {
        "mean_value": q(stats["mean"], stats["unit"]),
        "mean_absolute_error": q(stats["mean_absolute_error"], stats["unit"]),
        "relative_error": q(relative, "dimensionless"),
        "percentage_error": q(relative * 100, "percent"),
    }
    result = {key: computed[key] for key in contract.target.quantities if key in computed}
    return solved("repeated_measurement_solver", "repeated_measurement_statistics", apply_rounding(result, contract.rounding_policy), computed=computed)

