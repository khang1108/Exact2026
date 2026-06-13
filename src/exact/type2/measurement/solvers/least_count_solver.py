from __future__ import annotations

from exact.type2.measurement.diagnostics import solved
from exact.type2.measurement.formatting_policy import apply_rounding
from exact.type2.measurement.schemas import MeasurementContract, MeasurementQuantity
from exact.type2.measurement.unit_normalizer import q


def solve(contract: MeasurementContract) -> dict:
    least_count = contract.instrument["least_count"]
    assert isinstance(least_count, MeasurementQuantity)
    rule = contract.error_policy["least_count_rule"]
    error = least_count.value if rule == "full" else least_count.value / 2
    result = {"absolute_error": q(error, contract.target.unit or least_count.unit)}
    result = {key: value for key, value in result.items() if key in contract.target.quantities}
    return solved(
        "least_count_solver",
        f"least_count_{rule}",
        apply_rounding(result, contract.rounding_policy),
        normalized_inputs={"least_count": q(least_count.value, least_count.unit), "least_count_rule": rule},
    )

