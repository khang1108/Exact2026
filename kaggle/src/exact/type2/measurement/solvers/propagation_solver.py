from __future__ import annotations

from exact.type2.measurement.diagnostics import solved
from exact.type2.measurement.error_propagation import (
    evaluate_ast,
    is_additive_ast,
    propagated_absolute_error_for_sum,
    propagated_relative_error,
)
from exact.type2.measurement.formatting_policy import apply_rounding
from exact.type2.measurement.schemas import MeasurementContract
from exact.type2.measurement.unit_normalizer import q


def solve(contract: MeasurementContract) -> dict:
    symbols = {quantity.symbol: quantity for quantity in contract.measured_quantities.values()}
    ast = contract.derived_quantity["formula_ast"]
    unit = contract.target.unit or contract.derived_quantity.get("unit", "dimensionless")
    value = evaluate_ast(ast, symbols)
    if is_additive_ast(ast):
        absolute = propagated_absolute_error_for_sum(ast, symbols)
        relative = absolute / abs(value)
        contributions = {}
    else:
        relative, contributions = propagated_relative_error(ast, symbols)
        absolute = abs(value) * relative
    computed = {
        "value": q(value, unit),
        "absolute_error": q(absolute, unit),
        "relative_error": q(relative, "dimensionless"),
        "percentage_error": q(relative * 100, "percent"),
        "result_with_uncertainty": {
            "text": f"{value} ± {absolute} {unit}",
            "value": value,
            "absolute_error": absolute,
            "unit": unit,
        },
    }
    result = {key: computed[key] for key in contract.target.quantities if key in computed}
    rounded = apply_rounding(result, contract.rounding_policy)
    return solved(
        "propagation_solver",
        "first_order_relative_propagation" if not is_additive_ast(ast) else "absolute_error_addition",
        rounded,
        computed=computed,
        relative_inputs={symbol: q(value, "dimensionless") for symbol, value in contributions.items()},
        formula_chain=[contract.derived_quantity.get("formula", ""), "propagate uncertainty from formula_ast"],
    )

