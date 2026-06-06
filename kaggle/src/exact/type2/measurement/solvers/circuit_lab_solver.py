from __future__ import annotations

from exact.type2.measurement.schemas import MeasurementContract
from exact.type2.measurement.solvers.propagation_solver import solve as solve_propagation


def solve(contract: MeasurementContract) -> dict:
    result = solve_propagation(contract)
    result["solver"] = "circuit_lab_solver"
    result["selected_rule"] = f"{contract.derived_quantity.get('formula', 'circuit formula')} with relative uncertainty propagation"
    result["formula_chain"] = [
        contract.derived_quantity.get("formula", ""),
        "relative input uncertainties are added for product/quotient/powers",
        "absolute uncertainty = value * relative uncertainty",
    ]
    return result

