from __future__ import annotations

from exact.type2.measurement.diagnostics import unsolved
from exact.type2.measurement.schemas import MeasurementContract
from exact.type2.measurement.validator import validate_contract
from exact.type2.measurement.solvers import (
    circuit_lab_solver,
    direct_error_solver,
    least_count_solver,
    propagation_solver,
    repeated_measurement_solver,
    true_vs_measured_solver,
)


def solve_measurement_contract(contract: MeasurementContract) -> dict:
    validated, issue = validate_contract(contract)
    if issue or validated is None:
        return unsolved("measurement_error_solver", issue.reason if issue else "validation failed", missing=list(issue.missing) if issue else [])
    system_type = contract.system_type
    if system_type == "least_count_error":
        return least_count_solver.solve(contract)
    if system_type == "true_vs_measured_error":
        return true_vs_measured_solver.solve(contract)
    if system_type == "direct_uncertainty":
        return direct_error_solver.solve(contract)
    if system_type == "repeated_measurement":
        return repeated_measurement_solver.solve(contract)
    if system_type == "propagation":
        return propagation_solver.solve(contract)
    if system_type == "circuit_lab_propagation":
        return circuit_lab_solver.solve(contract)
    return unsolved("measurement_error_solver", f"unsupported system_type `{system_type}`")

