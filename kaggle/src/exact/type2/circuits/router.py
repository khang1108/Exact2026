from __future__ import annotations

from exact.type2.circuits.diagnostics import unsolved
from exact.type2.circuits.schemas import CircuitContract
from exact.type2.circuits.validator import validate_contract
from exact.type2.circuits.solvers import (
    ac_phasor_solver,
    ac_rms_solver,
    dc_resistor_network_solver,
    energy_time_solver,
    scalar_ohm_solver,
    transformer_solver,
)


def solve_circuit_contract(contract: CircuitContract) -> dict:
    validated, issue = validate_contract(contract)
    if issue or validated is None:
        return unsolved("circuits_router", issue.reason if issue else "validation failed", missing=list(issue.missing) if issue else [])
    system_type = validated.contract.system_type
    target = validated.contract.target.quantity
    if system_type == "single_resistor":
        return scalar_ohm_solver.solve(validated.contract)
    if system_type == "dc_resistor_network":
        return dc_resistor_network_solver.solve(validated.contract)
    if system_type == "energy_consumption":
        return energy_time_solver.solve(validated.contract)
    if system_type in {"series_ac_circuit", "series_rlc_circuit"}:
        if target in {"inductive_reactance", "capacitive_reactance"}:
            return ac_rms_solver.solve(validated.contract)
        return ac_phasor_solver.solve(validated.contract)
    if system_type == "ideal_transformer":
        return transformer_solver.solve(validated.contract)
    return unsolved("circuits_router", f"unsupported system_type `{system_type}`")

