from __future__ import annotations

from exact.type2.electromagnetism.diagnostics import unsolved
from exact.type2.electromagnetism.schemas import ElectromagnetismContract
from exact.type2.electromagnetism.validator import validate_contract
from exact.type2.electromagnetism.solvers import (
    faraday_lenz_solver,
    lc_energy_state_solver,
    phasor_ac_solver,
    reactance_solver,
    resonance_solver,
    solenoid_solver,
    transformer_solver,
)


def solve_electromagnetism_contract(contract: ElectromagnetismContract) -> dict:
    validated, issue = validate_contract(contract)
    if issue is not None or validated is None:
        return unsolved("electromagnetism_router", issue.reason if issue else "validation failed", missing=list(issue.missing) if issue else [])
    system_type = validated.contract.system_type
    target = validated.contract.target.quantity
    if system_type == "ideal_lc_oscillator":
        return lc_energy_state_solver.solve(validated.contract)
    if system_type == "series_rlc_circuit":
        if target in {"is_resonant", "resonance_frequency", "resonance_angular_frequency", "resonance_condition", "circuit_state_at_resonance"}:
            return resonance_solver.solve(validated.contract)
        return phasor_ac_solver.solve(validated.contract)
    if system_type == "ac_reactance":
        return reactance_solver.solve(validated.contract)
    if system_type == "long_solenoid":
        return solenoid_solver.solve(validated.contract)
    if system_type == "electromagnetic_induction":
        return faraday_lenz_solver.solve(validated.contract)
    if system_type == "ideal_transformer":
        return transformer_solver.solve(validated.contract)
    return unsolved("electromagnetism_router", f"unsupported system_type `{system_type}`")
