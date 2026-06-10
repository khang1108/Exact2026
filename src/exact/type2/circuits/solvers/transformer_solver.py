from __future__ import annotations

from exact.type2.circuits.diagnostics import solved, unsolved
from exact.type2.circuits.schemas import CircuitContract
from exact.type2.circuits.solvers._helpers import q, side


def solve(contract: CircuitContract) -> dict:
    np = side(contract.primary, "turns") if "turns" in contract.primary else None
    ns = side(contract.secondary, "turns") if "turns" in contract.secondary else None
    target = contract.target.quantity
    inter = {"turns_ratio": q(np / ns, "dimensionless")} if np is not None and ns is not None else {}
    if target == "secondary_voltage" and np is not None and ns is not None:
        return solved("transformer_solver", "ideal_transformer_secondary_voltage", q(side(contract.primary, "voltage_rms") * ns / np, "V"), intermediate_values=inter)
    if target == "primary_voltage" and np is not None and ns is not None:
        return solved("transformer_solver", "ideal_transformer_primary_voltage", q(side(contract.secondary, "voltage_rms") * np / ns, "V"), intermediate_values=inter)
    if target == "secondary_current" and np is not None and ns is not None:
        return solved("transformer_solver", "ideal_transformer_secondary_current", q(side(contract.primary, "current_rms") * np / ns, "A"), intermediate_values=inter)
    if target == "primary_current" and np is not None and ns is not None:
        return solved("transformer_solver", "ideal_transformer_primary_current", q(side(contract.secondary, "current_rms") * ns / np, "A"), intermediate_values=inter)
    if target == "turns_ratio" and np is not None and ns is not None:
        return solved("transformer_solver", "ideal_transformer_turns_ratio", q(np / ns, "dimensionless"), intermediate_values=inter)
    if target == "transformer_type" and np is not None and ns is not None:
        kind = "step_up" if ns > np else "step_down" if ns < np else "equal_voltage"
        return solved("transformer_solver", "ideal_transformer_type", {"value": kind, "unit": "categorical"}, intermediate_values=inter)
    if target == "primary_turns" and ns is not None:
        if "voltage_rms" in contract.primary and "voltage_rms" in contract.secondary:
            return solved("transformer_solver", "ideal_transformer_primary_turns", q(side(contract.primary, "voltage_rms") * ns / side(contract.secondary, "voltage_rms"), "dimensionless"), intermediate_values=inter)
        if "current_rms" in contract.primary and "current_rms" in contract.secondary:
            return solved("transformer_solver", "ideal_transformer_primary_turns", q(side(contract.secondary, "current_rms") * ns / side(contract.primary, "current_rms"), "dimensionless"), intermediate_values=inter)
    if target == "secondary_turns" and np is not None:
        if "voltage_rms" in contract.primary and "voltage_rms" in contract.secondary:
            return solved("transformer_solver", "ideal_transformer_secondary_turns", q(side(contract.secondary, "voltage_rms") * np / side(contract.primary, "voltage_rms"), "dimensionless"), intermediate_values=inter)
        if "current_rms" in contract.primary and "current_rms" in contract.secondary:
            return solved("transformer_solver", "ideal_transformer_secondary_turns", q(side(contract.primary, "current_rms") * np / side(contract.secondary, "current_rms"), "dimensionless"), intermediate_values=inter)
    return unsolved("transformer_solver", f"unsupported transformer target `{target}`")
