from __future__ import annotations

import math

from exact.type2.electromagnetism.diagnostics import solved, unsolved
from exact.type2.electromagnetism.schemas import ElectromagnetismContract
from exact.type2.electromagnetism.solvers._helpers import q, side


def solve(contract: ElectromagnetismContract) -> dict:
    target = contract.target.quantity
    np = side(contract.primary, "turns") if "turns" in contract.primary else None
    ns = side(contract.secondary, "turns") if "turns" in contract.secondary else None
    inter = {}
    if np is not None and ns is not None:
        inter["turns_ratio"] = q(np / ns, "dimensionless")
    if target == "turns_ratio" and np is not None and ns is not None:
        return solved("transformer_solver", "ideal_transformer_turns_ratio", q(np / ns, "dimensionless"), intermediate_values=inter)
    if target == "transformer_type" and np is not None and ns is not None:
        if ns > np:
            kind = "step_up"
        elif ns < np:
            kind = "step_down"
        else:
            kind = "isolation/equal-voltage transformer"
        return solved("transformer_solver", "ideal_transformer_type", {"value": kind, "unit": "categorical"}, intermediate_values=inter)
    if target == "secondary_voltage" and np is not None and ns is not None:
        vs = side(contract.primary, "voltage_rms") * ns / np
        return solved("transformer_solver", "ideal_transformer_secondary_voltage", q(vs, "V"), intermediate_values=inter)
    if target == "primary_voltage" and np is not None and ns is not None:
        vp = side(contract.secondary, "voltage_rms") * np / ns
        return solved("transformer_solver", "ideal_transformer_primary_voltage", q(vp, "V"), intermediate_values=inter)
    if target == "secondary_current" and np is not None and ns is not None:
        is_ = side(contract.primary, "current_rms") * np / ns
        return solved("transformer_solver", "ideal_transformer_secondary_current", q(is_, "A"), intermediate_values=inter)
    if target == "primary_current" and np is not None and ns is not None:
        ip = side(contract.secondary, "current_rms") * ns / np
        return solved("transformer_solver", "ideal_transformer_primary_current", q(ip, "A"), intermediate_values=inter)
    if target == "primary_turns" and ns is not None:
        np_value = _solve_primary_turns(contract, ns)
        if np_value is not None:
            return solved("transformer_solver", "ideal_transformer_primary_turns", q(np_value, "dimensionless"), intermediate_values=inter)
    if target == "secondary_turns" and np is not None:
        ns_value = _solve_secondary_turns(contract, np)
        if ns_value is not None:
            return solved("transformer_solver", "ideal_transformer_secondary_turns", q(ns_value, "dimensionless"), intermediate_values=inter)
    return unsolved("transformer_solver", f"insufficient transformer data for `{target}`")


def _solve_primary_turns(contract: ElectromagnetismContract, ns: float) -> float | None:
    if "voltage_rms" in contract.primary and "voltage_rms" in contract.secondary:
        return side(contract.primary, "voltage_rms") * ns / side(contract.secondary, "voltage_rms")
    if "current_rms" in contract.primary and "current_rms" in contract.secondary:
        return side(contract.secondary, "current_rms") * ns / side(contract.primary, "current_rms")
    return None


def _solve_secondary_turns(contract: ElectromagnetismContract, np: float) -> float | None:
    if "voltage_rms" in contract.primary and "voltage_rms" in contract.secondary:
        return side(contract.secondary, "voltage_rms") * np / side(contract.primary, "voltage_rms")
    if "current_rms" in contract.primary and "current_rms" in contract.secondary:
        return side(contract.primary, "current_rms") * np / side(contract.secondary, "current_rms")
    return None
