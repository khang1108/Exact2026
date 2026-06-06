from __future__ import annotations

import math

from exact.type2.electromagnetism.diagnostics import solved, unsolved
from exact.type2.electromagnetism.equation_graph import omega_from_frequency
from exact.type2.electromagnetism.schemas import ElectromagnetismContract
from exact.type2.electromagnetism.solvers._helpers import comp, q, source


def solve(contract: ElectromagnetismContract) -> dict:
    omega = omega_from_frequency(source(contract, "frequency"))
    target = contract.target.quantity
    xl = omega * comp(contract, "inductor", "inductance") if _has(contract, "inductor", "inductance") else None
    xc = 1 / (omega * comp(contract, "capacitor", "capacitance")) if _has(contract, "capacitor", "capacitance") else None
    inter = {"omega": q(omega, "rad/s")}
    if xl is not None:
        inter["XL"] = q(xl, "ohm")
    if xc is not None:
        inter["XC"] = q(xc, "ohm")
    if target == "inductive_reactance" and xl is not None:
        return solved("reactance_solver", "inductive_reactance", q(xl, "ohm"), intermediate_values=inter)
    if target == "capacitive_reactance" and xc is not None:
        return solved("reactance_solver", "capacitive_reactance", q(xc, "ohm"), intermediate_values=inter)
    if target == "net_reactance" and xl is not None and xc is not None:
        return solved("reactance_solver", "net_reactance", q(xl - xc, "ohm"), intermediate_values=inter)
    if target == "impedance_magnitude" and xl is not None and xc is not None:
        r = comp(contract, "resistor", "resistance")
        return solved("reactance_solver", "reactance_impedance_magnitude", q(math.sqrt(r**2 + (xl - xc) ** 2), "ohm"), intermediate_values=inter)
    return unsolved("reactance_solver", f"insufficient reactance data for `{target}`")


def _has(contract: ElectromagnetismContract, kind: str, prop: str) -> bool:
    return any(component.kind == kind and prop in component.properties for component in contract.components)
