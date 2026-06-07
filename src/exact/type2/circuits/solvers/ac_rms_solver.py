from __future__ import annotations

import math

from exact.type2.circuits.diagnostics import solved, unsolved
from exact.type2.circuits.schemas import CircuitContract
from exact.type2.circuits.solvers._helpers import prop, q, source


def solve(contract: CircuitContract) -> dict:
    omega = 2 * math.pi * source(contract, "frequency", "Hz")
    target_id = contract.target.component_id
    target = contract.target.quantity
    for component in contract.components:
        if target_id and component.id != target_id:
            continue
        if target == "inductive_reactance" and component.kind == "inductor":
            return solved("ac_rms_solver", "inductive_reactance", q(omega * prop(component, "inductance", "H"), "ohm"))
        if target == "capacitive_reactance" and component.kind == "capacitor":
            if omega == 0 or prop(component, "capacitance", "F") == 0:
                return unsolved("ac_rms_solver", "frequency or capacitance is zero")
            return solved("ac_rms_solver", "capacitive_reactance", q(1 / (omega * prop(component, "capacitance", "F")), "ohm"))
    return unsolved("ac_rms_solver", f"reactance target `{target}` could not be resolved")

