from __future__ import annotations

import math

from exact.type2.electromagnetism.diagnostics import solved
from exact.type2.electromagnetism.equation_graph import rlc_values
from exact.type2.electromagnetism.schemas import ElectromagnetismContract
from exact.type2.electromagnetism.solvers._helpers import comp, q, source


def solve(contract: ElectromagnetismContract) -> dict:
    l_h = comp(contract, "inductor", "inductance")
    c_f = comp(contract, "capacitor", "capacitance")
    f0 = 1 / (2 * math.pi * math.sqrt(l_h * c_f))
    target = contract.target.quantity
    if target == "resonance_frequency":
        return solved("resonance_solver", "series_rlc_resonance_frequency", q(f0, "Hz"))
    if target == "resonance_angular_frequency":
        return solved("resonance_solver", "series_rlc_resonance_angular_frequency", q(2 * math.pi * f0, "rad/s"))
    if target == "resonance_condition":
        return solved("resonance_solver", "series_rlc_resonance_condition", {"value": "XL = XC, equivalently omega0 = 1/sqrt(LC)", "unit": "conceptual"})
    if target == "circuit_state_at_resonance":
        return solved("resonance_solver", "series_rlc_state_at_resonance", {"value": "net reactance is zero; impedance equals R; phase angle is zero; power factor is one", "unit": "conceptual"})
    f = source(contract, "frequency")
    values = rlc_values(0.0, l_h, c_f, f)
    is_res = math.isclose(values["XL"], values["XC"], rel_tol=1e-3, abs_tol=1e-9)
    return solved(
        "resonance_solver",
        "series_rlc_resonance_check",
        {"value": is_res, "unit": "boolean", "reason": "XL = XC" if is_res else "XL != XC"},
        intermediate_values={"XL": q(values["XL"], "ohm"), "XC": q(values["XC"], "ohm"), "f0": q(f0, "Hz")},
    )
