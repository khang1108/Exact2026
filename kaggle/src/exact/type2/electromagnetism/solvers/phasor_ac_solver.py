from __future__ import annotations

import math

from exact.type2.electromagnetism.diagnostics import solved, unsolved
from exact.type2.electromagnetism.equation_graph import rlc_values
from exact.type2.electromagnetism.schemas import ElectromagnetismContract
from exact.type2.electromagnetism.solvers._helpers import comp, q, source


def solve(contract: ElectromagnetismContract) -> dict:
    r = comp(contract, "resistor", "resistance")
    values = rlc_values(r, comp(contract, "inductor", "inductance"), comp(contract, "capacitor", "capacitance"), source(contract, "frequency"))
    x = values["X"]
    relation = "current and voltage are in phase"
    character = "resistive/resonant"
    if x > 0:
        character, relation = "inductive", "current lags voltage"
    elif x < 0:
        character, relation = "capacitive", "current leads voltage"
    inter = {
        "omega": q(values["omega"], "rad/s"),
        "XL": q(values["XL"], "ohm"),
        "XC": q(values["XC"], "ohm"),
        "X": q(x, "ohm"),
        "Z": q(values["Z"], "ohm"),
    }
    target = contract.target.quantity
    if target == "impedance_magnitude":
        return solved("phasor_ac_solver", "series_rlc_impedance_magnitude", q(values["Z"], "ohm"), intermediate_values=inter)
    if target == "complex_impedance":
        return solved("phasor_ac_solver", "series_rlc_complex_impedance", {"value": f"{r:g} + j({x:g})", "unit": "ohm"}, intermediate_values=inter)
    if target == "phase_angle":
        unit = contract.target.unit or "degree"
        phi = math.degrees(values["phi"]) if unit == "degree" else values["phi"]
        return solved("phasor_ac_solver", "series_rlc_phasor_impedance", {"value": phi, "unit": unit, "interpretation": f"{character}; {relation}"}, intermediate_values=inter)
    if target == "power_factor":
        return solved("phasor_ac_solver", "series_rlc_power_factor", q(r / values["Z"], "dimensionless"), intermediate_values=inter)
    if target == "circuit_character":
        return solved("phasor_ac_solver", "series_rlc_circuit_character", {"value": character, "unit": "categorical", "current_relation": relation}, intermediate_values=inter)
    if target == "voltage_current_relation":
        return solved("phasor_ac_solver", "series_rlc_voltage_current_relation", {"value": relation, "unit": "categorical", "circuit_character": character}, intermediate_values=inter)
    if target in {"current_rms", "active_power"}:
        voltage = source(contract, "voltage_rms")
        current = voltage / values["Z"]
        inter["current_rms"] = q(current, "A")
        if target == "current_rms":
            return solved("phasor_ac_solver", "series_rlc_rms_current", q(current, "A"), intermediate_values=inter)
        power = voltage * current * r / values["Z"]
        return solved("phasor_ac_solver", "series_rlc_active_power", q(power, "W"), intermediate_values=inter)
    return unsolved("phasor_ac_solver", f"unsupported phasor target `{target}`", fallback_recommended=False)
