from __future__ import annotations

import math

from exact.type2.circuits.diagnostics import solved, unsolved
from exact.type2.circuits.schemas import CircuitContract
from exact.type2.circuits.solvers._helpers import prop, q, source


def solve(contract: CircuitContract) -> dict:
    f = source(contract, "frequency", "Hz")
    omega = 2 * math.pi * f
    r = sum(prop(c, "resistance", "ohm") for c in contract.components if c.kind == "resistor")
    xl = sum(omega * prop(c, "inductance", "H") for c in contract.components if c.kind == "inductor")
    if omega == 0 and any(c.kind == "capacitor" for c in contract.components):
        return unsolved("ac_phasor_solver", "frequency is zero with capacitors present")
    xc = sum(1 / (omega * prop(c, "capacitance", "F")) for c in contract.components if c.kind == "capacitor")
    x = xl - xc
    z = math.sqrt(r**2 + x**2)
    phi = math.atan2(x, r) if r or x else 0.0
    character, relation = "resistive/resonant", "current and voltage are in phase"
    if x > 0:
        character, relation = "inductive", "current lags voltage"
    elif x < 0:
        character, relation = "capacitive", "current leads voltage"
    inter = {
        "omega": q(omega, "rad/s"),
        "XL": q(xl, "ohm"),
        "XC": q(xc, "ohm"),
        "X": q(x, "ohm"),
        "Z_magnitude": q(z, "ohm"),
        "phase_angle": q(math.degrees(phi), "degree"),
    }
    target = contract.target.quantity
    if target in {"impedance", "impedance_magnitude"}:
        return solved("ac_phasor_solver", "series_ac_impedance", q(z, "ohm"), intermediate_values=inter)
    if target == "complex_impedance":
        return solved("ac_phasor_solver", "series_ac_complex_impedance", {"value": f"{r:g} + j({x:g})", "unit": "ohm"}, intermediate_values=inter)
    if target == "net_reactance":
        return solved("ac_phasor_solver", "series_ac_net_reactance", q(x, "ohm"), intermediate_values=inter)
    if target == "phase_angle":
        unit = contract.target.unit or "degree"
        return solved("ac_phasor_solver", "series_ac_phase", q(math.degrees(phi) if unit == "degree" else phi, unit), intermediate_values=inter)
    if target == "power_factor":
        return solved("ac_phasor_solver", "series_ac_power_factor", q(r / z if z else 1.0, "dimensionless"), intermediate_values=inter)
    if target == "circuit_character":
        return solved("ac_phasor_solver", "series_ac_character", {"value": character, "unit": "categorical", "interpretation": relation}, intermediate_values=inter)
    if target == "voltage_current_relation":
        return solved("ac_phasor_solver", "series_ac_voltage_current_relation", {"value": relation, "unit": "categorical", "circuit_character": character}, intermediate_values=inter)
    if target in {"current_rms", "active_power"}:
        voltage = source(contract, "voltage_rms", "V")
        if z == 0:
            return unsolved("ac_phasor_solver", "impedance is zero")
        current = voltage / z
        inter["current_rms"] = q(current, "A")
        if target == "current_rms":
            return solved("ac_phasor_solver", "series_ac_current_rms", q(current, "A"), intermediate_values=inter)
        return solved("ac_phasor_solver", "series_ac_active_power", q(voltage * current * (r / z if z else 1.0), "W"), intermediate_values=inter)
    return unsolved("ac_phasor_solver", f"unsupported AC target `{target}`")

