from __future__ import annotations

import math


MU0 = 4 * math.pi * 1e-7


def omega_from_frequency(frequency_hz: float) -> float:
    return 2 * math.pi * frequency_hz


def lc_omega(inductance_h: float, capacitance_f: float) -> float:
    return 1 / math.sqrt(inductance_h * capacitance_f)


def rlc_values(resistance_ohm: float, inductance_h: float, capacitance_f: float, frequency_hz: float) -> dict:
    omega = omega_from_frequency(frequency_hz)
    xl = omega * inductance_h
    xc = 1 / (omega * capacitance_f)
    x = xl - xc
    z = math.sqrt(resistance_ohm**2 + x**2)
    phi = math.atan2(x, resistance_ohm)
    return {"omega": omega, "XL": xl, "XC": xc, "X": x, "Z": z, "phi": phi}
