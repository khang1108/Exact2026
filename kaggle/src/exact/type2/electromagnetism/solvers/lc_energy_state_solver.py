from __future__ import annotations

import math

from exact.type2.electromagnetism.diagnostics import solved, unsolved
from exact.type2.electromagnetism.equation_graph import lc_omega
from exact.type2.electromagnetism.schemas import ElectromagnetismContract
from exact.type2.electromagnetism.solvers._helpers import comp, q, state


def solve(contract: ElectromagnetismContract) -> dict:
    l_h = comp(contract, "inductor", "inductance")
    c_f = comp(contract, "capacitor", "capacitance")
    omega = lc_omega(l_h, c_f)
    target = contract.target.quantity
    total_energy = _total_energy(contract, c_f)
    qmax = math.sqrt(2 * c_f * total_energy) if total_energy is not None else None
    imax = math.sqrt(2 * total_energy / l_h) if total_energy is not None else None
    inter = {"omega": q(omega, "rad/s"), "period": q(2 * math.pi / omega, "s")}
    if total_energy is not None:
        inter.update({"total_energy": q(total_energy, "J"), "Qmax": q(qmax, "C"), "Imax": q(imax, "A")})
    if target == "angular_frequency":
        return solved("lc_energy_state_solver", "lc_angular_frequency", q(omega, "rad/s"), intermediate_values=inter)
    if target == "frequency":
        return solved("lc_energy_state_solver", "lc_frequency", q(omega / (2 * math.pi), "Hz"), intermediate_values=inter)
    if target == "period":
        return solved("lc_energy_state_solver", "lc_period", q(2 * math.pi / omega, "s"), intermediate_values=inter)
    if target == "total_energy" and total_energy is not None:
        return solved("lc_energy_state_solver", "lc_total_energy", q(total_energy, "J"), intermediate_values=inter)
    if target == "maximum_current" and imax is not None:
        return solved("lc_energy_state_solver", "lc_energy_to_max_current", q(imax, "A"), intermediate_values=inter)
    if target == "maximum_charge" and qmax is not None:
        return solved("lc_energy_state_solver", "lc_energy_to_max_charge", q(qmax, "C"), intermediate_values=inter)
    if target in {"instantaneous_current", "instantaneous_charge", "capacitor_energy", "inductor_energy"}:
        if qmax is None:
            return unsolved("lc_energy_state_solver", "LC instantaneous/state target needs total energy or initial charge/voltage")
        t = state(contract, "time")
        charge = qmax * math.cos(omega * t)
        current = -omega * qmax * math.sin(omega * t)
        cap_e = charge**2 / (2 * c_f)
        ind_e = 0.5 * l_h * current**2
        inter.update({"q(t)": q(charge, "C"), "i(t)": q(current, "A"), "capacitor_energy": q(cap_e, "J"), "inductor_energy": q(ind_e, "J")})
        if target == "instantaneous_charge":
            return solved("lc_energy_state_solver", "lc_charge_time_state", q(charge, "C"), intermediate_values=inter)
        if target == "instantaneous_current":
            return solved("lc_energy_state_solver", "lc_current_time_state", q(current, "A"), intermediate_values=inter)
        if target == "capacitor_energy":
            return solved("lc_energy_state_solver", "lc_capacitor_energy_state", q(cap_e, "J"), intermediate_values=inter)
        return solved("lc_energy_state_solver", "lc_inductor_energy_state", q(ind_e, "J"), intermediate_values=inter)
    if target in {"energy_location", "phase_state"}:
        phase = contract.state.get("phase_fraction")
        frac = phase.value if hasattr(phase, "value") else 0.0
        label = _phase_label(float(frac) % 1.0)
        return solved("lc_energy_state_solver", "lc_phase_state_model", {"value": label, "unit": "conceptual"}, intermediate_values=inter)
    return unsolved("lc_energy_state_solver", f"unsupported LC target `{target}`", fallback_recommended=False)


def _total_energy(contract: ElectromagnetismContract, c_f: float) -> float | None:
    if "total_energy" in contract.state:
        return state(contract, "total_energy")
    if "capacitor_voltage" in contract.state:
        return 0.5 * c_f * state(contract, "capacitor_voltage") ** 2
    if "maximum_charge" in contract.state:
        return state(contract, "maximum_charge") ** 2 / (2 * c_f)
    return None


def _phase_label(frac: float) -> str:
    if math.isclose(frac, 0.0, abs_tol=1e-6):
        return "capacitor energy maximum; inductor energy zero; current zero"
    if math.isclose(frac, 0.25, abs_tol=1e-6):
        return "all energy is magnetic in the inductor; current magnitude is maximum"
    if math.isclose(frac, 0.5, abs_tol=1e-6):
        return "capacitor charged oppositely; current zero"
    if math.isclose(frac, 0.75, abs_tol=1e-6):
        return "all energy is magnetic in the inductor; current maximum in the opposite direction"
    return "mixed electric and magnetic energy state"
