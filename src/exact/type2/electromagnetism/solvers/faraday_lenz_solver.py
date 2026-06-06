from __future__ import annotations

from exact.type2.electromagnetism.diagnostics import partial, solved, unsolved
from exact.type2.electromagnetism.schemas import ElectromagnetismContract
from exact.type2.electromagnetism.solvers._helpers import flux, q, value


def solve(contract: ElectromagnetismContract) -> dict:
    turns = value(contract.coil.get("turn_count")) if "turn_count" in contract.coil else 1.0
    initial = flux(contract, "initial_flux")
    final = flux(contract, "final_flux")
    dt = flux(contract, "time_interval")
    rate = (final - initial) / dt
    magnitude = turns * abs(rate)
    target = contract.target.quantity
    inter = {"flux_change_rate": q(rate, "Wb/s"), "turn_count": q(turns, "dimensionless")}
    if target == "flux_change_rate":
        return solved("faraday_lenz_solver", "flux_change_rate", q(rate, "Wb/s"), intermediate_values=inter)
    if target == "induced_emf_magnitude":
        return solved("faraday_lenz_solver", "faraday_emf_magnitude", q(magnitude, "V"), intermediate_values=inter)
    direction = _direction(contract, rate)
    if target == "induced_emf":
        if direction is None:
            return partial(
                "faraday_lenz_solver",
                "orientation convention missing",
                {"magnitude": magnitude, "unit": "V", "direction": None},
                missing=["flux_change.direction", "convention.positive_emf_direction"],
                intermediate_values=inter,
            )
        return solved("faraday_lenz_solver", "faraday_lenz_emf", {"magnitude": magnitude, "unit": "V", "direction": direction}, intermediate_values=inter)
    if target == "induced_current_direction":
        if direction is None:
            return unsolved(
                "faraday_lenz_solver",
                "target asks for induced current direction but flux direction/convention is missing",
                missing=["flux_change.direction", "convention.positive_emf_direction"],
            )
        return solved("faraday_lenz_solver", "lenz_current_direction", {"value": direction, "unit": "categorical"}, intermediate_values=inter)
    return unsolved("faraday_lenz_solver", f"unsupported induction target `{target}`", fallback_recommended=False)


def _direction(contract: ElectromagnetismContract, rate: float) -> str | None:
    flux_dir = contract.flux_change.get("direction")
    positive = contract.convention.get("positive_emf_direction")
    if not isinstance(flux_dir, str) or not isinstance(positive, str):
        return None
    outward = flux_dir == "out_of_page"
    induced_outward = (rate < 0 and outward) or (rate > 0 and not outward)
    ccw_is_outward = positive == "counterclockwise"
    if induced_outward:
        return "counterclockwise" if ccw_is_outward else "clockwise"
    return "clockwise" if ccw_is_outward else "counterclockwise"
