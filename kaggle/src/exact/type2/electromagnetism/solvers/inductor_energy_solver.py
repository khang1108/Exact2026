from __future__ import annotations

import math

from exact.type2.electromagnetism.diagnostics import solved, unsolved
from exact.type2.electromagnetism.schemas import ElectromagnetismContract
from exact.type2.electromagnetism.solvers._helpers import comp, known, q


def solve(contract: ElectromagnetismContract) -> dict:
    target = contract.target.quantity
    inductance = comp(contract, "inductor", "inductance") if target != "inductance" else None
    current = known(contract, "current") if target != "current" else None
    energy = known(contract, "inductor_energy") if target != "inductor_energy" else None

    if target == "inductor_energy" and inductance is not None and current is not None:
        value = 0.5 * inductance * current**2
        return solved(
            "inductor_energy_solver",
            "single_inductor_energy_from_l_i",
            q(value, "J"),
            intermediate_values={"L": q(inductance, "H"), "I": q(current, "A")},
        )
    if target == "current" and inductance is not None and energy is not None:
        if inductance <= 0 or energy < 0:
            return unsolved("inductor_energy_solver", "current target requires L > 0 and W >= 0")
        value = math.sqrt(2 * energy / inductance)
        return solved(
            "inductor_energy_solver",
            "single_inductor_current_from_energy_l",
            q(value, "A"),
            intermediate_values={"W": q(energy, "J"), "L": q(inductance, "H")},
        )
    if target == "inductance" and current is not None and energy is not None:
        if current == 0:
            return unsolved("inductor_energy_solver", "inductance target requires nonzero current")
        value = 2 * energy / (current**2)
        return solved(
            "inductor_energy_solver",
            "single_inductor_inductance_from_energy_i",
            q(value, "H"),
            intermediate_values={"W": q(energy, "J"), "I": q(current, "A")},
        )
    return unsolved("inductor_energy_solver", f"insufficient inductor-energy data for `{target}`")
