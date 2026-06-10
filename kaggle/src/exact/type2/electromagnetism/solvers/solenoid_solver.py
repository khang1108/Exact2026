from __future__ import annotations

from exact.type2.electromagnetism.diagnostics import solved, unsolved
from exact.type2.electromagnetism.equation_graph import MU0
from exact.type2.electromagnetism.schemas import ElectromagnetismContract
from exact.type2.electromagnetism.solvers._helpers import geom, known, q


def solve(contract: ElectromagnetismContract) -> dict:
    mu_r = known(contract, "relative_permeability") if "relative_permeability" in contract.knowns else 1.0
    n = _turn_density(contract)
    target = contract.target.quantity
    current = known(contract, "current") if "current" in contract.knowns else None
    length = geom(contract, "length") if "length" in contract.geometry else None
    area = geom(contract, "cross_section_area") if "cross_section_area" in contract.geometry else None
    turns = geom(contract, "turn_count") if "turn_count" in contract.geometry else (n * length if length else None)
    b = MU0 * mu_r * n * current if current is not None else None
    inter = {"turn_density": q(n, "1/m"), "relative_permeability": q(mu_r, "dimensionless")}
    if b is not None:
        inter["B"] = q(b, "T")
    if target == "magnetic_field_inside" and b is not None:
        return solved("solenoid_solver", "long_solenoid_field", q(b, "T"), intermediate_values=inter)
    if target == "magnetic_flux_one_turn" and b is not None and area is not None:
        phi = b * area
        return solved("solenoid_solver", "long_solenoid_flux_one_turn", q(phi, "Wb"), intermediate_values={**inter, "Phi": q(phi, "Wb")})
    if target == "flux_linkage" and b is not None and area is not None and turns is not None:
        psi = turns * b * area
        return solved("solenoid_solver", "long_solenoid_flux_linkage", q(psi, "Wb_turn"), intermediate_values={**inter, "Psi": q(psi, "Wb_turn")})
    if target == "inductance" and area is not None and turns is not None and length is not None:
        inductance = MU0 * mu_r * turns**2 * area / length
        return solved("solenoid_solver", "long_solenoid_inductance", q(inductance, "H"), intermediate_values={**inter, "L": q(inductance, "H")})
    if target == "magnetic_energy_density" and b is not None:
        density = b**2 / (2 * MU0 * mu_r)
        return solved("solenoid_solver", "magnetic_energy_density", q(density, "J/m^3"), intermediate_values={**inter, "u_B": q(density, "J/m^3")})
    if target == "magnetic_energy" and area is not None and turns is not None and length is not None and current is not None:
        inductance = MU0 * mu_r * turns**2 * area / length
        energy = 0.5 * inductance * current**2
        return solved("solenoid_solver", "long_solenoid_magnetic_energy", q(energy, "J"), intermediate_values={**inter, "L": q(inductance, "H")})
    return unsolved("solenoid_solver", f"insufficient solenoid data for `{target}`")


def _turn_density(contract: ElectromagnetismContract) -> float:
    if "turn_density" in contract.geometry:
        return geom(contract, "turn_density")
    return geom(contract, "turn_count") / geom(contract, "length")
