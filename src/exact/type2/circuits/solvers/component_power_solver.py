from __future__ import annotations

from exact.type2.circuits.schemas import CircuitContract
from exact.type2.circuits.solvers.dc_resistor_network_solver import solve as solve_network


def solve(contract: CircuitContract) -> dict:
    return solve_network(contract)

