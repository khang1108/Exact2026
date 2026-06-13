from __future__ import annotations

from exact.type2.circuits.diagnostics import solved, unsolved
from exact.type2.circuits.schemas import CircuitContract
from exact.type2.circuits.solvers._helpers import q, resistance, source


def solve(contract: CircuitContract) -> dict:
    component = contract.components[0]
    r = resistance(component)
    voltage = source(contract, "voltage", "V") if "voltage" in contract.source else None
    current = source(contract, "current", "A") if "current" in contract.source else None
    target = contract.target.quantity
    if target in {"current", "component_current", "total_current"} and voltage is not None:
        return solved("scalar_ohm_solver", "single_resistor_ohm_current", q(voltage / r, "A"))
    if target in {"voltage", "component_voltage", "total_voltage"} and current is not None:
        return solved("scalar_ohm_solver", "single_resistor_ohm_voltage", q(current * r, "V"))
    if target in {"resistance", "equivalent_resistance"}:
        return solved("scalar_ohm_solver", "single_resistor_resistance", q(r, "ohm"))
    if target in {"power", "component_power", "total_power"}:
        if voltage is None and current is None:
            return unsolved("scalar_ohm_solver", "power needs voltage or current")
        power = voltage**2 / r if voltage is not None else current**2 * r
        return solved("scalar_ohm_solver", "single_resistor_power", q(power, "W"))
    return unsolved("scalar_ohm_solver", f"unsupported scalar target `{target}`")

