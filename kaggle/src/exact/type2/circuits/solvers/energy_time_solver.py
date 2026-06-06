from __future__ import annotations

from exact.type2.circuits.diagnostics import solved, unsolved
from exact.type2.circuits.schemas import CircuitContract
from exact.type2.circuits.solvers._helpers import known, q


def solve(contract: CircuitContract) -> dict:
    target_unit = contract.target.unit or "J"
    if "power" in contract.knowns:
        if target_unit in {"Wh", "kWh"}:
            energy_wh = known(contract, "power", "W") * known(contract, "time", "h")
            value = energy_wh / 1000 if target_unit == "kWh" else energy_wh
        else:
            value = known(contract, "power", "W") * known(contract, "time", "s")
    elif "voltage" in contract.knowns and "current" in contract.knowns:
        joules = known(contract, "voltage", "V") * known(contract, "current", "A") * known(contract, "time", "s")
        value = joules / 3.6e6 if target_unit == "kWh" else joules / 3600 if target_unit == "Wh" else joules
    else:
        return unsolved("energy_time_solver", "energy target needs power or voltage/current")
    return solved("energy_time_solver", "electrical_energy_time_conversion", q(value, target_unit), intermediate_values={"target_unit": target_unit})

