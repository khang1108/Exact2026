from __future__ import annotations

from typing import Any

from exact.type2.circuits.diagnostics import solved, unsolved
from exact.type2.circuits.schemas import CircuitComponent, CircuitContract
from exact.type2.circuits.solvers._helpers import q, resistance, source


def solve(contract: CircuitContract) -> dict:
    components = {component.id: component for component in contract.components}
    total_r = _equivalent(contract.topology, components)
    total_v = source(contract, "voltage", "V")
    state = {"total": {}, "components": {}, "branches": {}}
    _propagate(contract.topology, components, total_v, total_v / total_r, state)
    total_power = total_v * total_v / total_r
    state["total"] = {
        "resistance": q(total_r, "ohm"),
        "voltage": q(total_v, "V"),
        "current": q(total_v / total_r, "A"),
        "power": q(total_power, "W"),
    }
    result = _lookup(contract, state)
    if result is None:
        return unsolved("dc_resistor_network_solver", "target could not be resolved from circuit state")
    return solved(
        "dc_resistor_network_solver",
        "recursive_series_parallel_network",
        result,
        resolved_state=state,
        target_lookup={
            "quantity": contract.target.quantity,
            "scope": contract.target.scope,
            "component_id": contract.target.component_id,
            "branch_id": contract.target.branch_id,
        },
    )


def _equivalent(node: dict[str, Any] | str, components: dict[str, CircuitComponent]) -> float:
    if isinstance(node, str):
        return resistance(components[node])
    if node["type"] == "series":
        return sum(_equivalent(item, components) for item in node.get("items", ()))
    branch_rs = [sum(_equivalent(item, components) for item in branch.get("items", ())) for branch in node.get("branches", ())]
    return 1 / sum(1 / r for r in branch_rs)


def _propagate(node: dict[str, Any] | str, components: dict[str, CircuitComponent], voltage: float, current: float, state: dict) -> None:
    if isinstance(node, str):
        r = resistance(components[node])
        state["components"][node] = {
            "resistance": q(r, "ohm"),
            "voltage": q(voltage, "V"),
            "current": q(current, "A"),
            "power": q(voltage * current, "W"),
        }
        return
    if node["type"] == "series":
        for item in node.get("items", ()):
            item_r = _equivalent(item, components)
            _propagate(item, components, current * item_r, current, state)
        return
    for branch in node.get("branches", ()):
        branch_r = sum(_equivalent(item, components) for item in branch.get("items", ()))
        branch_i = voltage / branch_r
        branch_id = branch.get("id")
        if branch_id:
            state["branches"][branch_id] = {"resistance": q(branch_r, "ohm"), "voltage": q(voltage, "V"), "current": q(branch_i, "A")}
        for item in branch.get("items", ()):
            item_r = _equivalent(item, components)
            _propagate(item, components, branch_i * item_r, branch_i, state)


def _lookup(contract: CircuitContract, state: dict) -> dict | None:
    t = contract.target
    quantity = t.quantity
    if quantity in {"equivalent_resistance", "resistance"} and t.scope == "total":
        return state["total"]["resistance"]
    if quantity in {"total_voltage", "voltage"} and t.scope in {"total", "source"}:
        return state["total"]["voltage"]
    if quantity in {"total_current", "current"} and t.scope == "total":
        return state["total"]["current"]
    if quantity in {"total_power", "power"} and t.scope == "total":
        return state["total"]["power"]
    if t.scope == "component" and t.component_id in state["components"]:
        key = quantity.removeprefix("component_")
        return state["components"][t.component_id].get(key)
    if t.scope == "branch" and t.branch_id in state["branches"]:
        key = quantity.removeprefix("branch_")
        return state["branches"][t.branch_id].get(key)
    return None

