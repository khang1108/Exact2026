from __future__ import annotations

from exact.type2.electromagnetism.schemas import ElectromagnetismContract, EMQuantityValue


def value(item: EMQuantityValue | str | bool | None) -> float:
    if not isinstance(item, EMQuantityValue):
        raise KeyError("quantity value is missing")
    return float(item.value)


def comp(contract: ElectromagnetismContract, kind: str, prop: str) -> float:
    for component in contract.components:
        if component.kind == kind and prop in component.properties:
            return value(component.properties[prop])
    raise KeyError(f"{kind}.{prop}")


def source(contract: ElectromagnetismContract, name: str) -> float:
    return value(contract.source.get(name))


def known(contract: ElectromagnetismContract, name: str) -> float:
    return value(contract.knowns.get(name))


def geom(contract: ElectromagnetismContract, name: str) -> float:
    return value(contract.geometry.get(name))


def state(contract: ElectromagnetismContract, name: str) -> float:
    return value(contract.state.get(name))


def flux(contract: ElectromagnetismContract, name: str) -> float:
    return value(contract.flux_change.get(name))


def side(data: dict, name: str) -> float:
    return value(data.get(name))


def q(value_: float, unit: str) -> dict:
    return {"value": value_, "unit": unit}
