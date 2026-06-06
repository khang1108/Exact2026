from __future__ import annotations

from exact.type2.circuits.schemas import CircuitComponent, CircuitContract
from exact.type2.circuits.unit_normalizer import q, scalar


def source(contract: CircuitContract, name: str, unit: str | None = None) -> float:
    return scalar(contract.source.get(name), unit)


def known(contract: CircuitContract, name: str, unit: str | None = None) -> float:
    return scalar(contract.knowns.get(name), unit)


def side(data: dict, name: str) -> float:
    return scalar(data.get(name))


def resistance(component: CircuitComponent) -> float:
    return scalar(component.properties.get("resistance"), "ohm")


def prop(component: CircuitComponent, name: str, unit: str | None = None) -> float:
    return scalar(component.properties.get(name), unit)


__all__ = ["q", "source", "known", "side", "resistance", "prop"]

