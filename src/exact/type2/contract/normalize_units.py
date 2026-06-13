from __future__ import annotations

import pint

from exact.type2.solving.units import parse_quantity


def normalize_charge(value: float, unit: str) -> pint.Quantity:
    quantity = parse_quantity(value, unit)
    return quantity.to("C")


def normalize_distance(value: float, unit: str) -> pint.Quantity:
    quantity = parse_quantity(value, unit)
    return quantity.to("m")

