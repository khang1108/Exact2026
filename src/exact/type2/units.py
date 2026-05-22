from __future__ import annotations

import pint


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def parse_quantity(value: float, unit: str) -> pint.Quantity:
    return Q_(value, unit)
