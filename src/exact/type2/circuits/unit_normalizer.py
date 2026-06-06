from __future__ import annotations

from exact.type2.circuits.schemas import CircuitQuantity


def scalar(quantity: CircuitQuantity | str | bool | None, unit: str | None = None) -> float:
    if not isinstance(quantity, CircuitQuantity):
        raise KeyError("quantity is missing")
    value = float(quantity.value)
    if unit is None or quantity.unit == unit:
        return value
    src = _canonical_unit(quantity.unit)
    unit = _canonical_unit(unit)
    if src == unit:
        return value
    if unit == "s" and src in {"min", "minute", "minutes"}:
        return value * 60
    if unit == "s" and src in {"h", "hour", "hours"}:
        return value * 3600
    if unit == "h" and src in {"min", "minute", "minutes"}:
        return value / 60
    if unit == "h" and src == "s":
        return value / 3600
    if unit == "h" and src in {"day", "days"}:
        return value * 24
    if unit == "W" and src == "kW":
        return value * 1000
    if unit == "kW" and src == "W":
        return value / 1000
    if unit == "J" and src == "Wh":
        return value * 3600
    if unit == "J" and src == "kWh":
        return value * 3.6e6
    if unit == "Wh" and src == "J":
        return value / 3600
    if unit == "kWh" and src == "J":
        return value / 3.6e6
    raise ValueError(f"cannot normalize {src} to {unit}")


def _canonical_unit(unit: str | None) -> str | None:
    aliases = {
        "volt": "V",
        "ampere": "A",
        "second": "s",
        "hour": "h",
        "minute": "min",
        "watt": "W",
        "kilowatt": "kW",
        "ohm": "ohm",
        "farad": "F",
        "henry": "H",
        "hertz": "Hz",
    }
    return aliases.get(unit or "", unit)


def q(value: float, unit: str) -> dict:
    return {"value": value, "unit": unit}
