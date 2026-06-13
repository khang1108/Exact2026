from __future__ import annotations

from exact.type2.measurement.schemas import MeasurementQuantity, MeasuredQuantity, Uncertainty


ALIASES = {
    "centimeter": "cm",
    "millimeter": "mm",
    "meter": "m",
    "second": "s",
    "minute": "min",
    "hour": "h",
    "volt": "V",
    "ampere": "A",
    "ohm": "ohm",
    "watt": "W",
}


def canonical_unit(unit: str | None) -> str | None:
    return ALIASES.get(unit or "", unit)


def convert_value(value: float, source_unit: str, target_unit: str) -> float:
    src = canonical_unit(source_unit)
    dst = canonical_unit(target_unit)
    if src == dst:
        return value
    scale = {
        ("mm", "cm"): 0.1,
        ("cm", "mm"): 10.0,
        ("cm", "m"): 0.01,
        ("m", "cm"): 100.0,
        ("mm", "m"): 0.001,
        ("m", "mm"): 1000.0,
        ("min", "s"): 60.0,
        ("s", "min"): 1 / 60,
        ("h", "s"): 3600.0,
        ("s", "h"): 1 / 3600,
    }.get((src, dst))
    if scale is None:
        raise ValueError(f"cannot convert {source_unit} to {target_unit}")
    return value * scale


def normalize_quantity(quantity: MeasurementQuantity, unit: str) -> MeasurementQuantity:
    return MeasurementQuantity(convert_value(quantity.value, quantity.unit, unit), canonical_unit(unit) or unit, quantity.original)


def normalize_uncertainty(uncertainty: Uncertainty, unit: str) -> Uncertainty:
    return Uncertainty(convert_value(uncertainty.value, uncertainty.unit, unit), canonical_unit(unit) or unit, uncertainty.source)


def q(value: float, unit: str) -> dict:
    return {"value": value, "unit": unit}

