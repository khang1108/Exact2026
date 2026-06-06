from __future__ import annotations

from exact.type2.measurement.schemas import MeasuredQuantity


def absolute_uncertainty(quantity: MeasuredQuantity) -> float:
    if quantity.absolute_uncertainty is not None:
        return float(quantity.absolute_uncertainty.value)
    if quantity.relative_uncertainty is not None:
        return abs(quantity.value) * float(quantity.relative_uncertainty)
    if quantity.percentage_uncertainty is not None:
        return abs(quantity.value) * float(quantity.percentage_uncertainty) / 100
    raise KeyError(f"uncertainty missing for {quantity.symbol}")


def relative_uncertainty(quantity: MeasuredQuantity, denominator_policy: str = "measured_value") -> float:
    denominator = abs(quantity.value)
    if denominator_policy != "measured_value":
        denominator = abs(quantity.value)
    if denominator == 0:
        raise ZeroDivisionError("relative uncertainty denominator is zero")
    return absolute_uncertainty(quantity) / denominator

