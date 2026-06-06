from __future__ import annotations

from exact.type2.measurement.schemas import MeasurementQuantity
from exact.type2.measurement.unit_normalizer import normalize_quantity


def repeated_measurement_stats(measurements: tuple[MeasurementQuantity, ...], definition: str, unit: str | None = None) -> dict:
    target_unit = unit or measurements[0].unit
    values = [normalize_quantity(item, target_unit).value for item in measurements]
    mean = sum(values) / len(values)
    if definition == "mean_absolute_deviation_from_mean":
        error = sum(abs(value - mean) for value in values) / len(values)
    elif definition == "half_range":
        error = (max(values) - min(values)) / 2
    else:
        raise ValueError(f"unsupported mean_error_definition `{definition}`")
    return {"mean": mean, "mean_absolute_error": error, "unit": target_unit}

