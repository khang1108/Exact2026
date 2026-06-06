from __future__ import annotations

import math
from typing import Any


def apply_rounding(result: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    mode = policy.get("mode", "none")
    if mode == "none":
        return result
    rounded = dict(result)
    decimal_places = policy.get("decimal_places")
    percentage_places = policy.get("percentage_decimal_places")
    for key, item in list(rounded.items()):
        if not isinstance(item, dict) or "value" not in item:
            continue
        places = percentage_places if key == "percentage_error" and percentage_places is not None else decimal_places
        if places is not None and isinstance(item["value"], (int, float)):
            rounded[key] = {**item, "value": round(float(item["value"]), int(places))}
    if mode == "school_physics_default" and "absolute_error" in rounded and "value" in rounded:
        err = rounded["absolute_error"]["value"]
        if isinstance(err, (int, float)) and err != 0:
            err_rounded = _round_sig(float(err), int(policy.get("uncertainty_significant_figures", 1)))
            rounded["absolute_error"] = {**rounded["absolute_error"], "value": err_rounded}
            if policy.get("value_same_decimal_place_as_uncertainty"):
                places = max(0, -math.floor(math.log10(abs(err_rounded)))) if abs(err_rounded) < 1 else 0
                rounded["value"] = {**rounded["value"], "value": round(float(rounded["value"]["value"]), places)}
    if "result_with_uncertainty" in rounded and "value" in rounded and "absolute_error" in rounded:
        value = rounded["value"]["value"]
        err = rounded["absolute_error"]["value"]
        unit = rounded["value"].get("unit")
        rounded["result_with_uncertainty"] = {
            "text": f"{value} ± {err} {unit}",
            "value": value,
            "absolute_error": err,
            "unit": unit,
        }
    return rounded


def _round_sig(value: float, sig: int) -> float:
    if value == 0:
        return 0.0
    return round(value, sig - int(math.floor(math.log10(abs(value)))) - 1)

