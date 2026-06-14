from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from exact.config import Settings
from exact.type2.domains.thcb.schemas import ThcbContract, ThcbQuantity
from exact.type2.extraction.extractor import normalize_question
from exact.type2.extraction.llm_structured import build_llm_json_client


class ThcbQuantitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    unit: str = ""


class ThcbContractSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str = "UNKNOWN"
    target: str = "unknown"
    quantities: list[ThcbQuantitySpec] = Field(default_factory=list)
    readings: list[ThcbQuantitySpec] = Field(default_factory=list)
    relation: str | None = None
    requested_outputs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def extract_thcb_with_llm(question: str, settings: Settings | None = None) -> ThcbContract | None:
    client = build_llm_json_client(settings)
    if client is None:
        return None
    raw = client.complete_json_sync(
        messages=_build_thcb_contract_messages(question),
        temperature=(settings.llm_temperature if settings else 0.0),
        max_tokens=(settings.type2_extraction_max_tokens if settings else 768),
    )
    return _spec_to_contract(ThcbContractSpec.model_validate(raw))


def extract_thcb_heuristic(question: str) -> ThcbContract:
    text = normalize_question(question)
    lower = text.lower()
    family = _classify_family(lower)
    target = _classify_target(lower)
    quantities = _extract_named_quantities(text, lower)
    readings = _extract_readings(text, lower)
    requested = _requested_outputs(lower)
    relation = _classify_relation(lower)
    return ThcbContract(
        family=family,
        target=target,
        quantities=quantities,
        readings=readings,
        relation=relation,
        requested_outputs=requested,
        notes=["heuristic_thcb_contract"],
    )


def _build_thcb_contract_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Parse a THCB physics question into a strict JSON contract. Return one JSON object only. "
                "family must be MEASUREMENT_ERROR, ERROR_PROPAGATION, PARALLEL_CIRCUIT, SIMPLE_CIRCUIT, "
                "CONCEPTUAL_CIRCUIT, or UNKNOWN. Do not solve. "
                "target must be strictly one of: absolute_error, relative_error, percentage_error, "
                "power_absolute_error, power_relative_error, resistance_absolute_error, "
                "resistance_relative_error, voltage_absolute_error, current_absolute_error, "
                "random_error, maximum_possible, absolute_and_relative_error, equivalent_resistance, "
                "power, branch_currents, total_current, branch_power, current, conceptual. "
                "Pay close attention to what the question asks (e.g. absolute error in voltage vs power, or relative vs absolute). "
                "Use quantity names such as measured, actual, least_count, absolute_error, voltage, current, "
                "current_1, current_2, current_3, resistance_1, resistance_2, power_1, power_2, total_current, total_power. "
                "Use readings for repeated measurements. requested_outputs lists requested result names."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "Return shape: {\"family\":\"MEASUREMENT_ERROR\",\"target\":\"relative_error\","
                "\"quantities\":[{\"name\":\"measured\",\"value\":5.0,\"unit\":\"cm\"}],"
                "\"readings\":[],\"relation\":null,"
                "\"requested_outputs\":[\"relative_error\"],\"notes\":[]}"
            ),
        },
    ]


def _spec_to_contract(spec: ThcbContractSpec) -> ThcbContract:
    quantities: dict[str, ThcbQuantity] = {}
    for item in spec.quantities:
        key = item.name.strip().lower().replace(" ", "_")
        if not key:
            continue
        if key in quantities:
            suffix = 2
            while f"{key}_{suffix}" in quantities:
                suffix += 1
            key = f"{key}_{suffix}"
        quantities[key] = ThcbQuantity(key, item.value, item.unit)
    readings = [
        ThcbQuantity(item.name.strip() or "reading", item.value, item.unit)
        for item in spec.readings
    ]
    return ThcbContract(
        family=(spec.family or "").strip().upper() or "UNKNOWN",
        target=spec.target.strip().lower().replace(" ", "_") or "unknown",
        quantities=quantities,
        readings=readings,
        relation=spec.relation,
        requested_outputs=[item.strip().lower().replace(" ", "_") for item in spec.requested_outputs if item.strip()],
        notes=spec.notes,
    )


def _classify_family(lower: str) -> str:
    if "relative error in the power" in lower or "absolute error of the power" in lower:
        return "ERROR_PROPAGATION"
    if "absolute error of r" in lower or "absolute error of resistance" in lower:
        return "ERROR_PROPAGATION"
    if any(term in lower for term in ("least count", "uncertainty", "true value", "actual", "measured value", "measured result")):
        return "MEASUREMENT_ERROR"
    if "measure" in lower and any(term in lower for term in ("three", "readings", "obtains", "yielding", "taken")):
        return "MEASUREMENT_ERROR"
    if "parallel" in lower:
        return "PARALLEL_CIRCUIT"
    if any(term in lower for term in ("light bulb", "lamp", "resistor", "voltage source", "power consumption")):
        return "SIMPLE_CIRCUIT"
    return "UNKNOWN"


def _classify_target(lower: str) -> str:
    if "power" in lower and ("relative error" in lower or "relative uncertainty" in lower):
        return "power_relative_error"
    if "power" in lower and "absolute error" in lower:
        return "power_absolute_error"
    if "resistance" in lower and ("relative error" in lower or "relative uncertainty" in lower):
        return "resistance_relative_error"
    if "resistance" in lower and "absolute error" in lower:
        return "resistance_absolute_error"
    if ("voltage" in lower or "potential" in lower) and "absolute error" in lower:
        return "voltage_absolute_error"
    if "current" in lower and "absolute error" in lower:
        return "current_absolute_error"
    if "mean absolute error" in lower or "average absolute error" in lower or "random error" in lower:
        return "mean_and_mean_absolute_error" if "mean" in lower or "average" in lower else "random_error"
    if "maximum possible" in lower:
        return "maximum_possible"
    if "absolute error" in lower and "relative" in lower:
        return "absolute_and_relative_error"
    if "relative error" in lower or "relative uncertainty" in lower:
        return "relative_error"
    if "absolute error" in lower:
        return "absolute_error"
    if "total resistance" in lower or "equivalent resistance" in lower:
        return "equivalent_resistance"
    if "total power" in lower or "power consumption" in lower:
        return "power"
    if "current through each" in lower or "current flowing through each" in lower:
        return "branch_currents"
    if "total current" in lower or "flowing through the circuit" in lower:
        return "total_current"
    if "third branch" in lower:
        return "branch_current"
    if "power of each" in lower:
        return "branch_power"
    if "what happens" in lower or "how will" in lower or "how bright" in lower:
        return "conceptual"
    if "current" in lower:
        return "current"
    return "unknown"


def _requested_outputs(lower: str) -> list[str]:
    outputs: list[str] = []
    if "absolute error" in lower:
        outputs.append("absolute_error")
    if "relative error" in lower or "relative uncertainty" in lower:
        outputs.append("relative_error")
    if "mean value" in lower or "average" in lower or "mean " in lower:
        outputs.append("mean")
    if "mean absolute error" in lower or "average absolute error" in lower or "random error" in lower:
        outputs.append("mean_absolute_error")
    if "total current" in lower:
        outputs.append("total_current")
    if "current through each" in lower or "current flowing through each" in lower:
        outputs.append("branch_currents")
    return outputs


def _classify_relation(lower: str) -> str | None:
    if "resistance" in lower and "decreases" in lower and "current" in lower:
        return "resistance_down_current_up"
    if "total current increases" in lower or "current through one lamp" in lower and "increases" in lower:
        return "current_up_brightness_up"
    if "lower resistance" in lower and "bright" in lower:
        return "lower_resistance_brighter"
    return None


def _extract_named_quantities(text: str, lower: str) -> dict[str, ThcbQuantity]:
    quantities: dict[str, ThcbQuantity] = {}
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_₁₂₃]*)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%°/ΩΩ^0-9]+)?", text):
        key = _name_from_symbol(match.group(1))
        _add_quantity(quantities, key, float(match.group(2)), _clean_unit(match.group(3) or ""))
    for match in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*(?:±|\+-|\+/-)\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%°/ΩΩ^0-9]+)", text):
        value, err, unit = float(match.group(1)), float(match.group(2)), _clean_unit(match.group(3))
        key = _name_from_window(lower[max(0, match.start() - 45):match.start()])
        if key is None:
            prefix = text[max(0, match.start() - 8):match.start()]
            symbol_match = re.search(r"([A-Za-z])\s*=\s*$", prefix)
            if symbol_match:
                key = _name_from_symbol(symbol_match.group(1))
        _add_quantity(quantities, key or "measured", value, unit)
        _add_quantity(quantities, f"{key}_error" if key else "absolute_error", err, unit)
    for label, pattern in (
        ("least_count", r"least count(?: of)?\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%°/ΩΩ^0-9]+)"),
        ("actual", r"(?:actual|true) [A-Za-z ]*?(?:is|value is)\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%°/ΩΩ^0-9]+)"),
        ("measured", r"(?:measured|reads|reading is|measured it as|measured as)\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%°/ΩΩ^0-9]+)"),
        ("absolute_error", r"absolute error is\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%°/ΩΩ^0-9]+)"),
        ("total_current", r"total current (?:of|is)\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%°/ΩΩ^0-9]+)"),
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            _add_quantity(quantities, label, float(match.group(1)), _clean_unit(match.group(2)))
    _extract_context_values(text, lower, quantities)
    _duplicate_identical_branch_values(lower, quantities)
    return quantities


def _extract_context_values(text: str, lower: str, quantities: dict[str, ThcbQuantity]) -> None:
    for match in re.finditer(r"(?:d|lamp|branch|resistor)\s*([123₁₂₃])\D{0,35}?(?:is|draws|through)?\s*([-+]?\d+(?:\.\d+)?)\s*a\b", text, flags=re.IGNORECASE):
        _add_quantity(quantities, f"current_{_digit(match.group(1))}", float(match.group(2)), "A")
    for match in re.finditer(r"(?:r|resistance|resistor|lamp)\s*([123₁₂₃])?\D{0,20}?(?:=|of|has a resistance of|with a resistance of)\s*([-+]?\d+(?:\.\d+)?)\s*Ω", text, flags=re.IGNORECASE):
        index = _digit(match.group(1)) if match.group(1) else None
        _add_quantity(quantities, f"resistance_{index}" if index else "resistance", float(match.group(2)), "Ω")
    if "parallel" in lower:
        for match in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*Ω", text, flags=re.IGNORECASE):
            if not _covered_by_quantity(quantities, float(match.group(1)), "Ω"):
                _add_quantity(quantities, "resistance", float(match.group(1)), "Ω")
    for match in re.finditer(r"(?:voltage of|source of|supply voltage|applied|u\s*=)\s*([-+]?\d+(?:\.\d+)?)\s*v\b", text, flags=re.IGNORECASE):
        _add_quantity(quantities, "voltage", float(match.group(1)), "V")
    for match in re.finditer(r"(?:total of|total power|consume a total of)\s*([-+]?\d+(?:\.\d+)?)\s*w\b", text, flags=re.IGNORECASE):
        _add_quantity(quantities, "total_power", float(match.group(1)), "W")
    for match in re.finditer(r"(?:power consumption of|power.*?is|consumes)\s*([-+]?\d+(?:\.\d+)?)\s*w\b", text, flags=re.IGNORECASE):
        _add_quantity(quantities, "power", float(match.group(1)), "W")


def _digit(text: str | None) -> str:
    return (text or "").replace("₁", "1").replace("₂", "2").replace("₃", "3")


def _covered_by_quantity(quantities: dict[str, ThcbQuantity], value: float, unit: str) -> bool:
    return any(abs(item.value - value) < 1e-12 and item.unit == unit for item in quantities.values())


def _duplicate_identical_branch_values(lower: str, quantities: dict[str, ThcbQuantity]) -> None:
    if not any(term in lower for term in ("two lamps", "two identical", "each lamp", "both lamps")):
        return
    resistances = [key for key in quantities if key.startswith("resistance")]
    if len(resistances) == 1:
        item = quantities[resistances[0]]
        _add_quantity(quantities, "resistance", item.value, item.unit)


def _extract_readings(text: str, lower: str) -> list[ThcbQuantity]:
    if not any(term in lower for term in ("three", "readings", "obtains", "yielding", "taken")):
        return []
    readings: list[ThcbQuantity] = []
    for match in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*([A-Za-z°ΩΩ]+)", text):
        unit = _clean_unit(match.group(2))
        if unit.lower() in {"a", "v", "cm", "g", "kg", "c", "degc", "°c", "s"}:
            readings.append(ThcbQuantity("reading", float(match.group(1)), unit))
    return readings


def _add_quantity(quantities: dict[str, ThcbQuantity], key: str, value: float, unit: str) -> None:
    key = key.strip().lower().replace(" ", "_") or "value"
    if key in quantities:
        suffix = 2
        while f"{key}_{suffix}" in quantities:
            suffix += 1
        key = f"{key}_{suffix}"
    quantities[key] = ThcbQuantity(key, value, unit)


def _name_from_symbol(symbol: str) -> str:
    normalized = symbol.lower().replace("₁", "_1").replace("₂", "_2").replace("₃", "_3")
    return {
        "u": "voltage",
        "v": "voltage",
        "i": "current",
        "r": "resistance",
        "p": "power",
        "r1": "resistance_1",
        "r_1": "resistance_1",
        "r2": "resistance_2",
        "r_2": "resistance_2",
        "i1": "current_1",
        "i_1": "current_1",
        "i2": "current_2",
        "i_2": "current_2",
        "i3": "current_3",
        "i_3": "current_3",
    }.get(normalized, normalized)


def _name_from_window(window: str) -> str | None:
    if "voltage" in window:
        return "voltage"
    if "current" in window:
        return "current"
    if "resistance" in window:
        return "resistance"
    if "length" in window or "height" in window:
        return "measured"
    if "mass" in window:
        return "measured"
    return None


def _clean_unit(unit: str) -> str:
    unit = unit.strip().rstrip(".,;)")
    return unit.replace("Ω", "Ω").replace("Ohm", "Ω")
