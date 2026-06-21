from __future__ import annotations

import re

from exact.type2.solving.units import parse_quantity


TARGET_UNIT_MAP: dict[str, str] = {
    "acceleration": "m/s^2",
    "capacitance": "F",
    "charge": "C",
    "current": "A",
    "density": "kg/m^3",
    "distance": "m",
    "electric_field": "N/C",
    "electric_force": "N",
    "electric_potential": "V",
    "energy": "J",
    "force": "N",
    "frequency": "Hz",
    "heat": "J",
    "length": "m",
    "magnetic_field": "T",
    "mass": "kg",
    "potential": "V",
    "potential_energy": "J",
    "power": "W",
    "pressure": "Pa",
    "resistance": "ohm",
    "speed": "m/s",
    "stored_energy": "J",
    "temperature": "K",
    "time": "s",
    "voltage": "V",
    "volume": "m^3",
    "work": "J",
}


ALIASES: dict[str, str] = {
    "electrical_energy": "energy",
    "electric_force": "force",
    "electric_potential": "potential",
    "field_strength": "electric_field",
    "stored_energy": "energy",
}


def canonical_dimension(name: str | None) -> str:
    normalized = (name or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    normalized = re.sub(r"_\d+$", "", normalized)
    return ALIASES.get(normalized, normalized)


def expected_unit_for(target: str | None) -> str | None:
    return TARGET_UNIT_MAP.get(canonical_dimension(target))


def units_compatible(actual_unit: str | None, expected_unit: str | None) -> bool:
    if expected_unit is None:
        return True
    if actual_unit is None:
        return False
    if actual_unit == expected_unit:
        return True
    compatible_sets = (
        {"N/C", "V/m"},
        {"J", "N*m"},
        {"Pa", "N/m^2"},
    )
    if any(actual_unit in group and expected_unit in group for group in compatible_sets):
        return True
    try:
        parse_quantity(1.0, actual_unit).to(expected_unit)
        return True
    except Exception:
        return False
