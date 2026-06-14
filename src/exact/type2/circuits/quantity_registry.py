from __future__ import annotations


SUPPORTED_SYSTEM_TYPES = {
    "single_resistor",
    "dc_resistor_network",
    "energy_consumption",
    "series_ac_circuit",
    "series_rlc_circuit",
    "ideal_transformer",
}

SUPPORTED_SCOPES = {"total", "component", "branch", "source", "primary", "secondary"}

TARGET_UNITS = {
    "voltage": {"V"},
    "current": {"A"},
    "resistance": {"ohm"},
    "equivalent_resistance": {"ohm"},
    "total_voltage": {"V"},
    "total_current": {"A"},
    "component_voltage": {"V"},
    "component_current": {"A"},
    "branch_voltage": {"V"},
    "branch_current": {"A"},
    "power": {"W"},
    "component_power": {"W"},
    "total_power": {"W"},
    "electrical_energy": {"J", "Wh", "kWh"},
    "joule_heat": {"J", "Wh", "kWh"},
    "impedance": {"ohm"},
    "impedance_magnitude": {"ohm"},
    "complex_impedance": {"ohm"},
    "inductive_reactance": {"ohm"},
    "capacitive_reactance": {"ohm"},
    "net_reactance": {"ohm"},
    "current_rms": {"A"},
    "voltage_rms": {"V"},
    "phase_angle": {"degree", "radian"},
    "power_factor": {"dimensionless"},
    "circuit_character": {"categorical"},
    "voltage_current_relation": {"categorical"},
    "active_power": {"W"},
    "primary_voltage": {"V"},
    "secondary_voltage": {"V"},
    "primary_current": {"A"},
    "secondary_current": {"A"},
    "primary_turns": {"dimensionless"},
    "secondary_turns": {"dimensionless"},
    "turns_ratio": {"dimensionless"},
    "transformer_type": {"categorical"},
}

