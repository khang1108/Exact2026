from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantitySpec:
    quantity: str
    aliases: tuple[str, ...]
    unit: str
    type: str
    compatible_system_types: tuple[str, ...]
    requires: tuple[str, ...] = ()


SUPPORTED_SYSTEM_TYPES = {
    "inductor_energy",
    "ideal_lc_oscillator",
    "series_rlc_circuit",
    "ac_reactance",
    "long_solenoid",
    "magnetic_flux_system",
    "electromagnetic_induction",
    "ideal_transformer",
}


QUANTITY_REGISTRY: dict[str, QuantitySpec] = {
    "frequency": QuantitySpec("frequency", ("f",), "Hz", "scalar", ("ideal_lc_oscillator",), ("inductance", "capacitance")),
    "angular_frequency": QuantitySpec("angular_frequency", ("omega",), "rad/s", "scalar", ("ideal_lc_oscillator",), ("inductance", "capacitance")),
    "period": QuantitySpec("period", ("T",), "s", "scalar", ("ideal_lc_oscillator",), ("inductance", "capacitance")),
    "total_energy": QuantitySpec("total_energy", ("energy",), "J", "scalar", ("ideal_lc_oscillator",), ()),
    "capacitor_energy": QuantitySpec("capacitor_energy", ("electric energy",), "J", "scalar", ("ideal_lc_oscillator",), ()),
    "inductor_energy": QuantitySpec("inductor_energy", ("magnetic energy",), "J", "scalar", ("ideal_lc_oscillator", "inductor_energy"), ()),
    "maximum_current": QuantitySpec("maximum_current", ("current is greatest", "Imax"), "A", "scalar", ("ideal_lc_oscillator",), ()),
    "maximum_charge": QuantitySpec("maximum_charge", ("Qmax",), "C", "scalar", ("ideal_lc_oscillator",), ()),
    "instantaneous_current": QuantitySpec("instantaneous_current", ("i(t)",), "A", "scalar", ("ideal_lc_oscillator",), ("time",)),
    "instantaneous_charge": QuantitySpec("instantaneous_charge", ("q(t)",), "C", "scalar", ("ideal_lc_oscillator",), ("time",)),
    "energy_location": QuantitySpec("energy_location", ("where energy",), "conceptual", "conceptual", ("ideal_lc_oscillator",), ()),
    "phase_state": QuantitySpec("phase_state", ("state",), "conceptual", "conceptual", ("ideal_lc_oscillator",), ()),
    "inductive_reactance": QuantitySpec("inductive_reactance", ("XL", "coil reactance"), "ohm", "scalar", ("ac_reactance", "series_rlc_circuit"), ("inductance", "frequency")),
    "capacitive_reactance": QuantitySpec("capacitive_reactance", ("XC", "capacitor reactance"), "ohm", "scalar", ("ac_reactance", "series_rlc_circuit"), ("capacitance", "frequency")),
    "net_reactance": QuantitySpec("net_reactance", ("X",), "ohm", "scalar", ("ac_reactance", "series_rlc_circuit"), ("frequency",)),
    "impedance_magnitude": QuantitySpec("impedance_magnitude", ("|Z|",), "ohm", "scalar", ("series_rlc_circuit", "ac_reactance"), ("resistance", "frequency")),
    "complex_impedance": QuantitySpec("complex_impedance", ("Z",), "ohm", "complex", ("series_rlc_circuit",), ("resistance", "frequency")),
    "current_rms": QuantitySpec("current_rms", ("Irms",), "A", "scalar", ("series_rlc_circuit",), ("voltage_rms",)),
    "voltage_rms": QuantitySpec("voltage_rms", ("Urms",), "V", "scalar", ("series_rlc_circuit",), ("current_rms",)),
    "phase_angle": QuantitySpec("phase_angle", ("phi",), "degree", "scalar", ("series_rlc_circuit",), ("resistance", "frequency")),
    "power_factor": QuantitySpec("power_factor", ("cos phi",), "dimensionless", "scalar", ("series_rlc_circuit",), ("resistance", "frequency")),
    "active_power": QuantitySpec("active_power", ("real power",), "W", "scalar", ("series_rlc_circuit",), ("voltage_rms",)),
    "circuit_character": QuantitySpec("circuit_character", ("inductive", "capacitive"), "categorical", "conceptual", ("series_rlc_circuit",), ("frequency",)),
    "voltage_current_relation": QuantitySpec("voltage_current_relation", ("lead", "lag"), "categorical", "conceptual", ("series_rlc_circuit",), ("frequency",)),
    "is_resonant": QuantitySpec("is_resonant", ("resonant",), "boolean", "boolean", ("series_rlc_circuit",), ("inductance", "capacitance", "frequency")),
    "resonance_frequency": QuantitySpec("resonance_frequency", ("natural frequency",), "Hz", "scalar", ("series_rlc_circuit",), ("inductance", "capacitance")),
    "resonance_angular_frequency": QuantitySpec("resonance_angular_frequency", ("omega0",), "rad/s", "scalar", ("series_rlc_circuit",), ("inductance", "capacitance")),
    "resonance_condition": QuantitySpec("resonance_condition", ("XL equals XC",), "conceptual", "conceptual", ("series_rlc_circuit",), ("inductance", "capacitance")),
    "circuit_state_at_resonance": QuantitySpec("circuit_state_at_resonance", ("at resonance",), "conceptual", "conceptual", ("series_rlc_circuit",), ()),
    "magnetic_field_inside": QuantitySpec("magnetic_field_inside", ("B",), "T", "scalar", ("long_solenoid",), ("current",)),
    "magnetic_flux_one_turn": QuantitySpec("magnetic_flux_one_turn", ("flux through one turn",), "Wb", "scalar", ("long_solenoid",), ("current", "cross_section_area")),
    "flux_linkage": QuantitySpec("flux_linkage", ("N Phi",), "Wb_turn", "scalar", ("long_solenoid",), ("current", "cross_section_area")),
    "current": QuantitySpec("current", ("I",), "A", "scalar", ("inductor_energy",), ("inductance", "inductor_energy")),
    "inductance": QuantitySpec("inductance", ("L",), "H", "scalar", ("long_solenoid", "inductor_energy"), ("cross_section_area",)),
    "magnetic_energy_density": QuantitySpec("magnetic_energy_density", ("uB",), "J/m^3", "scalar", ("long_solenoid",), ("current",)),
    "magnetic_energy": QuantitySpec("magnetic_energy", ("W",), "J", "scalar", ("long_solenoid",), ("current",)),
    "induced_emf": QuantitySpec("induced_emf", ("emf",), "V", "scalar", ("electromagnetic_induction",), ("time_interval",)),
    "induced_emf_magnitude": QuantitySpec("induced_emf_magnitude", ("emf magnitude",), "V", "scalar", ("electromagnetic_induction",), ("time_interval",)),
    "induced_current_direction": QuantitySpec("induced_current_direction", ("current direction",), "categorical", "conceptual", ("electromagnetic_induction",), ("direction",)),
    "flux_change_rate": QuantitySpec("flux_change_rate", ("dPhi/dt",), "Wb/s", "scalar", ("electromagnetic_induction",), ("time_interval",)),
    "primary_voltage": QuantitySpec("primary_voltage", ("Vp",), "V", "scalar", ("ideal_transformer",), ()),
    "secondary_voltage": QuantitySpec("secondary_voltage", ("Vs",), "V", "scalar", ("ideal_transformer",), ()),
    "primary_current": QuantitySpec("primary_current", ("Ip",), "A", "scalar", ("ideal_transformer",), ()),
    "secondary_current": QuantitySpec("secondary_current", ("Is",), "A", "scalar", ("ideal_transformer",), ()),
    "primary_turns": QuantitySpec("primary_turns", ("Np",), "dimensionless", "scalar", ("ideal_transformer",), ()),
    "secondary_turns": QuantitySpec("secondary_turns", ("Ns",), "dimensionless", "scalar", ("ideal_transformer",), ()),
    "turns_ratio": QuantitySpec("turns_ratio", ("Np/Ns",), "dimensionless", "scalar", ("ideal_transformer",), ()),
    "transformer_type": QuantitySpec("transformer_type", ("step-up", "step-down"), "categorical", "conceptual", ("ideal_transformer",), ()),
}


def quantity_spec(quantity: str) -> QuantitySpec | None:
    return QUANTITY_REGISTRY.get(quantity)
