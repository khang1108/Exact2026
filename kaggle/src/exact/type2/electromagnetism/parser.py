from __future__ import annotations

import re

from exact.type2.electromagnetism.schemas import (
    ElectromagnetismContract,
    EMComponent,
    EMEvidence,
    EMQuantityValue,
    EMTarget,
)
from exact.type2.schemas import Extraction


def extract_electromagnetism_contract(extraction: Extraction) -> ElectromagnetismContract | None:
    """Parser boundary for electromagnetism solving.

    This may inspect natural-language text. Deterministic solvers receive only
    the returned contract.
    """
    text = extraction.normalized_question
    lower = text.lower()
    if _is_transformer(lower):
        return _transformer_contract(extraction)
    if _is_faraday(lower):
        return _faraday_contract(extraction)
    if _is_solenoid(lower):
        return _solenoid_contract(extraction)
    if _is_rlc(lower):
        return _rlc_contract(extraction)
    if _is_lc(lower):
        return _lc_contract(extraction)
    if _is_reactance(lower):
        return _reactance_contract(extraction)
    return None


def _lc_contract(extraction: Extraction) -> ElectromagnetismContract:
    lower = extraction.normalized_question.lower()
    target = _target(
        lower,
        {
            "maximum_current": ("maximum current", "current is greatest", "greatest current"),
            "maximum_charge": ("maximum charge",),
            "total_energy": ("total energy",),
            "instantaneous_current": ("instantaneous current",),
            "instantaneous_charge": ("instantaneous charge",),
            "energy_location": ("all energy", "energy is magnetic", "energy is electric"),
            "phase_state": ("quarter period", "half period", "phase", "state"),
            "frequency": ("frequency",),
            "period": ("period",),
            "angular_frequency": ("angular frequency", "omega"),
        },
        extraction.target or "maximum_current",
    )
    state = {}
    if "quarter period" in lower or "t/4" in lower:
        state["phase_fraction"] = EMQuantityValue(0.25, "dimensionless")
    if "half period" in lower or "t/2" in lower:
        state["phase_fraction"] = EMQuantityValue(0.5, "dimensionless")
    if "three quarter" in lower or "3t/4" in lower:
        state["phase_fraction"] = EMQuantityValue(0.75, "dimensionless")
    _copy_quantity(extraction, state, "voltage", "capacitor_voltage")
    _copy_quantity(extraction, state, "charge", "maximum_charge")
    _copy_quantity(extraction, state, "energy", "total_energy")
    _copy_quantity(extraction, state, "time", "time")
    return ElectromagnetismContract(
        system_type="ideal_lc_oscillator",
        target=EMTarget(target, _unit_for(target), _output_for(target)),
        components=tuple(_components(extraction, include_resistor=False)),
        state=state,
        assumptions={"ideal_components": "damping" not in lower and "resistance" not in lower, "resistance_ignored": "damping" not in lower},
        parse_confidence=0.72,
        evidence=(EMEvidence("LC circuit", {"system_type": "ideal_lc_oscillator"}),),
        unresolved=("RLC damping present" if "damping" in lower else "",),
    )


def _rlc_contract(extraction: Extraction) -> ElectromagnetismContract:
    lower = extraction.normalized_question.lower()
    if "natural frequency" in lower and not any(s in lower for s in ("is it", "does", "occur", "resonant?")):
        target = "resonance_frequency"
    else:
        target = _target(
            lower,
            {
                "is_resonant": ("is it resonant", "does resonance occur", "resonance occur", "xl equals xc", "xl = xc"),
                "phase_angle": ("phase angle",),
                "power_factor": ("power factor",),
                "current_rms": ("rms current", "current rms"),
                "active_power": ("active power", "real power"),
                "circuit_character": ("capacitive circuit", "inductive character", "circuit character", "inductive circuit"),
                "voltage_current_relation": ("current leads", "current lags", "voltage leads", "voltage lags"),
                "impedance_magnitude": ("impedance",),
                "resonance_frequency": ("resonance frequency", "natural frequency"),
            },
            extraction.target or "impedance_magnitude",
        )
    source = {}
    _copy_quantity(extraction, source, "frequency", "frequency")
    _copy_quantity(extraction, source, "voltage", "voltage_rms")
    return ElectromagnetismContract(
        system_type="series_rlc_circuit",
        target=EMTarget(target, _unit_for(target), _output_for(target)),
        components=tuple(_components(extraction, include_resistor=True)),
        source=source,
        assumptions={"steady_state_ac": True, "rms_values": bool(source.get("voltage_rms"))},
        parse_confidence=0.78,
        evidence=(EMEvidence("series RLC circuit", {"system_type": "series_rlc_circuit"}),),
    )


def _reactance_contract(extraction: Extraction) -> ElectromagnetismContract:
    lower = extraction.normalized_question.lower()
    target = _target(
        lower,
        {
            "inductive_reactance": ("inductive reactance", "reactance of the coil", "coil reactance"),
            "capacitive_reactance": ("capacitive reactance", "reactance of capacitor", "capacitor reactance"),
            "net_reactance": ("net reactance",),
            "impedance_magnitude": ("impedance",),
        },
        extraction.target or "inductive_reactance",
    )
    source = {}
    _copy_quantity(extraction, source, "frequency", "frequency")
    return ElectromagnetismContract(
        system_type="ac_reactance",
        target=EMTarget(target, _unit_for(target), _output_for(target)),
        components=tuple(_components(extraction, include_resistor=True)),
        source=source,
        assumptions={"steady_state_ac": True},
        parse_confidence=0.74,
        evidence=(EMEvidence("AC reactance", {"system_type": "ac_reactance"}),),
    )


def _solenoid_contract(extraction: Extraction) -> ElectromagnetismContract:
    lower = extraction.normalized_question.lower()
    target = _target(
        lower,
        {
            "magnetic_field_inside": ("magnetic field", "field inside"),
            "magnetic_flux_one_turn": ("flux through one turn", "one turn"),
            "flux_linkage": ("flux linkage", "total flux linkage"),
            "inductance": ("inductance", "self-inductance"),
            "magnetic_energy_density": ("energy density",),
            "magnetic_energy": ("magnetic energy", "stored energy"),
        },
        extraction.target or "magnetic_field_inside",
    )
    geometry, knowns = {}, {}
    for src, dst in (("turn_count", "turn_count"), ("turn_density", "turn_density"), ("length", "length"), ("area", "cross_section_area")):
        _copy_quantity(extraction, geometry, src, dst)
    _copy_quantity(extraction, knowns, "current", "current")
    knowns["relative_permeability"] = EMQuantityValue(1.0, "dimensionless")
    return ElectromagnetismContract(
        system_type="long_solenoid",
        target=EMTarget(target, _unit_for(target), _output_for(target)),
        geometry=geometry,
        knowns=knowns,
        assumptions={"long_solenoid_approximation": True, "uniform_field_inside": True, "edge_effects_ignored": True},
        parse_confidence=0.76,
        evidence=(EMEvidence("solenoid", {"system_type": "long_solenoid"}),),
        unresolved=("not a long solenoid" if ("toroid" in lower or "single loop" in lower) else "",),
    )


def _faraday_contract(extraction: Extraction) -> ElectromagnetismContract:
    lower = extraction.normalized_question.lower()
    target = _target(
        lower,
        {
            "induced_emf_magnitude": ("magnitude only", "emf magnitude"),
            "induced_current_direction": ("direction of induced current", "current direction"),
            "flux_change_rate": ("flux change rate",),
            "induced_emf": ("induced emf", "emf"),
        },
        extraction.target or "induced_emf",
    )
    coil, flux_change = {}, {}
    _copy_quantity(extraction, coil, "turn_count", "turn_count")
    _copy_quantity(extraction, flux_change, "flux", "initial_flux")
    _copy_quantity(extraction, flux_change, "flux_2", "final_flux")
    _copy_quantity(extraction, flux_change, "time", "time_interval")
    if "out of page" in lower:
        flux_change["direction"] = "out_of_page"
    elif "into page" in lower or "inward" in lower:
        flux_change["direction"] = "into_page"
    convention = {}
    if "counterclockwise" in lower:
        convention["positive_emf_direction"] = "counterclockwise"
    elif "clockwise" in lower:
        convention["positive_emf_direction"] = "clockwise"
    return ElectromagnetismContract(
        system_type="electromagnetic_induction",
        target=EMTarget(target, _unit_for(target), _output_for(target)),
        coil=coil,
        flux_change=flux_change,
        convention=convention,
        parse_confidence=0.7,
        evidence=(EMEvidence("changing magnetic flux", {"system_type": "electromagnetic_induction"}),),
    )


def _transformer_contract(extraction: Extraction) -> ElectromagnetismContract:
    lower = extraction.normalized_question.lower()
    target = _target(
        lower,
        {
            "secondary_voltage": ("secondary voltage", "output voltage"),
            "primary_voltage": ("primary voltage", "input voltage"),
            "secondary_current": ("secondary current",),
            "primary_current": ("primary current",),
            "turns_ratio": ("turns ratio",),
            "transformer_type": ("step-up", "step up", "step-down", "step down", "transformer type"),
        },
        extraction.target or "secondary_voltage",
    )
    primary, secondary = {}, {}
    for key, dst in (("primary_turns", "turns"), ("primary_voltage", "voltage_rms"), ("primary_current", "current_rms")):
        _copy_quantity(extraction, primary, key, dst)
    for key, dst in (("secondary_turns", "turns"), ("secondary_voltage", "voltage_rms"), ("secondary_current", "current_rms")):
        _copy_quantity(extraction, secondary, key, dst)
    return ElectromagnetismContract(
        system_type="ideal_transformer",
        target=EMTarget(target, _unit_for(target), _output_for(target)),
        primary=primary,
        secondary=secondary,
        assumptions={"ideal_transformer": True, "rms_values": True},
        parse_confidence=0.75,
        evidence=(EMEvidence("transformer", {"system_type": "ideal_transformer"}),),
    )


def _components(extraction: Extraction, *, include_resistor: bool) -> list[EMComponent]:
    components = []
    if "inductance" in extraction.quantities:
        components.append(EMComponent("L1", "inductor", {"inductance": _q(extraction, "inductance")}))
    if "capacitance" in extraction.quantities:
        components.append(EMComponent("C1", "capacitor", {"capacitance": _q(extraction, "capacitance")}))
    if include_resistor and "resistance" in extraction.quantities:
        components.append(EMComponent("R1", "resistor", {"resistance": _q(extraction, "resistance")}))
    return components


def _copy_quantity(extraction: Extraction, target: dict, source: str, dest: str) -> None:
    if source in extraction.quantities:
        target[dest] = _q(extraction, source)


def _q(extraction: Extraction, key: str) -> EMQuantityValue:
    quantity = extraction.quantities[key]
    return EMQuantityValue(float(quantity.value.magnitude), str(quantity.value.units), quantity.evidence)


def _target(lower: str, mapping: dict[str, tuple[str, ...]], fallback: str) -> str:
    normalized = fallback.strip().lower().replace(" ", "_") if fallback else ""
    if normalized in mapping:
        return normalized
    for quantity, markers in mapping.items():
        if any(marker in lower for marker in markers):
            return quantity
    return normalized or next(iter(mapping))


def _unit_for(target: str) -> str | None:
    return {
        "maximum_current": "A",
        "maximum_charge": "C",
        "total_energy": "J",
        "capacitor_energy": "J",
        "inductor_energy": "J",
        "frequency": "Hz",
        "period": "s",
        "angular_frequency": "rad/s",
        "instantaneous_current": "A",
        "instantaneous_charge": "C",
        "inductive_reactance": "ohm",
        "capacitive_reactance": "ohm",
        "net_reactance": "ohm",
        "impedance_magnitude": "ohm",
        "phase_angle": "degree",
        "power_factor": "dimensionless",
        "current_rms": "A",
        "active_power": "W",
        "is_resonant": "boolean",
        "resonance_frequency": "Hz",
        "magnetic_field_inside": "T",
        "magnetic_flux_one_turn": "Wb",
        "flux_linkage": "Wb_turn",
        "inductance": "H",
        "magnetic_energy_density": "J/m^3",
        "magnetic_energy": "J",
        "induced_emf": "V",
        "induced_emf_magnitude": "V",
        "flux_change_rate": "Wb/s",
        "secondary_voltage": "V",
        "primary_voltage": "V",
        "secondary_current": "A",
        "primary_current": "A",
        "turns_ratio": "dimensionless",
    }.get(target)


def _output_for(target: str):
    if target == "is_resonant":
        return "boolean"
    if target in {"energy_location", "phase_state", "circuit_character", "voltage_current_relation", "induced_current_direction", "transformer_type"}:
        return "conceptual"
    if target == "induced_emf":
        return "magnitude_direction"
    return "numeric"


def _is_lc(lower: str) -> bool:
    return "lc" in lower and "rlc" not in lower


def _is_rlc(lower: str) -> bool:
    return "rlc" in lower or "series rlc" in lower


def _is_reactance(lower: str) -> bool:
    return "reactance" in lower


def _is_solenoid(lower: str) -> bool:
    return "solenoid" in lower and "toroid" not in lower and "single loop" not in lower


def _is_faraday(lower: str) -> bool:
    return "induced emf" in lower or "faraday" in lower or "lenz" in lower or "magnetic flux" in lower


def _is_transformer(lower: str) -> bool:
    return "transformer" in lower or "primary" in lower and "secondary" in lower
