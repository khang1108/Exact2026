from __future__ import annotations

from exact.type2.electromagnetism.quantity_registry import SUPPORTED_SYSTEM_TYPES, quantity_spec
from exact.type2.electromagnetism.schemas import ElectromagnetismContract, EMValidationIssue, ValidatedEMContract


UNIT_COMPATIBILITY = {
    "A": {"A"},
    "C": {"C"},
    "F": {"F"},
    "H": {"H"},
    "Hz": {"Hz"},
    "J": {"J"},
    "J/m^3": {"J/m^3"},
    "T": {"T"},
    "V": {"V"},
    "W": {"W"},
    "Wb": {"Wb"},
    "Wb/s": {"Wb/s"},
    "Wb_turn": {"Wb_turn"},
    "degree": {"degree", "radian"},
    "ohm": {"ohm"},
    "s": {"s"},
    "rad/s": {"rad/s"},
    "boolean": {"boolean"},
    "dimensionless": {"dimensionless"},
    "categorical": {"categorical"},
    "conceptual": {"conceptual"},
}


def validate_contract(contract: ElectromagnetismContract) -> tuple[ValidatedEMContract | None, EMValidationIssue | None]:
    if contract.domain != "electromagnetism":
        return None, EMValidationIssue("domain is not electromagnetism", ("domain",))
    if contract.system_type not in SUPPORTED_SYSTEM_TYPES:
        return None, EMValidationIssue("system_type is missing or unsupported", ("system_type",))
    spec = quantity_spec(contract.target.quantity)
    if spec is None:
        return None, EMValidationIssue("target.quantity is missing or unsupported", ("target.quantity",))
    if contract.system_type not in spec.compatible_system_types:
        return None, EMValidationIssue(
            f"target `{contract.target.quantity}` is incompatible with `{contract.system_type}`",
            ("target.quantity", "system_type"),
        )
    unit_issue = _validate_target_unit(contract)
    if unit_issue is not None:
        return None, unit_issue
    issue = _validate_system_requirements(contract)
    if issue is not None:
        return None, issue
    return ValidatedEMContract(contract), None


def _validate_target_unit(contract: ElectromagnetismContract) -> EMValidationIssue | None:
    spec = quantity_spec(contract.target.quantity)
    if spec is None or contract.target.unit is None:
        return None
    allowed = UNIT_COMPATIBILITY.get(spec.unit, {spec.unit})
    if contract.target.unit not in allowed:
        return EMValidationIssue(
            f"target unit `{contract.target.unit}` is incompatible with `{contract.target.quantity}`",
            ("target.unit",),
        )
    if spec.type == "boolean" and contract.target.output not in {"boolean", "conceptual"}:
        return EMValidationIssue("target asks for boolean but output is not boolean", ("target.output",))
    if spec.type != "boolean" and contract.target.output == "boolean":
        return EMValidationIssue("target asks for boolean but solver target is not boolean", ("target.quantity",))
    return None


def _validate_system_requirements(contract: ElectromagnetismContract) -> EMValidationIssue | None:
    q = contract.target.quantity
    st = contract.system_type
    if st == "ideal_lc_oscillator":
        if not _component_value(contract, "inductor", "inductance"):
            return EMValidationIssue("ideal LC contract is missing inductance", ("components.inductor.inductance",))
        if not _component_value(contract, "capacitor", "capacitance"):
            return EMValidationIssue("ideal LC contract is missing capacitance", ("components.capacitor.capacitance",))
        if q in {"maximum_current", "total_energy"} and not (
            _state(contract, "capacitor_voltage") or _state(contract, "maximum_charge") or _state(contract, "total_energy")
        ):
            return EMValidationIssue(
                "LC energy target needs initial voltage, maximum charge, or total energy",
                ("state.capacitor_voltage", "state.maximum_charge", "state.total_energy"),
            )
        if q in {"instantaneous_current", "instantaneous_charge"} and not _state(contract, "time"):
            return EMValidationIssue("instantaneous LC target is missing time", ("state.time",))
        if q in {"energy_location", "phase_state"} and not (
            _state(contract, "phase_fraction") or _state(contract, "time_reference")
        ):
            return EMValidationIssue("LC state target is missing phase/state reference", ("state.phase_fraction",))
    if st == "series_rlc_circuit":
        if not _component_value(contract, "inductor", "inductance"):
            return EMValidationIssue("series RLC contract is missing inductance", ("components.inductor.inductance",))
        if not _component_value(contract, "capacitor", "capacitance"):
            return EMValidationIssue("series RLC contract is missing capacitance", ("components.capacitor.capacitance",))
        if q != "resonance_frequency" and q != "resonance_angular_frequency" and not _source(contract, "frequency"):
            return EMValidationIssue("series RLC target is missing source frequency", ("source.frequency",))
        if q not in {"is_resonant", "resonance_frequency", "resonance_angular_frequency", "resonance_condition", "circuit_state_at_resonance"}:
            if not _component_value(contract, "resistor", "resistance"):
                return EMValidationIssue("phasor target is missing resistance", ("components.resistor.resistance",))
            if q in {"current_rms", "active_power"} and not _source(contract, "voltage_rms"):
                return EMValidationIssue("RMS current/power target is missing RMS source voltage", ("source.voltage_rms",))
            if q in {"current_rms", "active_power"} and not contract.assumptions.get("rms_values"):
                return EMValidationIssue("source RMS/peak convention is missing", ("assumptions.rms_values",))
    if st == "ac_reactance":
        if q == "inductive_reactance" and not _component_value(contract, "inductor", "inductance"):
            return EMValidationIssue("inductive reactance requires an inductor", ("components.inductor.inductance",))
        if q == "capacitive_reactance" and not _component_value(contract, "capacitor", "capacitance"):
            return EMValidationIssue("capacitive reactance requires a capacitor", ("components.capacitor.capacitance",))
        if q in {"inductive_reactance", "capacitive_reactance", "net_reactance", "impedance_magnitude"} and not _source(contract, "frequency"):
            return EMValidationIssue("reactance target is missing frequency", ("source.frequency",))
        if q == "impedance_magnitude" and not _component_value(contract, "resistor", "resistance"):
            return EMValidationIssue("impedance magnitude requires resistance", ("components.resistor.resistance",))
    if st == "long_solenoid":
        if not (_geometry(contract, "turn_count") or _geometry(contract, "turn_density")):
            return EMValidationIssue("solenoid contract is missing turns or turn density", ("geometry.turn_count", "geometry.turn_density"))
        if _geometry(contract, "turn_count") and not _geometry(contract, "length") and q != "magnetic_field_inside":
            return EMValidationIssue("total turns N need solenoid length for geometry formulas", ("geometry.length",))
        if q in {"magnetic_flux_one_turn", "flux_linkage", "inductance"} and not _geometry(contract, "cross_section_area"):
            return EMValidationIssue("solenoid flux/inductance target is missing cross-section area", ("geometry.cross_section_area",))
        if q != "inductance" and not _known(contract, "current"):
            return EMValidationIssue("solenoid target is missing current", ("knowns.current",))
    if st == "electromagnetic_induction":
        if not _flux(contract, "time_interval"):
            return EMValidationIssue("Faraday target is missing time interval", ("flux_change.time_interval",))
        if not (_flux(contract, "initial_flux") and _flux(contract, "final_flux")):
            return EMValidationIssue("Faraday target is missing flux endpoints", ("flux_change.initial_flux", "flux_change.final_flux"))
        if q == "induced_current_direction" and not contract.flux_change.get("direction"):
            return EMValidationIssue("direction target is missing flux direction", ("flux_change.direction",))
        if q == "induced_current_direction" and not contract.convention.get("positive_emf_direction"):
            return EMValidationIssue("direction target is missing current/emf convention", ("convention.positive_emf_direction",))
    if st == "ideal_transformer":
        if not contract.primary or not contract.secondary:
            return EMValidationIssue("primary and secondary roles must be explicit", ("primary", "secondary"))
        if q in {"secondary_voltage", "primary_voltage", "turns_ratio", "transformer_type"} and not (
            _side(contract.primary, "turns") and _side(contract.secondary, "turns")
        ):
            return EMValidationIssue("transformer target is missing primary or secondary turns", ("primary.turns", "secondary.turns"))
        if q == "secondary_voltage" and not _side(contract.primary, "voltage_rms"):
            return EMValidationIssue("secondary voltage target is missing primary voltage", ("primary.voltage_rms",))
        if q == "primary_voltage" and not _side(contract.secondary, "voltage_rms"):
            return EMValidationIssue("primary voltage target is missing secondary voltage", ("secondary.voltage_rms",))
        if q == "secondary_current" and not _side(contract.primary, "current_rms"):
            return EMValidationIssue("secondary current target is missing primary current", ("primary.current_rms",))
        if q == "primary_current" and not _side(contract.secondary, "current_rms"):
            return EMValidationIssue("primary current target is missing secondary current", ("secondary.current_rms",))
    return None


def _component_value(contract: ElectromagnetismContract, kind: str, name: str):
    for component in contract.components:
        if component.kind == kind and name in component.properties:
            return component.properties[name]
    return None


def _source(contract: ElectromagnetismContract, name: str):
    return contract.source.get(name)


def _state(contract: ElectromagnetismContract, name: str):
    return contract.state.get(name)


def _known(contract: ElectromagnetismContract, name: str):
    return contract.knowns.get(name)


def _geometry(contract: ElectromagnetismContract, name: str):
    return contract.geometry.get(name)


def _flux(contract: ElectromagnetismContract, name: str):
    return contract.flux_change.get(name)


def _side(side: dict, name: str):
    return side.get(name)
