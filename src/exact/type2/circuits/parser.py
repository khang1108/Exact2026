from __future__ import annotations

import re

from exact.type2.circuits.schemas import CircuitComponent, CircuitContract, CircuitEvidence, CircuitQuantity, CircuitTarget
from exact.type2.schemas import Extraction


def extract_circuit_contract(extraction: Extraction) -> CircuitContract | None:
    text = extraction.normalized_question
    lower = text.lower()
    if ("capacitor" in lower or "capacitance" in lower) and _resistor_count(extraction) == 0 and "resistor" not in lower:
        return None
    if "transformer" in lower or ("primary" in lower and "secondary" in lower):
        return _transformer(extraction)
    if (
        any(token in lower for token in ("kwh", "for 30 minutes", "for 2 hours", "energy", "joule heat"))
        and not any(token in lower for token in ("magnetic field energy", "magnetic energy", "stored in the inductor"))
    ):
        return _energy(extraction)
    if "rlc" in lower or ("ac" in lower and any(x in lower for x in ("impedance", "phase", "power factor", "lead", "lag", "reactance"))):
        return _ac(extraction)
    if _resistor_count(extraction) > 1 or "series" in lower or "parallel" in lower:
        return _dc_network(extraction)
    if _resistor_count(extraction) == 1:
        return _single(extraction)
    return None


def _single(extraction: Extraction) -> CircuitContract:
    target = _target(extraction, {"current": ("current",), "voltage": ("voltage",), "resistance": ("resistance",), "power": ("power",)}, "current")
    return CircuitContract(
        system_type="single_resistor",
        source=_source(extraction),
        components=tuple(_components(extraction)),
        target=CircuitTarget(target, "component", _unit(target), component_id=_components(extraction)[0].id),
        assumptions={"ideal_wires": True},
        parse_confidence=0.7,
        evidence=(CircuitEvidence("single resistive component", {"system_type": "single_resistor"}),),
    )


def _dc_network(extraction: Extraction) -> CircuitContract:
    lower = extraction.normalized_question.lower()
    target = _target(
        extraction,
        {
            "component_current": ("current through", "current in lamp", "current through lamp"),
            "component_voltage": ("voltage across",),
            "component_power": ("power consumed by r", "power consumed by lamp"),
            "total_current": ("total current", "main wire"),
            "total_power": ("whole circuit", "total power", "supplied by the battery"),
            "equivalent_resistance": ("equivalent resistance",),
        },
        "total_current",
    )
    components = _components(extraction)
    component_id = _find_component_id(lower, components)
    scope = "component" if target.startswith("component_") else "total"
    topology = {}
    if "parallel" in lower or "same two terminals" in lower or "across the same" in lower:
        topology = {"type": "parallel", "branches": [{"id": f"B{i+1}", "items": [c.id]} for i, c in enumerate(components)]}
    elif "series" in lower or "successively" in lower:
        topology = {"type": "series", "items": [c.id for c in components]}
    return CircuitContract(
        system_type="dc_resistor_network",
        source=_source(extraction),
        components=tuple(components),
        topology=topology,
        target=CircuitTarget(target, scope, _unit(target), component_id=component_id if scope == "component" else None),
        parse_confidence=0.68,
        evidence=(CircuitEvidence("multi-component circuit", {"system_type": "dc_resistor_network"}),),
    )


def _energy(extraction: Extraction) -> CircuitContract:
    lower = extraction.normalized_question.lower()
    unit = "kWh" if "kwh" in lower else "J"
    knowns = {}
    for src, dst in (("power", "power"), ("time", "time"), ("voltage", "voltage"), ("current", "current")):
        if src in extraction.quantities:
            knowns[dst] = _q(extraction, src)
    return CircuitContract(system_type="energy_consumption", knowns=knowns, target=CircuitTarget("electrical_energy", "total", unit), parse_confidence=0.7)


def _ac(extraction: Extraction) -> CircuitContract:
    target = _target(
        extraction,
        {
            "capacitive_reactance": ("capacitive reactance", "reactance of capacitor"),
            "inductive_reactance": ("inductive reactance", "reactance of the coil", "coil reactance"),
            "impedance_magnitude": ("impedance",),
            "phase_angle": ("phase angle",),
            "power_factor": ("power factor",),
            "circuit_character": ("inductive or capacitive", "circuit character"),
            "voltage_current_relation": ("lead", "lag"),
            "current_rms": ("rms current",),
            "active_power": ("active power",),
        },
        extraction.target or "impedance_magnitude",
    )
    components = _components(extraction, ac=True)
    component_id = None
    if target == "capacitive_reactance":
        component_id = next((c.id for c in components if c.kind == "capacitor"), None)
    if target == "inductive_reactance":
        component_id = next((c.id for c in components if c.kind == "inductor"), None)
    source = _source(extraction)
    if "frequency" in extraction.quantities:
        source["frequency"] = _q(extraction, "frequency")
    if "voltage" in extraction.quantities:
        source["voltage_rms"] = _q(extraction, "voltage")
    return CircuitContract(
        system_type="series_rlc_circuit" if any(c.kind == "capacitor" for c in components) and any(c.kind == "inductor" for c in components) else "series_ac_circuit",
        source=source,
        components=tuple(components),
        topology={"type": "series", "items": [c.id for c in components]},
        target=CircuitTarget(target, "component" if component_id else "total", _unit(target), component_id=component_id),
        assumptions={"steady_state_ac": True, "rms_values": "voltage_rms" in source},
        parse_confidence=0.72,
    )


def _transformer(extraction: Extraction) -> CircuitContract:
    lower = extraction.normalized_question.lower()
    target = _target(extraction, {
        "secondary_voltage": ("secondary voltage",),
        "primary_voltage": ("primary voltage",),
        "secondary_current": ("secondary current",),
        "primary_current": ("primary current",),
        "transformer_type": ("step-up", "step up", "step-down", "step down"),
    }, extraction.target or "secondary_voltage")
    primary, secondary = {}, {}
    for key, dst in (("primary_turns", "turns"), ("primary_voltage", "voltage_rms"), ("primary_current", "current_rms")):
        if key in extraction.quantities:
            primary[dst] = _q(extraction, key)
    for key, dst in (("secondary_turns", "turns"), ("secondary_voltage", "voltage_rms"), ("secondary_current", "current_rms")):
        if key in extraction.quantities:
            secondary[dst] = _q(extraction, key)
    scope = "primary" if target.startswith("primary_") else "secondary" if target.startswith("secondary_") else "total"
    return CircuitContract(system_type="ideal_transformer", primary=primary, secondary=secondary, target=CircuitTarget(target, scope, _unit(target)), assumptions={"ideal_transformer": True, "rms_values": True}, parse_confidence=0.72)


def _components(extraction: Extraction, ac: bool = False) -> list[CircuitComponent]:
    comps = []
    for i, key in enumerate(k for k in extraction.quantities if k.startswith("resistance")):
        kind = "lamp" if "lamp" in extraction.normalized_question.lower() else "resistor"
        comps.append(CircuitComponent(f"L{i+1}" if kind == "lamp" else f"R{i+1}", kind, {"resistance": _q(extraction, key)}, model="resistive" if kind == "lamp" else None))
    if ac and "inductance" in extraction.quantities:
        comps.append(CircuitComponent("L1", "inductor", {"inductance": _q(extraction, "inductance")}))
    if ac and "capacitance" in extraction.quantities:
        comps.append(CircuitComponent("C1", "capacitor", {"capacitance": _q(extraction, "capacitance")}))
    return comps


def _source(extraction: Extraction) -> dict:
    source = {"id": "V1", "kind": "voltage_source", "mode": "dc"}
    if "voltage" in extraction.quantities:
        source["voltage"] = _q(extraction, "voltage")
    if "current" in extraction.quantities:
        source["current"] = _q(extraction, "current")
    return source


def _q(extraction: Extraction, key: str) -> CircuitQuantity:
    item = extraction.quantities[key]
    return CircuitQuantity(float(item.value.magnitude), str(item.value.units), item.evidence)


def _target(extraction: Extraction, mapping: dict[str, tuple[str, ...]], fallback: str) -> str:
    normalized = (fallback or "").strip().lower().replace(" ", "_")
    if normalized in mapping:
        return normalized
    lower = extraction.normalized_question.lower()
    for quantity, markers in mapping.items():
        if any(marker in lower for marker in markers):
            return quantity
    return normalized or next(iter(mapping))


def _unit(target: str) -> str:
    if "current" in target:
        return "A"
    if "voltage" in target:
        return "V"
    if "resistance" in target or "reactance" in target or "impedance" in target:
        return "ohm"
    if "power" in target:
        return "W"
    if target == "phase_angle":
        return "degree"
    if target == "power_factor":
        return "dimensionless"
    if target in {"circuit_character", "voltage_current_relation", "transformer_type"}:
        return "categorical"
    return "dimensionless"


def _resistor_count(extraction: Extraction) -> int:
    return len([key for key in extraction.quantities if key.startswith("resistance")])


def _find_component_id(lower: str, components: list[CircuitComponent]) -> str | None:
    for component in components:
        if component.id.lower() in lower:
            return component.id
    return components[0].id if len(components) == 1 else None
