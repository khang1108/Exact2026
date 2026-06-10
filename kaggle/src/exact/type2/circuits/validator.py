from __future__ import annotations

from exact.type2.circuits.quantity_registry import SUPPORTED_SCOPES, SUPPORTED_SYSTEM_TYPES, TARGET_UNITS
from exact.type2.circuits.schemas import CircuitContract, CircuitValidationIssue, ValidatedCircuitContract


def validate_contract(contract: CircuitContract) -> tuple[ValidatedCircuitContract | None, CircuitValidationIssue | None]:
    if contract.domain != "circuits":
        return None, CircuitValidationIssue("domain is not circuits", ("domain",))
    if contract.system_type not in SUPPORTED_SYSTEM_TYPES:
        return None, CircuitValidationIssue("system_type is unsupported", ("system_type",))
    if not contract.target.quantity:
        return None, CircuitValidationIssue("target.quantity is missing", ("target.quantity",))
    if contract.target.scope not in SUPPORTED_SCOPES:
        return None, CircuitValidationIssue("target.scope is missing or unsupported", ("target.scope",))
    unit_issue = _validate_target_unit(contract)
    if unit_issue:
        return None, unit_issue
    ids = [component.id for component in contract.components]
    if len(ids) != len(set(ids)):
        return None, CircuitValidationIssue("component IDs are duplicated")
    for component in contract.components:
        if component.kind in {"resistor", "lamp"}:
            if component.kind == "lamp" and component.model != "resistive":
                return None, CircuitValidationIssue("lamp component is not explicitly modeled as resistive", (f"components.{component.id}.model",))
            resistance = component.properties.get("resistance")
            if resistance is None:
                return None, CircuitValidationIssue("resistive component is missing resistance", (f"components.{component.id}.resistance",))
            if resistance.value <= 0 and not contract.assumptions.get("short_circuit_model"):
                return None, CircuitValidationIssue("resistance is zero/negative without short-circuit model", (f"components.{component.id}.resistance",))
    if contract.system_type == "single_resistor":
        return _validate_single_resistor(contract)
    issue = _validate_target_scope(contract)
    if issue:
        return None, issue
    if contract.system_type == "dc_resistor_network":
        return _validate_dc_network(contract)
    if contract.system_type == "energy_consumption":
        return _validate_energy(contract)
    if contract.system_type in {"series_ac_circuit", "series_rlc_circuit"}:
        return _validate_ac(contract)
    if contract.system_type == "ideal_transformer":
        return _validate_transformer(contract)
    return ValidatedCircuitContract(contract), None


def _validate_target_unit(contract: CircuitContract) -> CircuitValidationIssue | None:
    allowed = TARGET_UNITS.get(contract.target.quantity)
    if allowed is None:
        return CircuitValidationIssue("target.quantity is unsupported", ("target.quantity",))
    if contract.target.unit is None:
        return CircuitValidationIssue("target unit is missing", ("target.unit",))
    if contract.target.unit not in allowed:
        return CircuitValidationIssue(
            f"target unit `{contract.target.unit}` is incompatible with `{contract.target.quantity}`",
            ("target.unit",),
        )
    return None


def _validate_target_scope(contract: CircuitContract) -> CircuitValidationIssue | None:
    target = contract.target
    if len(contract.components) > 1 and target.quantity in {"power", "current", "voltage"}:
        return CircuitValidationIssue("ambiguous_target: multi-component circuit requires explicit target quantity/scope", ("target.quantity", "target.scope"))
    if target.scope == "component" and not target.component_id:
        return CircuitValidationIssue("target.component_id is required for component scope", ("target.component_id",))
    if target.component_id and target.component_id not in {component.id for component in contract.components}:
        return CircuitValidationIssue("referenced component does not exist", ("target.component_id",))
    if target.scope == "branch" and not target.branch_id:
        return CircuitValidationIssue("target.branch_id is required for branch scope", ("target.branch_id",))
    return None


def _validate_single_resistor(contract: CircuitContract):
    if len(contract.components) != 1:
        return None, CircuitValidationIssue("scalar Ohm solver requires exactly one component", ("components",))
    if contract.topology and len(contract.components) > 1:
        return None, CircuitValidationIssue("single_resistor cannot require network topology", ("topology",))
    if not (_source_value(contract, "voltage") or _source_value(contract, "current")):
        return None, CircuitValidationIssue("single-resistor solve is missing source voltage/current", ("source.voltage", "source.current"))
    return ValidatedCircuitContract(contract), None


def _validate_dc_network(contract: CircuitContract):
    if len(contract.components) > 1 and not contract.topology:
        return None, CircuitValidationIssue("multi-component circuit requires explicit topology", ("topology",))
    if not _source_value(contract, "voltage"):
        return None, CircuitValidationIssue("source voltage is missing for network solve", ("source.voltage",))
    known_components = {component.id for component in contract.components}
    branches = set()
    issue = _validate_topology_node(contract.topology, known_components, branches)
    if issue:
        return None, issue
    if contract.target.branch_id and contract.target.branch_id not in branches:
        return None, CircuitValidationIssue("referenced branch does not exist", ("target.branch_id",))
    return ValidatedCircuitContract(contract), None


def _validate_topology_node(node, known_components: set[str], branches: set[str]) -> CircuitValidationIssue | None:
    if not isinstance(node, dict) or node.get("type") not in {"series", "parallel"}:
        return CircuitValidationIssue("topology is unsupported", ("topology.type",))
    if node["type"] == "series":
        for item in node.get("items", ()):
            if isinstance(item, str):
                if item not in known_components:
                    return CircuitValidationIssue("topology contains unknown component ID", (item,))
            else:
                issue = _validate_topology_node(item, known_components, branches)
                if issue:
                    return issue
    if node["type"] == "parallel":
        for branch in node.get("branches", ()):
            branch_id = branch.get("id")
            if branch_id:
                branches.add(branch_id)
            for item in branch.get("items", ()):
                if isinstance(item, str):
                    if item not in known_components:
                        return CircuitValidationIssue("topology contains unknown component ID", (item,))
                else:
                    issue = _validate_topology_node(item, known_components, branches)
                    if issue:
                        return issue
    return None


def _validate_energy(contract: CircuitContract):
    if not contract.knowns.get("time"):
        return None, CircuitValidationIssue("energy target is missing time", ("knowns.time",))
    if not (contract.knowns.get("power") or (contract.knowns.get("voltage") and contract.knowns.get("current"))):
        return None, CircuitValidationIssue("energy target needs power or voltage/current", ("knowns.power", "knowns.voltage", "knowns.current"))
    return ValidatedCircuitContract(contract), None


def _validate_ac(contract: CircuitContract):
    if not _source_value(contract, "frequency"):
        return None, CircuitValidationIssue("AC frequency is missing", ("source.frequency",))
    q = contract.target.quantity
    if q in {"current_rms", "active_power"} and not contract.assumptions.get("rms_values"):
        return None, CircuitValidationIssue("RMS/peak convention is ambiguous", ("assumptions.rms_values",))
    if q in {"current_rms", "active_power"} and not _source_value(contract, "voltage_rms"):
        return None, CircuitValidationIssue("AC RMS target is missing voltage_rms", ("source.voltage_rms",))
    if q == "inductive_reactance" and not _component_with(contract, "inductor", "inductance", contract.target.component_id):
        return None, CircuitValidationIssue("inductive_reactance requires target inductor and frequency", ("target.component_id", "components.inductor.inductance"))
    if q == "capacitive_reactance" and not _component_with(contract, "capacitor", "capacitance", contract.target.component_id):
        return None, CircuitValidationIssue("capacitive_reactance requires target capacitor and frequency", ("target.component_id", "components.capacitor.capacitance"))
    if q not in {"inductive_reactance", "capacitive_reactance"} and not contract.components:
        return None, CircuitValidationIssue("AC target has no components", ("components",))
    return ValidatedCircuitContract(contract), None


def _validate_transformer(contract: CircuitContract):
    if not contract.primary or not contract.secondary:
        return None, CircuitValidationIssue("primary/secondary roles are ambiguous", ("primary", "secondary"))
    if contract.target.scope not in {"primary", "secondary", "total"}:
        return None, CircuitValidationIssue("target side must be explicit for transformer", ("target.scope",))
    np = contract.primary.get("turns")
    ns = contract.secondary.get("turns")
    q = contract.target.quantity
    if q in {"secondary_voltage", "primary_voltage", "secondary_current", "primary_current", "turns_ratio", "transformer_type"} and not (np and ns):
        return None, CircuitValidationIssue("transformer target is missing primary or secondary turns", ("primary.turns", "secondary.turns"))
    if q == "secondary_voltage" and not contract.primary.get("voltage_rms"):
        return None, CircuitValidationIssue("secondary_voltage requires primary voltage", ("primary.voltage_rms",))
    if q == "primary_voltage" and not contract.secondary.get("voltage_rms"):
        return None, CircuitValidationIssue("primary_voltage requires secondary voltage", ("secondary.voltage_rms",))
    if q == "secondary_current" and not contract.primary.get("current_rms"):
        return None, CircuitValidationIssue("secondary_current requires primary current", ("primary.current_rms",))
    if q == "primary_current" and not contract.secondary.get("current_rms"):
        return None, CircuitValidationIssue("primary_current requires secondary current", ("secondary.current_rms",))
    if q == "primary_turns" and not ns:
        return None, CircuitValidationIssue("primary_turns requires secondary turns", ("secondary.turns",))
    if q == "secondary_turns" and not np:
        return None, CircuitValidationIssue("secondary_turns requires primary turns", ("primary.turns",))
    return ValidatedCircuitContract(contract), None


def _source_value(contract: CircuitContract, name: str):
    return contract.source.get(name)


def _component_with(contract: CircuitContract, kind: str, prop: str, component_id: str | None = None) -> bool:
    for component in contract.components:
        if component_id and component.id != component_id:
            continue
        if component.kind == kind and prop in component.properties:
            return True
    return False
