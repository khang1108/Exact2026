from __future__ import annotations

from typing import Any

from exact.type2.capacitor_contract import extract_capacitor_contract, validate_capacitor_contract
from exact.type2.schemas import Extraction


def build_fallback_capacitor_context(extraction: Extraction) -> dict[str, Any] | None:
    """Build structured capacitor context for PoT/fallback prompts."""

    contract = extract_capacitor_contract(extraction)
    if contract is None:
        return None

    validated, issue = validate_capacitor_contract(contract)
    payload: dict[str, Any] = {
        "available": validated is not None,
        "reason": None if issue is None else issue.reason,
        "missing": [] if issue is None else list(issue.missing),
        "contract": {
            "domain": contract.domain,
            "system_type": contract.system_type,
            "target": {
                "quantity": getattr(contract.target, "quantity", None),
                "unit": getattr(contract.target, "unit", None),
                "capacitor_id": getattr(contract.target, "capacitor_id", None),
            },
            "knowns": {
                key: {"value": known.value, "unit": known.unit, "evidence": known.original}
                for key, known in contract.knowns.items()
            },
            "capacitors": [
                {
                    "id": cap.id,
                    "capacitance": {
                        "value": cap.capacitance.value,
                        "unit": cap.capacitance.unit,
                        "evidence": cap.capacitance.original,
                    },
                    "initial_voltage": None
                    if cap.initial_voltage is None
                    else {
                        "value": cap.initial_voltage.value,
                        "unit": cap.initial_voltage.unit,
                        "evidence": cap.initial_voltage.original,
                    },
                    "connection_sign": cap.connection_sign,
                }
                for cap in contract.capacitors
            ],
            "unresolved": list(contract.unresolved),
        },
    }
    if validated is not None:
        payload["normalized"] = {
            "knowns": {
                key: {"value": float(value.magnitude), "unit": str(value.units)}
                for key, value in validated.knowns.items()
            },
            "capacitors": [
                {
                    "id": cap_id,
                    "capacitance_F": float(capacitance.to("F").magnitude),
                    "initial_voltage_V": None if voltage is None else float(voltage.to("V").magnitude),
                    "connection_sign": sign,
                }
                for cap_id, capacitance, voltage, sign in validated.capacitors
            ],
        }
    return payload
