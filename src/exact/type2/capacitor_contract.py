from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pint

from exact.type2.schemas import Extraction, Type2SolveResult, Verification
from exact.type2.solving.units import parse_quantity, ureg


EPSILON_0 = 8.8541878128e-12 * ureg.farad / ureg.meter

CapacitorSystemType = Literal[
    "parallel_plate_capacitor",
    "single_capacitor",
    "series_capacitors",
    "parallel_capacitors",
    "connected_precharged_capacitors",
]
CapacitorTargetQuantity = Literal[
    "capacitance",
    "charge",
    "stored_energy",
    "relative_permittivity",
    "maximum_charge_before_breakdown",
    "breakdown_voltage",
    "final_voltage",
    "voltage_across_capacitor",
    "equivalent_capacitance",
]


@dataclass(frozen=True)
class CapacitorEvidence:
    text: str
    mapped_to: dict[str, Any]


@dataclass(frozen=True)
class CapacitorKnown:
    value: float
    unit: str
    original: str = ""


@dataclass(frozen=True)
class CapacitorElement:
    id: str
    capacitance: CapacitorKnown
    initial_voltage: CapacitorKnown | None = None
    connection_sign: int | None = None


@dataclass(frozen=True)
class CapacitorTarget:
    quantity: CapacitorTargetQuantity
    unit: str
    capacitor_id: str | None = None


@dataclass(frozen=True)
class CapacitorContract:
    domain: str
    system_type: CapacitorSystemType
    target: CapacitorTarget
    knowns: dict[str, CapacitorKnown] = field(default_factory=dict)
    capacitors: tuple[CapacitorElement, ...] = ()
    parse_confidence: float = 0.0
    evidence: tuple[CapacitorEvidence, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapacitorValidationIssue:
    reason: str
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatedCapacitorContract:
    contract: CapacitorContract
    knowns: dict[str, pint.Quantity]
    capacitors: tuple[tuple[str, pint.Quantity, pint.Quantity | None, int | None], ...]


def extract_capacitor_contract(extraction: Extraction) -> CapacitorContract | None:
    """Parser boundary for capacitor solving.

    This adapter may inspect natural language. The deterministic capacitor
    solver below consumes only the validated contract.
    """

    text = extraction.normalized_question
    lower = text.lower()
    if "capacitor" not in lower and "capacitance" not in lower:
        return None

    knowns = _extract_knowns(text)
    capacitors = _extract_capacitors(text)
    target = _target_from_text(text, extraction.target)
    if target is None:
        return None
    system_type = _system_type_from_text(text, target, capacitors)
    evidence = [CapacitorEvidence(text, {"target.quantity": target.quantity, "target.unit": target.unit})]
    for key, value in knowns.items():
        evidence.append(CapacitorEvidence(value.original, {f"knowns.{key}.value": value.value, f"knowns.{key}.unit": value.unit}))
    return CapacitorContract(
        domain="capacitors",
        system_type=system_type,
        target=target,
        knowns=knowns,
        capacitors=tuple(capacitors),
        parse_confidence=0.84,
        evidence=tuple(evidence),
    )


def validate_capacitor_contract(
    contract: CapacitorContract,
) -> tuple[ValidatedCapacitorContract | None, CapacitorValidationIssue | None]:
    if contract.domain != "capacitors":
        return None, CapacitorValidationIssue("domain is not `capacitors`")
    if contract.system_type not in {
        "parallel_plate_capacitor",
        "single_capacitor",
        "series_capacitors",
        "parallel_capacitors",
        "connected_precharged_capacitors",
    }:
        return None, CapacitorValidationIssue("unsupported capacitor system_type")
    if contract.target.quantity not in {
        "capacitance",
        "charge",
        "stored_energy",
        "relative_permittivity",
        "maximum_charge_before_breakdown",
        "breakdown_voltage",
        "final_voltage",
        "voltage_across_capacitor",
        "equivalent_capacitance",
    }:
        return None, CapacitorValidationIssue("unsupported capacitor target")

    knowns: dict[str, pint.Quantity] = {}
    for key, known in contract.knowns.items():
        try:
            knowns[key] = _normalize_known(key, known)
        except Exception as exc:
            return None, CapacitorValidationIssue(f"known `{key}` has invalid units: {exc}")

    caps = []
    seen_ids = set()
    for index, cap in enumerate(contract.capacitors):
        if cap.id in seen_ids:
            return None, CapacitorValidationIssue(f"duplicated capacitor id `{cap.id}`")
        seen_ids.add(cap.id)
        c = _quantity(cap.capacitance).to("F")
        if float(c.magnitude) <= 0:
            return None, CapacitorValidationIssue(f"capacitance `{cap.id}` must be positive")
        u = _quantity(cap.initial_voltage).to("V") if cap.initial_voltage is not None else None
        if contract.system_type == "connected_precharged_capacitors" and cap.connection_sign not in {-1, 1}:
            return None, CapacitorValidationIssue("connected capacitor polarity/sign is ambiguous", (f"capacitors.{index}.connection_sign",))
        caps.append((cap.id, c, u, cap.connection_sign))

    required = _required_knowns(contract)
    missing = tuple(key for key in required if key not in knowns)
    if missing:
        return None, CapacitorValidationIssue(
            f"{missing[0]} is required for {contract.target.quantity}",
            tuple(f"knowns.{key}" for key in missing),
        )

    for key in ("area", "plate_distance", "capacitance"):
        if key in knowns and float(knowns[key].magnitude) <= 0:
            return None, CapacitorValidationIssue(f"{key} must be positive")
    if "breakdown_field" in knowns and float(knowns["breakdown_field"].magnitude) <= 0:
        return None, CapacitorValidationIssue("breakdown_field must be positive")

    if contract.target.quantity == "voltage_across_capacitor":
        if contract.target.capacitor_id is None:
            return None, CapacitorValidationIssue(
                "series capacitor target.capacitor_id is required",
                ("target.capacitor_id",),
            )
        if contract.target.capacitor_id not in seen_ids:
            return None, CapacitorValidationIssue(f"unknown target capacitor `{contract.target.capacitor_id}`")

    if contract.system_type in {"series_capacitors", "parallel_capacitors", "connected_precharged_capacitors"}:
        if len(caps) < 2:
            return None, CapacitorValidationIssue("at least two capacitors are required", ("capacitors",))
    if contract.system_type == "connected_precharged_capacitors":
        for index, (_, _, voltage, _) in enumerate(caps):
            if voltage is None:
                return None, CapacitorValidationIssue("initial_voltage is required", (f"capacitors.{index}.initial_voltage",))

    return ValidatedCapacitorContract(contract, knowns, tuple(caps)), None


def solve_capacitor_contract(extraction: Extraction) -> Type2SolveResult | None:
    contract = extract_capacitor_contract(extraction)
    if contract is None:
        return None
    if not _contract_solver_should_own(contract):
        return None
    validated, issue = validate_capacitor_contract(contract)
    if issue is not None or validated is None:
        return _failure_result(extraction, issue or CapacitorValidationIssue("invalid capacitor contract"), contract)
    solved = _solve_validated_capacitor(validated)
    if solved is None:
        return _failure_result(extraction, CapacitorValidationIssue("no deterministic capacitor rule matched"), contract)
    answer, unit, value, diagnostics = solved
    return Type2SolveResult(
        answer=answer,
        unit=unit,
        value=value if isinstance(value, pint.Quantity) else None,
        formula=None,
        extraction=extraction,
        verification=Verification(True, f"Solved by deterministic capacitor contract rule `{diagnostics['selected_rule']}`."),
        cot=[
            "Parsed a formal capacitor contract.",
            "Validated target, required knowns, units, capacitor IDs, and polarity fields.",
            "Solved using deterministic capacitor physics rules without reading raw problem text.",
        ],
        premises=[f"diagnostics={diagnostics}"],
        confidence=0.94,
        error=None,
    )


def _solve_validated_capacitor(
    validated: ValidatedCapacitorContract,
) -> tuple[str, str | None, pint.Quantity | None, dict[str, Any]] | None:
    contract = validated.contract
    target = contract.target.quantity
    k = validated.knowns
    selected_rule = target
    chain = []

    if target == "relative_permittivity":
        value = (k["capacitance"] * k["plate_distance"] / (EPSILON_0 * k["area"])).to_base_units()
        if not value.dimensionless:
            return None
        unit = "dimensionless"
        chain = ["epsilon_r = C * d / (epsilon_0 * A)"]
        return _solved(value, unit, selected_rule, chain, validated)

    if target == "breakdown_voltage":
        value = (k["breakdown_field"] * k["plate_distance"]).to("V")
        chain = ["U_max = E_breakdown * d"]
        return _solved(value, "V", selected_rule, chain, validated)

    if target == "maximum_charge_before_breakdown":
        eps_r = k.get("relative_permittivity", 1 * ureg.dimensionless)
        value = (EPSILON_0 * eps_r * k["area"] * k["breakdown_field"]).to("C")
        chain = [
            "C = epsilon_0 * epsilon_r * A / d",
            "U_max = E_breakdown * d",
            "Q_max = C * U_max = epsilon_0 * epsilon_r * A * E_breakdown",
        ]
        return _solved(value, "C", selected_rule, chain, validated)

    if target == "final_voltage":
        numerator = 0 * ureg.coulomb
        denominator = 0 * ureg.farad
        for _, capacitance, voltage, sign in validated.capacitors:
            if voltage is None or sign is None:
                return None
            numerator += sign * capacitance * voltage
            denominator += capacitance
        if float(denominator.magnitude) == 0:
            return None
        value = (numerator / denominator).to("V")
        chain = ["U_f = sum(connection_sign_i * C_i * U_i) / sum(C_i)"]
        return _solved(value, "V", selected_rule, chain, validated)

    if target == "voltage_across_capacitor":
        total_voltage = k["total_voltage"]
        if not validated.capacitors:
            return None
        reciprocal_sum = sum((1 / cap[1] for cap in validated.capacitors), 0 / ureg.farad)
        if float(reciprocal_sum.magnitude) == 0:
            return None
        c_eq = (1 / reciprocal_sum).to("F")
        charge = (c_eq * total_voltage).to("C")
        cap_id = contract.target.capacitor_id
        for current_id, capacitance, _, _ in validated.capacitors:
            if current_id == cap_id:
                value = (charge / capacitance).to("V")
                chain = ["1 / C_eq = sum(1 / C_i)", "Q = C_eq * U_total", "U_i = Q / C_i"]
                return _solved(value, "V", selected_rule, chain, validated)
        return None

    if target == "equivalent_capacitance" and contract.system_type == "series_capacitors":
        if not validated.capacitors:
            return None
        reciprocal_sum = sum((1 / cap[1] for cap in validated.capacitors), 0 / ureg.farad)
        if float(reciprocal_sum.magnitude) == 0:
            return None
        value = (1 / reciprocal_sum).to("F")
        return _solved(value, "F", "series_equivalent_capacitance", ["1 / C_eq = sum(1 / C_i)"], validated)

    if target == "equivalent_capacitance" and contract.system_type == "parallel_capacitors":
        value = sum((cap[1] for cap in validated.capacitors), 0 * ureg.farad).to("F")
        return _solved(value, "F", "parallel_equivalent_capacitance", ["C_eq = sum(C_i)"], validated)

    if target == "capacitance" and contract.system_type == "parallel_plate_capacitor":
        eps_r = k.get("relative_permittivity", 1 * ureg.dimensionless)
        value = (EPSILON_0 * eps_r * k["area"] / k["plate_distance"]).to("F")
        return _solved(value, "F", "parallel_plate_capacitance", ["C = epsilon_0 * epsilon_r * A / d"], validated)

    if target == "charge":
        value = (k["capacitance"] * k["voltage"]).to("C")
        return _solved(value, "C", "charge_stored", ["Q = C * U"], validated)

    if target == "stored_energy":
        value = (0.5 * k["capacitance"] * k["voltage"] ** 2).to("J")
        return _solved(value, "J", "stored_energy", ["W = 1/2 * C * U^2"], validated)

    return None


def _failure_result(
    extraction: Extraction,
    issue: CapacitorValidationIssue,
    contract: CapacitorContract,
) -> Type2SolveResult:
    diagnostics = {
        "solver": "deterministic_capacitor_solver",
        "status": "unsolved",
        "reason": issue.reason,
        "missing": list(issue.missing),
        "fallback_recommended": True,
        "target": contract.target.quantity,
        "system_type": contract.system_type,
    }
    return Type2SolveResult(
        answer="",
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(False, issue.reason),
        cot=["Capacitor contract validation failed; deterministic solver refused to guess."],
        premises=[f"diagnostics={diagnostics}"],
        confidence=0.0,
        error="type2_capacitor_contract_unsolved",
    )


def _contract_solver_should_own(contract: CapacitorContract) -> bool:
    if contract.target.quantity in {
        "relative_permittivity",
        "maximum_charge_before_breakdown",
        "breakdown_voltage",
        "final_voltage",
        "voltage_across_capacitor",
        "equivalent_capacitance",
    }:
        return True
    if contract.target.quantity == "capacitance" and {"area", "plate_distance"}.issubset(contract.knowns):
        return True
    if contract.target.quantity in {"charge", "stored_energy"} and {"capacitance", "voltage"}.issubset(contract.knowns):
        return True
    return False


def _solved(
    value: pint.Quantity,
    unit: str,
    selected_rule: str,
    formula_chain: list[str],
    validated: ValidatedCapacitorContract,
) -> tuple[str, str, pint.Quantity, dict[str, Any]]:
    converted = value.to(unit) if unit != "dimensionless" else value.to_base_units()
    if unit == "dimensionless" and not converted.dimensionless:
        raise ValueError("dimensionless target returned dimensional value")
    magnitude = float(converted.magnitude)
    diagnostics = {
        "solver": "deterministic_capacitor_solver",
        "status": "solved",
        "selected_rule": selected_rule,
        "formula_chain": formula_chain,
        "normalized_inputs": _normalized_inputs(validated),
        "result": {"value": magnitude, "unit": unit},
    }
    return _format_number(magnitude), None if unit == "dimensionless" else unit, converted, diagnostics


def _normalized_inputs(validated: ValidatedCapacitorContract) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for key, value in validated.knowns.items():
        unit = _canonical_unit_for_key(key)
        inputs[key] = {"value": float(value.to(unit).magnitude), "unit": unit}
    if validated.capacitors:
        inputs["capacitors"] = [
            {
                "id": cap_id,
                "capacitance": {"value": float(capacitance.to("F").magnitude), "unit": "F"},
                "initial_voltage": None if voltage is None else {"value": float(voltage.to("V").magnitude), "unit": "V"},
                "connection_sign": sign,
            }
            for cap_id, capacitance, voltage, sign in validated.capacitors
        ]
    return inputs


def _target_from_text(text: str, extraction_target: str | None) -> CapacitorTarget | None:
    lower = text.lower()
    if any(marker in lower for marker in ("dielectric constant", "relative permittivity", "epsilon r", "epsilon_r", "κ", "kappa", "permittivity ratio")):
        return CapacitorTarget("relative_permittivity", "dimensionless")
    if any(marker in lower for marker in ("maximum charge", "largest safe charge", "breakdown", "charge limit", "maximum charge the plates can hold")):
        return CapacitorTarget("maximum_charge_before_breakdown", "C")
    if "final voltage" in lower or "final potential" in lower or "after connecting" in lower or "afterwards" in lower:
        return CapacitorTarget("final_voltage", "V")
    if "voltage across" in lower and "series" in lower:
        cap_id = _requested_capacitor_id(text)
        return CapacitorTarget("voltage_across_capacitor", "V", cap_id)
    # Normalize the upstream target so LLM phrasings ("stored energy",
    # "energy_stored") match the same buckets the heuristic extractor emits.
    norm = (extraction_target or "").strip().lower().replace("_", " ")
    if "energy" in norm or "energy stored" in lower or "stored energy" in lower:
        return CapacitorTarget("stored_energy", "J")
    if "capacitance" in norm:
        return CapacitorTarget("capacitance", "F")
    if "charge" in norm:
        return CapacitorTarget("charge", "C")
    if ("voltage" in norm or "potential" in norm) and "series" in lower:
        return CapacitorTarget("voltage_across_capacitor", "V", _requested_capacitor_id(text))
    return None


def _system_type_from_text(
    text: str,
    target: CapacitorTarget,
    capacitors: list[CapacitorElement],
) -> CapacitorSystemType:
    lower = text.lower()
    if "after connecting" in lower or "afterwards" in lower or "terminals" in lower or "plates are connected" in lower:
        return "connected_precharged_capacitors"
    if "series" in lower:
        return "series_capacitors"
    if "parallel" in lower and len(capacitors) >= 2 and "parallel-plate" not in lower and "parallel plate" not in lower:
        return "parallel_capacitors"
    if target.quantity in {"relative_permittivity", "maximum_charge_before_breakdown", "breakdown_voltage", "capacitance"}:
        return "parallel_plate_capacitor"
    return "single_capacitor"


def _extract_knowns(text: str) -> dict[str, CapacitorKnown]:
    knowns: dict[str, CapacitorKnown] = {}
    for key, aliases in {
        "capacitance": ("capacitance", "c"),
        "voltage": ("voltage", "potential difference", "u"),
        "area": ("area",),
        "plate_distance": ("distance between the plates", "plate separation", "separated by", "distance"),
        "breakdown_field": ("breakdown field", "dielectric strength", "electric field strength"),
        "total_voltage": ("total voltage", "uab", "voltage uab"),
    }.items():
        value = _find_known(text, aliases)
        if value is not None:
            knowns[key] = value

    radius = _find_radius(text)
    if radius is not None and "area" not in knowns:
        r = _quantity(radius).to("m")
        area = 3.141592653589793 * r.magnitude * r.magnitude
        knowns["area"] = CapacitorKnown(area, "m^2", radius.original)
    if "air" in text.lower() and "relative_permittivity" not in knowns:
        knowns["relative_permittivity"] = CapacitorKnown(1.0, "dimensionless", "air")
    return knowns


def _find_known(text: str, aliases: tuple[str, ...]) -> CapacitorKnown | None:
    unit_pattern = r"(?:F|uF|µF|μF|nF|pF|V|kV|mV|C|uC|µC|μC|nC|J|nJ|uJ|µJ|mJ|m\^2|cm\^2|mm\^2|m²|cm²|mm²|m|cm|mm|V/m|N/C)"
    for alias in aliases:
        patterns = (
            rf"\b{re.escape(alias)}\s*(?:=|is|of)?\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+|×10\^[-+]?\d+|x10\^[-+]?\d+)?)\s*(?P<unit>{unit_pattern})\b",
            rf"\b{re.escape(alias)}\b[^\d]{{0,60}}(?P<value>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+|×10\^[-+]?\d+|x10\^[-+]?\d+)?)\s*(?P<unit>{unit_pattern})\b",
            rf"(?P<value>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+|×10\^[-+]?\d+|x10\^[-+]?\d+)?)\s*(?P<unit>{unit_pattern})\s+(?:{re.escape(alias)})\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return CapacitorKnown(_parse_number(match.group("value")), _clean_unit(match.group("unit")), match.group(0))
    return None


def _find_radius(text: str) -> CapacitorKnown | None:
    match = re.search(
        r"(?:radius|r)\s*(?:R\s*)?(?:=|is|of)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>m|cm|mm)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return CapacitorKnown(float(match.group("value")), _clean_unit(match.group("unit")), match.group(0))


def _extract_capacitors(text: str) -> list[CapacitorElement]:
    caps: dict[str, dict[str, Any]] = {}
    for match in re.finditer(
        r"\b(?P<id>C[_A-Za-z0-9]*)\s*=\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>uF|µF|μF|nF|pF|F)\b",
        text,
        flags=re.IGNORECASE,
    ):
        cap_id = _cap_id(match.group("id"))
        caps.setdefault(cap_id, {})["capacitance"] = CapacitorKnown(float(match.group("value")), _clean_unit(match.group("unit")), match.group(0))
    for match in re.finditer(
        r"\bU(?P<suffix>[_A-Za-z0-9]*)\s*=\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>V|kV|mV)\b",
        text,
        flags=re.IGNORECASE,
    ):
        suffix = match.group("suffix").strip("_")
        cap_id = _cap_id(f"C{suffix}" if suffix else "C1")
        caps.setdefault(cap_id, {})["initial_voltage"] = CapacitorKnown(float(match.group("value")), _clean_unit(match.group("unit")), match.group(0))
    lower = text.lower()
    opposite = any(marker in lower for marker in ("opposite-polarity", "positive-to-negative", "opposite polarity"))
    for index, cap_id in enumerate(sorted(caps)):
        caps[cap_id].setdefault("connection_sign", -1 if opposite and index == 1 else 1)
    return [
        CapacitorElement(
            id=cap_id,
            capacitance=values["capacitance"],
            initial_voltage=values.get("initial_voltage"),
            connection_sign=values.get("connection_sign"),
        )
        for cap_id, values in sorted(caps.items())
        if "capacitance" in values
    ]


def _requested_capacitor_id(text: str) -> str | None:
    match = re.search(r"voltage\s+across\s+(?:capacitor\s+)?(?P<id>C[_A-Za-z0-9]*)", text, flags=re.IGNORECASE)
    if match:
        return _cap_id(match.group("id"))
    return None


def _required_knowns(contract: CapacitorContract) -> tuple[str, ...]:
    target = contract.target.quantity
    if target == "relative_permittivity":
        return ("capacitance", "area", "plate_distance")
    if target == "breakdown_voltage":
        return ("breakdown_field", "plate_distance")
    if target == "maximum_charge_before_breakdown":
        return ("area", "breakdown_field")
    if target == "voltage_across_capacitor":
        return ("total_voltage",)
    if target == "capacitance" and contract.system_type == "parallel_plate_capacitor":
        return ("area", "plate_distance")
    if target == "charge":
        return ("capacitance", "voltage")
    if target == "stored_energy":
        return ("capacitance", "voltage")
    return ()


def _normalize_known(key: str, known: CapacitorKnown) -> pint.Quantity:
    unit = _canonical_unit_for_key(key)
    value = _quantity(known).to(unit)
    return value


def _canonical_unit_for_key(key: str) -> str:
    return {
        "capacitance": "F",
        "voltage": "V",
        "total_voltage": "V",
        "area": "m^2",
        "plate_distance": "m",
        "breakdown_field": "V/m",
        "relative_permittivity": "dimensionless",
    }.get(key, "")


def _quantity(known: CapacitorKnown | None) -> pint.Quantity:
    if known is None:
        raise ValueError("missing quantity")
    return parse_quantity(known.value, _clean_unit(known.unit))


def _clean_unit(unit: str) -> str:
    return unit.strip().replace("μ", "u").replace("µ", "u").replace("²", "^2")


def _parse_number(value: str) -> float:
    text = value.replace(" ", "").replace("×", "x")
    if "x10" in text:
        base, exponent = text.split("x10", 1)
        return float(base) * (10 ** int(exponent.replace("^", "")))
    return float(text)


def _cap_id(raw: str) -> str:
    text = raw.strip().replace("_", "")
    if len(text) == 1 and text.upper() == "C":
        return "C1"
    return "C" + text[1:].upper()


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
