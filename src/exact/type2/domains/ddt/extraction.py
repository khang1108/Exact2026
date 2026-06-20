from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from exact.config import Settings
from exact.type2.domains.ddt.schemas import DdtContract, DdtQuantity
from exact.type2.extraction.extractor import normalize_question
from exact.type2.extraction.llm_structured import build_llm_json_client


FAMILIES = {
    "SOLENOID_FIELD",
    "SOLENOID_INDUCTANCE",
    "INDUCED_EMF",
    "MAGNETIC_FLUX",
    "RLC_REACTANCE",
    "RLC_IMPEDANCE",
    "RLC_RESONANCE",
    "LC_ENERGY",
    "CONCEPTUAL_SOLENOID",
    "UNKNOWN",
}


class DdtQuantitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    unit: str = ""


class DdtContractSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str = "UNKNOWN"
    target: str = "unknown"
    quantities: list[DdtQuantitySpec] = Field(default_factory=list)
    relation: str | None = None
    notes: list[str] = Field(default_factory=list)


def build_ddt_contract(question: str, settings: Settings | None = None) -> DdtContract:
    heuristic = extract_ddt_heuristic(question)
    llm = extract_ddt_with_llm(question, settings)
    return reconcile_ddt_contracts(llm, heuristic)


def extract_ddt_with_llm(question: str, settings: Settings | None = None) -> DdtContract | None:
    client = build_llm_json_client(settings)
    if client is None:
        return None
    raw = client.complete_json_sync(
        messages=_build_messages(question),
        temperature=(settings.llm_temperature if settings else 0.0),
        max_tokens=(settings.type2_extraction_max_tokens if settings else 768),
    )
    spec = DdtContractSpec.model_validate(raw)
    return _spec_to_contract(spec, "llm")


def extract_ddt_heuristic(question: str) -> DdtContract:
    text = normalize_question(question)
    lower = text.lower()
    quantities = _extract_quantities(text)
    family = _classify_family(lower)
    target = _classify_target(lower)
    relation = _classify_relation(lower)
    return DdtContract(family, target, quantities, relation, ["heuristic_ddt_contract"], "heuristic")


def reconcile_ddt_contracts(llm: DdtContract | None, heuristic: DdtContract) -> DdtContract:
    if llm is None or llm.family == "UNKNOWN":
        heuristic.notes.append("llm_contract_missing_or_unknown")
        return heuristic
    quantities = dict(heuristic.quantities)
    agreed = []
    conflicts = []
    for key, value in llm.quantities.items():
        if key in quantities:
            if _compatible(quantities[key], value):
                agreed.append(key)
            else:
                conflicts.append(key)
            continue
        quantities[key] = value
    family = llm.family if llm.family == heuristic.family or heuristic.family == "UNKNOWN" else heuristic.family
    target = llm.target if llm.target == heuristic.target or heuristic.target == "unknown" else heuristic.target
    notes = [
        "reconciled_llm_and_heuristic_contract",
        f"agreed={','.join(agreed) or '-'}",
        f"conflicts={','.join(conflicts) or '-'}",
        *heuristic.notes,
        *llm.notes,
    ]
    return DdtContract(family, target, quantities, llm.relation or heuristic.relation, notes, "llm_heuristic_reconciled")


def _build_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Parse a DDT solenoid/RLC/LC physics question into a strict JSON contract. Return one JSON object only. "
                "family must be one of SOLENOID_FIELD, SOLENOID_INDUCTANCE, INDUCED_EMF, MAGNETIC_FLUX, "
                "RLC_REACTANCE, RLC_IMPEDANCE, RLC_RESONANCE, LC_ENERGY, CONCEPTUAL_SOLENOID, UNKNOWN. "
                "Use quantity names only from: N, length, turn_density, I, I_initial, I_final, L, B, area, flux, "
                "time, frequency, R, C, Z, X_L, X_C, U, Q. Do not solve. "
                "CRITICAL: All values must be evaluated float numbers. NEVER output mathematical expressions (e.g. 4*Math.PI or fractions) as values."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "Return shape: {\"family\":\"SOLENOID_FIELD\",\"target\":\"magnetic_field\","
                "\"quantities\":[{\"name\":\"N\",\"value\":1000,\"unit\":\"turns\"}],"
                "\"relation\":null,\"notes\":[]}"
            ),
        },
    ]


def _spec_to_contract(spec: DdtContractSpec, source: str) -> DdtContract:
    quantities: dict[str, DdtQuantity] = {}
    for item in spec.quantities:
        key = _canon_name(item.name)
        if key:
            quantities[key] = DdtQuantity(key, item.value, item.unit)
    family = (spec.family or "").strip().upper() or "UNKNOWN"
    if family not in FAMILIES:
        family = "UNKNOWN"
    return DdtContract(family, _canon_target(spec.target), quantities, spec.relation, spec.notes, source)


def _classify_family(lower: str) -> str:
    if any(term in lower for term in ("unit of", "what quantity", "what happens", "depend", "application", "characteristics")):
        return "CONCEPTUAL_SOLENOID"
    if "induced electromotive force" in lower or "emf" in lower:
        return "INDUCED_EMF"
    if "magnetic flux" in lower or "flux linkage" in lower:
        return "MAGNETIC_FLUX"
    if "turn density" in lower or "turns per meter" in lower or "turns per unit length" in lower or "turns per meter length" in lower:
        return "SOLENOID_FIELD"
    if "inductance" in lower and "solenoid" in lower and "energy" not in lower:
        return "SOLENOID_INDUCTANCE"
    if "solenoid" in lower and "magnetic field" in lower:
        return "SOLENOID_FIELD"
    if "reactance" in lower:
        return "RLC_REACTANCE"
    if "impedance" in lower:
        return "RLC_IMPEDANCE"
    if "series" in lower and "circuit" in lower and ("frequency" in lower or " f " in lower) and ("inductor" in lower or " l " in lower):
        return "RLC_IMPEDANCE"
    if "power" in lower and (" z " in lower or "impedance" in lower or "resonant" in lower):
        return "RLC_RESONANCE"
    if "resonance" in lower or "power factor" in lower:
        return "RLC_RESONANCE"
    if "energy" in lower and any(term in lower for term in ("inductor", "lc", "capacitor", "solenoid")):
        return "LC_ENERGY"
    return "UNKNOWN"


def _classify_target(lower: str) -> str:
    if "magnetic field energy density" in lower or "energy density" in lower:
        return "energy_density"
    if "magnetic field energy" in lower:
        return "energy"
    if "flux linkage" in lower or ("magnetic flux" in lower and "entire solenoid" in lower):
        return "flux_linkage"
    if "magnetic flux" in lower:
        return "magnetic_flux"
    if "calculate the magnetic field" in lower or "magnetic field b" in lower:
        return "magnetic_field"
    if "turn density" in lower or "turns per meter" in lower or "turns per unit length" in lower:
        return "turn_density"
    if "what is the induced electromotive force" in lower or "calculate the induced electromotive force" in lower:
        return "voltage"
    if "determine the rms voltage" in lower or "rms voltage" in lower:
        return "voltage"
    if "inductance" in lower:
        return "inductance"
    if "electromotive force" in lower or "emf" in lower:
        return "voltage"
    if "magnetic field" in lower:
        return "magnetic_field"
    if "capacitive reactance" in lower:
        return "capacitive_reactance"
    if "inductive reactance" in lower:
        return "inductive_reactance"
    if "power factor" in lower:
        return "power_factor"
    if "power" in lower:
        return "power"
    if "impedance" in lower:
        return "impedance"
    if "current" in lower:
        return "current"
    if "voltage" in lower:
        return "voltage"
    if "energy" in lower:
        return "energy"
    return "conceptual" if "?" in lower else "unknown"


def _classify_relation(lower: str) -> str | None:
    if "double" in lower and "turn" in lower and "magnetic field" in lower:
        return "turns_double_field_double"
    if "external magnetic field" in lower and "ideal solenoid" in lower:
        return "ideal_solenoid_external_field_zero"
    if "current is suddenly disconnected" in lower:
        return "disconnect_induced_emf_opposes_change"
    if "unit of inductance" in lower:
        return "unit_inductance_henry"
    if "unit of induced electromotive force" in lower:
        return "unit_emf_volt"
    if "does not depend" in lower and "self-inductance" in lower:
        return "self_inductance_not_current"
    return None


def _extract_quantities(text: str) -> dict[str, DdtQuantity]:
    quantities: dict[str, DdtQuantity] = {}
    normalized_text = (
        text.replace("μ", "u")
        .replace("µ", "u")
        .replace("²", "^2")
        .replace("⁻", "-")
        .replace("×", "x")
    )
    normalized_text = re.sub(
        r"(?P<base>[-+]?\d+(?:\.\d+)?)\s*x\s*10\^?(?P<exp>[-+]?\d+)",
        r"\g<base>e\g<exp>",
        normalized_text,
        flags=re.IGNORECASE,
    )
    for match in re.finditer(r"\b([A-Za-z][A-Za-z_]*|X_L|X_C)\s*=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*([A-Za-z/%ΩΩ^0-9]+)?", normalized_text, flags=re.IGNORECASE):
        _put(quantities, _canon_name(match.group(1)), float(match.group(2)), _unit(match.group(3) or ""))
    for label, pattern in (
        ("length", r"([-+]?\d+(?:\.\d+)?)\s*m\s+long"),
        ("length", r"length of\s+([-+]?\d+(?:\.\d+)?)\s*m"),
        ("N", r"has\s+([-+]?\d+(?:\.\d+)?)\s+turns"),
        ("N", r"consists of\s+([-+]?\d+(?:\.\d+)?)\s+turns"),
        ("N", r"([-+]?\d+(?:\.\d+)?)\s+turns"),
        ("turn_density", r"turn density of\s+([-+]?\d+(?:\.\d+)?)\s*turns/m"),
        ("turn_density", r"number of turns per meter is\s+([-+]?\d+(?:\.\d+)?)"),
        ("turn_density", r"turn density of\s+n\s*=\s*([-+]?\d+(?:\.\d+)?)\s*turns/m"),
        ("area", r"(?:area|cross-sectional area).*?([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(cm\^2|m\^2)"),
        ("I", r"(?:current|carries a current|electric current).*?([-+]?\d+(?:\.\d+)?)\s*A"),
        ("L", r"inductance(?: L)?(?: of| is| =)?\s*([-+]?\d+(?:\.\d+)?)\s*H"),
        ("L", r"self-inductance.*?([-+]?\d+(?:\.\d+)?)\s*H"),
        ("B", r"(?:magnetic flux density|magnetic field).*?([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*T"),
        ("flux", r"magnetic flux(?: of)?\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*Wb"),
        ("flux", r"flux per turn is\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*Wb"),
        ("time", r"(?:in|over|over a period of|period of|time interval is)\s*([-+]?\d+(?:\.\d+)?)\s*s"),
        ("frequency", r"(?:frequency|f)\s*(?:=|of)?\s*([-+]?\d+(?:\.\d+)?)\s*Hz"),
        ("U", r"(?:emf|electromotive force)(?:\s+is)?\s*([-+]?\d+(?:\.\d+)?)\s*V"),
        ("U", r"voltage of\s*([-+]?\d+(?:\.\d+)?)\s*V"),
    ):
        for m in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
            _put(quantities, label, float(m.group(1)), _unit(m.group(2) if m.lastindex and m.lastindex >= 2 else ""))
    current_change = re.search(r"current.*?(?:increases|decreases).*?from\s+([-+]?\d+(?:\.\d+)?)\s*A?\s+to\s+([-+]?\d+(?:\.\d+)?)", normalized_text, flags=re.IGNORECASE)
    if current_change:
        _put(quantities, "I_initial", float(current_change.group(1)), "A")
        _put(quantities, "I_final", float(current_change.group(2)), "A")
    return quantities


def _put(q: dict[str, DdtQuantity], key: str, value: float, unit: str) -> None:
    if key:
        q[key] = DdtQuantity(key, value, unit)


def _canon_name(name: str) -> str:
    n = name.strip().lower().replace(" ", "_")
    aliases = {
        "n": "turn_density",
        "turns": "N",
        "u": "U",
        "v": "U",
        "voltage": "U",
        "i": "I",
        "current": "I",
        "l": "L",
        "inductance": "L",
        "b": "B",
        "r": "R",
        "z": "Z",
        "c": "C",
        "q": "Q",
        "xl": "X_L",
        "x_l": "X_L",
        "xc": "X_C",
        "x_c": "X_C",
    }
    return aliases.get(n, n)


def _canon_target(target: str) -> str:
    return target.strip().lower().replace(" ", "_")


def _unit(unit: str) -> str:
    return unit.strip().replace("Ω", "Ω").replace("Ohm", "Ω")


def _compatible(left: DdtQuantity, right: DdtQuantity) -> bool:
    return left.unit == right.unit and abs(left.value - right.value) <= max(1e-9, 0.02 * max(abs(left.value), abs(right.value), 1.0))
