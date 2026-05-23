from __future__ import annotations

import re
import unicodedata

from exact.type2.schemas import Extraction, Quantity, Type2QuestionKind
from exact.type2.solving.units import parse_quantity


UNIT_REPLACEMENTS = {
    "μ": "u",
    "µ": "u",
    "Ω": "ohm",
    "Ω": "ohm",
    "×": "x",
    "−": "-",
    "⁻": "-",
    "²": "^2",
    "³": "^3",
    "π": "pi",
    "°": " degree",
}


def normalize_question(question: str) -> str:
    text = unicodedata.normalize("NFKC", question)
    for old, new in UNIT_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"10\s*\^\s*([+-]?\d+)", r"1e\1", text)
    text = re.sub(r"10\s*-\s*(\d+)", r"1e-\1", text)
    text = re.sub(r"(\d)\s*x\s*1e", r"\1e", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_unit(unit: str) -> str:
    unit = unit.strip()
    for old, new in UNIT_REPLACEMENTS.items():
        unit = unit.replace(old, new)
    replacements = {
        "Ohm": "ohm",
        "ohms": "ohm",
        "F": "farad",
        "H": "henry",
    }
    return replacements.get(unit, unit)


CONCEPTUAL_MARKERS = (
    "why ",
    "explain",
    "state ",
    "define",
    "what happens",
    "where is",
    "how does",
    "shape of the graph",
    "directly proportional",
)

NUMERICAL_MARKERS = (
    "calculate",
    "determine",
    "find",
    "what is the value",
    "magnitude",
    "round",
)


def classify_type2_question(question: str) -> Type2QuestionKind:
    lower = question.lower()
    has_concept = any(marker in lower for marker in CONCEPTUAL_MARKERS)
    has_number = any(char.isdigit() for char in question)
    has_numeric_intent = any(marker in lower for marker in NUMERICAL_MARKERS)

    if has_concept and (has_number or has_numeric_intent):
        return Type2QuestionKind.MIXED
    if has_concept and not has_numeric_intent:
        return Type2QuestionKind.CONCEPTUAL
    return Type2QuestionKind.NUMERICAL



SYMBOL_TO_NAME = {
    "u": "voltage",
    "v": "voltage",
    "i": "current",
    "r": "resistance",
    "p": "power",
    "c": "capacitance",
    "q": "charge",
    "e": "energy_or_field",
    "w": "energy",
    "l": "inductance",
    "f": "frequency",
    "z": "impedance",
}

UNIT_TO_NAME = {
    "v": "voltage",
    "volt": "voltage",
    "a": "current",
    "ampere": "current",
    "ohm": "resistance",
    "w": "power",
    "j": "energy",
    "mj": "energy",
    "nj": "energy",
    "uj": "energy",
    "f": "capacitance",
    "uf": "capacitance",
    "nf": "capacitance",
    "pf": "capacitance",
    "c": "charge",
    "uc": "charge",
    "nc": "charge",
    "pc": "charge",
    "h": "inductance",
    "mh": "inductance",
    "hz": "frequency",
    "rad/s": "angular_frequency",
    "v/m": "electric_field",
    "n/c": "electric_field",
    "n": "force",
    "t": "magnetic_field",
    "wb": "magnetic_flux",
    "cm": "length",
    "mm": "length",
    "m": "length",
    "degree": "angle",
    "deg": "angle",
}

NAME_PATTERNS = (
    ("potential difference", "voltage"),
    ("voltage", "voltage"),
    ("current", "current"),
    ("resistance", "resistance"),
    ("impedance", "impedance"),
    ("power", "power"),
    ("capacitance", "capacitance"),
    ("capacitor", "capacitance"),
    ("charge", "charge"),
    ("energy", "energy"),
    ("electric field", "electric_field"),
    ("electric force", "force"),
    ("force", "force"),
    ("inductance", "inductance"),
    ("inductor", "inductance"),
    ("frequency", "frequency"),
    ("angular frequency", "angular_frequency"),
    ("magnetic field", "magnetic_field"),
    ("magnetic flux", "magnetic_flux"),
    ("distance", "length"),
    ("separated", "length"),
    ("apart", "length"),
    ("angle", "angle"),
)

TARGET_PATTERNS = (
    ("magnetic field energy", "energy"),
    ("electric field energy", "energy"),
    ("stored energy", "energy"),
    ("energy", "energy"),
    ("current", "current"),
    ("voltage", "voltage"),
    ("potential difference", "voltage"),
    ("resistance", "resistance"),
    ("impedance", "impedance"),
    ("power", "power"),
    ("capacitance", "capacitance"),
    ("charge", "charge"),
    ("electric field", "electric_field"),
    ("field strength", "electric_field"),
    ("electric force", "force"),
    ("force", "force"),
    ("frequency", "frequency"),
    ("angular frequency", "angular_frequency"),
    ("inductance", "inductance"),
    ("magnetic field", "magnetic_field"),
    ("magnetic flux", "magnetic_flux"),
    ("absolute error", "absolute_error"),
    ("relative error", "relative_error"),
    ("percentage error", "relative_error"),
)

NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?"
UNIT = r"(?:uF|nF|pF|mF|F|uC|nC|pC|mC|C|mA|A|mV|kV|V|kohm|ohm|mW|W|mJ|uJ|nJ|J|mH|H|Hz|rad/s|V/m|N/C|mN|N|T|mT|Wb|uWb|cm|mm|m|s|%|degree|degrees|deg|°)"

SYMBOL_VALUE_RE = re.compile(
    rf"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s*=\s*(?P<value>{NUMBER})\s*(?P<unit>{UNIT})\b",
    re.IGNORECASE,
)
VALUE_UNIT_RE = re.compile(rf"\b(?P<value>{NUMBER})\s*(?P<unit>{UNIT})\b", re.IGNORECASE)


def extract_type2(question: str) -> Extraction:
    normalized = normalize_question(question)
    lower = normalized.lower()
    quantities: dict[str, Quantity] = {}
    notes: list[str] = []

    consumed_spans: list[tuple[int, int]] = []
    for match in SYMBOL_VALUE_RE.finditer(normalized):
        name = _name_from_symbol(match.group("symbol"), match.group("unit"), lower)
        _add_quantity(quantities, name, match, consumed_spans)

    for match in VALUE_UNIT_RE.finditer(normalized):
        if any(start <= match.start() and match.end() <= end for start, end in consumed_spans):
            continue
        name = _name_from_context(match, lower)
        if name is None:
            notes.append(f"Could not map quantity: {match.group(0)}")
            continue
        _add_quantity(quantities, name, match, consumed_spans)

    _add_implicit_duplicate_quantities(quantities, lower)

    target = detect_target(normalized)
    if target == "energy_or_field":
        target = _disambiguate_energy_or_field(normalized)

    return Extraction(
        kind=classify_type2_question(normalized),
        normalized_question=normalized,
        target=target,
        quantities=quantities,
        notes=tuple(notes),
    )


def detect_target(question: str) -> str | None:
    lower = question.lower()
    object_match = re.search(
        r"\b(?:calculate|determine|find|what is|what is the)\s+(?P<object>.+?)(?:\s+when|\s+given|\s+if|[?.]|$)",
        lower,
    )
    search_space = object_match.group("object") if object_match else lower.split(" given ")[0].split(" when ")[0].split(" if ")[0]

    for phrase, name in TARGET_PATTERNS:
        if phrase in search_space:
            return name

    match = re.search(r"\b(?:calculate|determine|find)\s+(?:the\s+)?(?P<symbol>[A-Za-z])\b", lower)
    if match:
        return SYMBOL_TO_NAME.get(match.group("symbol").lower())
    return None


def _name_from_symbol(symbol: str, unit: str, question: str) -> str:
    symbol_name = SYMBOL_TO_NAME.get(symbol[:1].lower())
    if symbol_name == "energy_or_field":
        return _disambiguate_energy_or_field_by_unit(unit)
    if symbol_name:
        return symbol_name
    return _name_from_unit(unit) or _name_from_nearby_text(question) or symbol.lower()


def _name_from_context(match: re.Match[str], question: str) -> str | None:
    unit_name = _name_from_unit(match.group("unit"))
    window = question[max(0, match.start() - 80) : min(len(question), match.end() + 80)]
    nearby_name = _name_from_nearby_text(window)
    return unit_name or nearby_name


def _name_from_unit(unit: str) -> str | None:
    key = normalize_unit(unit).lower()
    if key in {"degrees", "°"}:
        key = "degree"
    return UNIT_TO_NAME.get(key)


def _name_from_nearby_text(text: str) -> str | None:
    for phrase, name in NAME_PATTERNS:
        if phrase in text:
            return name
    return None


def _disambiguate_energy_or_field(question: str) -> str:
    lower = question.lower()
    if "electric field" in lower or "field strength" in lower:
        return "electric_field"
    return "energy"


def _disambiguate_energy_or_field_by_unit(unit: str) -> str:
    unit_name = _name_from_unit(unit)
    if unit_name in {"electric_field", "energy"}:
        return unit_name
    return "energy"


def _add_quantity(
    quantities: dict[str, Quantity],
    name: str,
    match: re.Match[str],
    consumed_spans: list[tuple[int, int]],
) -> None:
    unit = normalize_unit(match.group("unit"))
    value = float(match.group("value"))
    try:
        quantity = parse_quantity(value, unit)
    except Exception:
        return

    key = name
    if key in quantities:
        suffix = 2
        while f"{key}_{suffix}" in quantities:
            suffix += 1
        key = f"{key}_{suffix}"

    quantities[key] = Quantity(
        name=name,
        value=quantity,
        evidence=match.group(0),
        confidence=0.9,
    )
    consumed_spans.append(match.span())


def _add_implicit_duplicate_quantities(quantities: dict[str, Quantity], question: str) -> None:
    if "each" in question and "force" in quantities and "force_2" not in quantities:
        original = quantities["force"]
        quantities["force_2"] = Quantity(
            name="force",
            value=original.value,
            evidence=f"each: {original.evidence}",
            confidence=0.75,
        )
