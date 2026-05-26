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
    "℃": "degC",
    "°C": "degC",
    "°": " degree",
    "·": "*",
    "⋅": "*",
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
    "describe",
    "which ",
    "where ",
    "when ",
    "what happens",
    "what happens",
    "where is",
    "how does",
    "how do",
    "how would",
    "shape of the graph",
    "directly proportional",
    "maximum",
    "minimum",
    "increase",
    "decrease",
    "larger",
    "smaller",
    "stored in",
)

NUMERICAL_MARKERS = (
    "calculate",
    "determine",
    "find",
    "what is the value",
    "magnitude",
    "round",
)

QUALITATIVE_TARGET_MARKERS = (
    "maximum",
    "minimum",
    "increases",
    "decreases",
    "increase",
    "decrease",
    "larger",
    "smaller",
    "stored",
    "where",
    "which",
    "when",
)

QUALITATIVE_QUESTION_MARKERS = (
    "why ",
    "explain",
    "where",
    "which",
    "when",
    "how does",
    "how do",
    "how would",
    "what happens",
    "maximum",
    "minimum",
)


def classify_type2_question(question: str) -> Type2QuestionKind:
    lower = question.lower()
    has_concept = any(marker in lower for marker in CONCEPTUAL_MARKERS)
    has_number = any(char.isdigit() for char in question)
    has_numeric_intent = any(marker in lower for marker in NUMERICAL_MARKERS)
    has_qualitative_target = any(marker in lower for marker in QUALITATIVE_TARGET_MARKERS)
    has_qualitative_question = any(marker in lower for marker in QUALITATIVE_QUESTION_MARKERS)

    if not has_number and not has_numeric_intent:
        return Type2QuestionKind.CONCEPTUAL
    if has_numeric_intent and not has_number and has_qualitative_target:
        return Type2QuestionKind.CONCEPTUAL
    if has_concept and has_numeric_intent and has_qualitative_question:
        return Type2QuestionKind.MIXED

    if has_concept and (has_number or has_numeric_intent) and has_qualitative_question:
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
    "eta": "efficiency",
    "t": "time",
    "s": "length",
    "d": "length",
    "l": "inductance",
    "f": "frequency",
    "z": "impedance",
    "m": "mass",
    "rho": "density",
    "lambda": "length",
}

UNIT_TO_NAME = {
    "v": "voltage",
    "volt": "voltage",
    "a": "current",
    "ampere": "current",
    "ohm": "resistance",
    "w": "power",
    "kw": "power",
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
    "h": "time",
    "henry": "inductance",
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
    "s": "time",
    "min": "time",
    "m^2": "area",
    "m^3": "volume",
    "l": "volume",
    "kg": "mass",
    "g": "mass",
    "kg/m^3": "density",
    "kg/m³": "density",
    "pa": "pressure",
    "kpa": "pressure",
    "m/s": "speed",
    "km/h": "speed",
    "j/kg": "heat_of_combustion",
    "j/(kg*degc)": "specific_heat_capacity",
    "j/(kg*degree)": "specific_heat_capacity",
    "degc": "temperature",
    "k": "temperature",
    "degree": "angle",
    "deg": "angle",
}

NAME_PATTERNS = (
    ("potential difference", "voltage"),
    ("electric potential", "voltage"),
    ("voltage", "voltage"),
    ("current", "current"),
    ("resistance", "resistance"),
    ("impedance", "impedance"),
    ("power", "power"),
    ("luminosity", "power"),
    ("specific heat capacity", "specific_heat_capacity"),
    ("calorific value", "heat_of_combustion"),
    ("heat value", "heat_of_combustion"),
    ("density", "density"),
    ("pressure", "pressure"),
    ("mass", "mass"),
    ("temperature change", "temperature"),
    ("temperature rise", "temperature"),
    ("temperature increase", "temperature"),
    ("initial temperature", "temperature"),
    ("final temperature", "temperature"),
    ("temperature", "temperature"),
    ("speed of light", "speed"),
    ("velocity", "speed"),
    ("speed", "speed"),
    ("wavelength", "length"),
    ("capacitance", "capacitance"),
    ("capacitor", "capacitance"),
    ("efficiency", "efficiency"),
    ("charge", "charge"),
    ("energy", "energy"),
    ("dipole moment", "electric_dipole_moment"),
    ("electric field", "electric_field"),
    ("electric force", "force"),
    ("force", "force"),
    ("inductance", "inductance"),
    ("inductor", "inductance"),
    ("frequency", "frequency"),
    ("angular frequency", "angular_frequency"),
    ("magnetic field", "magnetic_field"),
    ("magnetic flux", "magnetic_flux"),
    ("turn density", "turn_density"),
    ("number of turns per unit length", "turn_density"),
    ("surface charge density", "surface_charge_density"),
    ("linear charge density", "linear_charge_density"),
    ("relative permittivity", "relative_permittivity"),
    ("susceptibility", "electric_susceptibility"),
    ("time", "time"),
    ("area", "area"),
    ("distance", "length"),
    ("length", "length"),
    ("radius", "length"),
    ("separated", "length"),
    ("apart", "length"),
    ("angle", "angle"),
)

TARGET_PATTERNS = (
    ("magnetic field energy", "energy"),
    ("electric field energy", "energy"),
    ("stored energy", "energy"),
    ("energy", "energy"),
    ("efficiency", "efficiency"),
    ("current", "current"),
    ("voltage", "voltage"),
    ("potential difference", "voltage"),
    ("electric potential", "voltage"),
    ("resistance", "resistance"),
    ("impedance", "impedance"),
    ("power", "power"),
    ("capacitance", "capacitance"),
    ("net electric force", "force"),
    ("net force", "force"),
    ("force acting", "force"),
    ("electric force", "force"),
    ("force", "force"),
    ("charge", "charge"),
    ("electric field", "electric_field"),
    ("field strength", "electric_field"),
    ("frequency", "frequency"),
    ("angular frequency", "angular_frequency"),
    ("inductance", "inductance"),
    ("magnetic field", "magnetic_field"),
    ("magnetic flux", "magnetic_flux"),
    ("flux density", "flux_density"),
    ("relative permittivity", "relative_permittivity"),
    ("susceptibility", "electric_susceptibility"),
    ("dipole moment", "electric_dipole_moment"),
    ("momentum", "momentum"),
    ("specific heat capacity", "specific_heat_capacity"),
    ("calorific value", "heat_of_combustion"),
    ("heat value", "heat_of_combustion"),
    ("density", "density"),
    ("pressure", "pressure"),
    ("mass", "mass"),
    ("temperature change", "temperature"),
    ("temperature rise", "temperature"),
    ("temperature increase", "temperature"),
    ("initial temperature", "temperature"),
    ("final temperature", "temperature"),
    ("temperature", "temperature"),
    ("speed of light", "speed"),
    ("velocity", "speed"),
    ("speed", "speed"),
    ("wavelength", "length"),
    ("time", "time"),
    ("area", "area"),
    ("distance", "length"),
    ("length", "length"),
    ("radius", "length"),
    ("absolute error", "absolute_error"),
    ("relative error", "relative_error"),
    ("percentage error", "relative_error"),
)

NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?"
UNIT = r"(?:J/\(kg\*degC\)|J/\(kg\*degree\)|J/kg|kg/m\^3|kg/m³|m\^3|km/h|m/s|uF|nF|pF|mF|F|uC|nC|pC|mC|C|mA|A|mV|kV|V|kohm|ohm|mW|kW|W|mJ|uJ|nJ|J|mH|H|Hz|rad/s|V/m|N/C|mN|N|kPa|Pa|T|mT|Wb|uWb|cm|mm|kg|g|L|m\^2|m|s|h|min|%|degC|degree|degrees|deg|°)"

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
    unit_name = _name_from_unit(unit)
    if unit_name == "length" and symbol.upper() in {"AB", "AC", "BC", "CA", "CB"}:
        return "length"
    symbol_name = SYMBOL_TO_NAME.get(symbol[:1].lower())
    if symbol_name == "energy_or_field":
        return _disambiguate_energy_or_field_by_unit(unit)
    if symbol_name:
        return symbol_name
    return unit_name or _name_from_nearby_text(question) or symbol.lower()


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
