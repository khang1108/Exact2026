from __future__ import annotations

from dataclasses import dataclass, field
import re

from exact.type2.extraction.extractor import extract_type2, normalize_question


@dataclass(frozen=True)
class CompatQuantity:
    name: str
    symbol: str | None = None
    base_symbol: str | None = None
    index: str | None = None
    value: float | None = None
    unit: str | None = None
    dimension: str | None = None
    evidence: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class CompatRelation:
    type: str
    symbols: list[str] = field(default_factory=list)
    value: float | None = None
    unit: str | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class CompatVectorContributionGroup:
    count: int
    quantity_dimension: str | None = None
    magnitude_symbol: str | None = None
    angle_between_deg: float | None = None
    relation: str = "equal_magnitude"
    evidence: str | None = None


@dataclass(frozen=True)
class CompatExtraction:
    question_text: str
    quantities: list[CompatQuantity] = field(default_factory=list)
    relations: list[CompatRelation] = field(default_factory=list)
    vector_contribution_groups: list[CompatVectorContributionGroup] = field(default_factory=list)


_ASSIGNMENT_RE = re.compile(
    r"(?P<lhs>(?:[A-Za-z](?:_\{?\d+\}?|\d+)?\s*=\s*)+)"
    r"(?P<value>[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*(?:e|E)\s*[-+]?\d+)?)\s*"
    r"(?P<unit>J/\(kg\*degC\)|kg/m\^3|m/s\^2|m/s|N/C|V/m|uF|nF|pF|mF|uC|nC|pC|mC|mA|kV|mV|kW|mJ|uJ|nJ|mT|degC|degree|degrees|kg|cm|mm|m|s|h|min|%|F|C|A|V|W|J|H|Hz|N|T|Wb|g|L|ohm)?",
    re.IGNORECASE,
)
_SYMBOL_RE = re.compile(r"(?P<base>[A-Za-z])(?:_\{?(?P<index>\d+)\}?|(?P<index_plain>\d+))?")
_POINT_DISTANCE_RE = re.compile(
    r"(?P<segment>[A-Z]{2})\s*=\s*(?P<value>[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*(?:e|E)\s*[-+]?\d+)?)\s*(?P<unit>m|cm|mm|km)",
    re.IGNORECASE,
)


def extract_question(problem_text: str, settings=None) -> CompatExtraction:
    text = normalize_notation_text(problem_text)
    quantities: list[CompatQuantity] = []
    relations: list[CompatRelation] = []

    for match in _ASSIGNMENT_RE.finditer(text):
        unit = _normalize_unit(match.group("unit"))
        value = _parse_number(match.group("value"))
        symbols = [part.strip() for part in match.group("lhs").split("=") if part.strip()]
        extracted: list[CompatQuantity] = []
        for symbol in symbols:
            base, index = _split_indexed_symbol(symbol)
            dimension = resolve_canonical_dimension(symbol, unit, text)
            quantity = CompatQuantity(
                name=f"{dimension}_{index}" if dimension and index else dimension or symbol.lower(),
                symbol=symbol,
                base_symbol=base,
                index=index,
                value=value,
                unit=unit,
                dimension=dimension,
                evidence=match.group(0),
                confidence=0.9,
            )
            quantities.append(quantity)
            extracted.append(quantity)
        if len(extracted) > 1:
            relations.append(
                CompatRelation(
                    type="equal_quantities",
                    symbols=[item.symbol or item.name for item in extracted],
                    value=value,
                    unit=unit,
                    evidence=match.group(0),
                )
            )

    quantities.extend(_phrase_quantities(text, quantities))
    quantities.extend(_point_distance_quantities(text, quantities))
    relations.extend(_circuit_relations(text, quantities))
    vector_groups = _vector_groups(text)
    return CompatExtraction(
        question_text=text,
        quantities=_dedupe_quantities(quantities),
        relations=relations,
        vector_contribution_groups=vector_groups,
    )


def normalize_notation_text(text: str) -> str:
    normalized = normalize_question(text).replace("\\Omega", "ohm").replace("Ω", "ohm")
    normalized = normalized.replace("×10^", "e").replace("× 10^", "e")
    normalized = normalized.replace("×10", "e").replace("× 10", "e")
    normalized = normalized.replace("∗10^", "e").replace("·10^", "e")
    normalized = normalized.replace("⁻", "-").replace("−", "-")
    return normalized


def resolve_canonical_dimension(symbol: str, unit: str | None, context: str | None = None) -> str | None:
    unit_key = (unit or "").lower()
    base, _ = _split_indexed_symbol(symbol)
    symbol_key = base.lower()
    by_unit = {
        "n/c": "electric_field",
        "v/m": "electric_field",
        "j": "energy",
        "mj": "energy",
        "uj": "energy",
        "nj": "energy",
        "s": "time",
        "min": "time",
        "h": "time",
        "m": "length",
        "cm": "length",
        "mm": "length",
        "km": "length",
        "m/s": "speed",
        "v": "voltage",
        "mv": "voltage",
        "kv": "voltage",
        "a": "current",
        "ma": "current",
        "ua": "current",
        "ohm": "resistance",
        "f": "capacitance",
        "uf": "capacitance",
        "nf": "capacitance",
        "pf": "capacitance",
        "c": "charge",
        "uc": "charge",
        "nc": "charge",
        "pc": "charge",
        "mc": "charge",
        "n": "force",
        "t": "magnetic_field",
        "mt": "magnetic_field",
        "w": "power",
        "kw": "power",
    }
    if unit_key in by_unit:
        return by_unit[unit_key]
    by_symbol = {
        "e": "energy",
        "t": "time",
        "s": "length",
        "d": "length",
        "r": "resistance",
        "i": "current",
        "u": "voltage",
        "v": "voltage",
        "c": "capacitance",
        "q": "charge",
        "b": "magnetic_field",
        "p": "power",
        "f": "force",
    }
    return by_symbol.get(symbol_key)


def _phrase_quantities(text: str, existing: list[CompatQuantity]) -> list[CompatQuantity]:
    patterns = [
        (r"across a (?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>V|mV|kV) (?:battery|source|power supply)", "voltage", "V"),
        (r"(?:a charge of|with charge|has charge) (?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>C|uC|nC|pC|mC)", "charge", "Q"),
        (r"moves at (?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>m/s)", "speed", "v"),
        (r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>T|mT) magnetic field", "magnetic_field", "B"),
        (r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>m|cm|mm|km) away", "length", "d"),
    ]
    found: list[CompatQuantity] = []
    for pattern, dimension, symbol in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            unit = _normalize_unit(match.group("unit"))
            quantity = CompatQuantity(
                name=dimension,
                symbol=symbol,
                base_symbol=symbol,
                value=_parse_number(match.group("value")),
                unit=unit,
                dimension=dimension,
                evidence=match.group(0),
                confidence=0.75,
            )
            if not _is_duplicate(quantity, [*existing, *found]):
                found.append(quantity)
    return found


def _point_distance_quantities(text: str, existing: list[CompatQuantity]) -> list[CompatQuantity]:
    found: list[CompatQuantity] = []
    for match in _POINT_DISTANCE_RE.finditer(text):
        segment = match.group("segment").upper()
        unit = _normalize_unit(match.group("unit"))
        quantity = CompatQuantity(
            name="length",
            symbol=segment,
            base_symbol=segment,
            value=_parse_number(match.group("value")),
            unit=unit,
            dimension="length",
            evidence=match.group(0),
            confidence=0.85,
        )
        if not _is_duplicate(quantity, [*existing, *found]):
            found.append(quantity)
    return found


def _circuit_relations(text: str, quantities: list[CompatQuantity]) -> list[CompatRelation]:
    relations: list[CompatRelation] = []
    explicit = [q.symbol for q in quantities if q.symbol]
    indexed = [symbol for symbol in explicit if symbol and _split_indexed_symbol(symbol)[1] is not None]
    patterns = [
        (r"(?P<a>[A-Za-z]\d?)\s+and\s+(?P<b>[A-Za-z]\d?)\s+are\s+(?:connected\s+)?in\s+parallel", "connected_in_parallel"),
        (r"(?P<a>[A-Za-z]\d?)\s+and\s+(?P<b>[A-Za-z]\d?)\s+are\s+(?:connected\s+)?in\s+series", "connected_in_series"),
        (r"two\s+(?:resistors|capacitors).+?are\s+in\s+parallel", "connected_in_parallel"),
        (r"two\s+(?:resistors|capacitors).+?are\s+in\s+series", "connected_in_series"),
    ]
    for pattern, relation_type in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            symbols = [value for value in (match.groupdict().get("a"), match.groupdict().get("b")) if value]
            if not symbols and len(indexed) >= 2:
                symbols = indexed[:2]
            if symbols:
                relations.append(CompatRelation(type=relation_type, symbols=symbols, evidence=match.group(0)))
    return relations


def _vector_groups(text: str) -> list[CompatVectorContributionGroup]:
    same_direction_match = re.search(
        r"same direction.*?magnitudes? of (?P<a>[-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*N\s+and\s+(?P<b>[-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*N",
        text,
        flags=re.IGNORECASE,
    )
    if same_direction_match is not None:
        return [
            CompatVectorContributionGroup(
                count=2,
                quantity_dimension="force",
                relation="same_direction",
                evidence=same_direction_match.group(0),
            )
        ]

    match = re.search(
        r"two equal (?P<kind>electric fields|forces|vectors|contributions)(?: each)? of magnitude (?P<symbol>[A-Za-z]\d?)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return []
    kind = match.group("kind").lower()
    dimension = "electric_field" if "electric field" in kind else "force" if "force" in kind else None
    tail = text[match.end():]
    angle_match = re.search(r"(?P<angle>\d+(?:\.\d+)?)\s*degrees?", tail, flags=re.IGNORECASE)
    angle = float(angle_match.group("angle")) if angle_match else None
    return [
        CompatVectorContributionGroup(
            count=2,
            quantity_dimension=dimension,
            magnitude_symbol=match.group("symbol"),
            angle_between_deg=angle,
            evidence=match.group(0),
        )
    ]


def _split_indexed_symbol(symbol: str) -> tuple[str, str | None]:
    match = _SYMBOL_RE.fullmatch(symbol.strip())
    if match is None:
        return symbol.strip(), None
    return match.group("base"), match.group("index") or match.group("index_plain")


def _normalize_unit(unit: str | None) -> str | None:
    return unit.replace("µ", "u").replace("μ", "u") if unit else None


def _parse_number(text: str) -> float:
    normalized = text.replace(" ", "")
    return float(normalized)


def _dedupe_quantities(quantities: list[CompatQuantity]) -> list[CompatQuantity]:
    deduped: list[CompatQuantity] = []
    for quantity in quantities:
        if not _is_duplicate(quantity, deduped):
            deduped.append(quantity)
    return deduped


def _is_duplicate(candidate: CompatQuantity, existing: list[CompatQuantity]) -> bool:
    return any(
        candidate.symbol == item.symbol
        and candidate.dimension == item.dimension
        and candidate.value == item.value
        and candidate.unit == item.unit
        for item in existing
    )


__all__ = [
    "CompatExtraction",
    "CompatQuantity",
    "CompatRelation",
    "CompatVectorContributionGroup",
    "extract_question",
    "extract_type2",
    "normalize_notation_text",
    "resolve_canonical_dimension",
]
