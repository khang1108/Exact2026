from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact.config import Settings, get_settings
from exact.type2.domains.ddt.extraction import extract_ddt_heuristic
from exact.type2.domains.nl_energy.classifier import classify_nl_energy_family
from exact.type2.domains.thcb.extraction import extract_thcb_heuristic
from exact.type2.extraction.llm_structured import JsonClient, build_llm_json_client


PipelineDomain = Literal["LD", "TD", "THCB", "DDT", "CH", "DT", "NL_ENERGY", "CIRCUIT", "ELECTROMAGNETISM", "GENERIC"]
Type2Domain = Literal["numerical", "conceptual", "mixed"]
VALID_DOMAINS = {"numerical", "conceptual", "mixed"}


@dataclass(frozen=True)
class DomainRoute:
    domain: Type2Domain
    source: str
    confidence: float | None = None
    reason: str | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DomainRouteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None

    @field_validator("domain")
    @classmethod
    def domain_must_be_supported(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(VALID_DOMAINS)}")
        return normalized


def route_domain(question_id: str | None, question_text: str) -> PipelineDomain:
    """Route a Type 2 question to a specific domain pipeline from content only."""

    return route_domain_heuristic(question_id, question_text)


def route_domain_with_metadata(
    question_id: str | None,
    question_text: str,
    settings: Settings | None = None,
    client: JsonClient | None = None,
) -> DomainRoute:
    """Use the configured LLM to route Type 2 domain ownership, then fallback."""

    settings = settings or get_settings()
    if settings.type2_use_llm_domain_routing:
        route = _try_llm_domain_route(question_text, settings=settings, client=client)
        if route is not None:
            return route

    fallback = route_question_kind_heuristic(question_id, question_text)
    reason = "LLM domain routing disabled" if not settings.type2_use_llm_domain_routing else "LLM route unavailable"
    return DomainRoute(domain=fallback, source="heuristic", fallback_reason=reason)


def route_domain_heuristic(question_id: str | None, question_text: str) -> PipelineDomain:
    """Route Type 2 pipeline domain from dataset ID when present, then text."""

    id_route = _route_from_question_id(question_id)
    if id_route is not None:
        return id_route

    if _looks_like_ch_question(question_text):
        return "CH"

    if _looks_like_thcb_question(question_text):
        return "THCB"

    if _looks_like_td_question(question_text):
        return "TD"

    if _looks_like_ddt_question(question_text):
        return "DDT"

    if _looks_like_nl_energy_question(question_text):
        return "NL_ENERGY"

    if _looks_like_ld_question(question_text):
        return "LD"

    text = question_text.lower()
    nl_keywords = [
        "capacitor energy",
        "electric field energy",
        "magnetic field energy",
        "inductor energy",
        "lc circuit",
        "oscillation",
        "maximum current",
        "maximum charge",
        "si unit of energy",
        "energy versus current",
        "energy versus capacitance",
    ]

    for keyword in nl_keywords:
        if keyword in text:
            return "NL_ENERGY"

    if re.search(r"\b[iqvu]\(t\).*?(sin|cos)", text):
        return "NL_ENERGY"

    circuit_keywords = [
        "ohm",
        "resistor",
        "resistance",
        "series circuit",
        "parallel circuit",
        "dc circuit",
        "ac circuit",
        "rms current",
        "rms voltage",
        "impedance",
        "reactance",
        "power factor",
        "kirchhoff",
        "wheatstone bridge",
        "transformer",
    ]
    if _has_any(text, circuit_keywords):
        return "CIRCUIT"

    electromagnetism_keywords = [
        "magnetic field",
        "magnetic flux",
        "faraday",
        "lenz",
        "solenoid",
        "inductor",
        "inductance",
        "motional emf",
        "electromagnetic induction",
        "flux linkage",
        "magnetic moment",
        "lorentz force",
        "hall effect",
    ]
    if _has_any(text, electromagnetism_keywords):
        return "ELECTROMAGNETISM"

    return "GENERIC"


def _route_from_question_id(question_id: str | None) -> PipelineDomain | None:
    if question_id is None:
        return None
    normalized = question_id.strip().upper().replace("-", "_")
    if not normalized:
        return None

    for prefix, domain in (
        ("THCB", "THCB"),
        ("DDT", "DDT"),
        ("CH", "CH"),
        ("DT", "DT"),
        ("NL_ENERGY", "NL_ENERGY"),
        ("NL", "NL_ENERGY"),
        ("LD", "LD"),
        ("TD", "TD"),
    ):
        if (
            normalized == prefix
            or normalized.startswith(f"{prefix}_")
            or re.match(rf"^{prefix}\d", normalized)
            or re.search(rf"(?:^|_){prefix}\d", normalized)
        ):
            return cast(PipelineDomain, domain)
    return None


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _looks_like_thcb_question(question_text: str) -> bool:
    contract = extract_thcb_heuristic(question_text)
    if contract.family == "UNKNOWN" or contract.target == "unknown":
        return False

    if contract.family in {"MEASUREMENT_ERROR", "ERROR_PROPAGATION"}:
        return bool(contract.quantities or contract.readings or contract.requested_outputs)

    if contract.family in {"PARALLEL_CIRCUIT", "SIMPLE_CIRCUIT", "CONCEPTUAL_CIRCUIT"}:
        return bool(contract.quantities or contract.relation or contract.requested_outputs)

    return False


def _looks_like_ddt_question(question_text: str) -> bool:
    contract = extract_ddt_heuristic(question_text)
    if contract.family == "UNKNOWN":
        return False
    return bool(contract.quantities or contract.relation or contract.target != "unknown")


def _looks_like_nl_energy_question(question_text: str) -> bool:
    return classify_nl_energy_family(question_text) != "UNKNOWN"


def _looks_like_ch_question(question_text: str) -> bool:
    from exact.type2.domains.ch.solver import solve_ch_resonance

    return solve_ch_resonance(None, question_text) is not None


def _looks_like_ld_question(question_text: str) -> bool:
    text = question_text.lower()
    has_charge = any(term in text for term in ("point charge", "charges", "electric charge", "coulomb"))
    has_field_or_potential = any(
        term in text
        for term in (
            "electric field",
            "field intensity",
            "electric potential",
            "electric force",
            "resultant field",
        )
    )
    has_geometry = any(
        term in text
        for term in (
            "midpoint",
            "equilateral",
            "triangle",
            "perpendicular bisector",
            "distance from each charge",
            "equidistant",
            "at point",
        )
    )
    return has_charge and (has_field_or_potential or has_geometry)


def _looks_like_td_question(question_text: str) -> bool:
    text = question_text.lower()
    capacitor_terms = (
        "capacitor",
        "capacitance",
        "dielectric",
        "parallel plate",
        "stored energy",
        "electric field between plates",
        "breakdown voltage",
    )
    if any(term in text for term in capacitor_terms):
        return True

    static_circuit_terms = (
        "dc circuit",
        "series circuit",
        "parallel circuit",
        "ohm's law",
        "kirchhoff",
        "equivalent resistance",
    )
    return any(term in text for term in static_circuit_terms)


def route_question_kind_heuristic(question_id: str | None, question_text: str) -> Type2Domain:
    text = question_text.lower()
    conceptual_keywords = [
        "explain", "why", "describe", "which of", "what is the relationship",
        "conceptual", "theory", "meaning", "define", "statement is correct",
        "is it true", "how does", "what happens"
    ]
    for kw in conceptual_keywords:
        if kw in text:
            return "conceptual"

    return "numerical"


def _try_llm_domain_route(
    question_text: str,
    settings: Settings,
    client: JsonClient | None = None,
) -> DomainRoute | None:
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    try:
        raw = client.complete_json_sync(
            messages=_build_domain_route_messages(question_text),
            temperature=settings.llm_temperature,
            max_tokens=settings.type2_domain_routing_max_tokens,
        )
        spec = DomainRouteSpec.model_validate(raw)
    except Exception:
        return None

    return DomainRoute(
        domain=spec.domain,  # type: ignore[arg-type]
        source="llm",
        confidence=spec.confidence,
        reason=spec.reason,
    )


def _build_domain_route_messages(question_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You classify a Type 2 physics question into one of three categories: "
                "numerical, conceptual, mixed. "
                "Return JSON only with keys domain, confidence, reason. "
                "domain must be exactly one of: numerical, conceptual, mixed. "
                "Use numerical when the expected final answer is a number, unit-bearing value, "
                "or numeric expression. Use conceptual when the expected final answer is qualitative "
                "text, a choice of behavior, a unit-name explanation, or a descriptive statement. "
                "Use mixed when the problem asks for both numeric work and qualitative reasoning. "
                "Classify from the question content only; do not rely on dataset ID prefixes."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question_text}",
        },
    ]
