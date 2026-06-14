from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact.config import Settings, get_settings
from exact.type2.extraction.llm_structured import JsonClient, build_llm_json_client


PipelineDomain = Literal["LD", "TD", "NL_ENERGY", "CIRCUIT", "ELECTROMAGNETISM", "GENERIC"]
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
    """Route Type 2 pipeline domain from question text only."""

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


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


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
