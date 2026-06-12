from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact.config import Settings, get_settings
from exact.type2.extraction.llm_structured import JsonClient, build_llm_json_client


Type2Domain = Literal["LD", "TD", "NL_ENERGY", "GENERIC"]
VALID_DOMAINS = {"LD", "TD", "NL_ENERGY", "GENERIC"}


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
        normalized = value.strip().upper()
        if normalized not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(VALID_DOMAINS)}")
        return normalized


def route_domain(question_id: str | None, question_text: str) -> Type2Domain:
    """Route a Type 2 question to a specific domain pipeline based on ID or content."""

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

    fallback = route_domain(question_id, question_text)
    reason = "LLM domain routing disabled" if not settings.type2_use_llm_domain_routing else "LLM route unavailable"
    return DomainRoute(domain=fallback, source="heuristic", fallback_reason=reason)


def route_domain_heuristic(question_id: str | None, question_text: str) -> Type2Domain:
    # Prefix routing
    if question_id:
        if "LD" in question_id or "DT" in question_id:
            return "LD"
        if "TD" in question_id:
            return "TD"
        if "NL" in question_id:
            return "NL_ENERGY"

    # Conservative keyword fallback for NL_ENERGY if no ID is present
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
    
    for kw in nl_keywords:
        if kw in text:
            return "NL_ENERGY"
            
    # Regex fallback for I(t), q(t), U(t) etc with sin/cos
    if re.search(r"\b[iqvu]\(t\).*?(sin|cos)", text):
        return "NL_ENERGY"

    return "GENERIC"


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
                "You route Type 2 physics questions to exactly one solver domain. "
                "Return JSON only with keys domain, confidence, reason. "
                "Allowed domain values: LD, TD, NL_ENERGY, GENERIC. "
                "LD is for electrostatics/vector Coulomb-force or electric-field point-charge problems. "
                "TD is for static capacitor/dielectric/capacitance problems. "
                "NL_ENERGY is for LC oscillations, time-varying capacitor/inductor energy, "
                "energy graph shape, maximum current/charge energy, or conceptual SI energy unit questions. "
                "GENERIC is for all other Type 2 physics problems. "
                "Classify from the question content only; do not rely on dataset ID prefixes."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question_text}",
        },
    ]
