from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact.config import Settings, get_settings
from exact.type2.extraction.llm_structured import JsonClient, build_llm_json_client


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
    # Prefix routing mapping to kinds if possible, else heuristic keywords
    if question_id:
        if "LD" in question_id or "DT" in question_id:
            return "numerical"
        if "TD" in question_id:
            return "numerical"
        if "NL" in question_id:
            return "numerical"

    text = question_text.lower()
    # If question asks to explain or why, classify as conceptual
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
