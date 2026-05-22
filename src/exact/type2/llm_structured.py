from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact.config import Settings, get_settings
from exact.llm_client import LLMClient, build_json_client_from_settings
from exact.type2.extractor import normalize_question


class JsonClient(Protocol):
    def complete_json_sync(
        self,
        messages,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]: ...


class ExtractionQuantitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    unit: str
    evidence: str | None = None

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be empty")
        return value


class Type2ExtractionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(default="numerical")
    target: str | None = None
    quantities: list[ExtractionQuantitySpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PotCodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    explanation: str | None = None
    answer_unit: str | None = None

    @field_validator("code")
    @classmethod
    def code_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("code must not be empty")
        return value


def build_llm_json_client(settings: Settings | None = None) -> JsonClient | None:
    settings = settings or get_settings()
    if settings.mock_llm:
        return None
    try:
        client = build_json_client_from_settings(settings)
    except Exception:
        return None
    return client


def parse_with_llm(
    question: str,
    client: JsonClient | None = None,
    settings: Settings | None = None,
) -> Type2ExtractionSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    raw = client.complete_json_sync(
        messages=_build_extraction_messages(question),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    spec = Type2ExtractionSpec.model_validate(raw)
    spec.notes.append(f"normalized_question={normalize_question(question)}")
    return spec


def generate_pot_code(
    question: str,
    explanation: str,
    client: JsonClient | None = None,
    settings: Settings | None = None,
) -> PotCodeSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    raw = client.complete_json_sync(
        messages=_build_pot_messages(question, explanation),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return PotCodeSpec.model_validate(raw)


def repair_pot_code(
    question: str,
    original_code: str,
    error_message: str,
    client: JsonClient | None = None,
    settings: Settings | None = None,
) -> PotCodeSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    raw = client.complete_json_sync(
        messages=_build_repair_messages(question, original_code, error_message),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return PotCodeSpec.model_validate(raw)


def _build_extraction_messages(question: str):
    return [
        {
            "role": "system",
            "content": (
                "You extract structured physics quantities from a Type 2 educational question. "
                "Return JSON only. Use canonical names such as voltage, current, resistance, "
                "capacitance, charge, energy, electric_field, force, inductance, frequency."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return JSON with keys kind, target, quantities, notes.\n"
                "Each quantity must contain name, value, unit, evidence.\n"
                "If the question is conceptual, set kind to conceptual.\n"
                f"Question: {question}"
            ),
        },
    ]


def _build_pot_messages(question: str, explanation: str):
    return [
        {
            "role": "system",
            "content": (
                "You write a short Python program to solve a physics question. "
                "Return JSON only with keys code, explanation, answer_unit. "
                "The code must define ans = <numeric result> and may define ans_unit. "
                "Use pint for units and sympy only if needed. Do not print."
            ),
        },
        {
            "role": "user",
            "content": (
                "Solve the following question using Python.\n"
                "Include only code inside a single Python code block in the JSON field `code`.\n"
                "Question:\n"
                f"{question}\n\n"
                f"Context:\n{explanation}"
            ),
        },
    ]


def _build_repair_messages(question: str, original_code: str, error_message: str):
    return [
        {
            "role": "system",
            "content": (
                "You repair Python code for a physics question. Return JSON only with keys code, explanation, answer_unit. "
                "Keep the solution short. The code must define ans."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Original code:\n{original_code}\n\n"
                f"Error:\n{error_message}\n\n"
                "Return a repaired Python program only."
            ),
        },
    ]
