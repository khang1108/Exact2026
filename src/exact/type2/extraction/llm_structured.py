from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact.config import Settings, get_settings
from exact.llm_client import LLMClient, build_json_client_from_settings
from exact.prompts.prompts import Type2JsonFewShotPoTPrompt
from exact.type2.extraction.extractor import normalize_question


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
    formula_ids_used: list[str] = Field(default_factory=list)

    @field_validator("code")
    @classmethod
    def code_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("code must not be empty")
        return value


class FormulaChoiceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formula_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FinalExplanationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str
    premises: list[str] = Field(default_factory=list)
    cot: list[str] = Field(default_factory=list)


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
    formula_context: str = "",
    client: JsonClient | None = None,
    settings: Settings | None = None,
) -> PotCodeSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    raw = client.complete_json_sync(
        messages=_build_pot_messages(question, explanation, formula_context),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return PotCodeSpec.model_validate(raw)


def select_formula_ids(
    question: str,
    extraction_summary: str,
    formula_summaries: list[dict[str, Any]],
    client: JsonClient | None = None,
    settings: Settings | None = None,
) -> FormulaChoiceSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    raw = client.complete_json_sync(
        messages=_build_formula_selection_messages(question, extraction_summary, formula_summaries),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return FormulaChoiceSpec.model_validate(raw)


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


def generate_final_explanation(
    question: str,
    answer: str,
    unit: str | None,
    formula_context: str,
    code_explanation: str,
    formula_ids_used: list[str],
    client: JsonClient | None = None,
    settings: Settings | None = None,
) -> FinalExplanationSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    raw = client.complete_json_sync(
        messages=_build_final_explanation_messages(
            question,
            answer,
            unit,
            formula_context,
            code_explanation,
            formula_ids_used,
        ),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return FinalExplanationSpec.model_validate(raw)


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


def _build_formula_selection_messages(
    question: str,
    extraction_summary: str,
    formula_summaries: list[dict[str, Any]],
):
    return [
        {
            "role": "system",
            "content": (
                "You select physics formula IDs from a provided formula bank. "
                "Return JSON only with keys formula_ids and notes. "
                "Do not invent formula IDs. Prefer formulas whose variables and conditions match."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n"
                f"{question}\n\n"
                "Extracted variables:\n"
                f"{extraction_summary}\n\n"
                "Available formula bank entries:\n"
                f"{formula_summaries}\n\n"
                "Return the best formula_ids in priority order."
            ),
        },
    ]


def _build_pot_messages(question: str, explanation: str, formula_context: str = ""):
    few_shot = Type2JsonFewShotPoTPrompt.examples()
    return [
        {
            "role": "system",
            "content": (
                "You write a short Python program to solve a physics question. "
                "Return JSON only with keys code, explanation, answer_unit, formula_ids_used. "
                "The code must define ans = <numeric result> and ans_unit = <unit string>. "
                "Use pint for units and sympy only if needed. Do not print. "
                "Use the supplied formula bank context when it applies, and check units before finalizing. "
                "For vector quantities such as electric force/field, never add magnitudes as scalars unless "
                "the directions are explicitly the same. If geometry is given, compute components or use the "
                "matching resultant/vector formula from the formula context."
            ),
        },
        {
            "role": "user",
            "content": (
                "Solve the following question using Python.\n"
                "Include only code inside a single Python code block in the JSON field `code`.\n"
                "Question:\n"
                f"{question}\n\n"
                f"Failure/context:\n{explanation}\n\n"
                f"Formula bank context:\n{formula_context or 'No formula context available.'}\n\n"
                f"{few_shot}\n\n"
                "Checklist:\n"
                "- Verify that extracted variables match the requested target.\n"
                "- Use a formula only when its conditions match the question.\n"
                "- Convert units consistently before computing.\n"
                "- For net electric force/field in a triangle or angled geometry, account for vector directions.\n"
                "- In an equilateral triangle, two equal forces on one vertex charge have a 60 degree included angle, so the resultant magnitude is sqrt(3) times one force.\n"
                "- Return JSON strings using escaped newlines; do not use Python triple quotes inside JSON.\n"
                "- Do not import numpy.\n"
                "- Define ans as the final numeric magnitude.\n"
                "- Define ans_unit as the final unit string.\n"
                "- Set formula_ids_used to IDs from the supplied formula context."
            ),
        },
    ]


def _build_final_explanation_messages(
    question: str,
    answer: str,
    unit: str | None,
    formula_context: str,
    code_explanation: str,
    formula_ids_used: list[str],
):
    return [
        {
            "role": "system",
            "content": (
                "You produce concise evidence for a verified physics answer. "
                "Return JSON only with keys explanation, premises, cot. "
                "Do not change the answer. Ground the explanation in the supplied formulas."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Verified answer: {answer}{(' ' + unit) if unit else ''}\n\n"
                f"Formula IDs used: {formula_ids_used}\n\n"
                f"Formula context:\n{formula_context}\n\n"
                f"Code explanation:\n{code_explanation}\n\n"
                "Write a short explanation and evidence premises."
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
