from __future__ import annotations

import json
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
    except Exception as exc:
        print(f"[!] Error building LLM JSON client: {exc}", flush=True)
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
    return _validate_pot_code_spec(raw)


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
    return _validate_pot_code_spec(raw)


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
    return _validate_final_explanation_spec(raw)


def _build_extraction_messages(question: str):
    return [
        {
            "role": "system",
            "content": (
                "You extract structured physics quantities from a Type 2 educational question. "
                "Return JSON only. Use canonical names such as voltage, current, resistance, "
                "capacitance, charge, energy, electric_field, force, inductance, frequency, "
                "speed, mass, density, volume, pressure, temperature, specific_heat_capacity, "
                "heat_of_combustion, efficiency."
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
                "Return strict JSON only with keys code, explanation, answer_unit, formula_ids_used. "
                "Do not wrap the response in markdown or code fences. "
                "The code must be a JSON string, not a nested JSON object. "
                "Inside the code string, escape newlines as \\n; do not emit literal line breaks inside any JSON string value. "
                "The code must define ans = <numeric result> and ans_unit = <unit string>. "
                "Use pint for units and sympy only if needed. Do not print. "
                "Use the supplied formula bank context when it applies, and check units before finalizing. "
                "Prefer formula IDs from the top of the supplied context, but use multiple retrieved IDs when "
                "a multi-step solution needs an intermediate value. "
                "For vector quantities such as electric force/field, never add magnitudes as scalars unless "
                "the directions are explicitly the same. If geometry is given, compute components or use the "
                "matching resultant/vector formula from the formula context. "
                "The field formula_ids_used must be a JSON array of strings copied from the supplied formula context. "
                "When a question contains multiple distinct charges or source/target roles (for example q1, q2, q3, q'), "
                "keep them separate in code; do not collapse them into one generic q unless the problem explicitly says "
                "all charges are identical and interchangeable."
            ),
        },
        {
            "role": "user",
            "content": (
                "Solve the following question using Python.\n"
                "Return one JSON object only.\n"
                "Do not use Markdown code fences anywhere in the response.\n"
                "Question:\n"
                f"{question}\n\n"
                f"Failure/context:\n{explanation}\n\n"
                f"Formula bank context:\n{formula_context or 'No formula context available.'}\n\n"
                f"{few_shot}\n\n"
                "Checklist:\n"
                "- Verify that extracted variables match the requested target.\n"
                "- Use a formula only when its target, required variables, and conditions match the question.\n"
                "- If no single formula directly solves the target, combine the smallest valid chain of supplied formulas.\n"
                "- Convert units consistently before computing.\n"
                "- For geometry, identify whether the problem is a rectangle, triangle, circle, sphere, cylinder, or right triangle before selecting a formula.\n"
                "- For net electric force/field in a triangle or angled geometry, account for vector directions.\n"
                "- In an equilateral triangle, two equal forces on one vertex charge have a 60 degree included angle, so the resultant magnitude is sqrt(3) times one force.\n"
                "- When the problem has multiple distinct charges or labeled roles, keep the source and target charges separate; do not reuse one q variable for all roles.\n"
                "- Return JSON strings using escaped newlines only; do not use Python triple quotes or literal newlines inside any JSON string.\n"
                "- Do not import numpy.\n"
                "- Define ans as the final numeric magnitude.\n"
                "- Define ans_unit as the final unit string.\n"
                "- Set formula_ids_used to a JSON list of IDs from the supplied formula context.\n"
                "- The output must be valid for strict json.loads without manual cleanup."
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
                "You repair Python code for a physics question. Return strict JSON only with keys "
                "code, explanation, answer_unit, formula_ids_used. Do not return Markdown or code fences. "
                "Keep the solution short. The code must define ans and ans_unit. "
                "The code field must be a JSON string with escaped newlines (\\n), not literal line breaks. "
                "formula_ids_used must be a JSON array of strings, never a string."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Original code:\n{original_code}\n\n"
                f"Error:\n{error_message}\n\n"
                "Return one JSON object only. Put the full repaired Python program in the `code` string. "
                "Make the JSON parseable by strict json.loads without preprocessing."
            ),
        },
    ]


def _validate_pot_code_spec(raw: dict[str, Any]) -> PotCodeSpec:
    normalized = _normalize_pot_code_raw(raw)
    try:
        return PotCodeSpec.model_validate(normalized)
    except Exception:
        recovered = _recover_malformed_pot_code(normalized)
        if recovered is None:
            raise
        return PotCodeSpec.model_validate(recovered)

def _validate_final_explanation_spec(raw: dict[str, Any]) -> FinalExplanationSpec:
    normalized = _normalize_final_explanation_raw(raw)
    return FinalExplanationSpec.model_validate(normalized)


def _normalize_final_explanation_raw(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    normalized = dict(raw)
    explanation = normalized.get("explanation")
    if not isinstance(explanation, str):
        normalized["explanation"] = _stringify_value(explanation) if explanation is not None else "Generated explanation."
    normalized["premises"] = _string_list(normalized.get("premises"))
    normalized["cot"] = _string_list(normalized.get("cot"))
    return normalized


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_stringify_value(item) for item in value]
    return [_stringify_value(value)]


def _stringify_value(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _normalize_pot_code_raw(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    normalized = dict(raw)
    formula_ids = normalized.get("formula_ids_used")
    if isinstance(formula_ids, str):
        normalized["formula_ids_used"] = _parse_formula_ids_used(formula_ids)
    return normalized


def _parse_formula_ids_used(value: str) -> list[str]:
    text = value.strip()
    if not text or any(token in text for token in ("=", "*", "/", "+", "-", "(", ")")):
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _recover_malformed_pot_code(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    code = raw.get("code")
    if isinstance(code, str) and code.strip():
        return None

    recovered_code = _find_code_like_text(raw)
    if recovered_code is None:
        return None

    formula_ids = raw.get("formula_ids_used")
    return {
        "code": recovered_code,
        "explanation": raw.get("explanation") if isinstance(raw.get("explanation"), str) else "Recovered code from malformed LLM JSON.",
        "answer_unit": raw.get("answer_unit") if isinstance(raw.get("answer_unit"), str) else raw.get("ans_unit") if isinstance(raw.get("ans_unit"), str) else None,
        "formula_ids_used": formula_ids if isinstance(formula_ids, list) else [],
    }


def _find_code_like_text(raw: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    for key, value in raw.items():
        if isinstance(key, str):
            candidates.append(key)
        if isinstance(value, str):
            candidates.append(value)
    for candidate in candidates:
        code = _clean_code_candidate(candidate)
        if _looks_like_python_solution(code):
            return code
    return None


def _clean_code_candidate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("python"):
            stripped = stripped[6:].strip()
    stripped = stripped.strip("`").strip()
    return stripped


def _looks_like_python_solution(text: str) -> bool:
    return "ans" in text and ("import " in text or "ureg" in text) and "=" in text
