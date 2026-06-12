from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact.config import Settings, get_settings
from exact.llm_client import build_json_client_from_settings
from exact.prompts.prompts import Type2JsonFewShotPoTPrompt
from exact.type2.extraction.extractor import normalize_question


class JsonClient(Protocol):
    def complete_json_sync(
        self,
        messages,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]: ...

    def complete_json_batch_sync(
        self,
        messages,
        n: int,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> list[dict[str, Any]]: ...


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


class Type2QuestionKindSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None

    @field_validator("kind")
    @classmethod
    def kind_must_be_supported(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"numerical", "conceptual", "mixed"}:
            raise ValueError("kind must be one of: numerical, conceptual, mixed")
        return normalized


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
    missing_variables: list[str] = Field(default_factory=list)
    solution_plan: list[str] = Field(default_factory=list)
    confidence: float | None = None
    notes: list[str] = Field(default_factory=list)


class FinalExplanationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str
    premises: list[str] = Field(default_factory=list)
    cot: list[str] = Field(default_factory=list)


class DirectAnswerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    unit: str | None = None
    explanation: str
    premises: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


def build_llm_json_client(settings: Settings | None = None) -> JsonClient | None:
    settings = settings or get_settings()
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
        max_tokens=settings.type2_extraction_max_tokens,
    )
    spec = Type2ExtractionSpec.model_validate(raw)
    spec.notes.append(f"normalized_question={normalize_question(question)}")
    return spec


def classify_question_kind_with_llm(
    question: str,
    client: JsonClient | None = None,
    settings: Settings | None = None,
) -> Type2QuestionKindSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    raw = client.complete_json_sync(
        messages=_build_question_kind_messages(question),
        temperature=settings.llm_temperature,
        max_tokens=settings.type2_question_kind_max_tokens,
    )
    return Type2QuestionKindSpec.model_validate(raw)


def generate_pot_code(
    question: str,
    explanation: str,
    formula_context: str = "",
    client: JsonClient | None = None,
    settings: Settings | None = None,
    debug_metadata: dict[str, Any] | None = None,
) -> PotCodeSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    messages = _build_pot_messages(question, explanation, formula_context)
    _log_type2_prompt(
        settings,
        stage="pot_code",
        question=question,
        messages=messages,
        max_tokens=settings.type2_pot_code_max_tokens,
        metadata=debug_metadata,
    )
    raw = client.complete_json_sync(
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.type2_pot_code_max_tokens,
    )
    return _validate_pot_code_spec(raw)


def generate_pot_code_candidates(
    question: str,
    explanation: str,
    formula_context: str = "",
    candidate_count: int = 1,
    client: JsonClient | None = None,
    settings: Settings | None = None,
    temperature: float | None = None,
    debug_metadata: dict[str, Any] | None = None,
) -> list[PotCodeSpec] | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    candidate_count = max(1, int(candidate_count))
    messages = _build_pot_messages(question, explanation, formula_context)
    metadata = dict(debug_metadata or {})
    metadata["candidate_count"] = candidate_count
    _log_type2_prompt(
        settings,
        stage="pot_code_batch" if candidate_count > 1 else "pot_code",
        question=question,
        messages=messages,
        max_tokens=settings.type2_pot_code_max_tokens,
        metadata=metadata,
    )
    effective_temperature = settings.llm_temperature if temperature is None else temperature

    batch_method = getattr(client, "complete_json_batch_sync", None)
    if callable(batch_method):
        raw_items = batch_method(
            messages=messages,
            n=candidate_count,
            temperature=effective_temperature,
            max_tokens=settings.type2_pot_code_max_tokens,
        )
        return [_validate_pot_code_spec(raw) for raw in raw_items]

    candidates: list[PotCodeSpec] = []
    for _ in range(candidate_count):
        raw = client.complete_json_sync(
            messages=messages,
            temperature=effective_temperature,
            max_tokens=settings.type2_pot_code_max_tokens,
        )
        candidates.append(_validate_pot_code_spec(raw))
    return candidates


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
        max_tokens=settings.type2_formula_selection_max_tokens,
    )
    return FormulaChoiceSpec.model_validate(raw)


def repair_pot_code(
    question: str,
    original_code: str,
    error_message: str,
    client: JsonClient | None = None,
    settings: Settings | None = None,
    debug_metadata: dict[str, Any] | None = None,
) -> PotCodeSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    messages = _build_repair_messages(question, original_code, error_message)
    _log_type2_prompt(
        settings,
        stage="pot_repair",
        question=question,
        messages=messages,
        max_tokens=settings.type2_pot_repair_max_tokens,
        metadata=debug_metadata,
    )
    raw = client.complete_json_sync(
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.type2_pot_repair_max_tokens,
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
        max_tokens=settings.type2_final_explanation_max_tokens,
    )
    return _validate_final_explanation_spec(raw)


def generate_direct_answer(
    question: str,
    failure_context: str,
    client: JsonClient | None = None,
    settings: Settings | None = None,
    temperature: float | None = None,
) -> DirectAnswerSpec | None:
    settings = settings or get_settings()
    client = client or build_llm_json_client(settings)
    if client is None:
        return None

    raw = client.complete_json_sync(
        messages=_build_direct_answer_messages(question, failure_context),
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.type2_agent_loop_max_tokens,
    )
    normalized = _normalize_direct_answer_raw(raw)
    return DirectAnswerSpec.model_validate(normalized)


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


def _build_question_kind_messages(question: str):
    return [
        {
            "role": "system",
            "content": (
                "You classify the answer mode of a Type 2 physics question. "
                "Return JSON only with keys kind, confidence, reason. "
                "kind must be exactly one of: numerical, conceptual, mixed. "
                "Use numerical when the expected final answer is a number, unit-bearing value, "
                "or numeric expression. Use conceptual when the expected final answer is qualitative "
                "text, a choice of behavior, a unit-name explanation, or a descriptive statement. "
                "Use mixed when the problem asks for both numeric work and qualitative reasoning."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}",
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
                "Return JSON only with keys formula_ids, missing_variables, solution_plan, confidence, notes. "
                "Do not invent formula IDs. Prefer formulas whose variables and conditions match. "
                "The solution_plan must be short, concrete, and usable by a teacher explaining the solution."
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
                "Return the best formula_ids in priority order, list any missing_variables, "
                "and include a concise step-by-step solution_plan."
            ),
        },
    ]


def _build_pot_messages(question: str, explanation: str, formula_context: str = ""):
    few_shot = Type2JsonFewShotPoTPrompt.examples()
    return [
        {
            "role": "system",
            "content": (
                "You solve a physics problem by writing one short Python/Pint program. "
                "Return strict JSON only with exactly these keys: code, explanation, answer_unit, formula_ids_used. "
                "No Markdown, no code fences, no extra prose. "
                "`code` must be a JSON string with escaped newlines (\\n), never literal newlines; do not use Python triple quotes. "
                "The program must define ans as the final numeric magnitude and ans_unit as the final unit string. "
                "Use pint for units and sympy only if genuinely needed; Do not import numpy and do not print. "
                "Follow the Formula selector plan when present, and use only formulas whose IDs appear in the supplied context. "
                "Set formula_ids_used to a JSON array of copied formula IDs. "
                "Preserve distinct physical objects and roles: q1, q2, q3, source charge, target charge, and test charge "
                "must not be collapsed unless the problem explicitly says they are identical; keep them separate in code, "
                "and keep the source and target charges separate. "
                "For vector force/field questions, compute components or use a retrieved resultant/vector formula; "
                "never add magnitudes as scalars unless directions are explicitly the same. "
                "Before finalizing, check that units convert to answer_unit and that the result answers the requested target."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n"
                f"{question}\n\n"
                f"Solver context or previous failure:\n{explanation}\n\n"
                f"Formula bank context:\n{formula_context or 'No formula context available.'}\n\n"
                "Strict output contract:\n"
                "- Return one JSON object only.\n"
                "- code: complete Python program as one JSON string using escaped newlines.\n"
                "- explanation: one concise sentence describing the physics computation, not implementation details.\n"
                "- answer_unit: unit string assigned to ans_unit.\n"
                "- formula_ids_used: array of formula IDs copied from the supplied context.\n\n"
                "Few-shot JSON examples:\n"
                f"{few_shot}\n\n"
                "Now solve the question. The output must be valid for strict json.loads without cleanup."
            ),
        },
    ]


def _log_type2_prompt(
    settings: Settings,
    *,
    stage: str,
    question: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not settings.type2_debug_log_pot_prompts:
        return

    log_path = Path(settings.type2_debug_pot_prompt_log_path)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "question": question,
        "temperature": settings.llm_temperature,
        "max_tokens": max_tokens,
        "metadata": metadata or {},
        "messages": messages,
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


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
                "You are an encouraging, experienced, and highly clear physics teacher. "
                "Produce a warm, pedagogical, yet highly concise explanation (max 3-4 sentences) "
                "for a verified physics answer. Start with the physical intuition in a very natural, "
                "human-like tone, then connect it to the math/formulas used. "
                "Return JSON only with keys explanation, premises, cot. "
                "Do not change the answer. Ground the explanation in the supplied formulas. "
                "CRITICAL: Avoid LaTeX formatting, LaTeX wrappers like \\( \\), and LaTeX backslash commands like \\frac. "
                "Write all equations and mathematical terms in clean, standard plain text (e.g., use 'W = 1/2 * L * I^2' "
                "instead of LaTeX fraction and symbol syntax). Do not output any backslashes in your text. "
                "Do not mention Python, code, JSON, or implementation details; explain the physics solution as a teacher."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Verified answer: {answer}{(' ' + unit) if unit else ''}\n\n"
                f"Formula IDs used: {formula_ids_used}\n\n"
                f"Formula context:\n{formula_context}\n\n"
                f"Structured solution context:\n{code_explanation}\n\n"
                "Write a short explanation and evidence premises. Prefer this shape: physical intuition, "
                "formula choice, substitution/numeric work, final answer."
            ),
        },
    ]


def _build_direct_answer_messages(question: str, failure_context: str):
    return [
        {
            "role": "system",
            "content": (
                "You are an expert physics solver for the Type 2 pipeline. "
                "The normal deterministic or Program-of-Thought path did not finish cleanly, so produce the best direct answer. "
                "Return JSON only with keys answer, unit, explanation, premises, confidence. "
                "For numeric answers, put only the numeric magnitude in answer and the unit string in unit. "
                "For conceptual answers, put a short phrase in answer and null in unit. "
                "Use your physics knowledge to infer the correct method and solve from the question when possible. "
                "Do not return a refusal or a guardrail-style response just because some structured context is missing. "
                "If a detail is ambiguous, make the most likely assumption and state it briefly in the explanation. "
                "Keep explanation concise. Do not mention pipeline internals, code, JSON, or failures."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Previous failure context:\n{failure_context}\n\n"
                "Return one JSON object only. Solve the problem instead of describing why the pipeline failed."
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


def _normalize_direct_answer_raw(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    normalized = dict(raw)
    answer = normalized.get("answer")
    normalized["answer"] = _stringify_value(answer).strip() if answer is not None else ""
    explanation = normalized.get("explanation")
    normalized["explanation"] = (
        _stringify_value(explanation).strip()
        if explanation is not None
        else "Generated a direct fallback answer."
    )
    unit = normalized.get("unit")
    normalized["unit"] = str(unit).strip() if unit is not None and str(unit).strip() else None
    normalized["premises"] = _string_list(normalized.get("premises"))
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
