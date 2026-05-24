"""Type 1 logic pipeline.

The default path is deterministic symbolic reasoning. An injected LLM client is
still supported for experiments and backwards-compatible tests, but production
code should treat that as parser/fallback plumbing rather than the core proof.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from exact.config import Settings, get_settings
from exact.datasets.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.json_utils import parse_json_object
from exact.logic.explain import explain_result, kb_to_fol_like_text
from exact.logic.kb import build_kb_from_parsed_premises
from exact.logic.llm_translator import JsonLLMClient, translate_with_fallback
from exact.logger import get_request_logger
from exact.symbolic_solvers import ForwardChainSolver


class SyncLLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


def run_type1_pipeline(
    request: PredictionRequest,
    llm_client: SyncLLMClient | None = None,
    translator_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Answer a Type 1 logic query with symbolic proof where possible."""

    logger = get_request_logger(
        __name__,
        request_id=request.id,
        task_type=TaskType.TYPE1_LOGIC.value,
    )
    logger.info("Start Type 1 pipeline")

    if llm_client is not None:
        response = _run_injected_llm_path(request, llm_client)
        if response is not None:
            logger.info("Injected LLM response accepted")
            return response
        logger.info("Injected LLM response invalid; falling back to symbolic path")

    settings = settings or get_settings()
    premises = request.premises_nl or []
    parsed_premises, query, translation_warnings = translate_with_fallback(
        premises=premises,
        question=request.question,
        llm_client=translator_client,
        settings=settings,
    )
    kb = build_kb_from_parsed_premises(
        premises,
        parsed_premises,
        parser_version="llm_translator_v1" if settings.llm_base_url else "heuristic_horn_v1",
        extra_warnings=translation_warnings,
    )
    result = ForwardChainSolver().solve(kb, query.claim)
    explanation, cot, cited_premises = explain_result(result, kb)

    confidence = {
        "Yes": 0.78,
        "No": 0.76,
        "Unknown": 0.35,
    }[result.label]

    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=QuestionType.YES_NO_UNCERTAIN,
        answer=result.label,
        explanation=explanation,
        fol=kb_to_fol_like_text(kb) or None,
        cot=cot,
        premises=cited_premises,
        confidence=confidence,
        error="; ".join(result.warnings) if result.warnings else None,
    )


def _run_injected_llm_path(
    request: PredictionRequest,
    llm_client: SyncLLMClient,
) -> PredictionResponse | None:
    prompt = _build_fallback_prompt(request)
    try:
        data = parse_json_object(llm_client.generate(prompt))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    answer = _normalize_answer(str(data.get("answer", "")))
    if not answer:
        return None

    confidence = data.get("confidence", 0.25)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.25

    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=QuestionType.YES_NO_UNCERTAIN,
        answer=answer,
        explanation=str(data.get("explanation") or "Answered by injected LLM fallback."),
        fol=data.get("fol"),
        cot=_as_string_list(data.get("cot")),
        premises=_as_string_list(data.get("premises")),
        confidence=max(0.0, min(1.0, confidence)),
        error=None,
    )


def _build_fallback_prompt(request: PredictionRequest) -> str:
    premises = "\n".join(f"P{idx + 1}: {premise}" for idx, premise in enumerate(request.premises_nl or []))
    return (
        "Answer the logic question using only the premises. Return JSON with "
        "answer, explanation, fol, cot, premises, confidence.\n\n"
        f"{premises}\n\nQuestion: {request.question}"
    )


def _normalize_answer(answer: str) -> str:
    value = answer.strip().lower()
    if value in {"yes", "true", "entailed"}:
        return "Yes"
    if value in {"no", "false", "contradicted"}:
        return "No"
    if value in {"unknown", "uncertain", "not enough information"}:
        return "Unknown"
    if len(answer.strip()) == 1 and answer.strip().upper() in {"A", "B", "C", "D"}:
        return answer.strip().upper()
    return answer.strip()


def _as_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
