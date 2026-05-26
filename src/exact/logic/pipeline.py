"""Type 1 logic pipeline.

The default path is deterministic symbolic reasoning. An injected LLM client is
still supported for experiments and backwards-compatible tests, but production
code should treat that as parser/fallback plumbing rather than the core proof.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from exact.config import Settings, get_settings
from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.logic.explain import explain_result, kb_to_fol_like_text
from exact.logic.kb import build_kb_from_parsed_premises
from exact.logic.ir import Atom, SolveResult
from exact.logic.kb import KnowledgeBase
from exact.logic.llm_translator import JsonLLMClient, translate_with_fallback
from exact.logic.parser import atom_from_text
from exact.logger import get_request_logger
from exact.symbolic_solvers import ForwardChainSolver


class SyncLLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


_OPTION_RE = re.compile(
    r"(?:^|\n)\s*([A-D])\.\s*(.*?)(?=(?:\n\s*[A-D]\.\s*)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_OPTION_ORDER = ("A", "B", "C", "D")


def extract_options(question: str) -> list[tuple[str, str]]:
    options = [
        (match.group(1).upper(), " ".join(match.group(2).split()))
        for match in _OPTION_RE.finditer(question)
    ]
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for label, text in options:
        if label in seen or not text:
            continue
        seen.add(label)
        deduped.append((label, text))
    return deduped


def strip_options_from_question(question: str) -> str:
    first_option = _OPTION_RE.search(question)
    stem = question[: first_option.start()] if first_option else question
    return " ".join(stem.split())


def build_goals_for_mcq(options: list[tuple[str, str]]) -> list[tuple[str, Atom]]:
    return [(label, atom_from_text(text)) for label, text in options]


def evaluate_mcq_options(
    kb: KnowledgeBase,
    goals: list[tuple[str, Atom]],
    solver: ForwardChainSolver | None = None,
) -> dict[str, SolveResult]:
    solver = solver or ForwardChainSolver()
    return {label: solver.solve(kb, goal) for label, goal in goals}


def decide_mcq_winner(results: dict[str, SolveResult], question_stem: str) -> str:
    if not results:
        return "A"

    ordered_labels = [label for label in _OPTION_ORDER if label in results]
    entailed = [label for label in ordered_labels if results[label].label == "Yes"]
    if len(entailed) == 1:
        return entailed[0]

    candidates = entailed or ordered_labels
    rank = {"Yes": 0, "Unknown": 1, "No": 2}

    def score(label: str) -> tuple[int, int, int]:
        result = results[label]
        proof_size = len(result.supporting_premises) if result.supporting_premises else 999
        return rank.get(result.label, 3), proof_size, _OPTION_ORDER.index(label)

    if entailed and "fewest premises" in question_stem.lower():
        return min(entailed, key=lambda label: (len(results[label].supporting_premises), _OPTION_ORDER.index(label)))

    return min(candidates, key=score)


def run_type1_pipeline(
    request: PredictionRequest,
    llm_client: SyncLLMClient | None = None,
    translator_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
    allow_heuristic_fallback: bool = True,
    question_type: QuestionType | None = None,
) -> PredictionResponse:
    """
    Answer a Type 1 logic query with symbolic proof where possible.

    Args:
        request (PredictionRequest): Đại diện cho request cần xử lý
        llm_client (SyncLLMClient): Đại diện cho LLMClient để gọi API
        translator_client (JsonLLMClient): Client dùng để dịch NL -> Logical Form
        settings (Settings): Các settings cho pipeline
        allow_heuristic_fallback (bool): Có dùng heuristic không

    Returns:
        PredictionResponse: Đại diện cho dự đoán
    """

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
    routed_question_type = question_type or QuestionType.YES_NO_UNCERTAIN
    premises = request.premises_nl or []
    parsed_premises, query, translation_warnings = translate_with_fallback(
        premises=premises,
        question=request.question,
        llm_client=translator_client,
        settings=settings,
        allow_heuristic_fallback=allow_heuristic_fallback,
    )
    kb = build_kb_from_parsed_premises(
        premises,
        parsed_premises,
        parser_version="llm_translator_v1" if settings.llm_base_url else "heuristic_horn_v1",
        extra_warnings=translation_warnings,
    )

    if routed_question_type == QuestionType.MCQ:
        return _run_mcq_path(request, kb, routed_question_type)

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
        question_type=routed_question_type,
        answer=result.label,
        explanation=explanation,
        fol=kb_to_fol_like_text(kb) or None,
        cot=cot,
        premises=cited_premises,
        confidence=confidence,
        error="; ".join(result.warnings) if result.warnings else None,
    )


def _run_mcq_path(
    request: PredictionRequest,
    kb: KnowledgeBase,
    question_type: QuestionType,
) -> PredictionResponse:
    options = extract_options(request.question)
    if not options:
        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE1_LOGIC,
            question_type=question_type,
            answer="A",
            explanation="No multiple-choice options were parsed, so the system returned the default option A.",
            fol=kb_to_fol_like_text(kb) or None,
            cot=["No option labels A-D were available for symbolic evaluation."],
            premises=[],
            confidence=0.05,
            error="MCQ routed but no options were parsed.",
        )

    stem = strip_options_from_question(request.question)
    goals = build_goals_for_mcq(options)
    results = evaluate_mcq_options(kb, goals)
    winner = decide_mcq_winner(results, stem)
    winning_result = results[winner]
    explanation, proof_cot, cited_premises = explain_result(winning_result, kb)
    if winning_result.label == "Unknown":
        explanation = "The selected option has the best available symbolic ranking, but no proof was found."
    option_summary = _format_mcq_option_summary(results)
    no_entailed_option = all(result.label != "Yes" for result in results.values())

    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=question_type,
        answer=winner,
        explanation=f"Option {winner} is selected. {explanation} {option_summary}",
        fol=kb_to_fol_like_text(kb) or None,
        cot=[*proof_cot, option_summary],
        premises=cited_premises,
        confidence=_mcq_confidence(results, winner),
        error="No MCQ option was symbolically entailed." if no_entailed_option else None,
    )


def _format_mcq_option_summary(results: dict[str, SolveResult]) -> str:
    parts = [f"{label}: {results[label].label}" for label in _OPTION_ORDER if label in results]
    return "Option entailment results: " + ", ".join(parts) + "."


def _mcq_confidence(results: dict[str, SolveResult], winner: str) -> float:
    if results[winner].label == "Yes":
        entailed_count = sum(result.label == "Yes" for result in results.values())
        return 0.72 if entailed_count == 1 else 0.58
    if results[winner].label == "Unknown":
        return 0.28
    return 0.12


def _run_injected_llm_path(
    request: PredictionRequest,
    llm_client: SyncLLMClient,
) -> PredictionResponse | None:
    prompt = _build_fallback_prompt(request)
    try:
        data = _parse_json_object(llm_client.generate(prompt))
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
    premises = "\n".join(
        f"P{idx + 1}: {premise}" for idx, premise in enumerate(request.premises_nl or [])
    )
    return (
        "Answer the logic question using only the premises. Return JSON with "
        "answer, explanation, fol, cot, premises, confidence.\n\n"
        f"{premises}\n\nQuestion: {request.question}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty LLM output")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM output did not contain a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON output must be an object")
    return parsed


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
