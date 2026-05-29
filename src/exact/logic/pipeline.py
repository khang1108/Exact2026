"""Type 1 logic pipeline.

Type 1 uses an LLM semantic parser to build logic IR, then deterministic
symbolic reasoning to produce the answer. There is no local parser substitute:
missing or invalid LLM output is logged and raised.
"""

from __future__ import annotations

import re
from collections import Counter

from exact.config import Settings, get_settings
from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.llm_client import build_json_client_from_settings
from exact.logic.explain import explain_result, kb_to_fol_like_text
from exact.logic.kb import get_or_build_kb_candidates
from exact.logic.ir import Atom, SolveResult
from exact.logic.kb import KnowledgeBase
from exact.logic.llm_translator import (
    JsonLLMClient,
    translate_mcq_options_with_llm,
    translate_query_only_with_llm,
)
from exact.logic.parser import atom_from_text
from exact.logger import get_logger, get_request_logger

logger = get_logger(__name__)
from exact.symbolic_solvers import ForwardChainSolver


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
    translator_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
    question_type: QuestionType | None = None,
) -> PredictionResponse:
    """
    Answer a Type 1 logic query with symbolic proof where possible.

    Args:
        request (PredictionRequest): Đại diện cho request cần xử lý
        translator_client (JsonLLMClient): Client dùng để dịch NL -> Logical Form
        settings (Settings): Các settings cho pipeline

    Returns:
        PredictionResponse: Đại diện cho dự đoán
    """

    logger = get_request_logger(
        __name__,
        request_id=request.id,
        task_type=TaskType.TYPE1_LOGIC.value,
    )
    logger.info("Start Type 1 pipeline")

    settings = settings or get_settings()
    translator_client = translator_client or build_json_client_from_settings(settings)
    if translator_client is None:
        logger.error("Type 1 LLM-only pipeline has no configured JSON LLM client")
        raise RuntimeError(
            "Type 1 requires a JSON LLM client. Configure EXACT_LLM_BASE_URL for an "
            "OpenAI-compatible server or EXACT_LLM_PROVIDER=local for a Transformers model."
        )

    routed_question_type = question_type or QuestionType.YES_NO_UNCERTAIN
    premises = request.premises_nl or []
    samples = max(1, settings.type1_translation_samples)
    sampling_temperature = (
        settings.type1_sampling_temperature if samples > 1 else settings.llm_temperature
    )

    if routed_question_type == QuestionType.MCQ:
        try:
            kb_candidates, candidate_warnings = get_or_build_kb_candidates(
                premises,
                translator_client,
                settings,
                samples=samples,
                sampling_temperature=sampling_temperature,
            )
        except Exception as exc:
            logger.exception("Type 1 MCQ LLM premise translation failed")
            raise RuntimeError(
                f"Type 1 MCQ LLM premise translation failed for request {request.id}: {exc}"
            ) from exc
        return _run_mcq_vote_path(
            request, kb_candidates, routed_question_type, candidate_warnings,
            translator_client=translator_client, settings=settings,
        )

    try:
        kb_candidates, candidate_warnings = get_or_build_kb_candidates(
            premises,
            translator_client,
            settings,
            samples=samples,
            sampling_temperature=sampling_temperature,
        )
    except Exception as exc:
        logger.exception("Type 1 LLM premise translation failed")
        raise RuntimeError(
            f"Type 1 LLM premise translation failed for request {request.id}: {exc}"
        ) from exc

    candidate_results: list[tuple[KnowledgeBase, SolveResult]] = []
    query_errors: list[str] = []
    for index, kb in enumerate(kb_candidates, start=1):
        try:
            query = translate_query_only_with_llm(
                question=request.question,
                predicate_names=kb.predicate_names,
                llm_client=translator_client,
                settings=settings,
            )
            candidate_results.append((kb, ForwardChainSolver().solve(kb, query.claim)))
        except Exception as exc:
            message = f"query candidate {index}/{len(kb_candidates)} failed: {exc}"
            logger.exception("Type 1 LLM query translation failed: %s", message)
            query_errors.append(message)

    if not candidate_results:
        raise RuntimeError(
            f"Type 1 LLM query translation failed for request {request.id}: "
            + "; ".join((*candidate_warnings, *query_errors))
        )

    winner_label, vote_summary, vote_confidence = _vote_labels(
        [result.label for _, result in candidate_results],
        tie_order=("Unknown", "No", "Yes"),
    )
    kb, result = next(
        (candidate_kb, candidate_result)
        for candidate_kb, candidate_result in candidate_results
        if candidate_result.label == winner_label
    )
    explanation, cot, cited_premises = explain_result(result, kb)
    vote_line = f"symbolic_consistency_vote: {vote_summary}"
    warnings = (*candidate_warnings, *query_errors, *result.warnings)

    response = PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=routed_question_type,
        answer=result.label,
        explanation=f"{explanation} {vote_line}",
        fol=kb_to_fol_like_text(kb) or None,
        cot=[*cot, vote_line],
        premises=cited_premises,
        confidence=vote_confidence,
        error="; ".join(warnings) if warnings else None,
    )

    if result.label == "Unknown" and settings.type1_enable_cot_fallback:
        return _run_cot_unknown_fallback(
            request=request,
            response=response,
            llm_client=translator_client,
            settings=settings,
            logger=logger,
        )

    return response


def _run_cot_unknown_fallback(
    request: PredictionRequest,
    response: PredictionResponse,
    llm_client: JsonLLMClient,
    settings: Settings,
    logger,
) -> PredictionResponse:
    """Use LLM reasoning when symbolic consistency voting cannot prove a label."""

    messages = _build_cot_unknown_messages(request, response)
    try:
        raw = llm_client.complete_json_sync(
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        answer = _normalize_cot_answer(raw.get("answer"))
        explanation = str(raw.get("explanation") or "").strip()
        cot = _as_string_list(raw.get("cot"))
        confidence = _bounded_confidence(raw.get("confidence"), default=response.confidence or 0.35)
    except Exception as exc:
        logger.exception("Type 1 CoT fallback after symbolic Unknown failed")
        return response.model_copy(
            update={
                "error": _join_errors(response.error, f"CoT fallback failed: {exc}"),
            }
        )

    fallback_line = f"cot_fallback_after_symbolic_unknown: answer={answer}"
    return response.model_copy(
        update={
            "answer": answer,
            "explanation": (
                explanation
                or f"Symbolic reasoning returned Unknown, then the LLM reasoning fallback selected {answer}."
            ),
            "cot": [*(response.cot or []), *cot, fallback_line],
            "confidence": confidence,
            "error": response.error,
        }
    )


def _build_cot_unknown_messages(
    request: PredictionRequest,
    response: PredictionResponse,
) -> list[dict[str, str]]:
    premises = "\n".join(
        f"P{idx + 1}: {premise}" for idx, premise in enumerate(request.premises_nl or [])
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a careful educational logic reasoner. Return JSON only. "
                "Use the given premises only. The answer must be Yes, No, or Unknown."
            ),
        },
        {
            "role": "user",
            "content": (
                "The symbolic solver returned Unknown. Re-check the question with concise reasoning.\n"
                'Return JSON shaped as {"answer":"Yes|No|Unknown","explanation":"...",'
                '"cot":["step"],"confidence":0.0}.\n\n'
                f"Premises:\n{premises}\n\n"
                f"Question:\n{request.question}\n\n"
                f"Symbolic result:\nanswer={response.answer}\n"
                f"explanation={response.explanation}\n"
            ),
        },
    ]


def _normalize_cot_answer(value: object) -> str:
    answer = str(value or "").strip().lower()
    if answer in {"yes", "true", "entailed"}:
        return "Yes"
    if answer in {"no", "false", "contradicted"}:
        return "No"
    if answer in {"unknown", "uncertain", "not enough information"}:
        return "Unknown"
    raise ValueError(f"CoT fallback returned unsupported answer: {value!r}")


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _bounded_confidence(value: float, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def _join_errors(*errors: str | None) -> str | None:
    parts = [error for error in errors if error]
    return "; ".join(parts) if parts else None


def _run_mcq_path(
    request: PredictionRequest,
    kb: KnowledgeBase,
    question_type: QuestionType,
    translator_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
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

    # Translate options via LLM (using KB predicate vocabulary) then fall back per-option to text parser.
    translated: dict[str, Atom] = {}
    if translator_client is not None:
        try:
            translated = translate_mcq_options_with_llm(
                request.question, options, kb.predicate_names,
                translator_client, settings,
            )
        except Exception as exc:
            logger.debug("MCQ LLM option translation skipped: %s", exc)
    goals = [(label, translated.get(label) or atom_from_text(text)) for label, text in options]
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


def _run_mcq_vote_path(
    request: PredictionRequest,
    kb_candidates: tuple[KnowledgeBase, ...],
    question_type: QuestionType,
    candidate_warnings: tuple[str, ...] = (),
    translator_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> PredictionResponse:
    responses: list[PredictionResponse] = [
        _run_mcq_path(request, kb, question_type, translator_client, settings)
        for kb in kb_candidates
    ]
    winner, vote_summary, confidence = _vote_labels(
        [response.answer for response in responses],
        tie_order=_OPTION_ORDER,
    )
    selected = next(response for response in responses if response.answer == winner)
    vote_line = f"symbolic_consistency_vote: {vote_summary}"
    warnings = [warning for warning in candidate_warnings if warning]
    if selected.error:
        warnings.append(selected.error)

    return selected.model_copy(
        update={
            "confidence": confidence,
            "explanation": f"{selected.explanation} {vote_line}",
            "cot": [*(selected.cot or []), vote_line],
            "error": "; ".join(warnings) if warnings else None,
        }
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


def _vote_labels(labels: list[str], tie_order: tuple[str, ...]) -> tuple[str, str, float]:
    if not labels:
        raise ValueError("Cannot vote over an empty label list")

    counts = Counter(labels)
    tie_rank = {label: index for index, label in enumerate(tie_order)}
    winner = min(
        counts,
        key=lambda label: (-counts[label], tie_rank.get(label, len(tie_rank)), label),
    )
    ordered_labels = sorted(
        counts,
        key=lambda label: (-counts[label], tie_rank.get(label, len(tie_rank)), label),
    )
    summary = ", ".join(f"{label}={counts[label]}" for label in ordered_labels)
    return winner, summary, counts[winner] / len(labels)
