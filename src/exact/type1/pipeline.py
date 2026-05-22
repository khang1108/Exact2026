from __future__ import annotations

import json
from typing import Any

from exact.baselines.llm_client import get_default_llm_client
from exact.datasets.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.logger import get_request_logger
from exact.type1.answer_normalizer import normalize_type1_answer
from exact.type1.prompt_builder import build_type1_prompt


def run_type1_pipeline(
    request: PredictionRequest,
    llm_client: Any | None = None,
) -> PredictionResponse:
    logger = get_request_logger(
        __name__,
        request_id=request.id,
        task_type=TaskType.TYPE1_LOGIC.value,
    )
    logger.info("Start Type 1 pipeline")

    premises = request.premises_nl or []
    prompt = build_type1_prompt(premises, request.question)

    try:
        client = llm_client or get_default_llm_client()
        text = client.generate(prompt)
        data = _parse_json_object(text)

        answer = normalize_type1_answer(str(data.get("answer", "")))
        explanation = str(data.get("explanation", "")).strip()
        cot = _as_list(data.get("cot"))
        used_premises = _as_list(data.get("premises"))
        fol = data.get("fol")
        confidence = _safe_float(data.get("confidence"), default=0.75)

        if not explanation:
            explanation = "The answer is derived from the provided premises."

        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE1_LOGIC,
            question_type=_infer_type1_question_type(request.question),
            answer=answer,
            explanation=explanation,
            fol=str(fol) if fol else None,
            cot=cot,
            premises=used_premises,
            confidence=confidence,
        )

    except Exception as exc:
        logger.warning("Type 1 failed, using fallback: %s", exc)
        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE1_LOGIC,
            question_type=_infer_type1_question_type(request.question),
            answer="Uncertain",
            explanation=(
                "The system could not derive a reliable conclusion from the given premises, "
                "so it returns Uncertain as a conservative answer."
            ),
            fol=None,
            cot=[
                "Attempted to reason from the provided premises.",
                "No reliable derivation was produced.",
            ],
            premises=[],
            confidence=0.2,
            error=str(exc),
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
        return max(0.0, min(1.0, number))
    except Exception:
        return default


def _infer_type1_question_type(question: str) -> QuestionType:
    q = question.lower()
    if "\na." in q or " a." in q or "\na)" in q or " a)" in q:
        return QuestionType.MCQ
    if q.startswith(("does", "do", "is", "are", "can", "will", "has", "have")):
        return QuestionType.YES_NO_UNCERTAIN
    return QuestionType.OPEN_ENDED
