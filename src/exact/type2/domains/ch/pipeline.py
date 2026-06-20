from __future__ import annotations

from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.config import Settings
from exact.type2.domains.ch.solver import ChAnswer, solve_ch_resonance


def run_ch_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse | None:
    answer = solve_ch_resonance(request.id, request.question)
    if answer is None:
        return None
    return _to_response(request, answer)


def _to_response(request: PredictionRequest, answer: ChAnswer) -> PredictionResponse:
    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=QuestionType.NUMERICAL,
        answer=answer.answer,
        explanation=answer.explanation,
        unit=answer.unit,
        fol=None,
        cot=[f"Matched CH deterministic rule `{answer.rule}`."],
        premises=[answer.explanation],
        confidence=answer.confidence,
        error=None,
        routing_diagnostics={
            "domain": "CH",
            "solver": "ch_resonance_deterministic_solver",
            "rule": answer.rule,
            "fallback_used": False,
        },
    )
