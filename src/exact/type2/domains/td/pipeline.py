from __future__ import annotations

from exact.common.schemas import PredictionRequest, PredictionResponse
from exact.common.schemas import QuestionType, TaskType
from exact.config import Settings
from exact.type2.domains.td.solver import solve_td_capacitor_late_range
from exact.type2.pipeline import run_generic_pipeline

def run_td_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Run the TD (capacitor/static) specific pipeline.

    Specialized TD capacitor state-change rules run first. Generic Type 2 then
    handles the direct scalar and contract-backed cases.
    """
    answer = solve_td_capacitor_late_range(request.id, request.question)
    if answer is not None:
        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE2_PHYSICS,
            question_type=QuestionType.NUMERICAL,
            answer=answer.answer,
            explanation=answer.explanation,
            unit=answer.unit,
            fol=None,
            cot=[f"Matched TD capacitor deterministic rule `{answer.rule}`."],
            premises=[answer.explanation],
            confidence=answer.confidence,
            error=None,
            routing_diagnostics={
                "domain": "TD",
                "solver": "td_capacitor_deterministic_solver",
                "rule": answer.rule,
                "fallback_used": False,
            },
        )
    return run_generic_pipeline(request, settings)
