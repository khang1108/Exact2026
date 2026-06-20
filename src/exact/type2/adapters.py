from __future__ import annotations

from exact.common.schemas import OfficialPredictionResponse, PredictionResponse, QuestionType, TaskType
from exact.type2.schemas import Type2InternalResult


def to_prediction_response(result: Type2InternalResult) -> PredictionResponse:
    return PredictionResponse(
        query_id=result.query_id,
        answer=result.answer,
        unit=result.unit,
        explanation=result.explanation,
        premises_used=[],
        reasoning=None,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=QuestionType.NUMERICAL if result.answer else QuestionType.OPEN_ENDED,
        confidence=result.confidence,
        routing_diagnostics=result.diagnostics,
    )


def to_official_type2_response(result: Type2InternalResult) -> list[OfficialPredictionResponse]:
    return [
        OfficialPredictionResponse(
            query_id=result.query_id,
            answer=result.answer,
            unit=result.unit,
            explanation=result.explanation or "The system produced an answer but could not generate a detailed explanation.",
            premises_used=[],
            reasoning=None,
        )
    ]
