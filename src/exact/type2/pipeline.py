from __future__ import annotations

from exact.datasets.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.logger import get_request_logger


TYPE2_NOT_IMPLEMENTED_MESSAGE = (
    "Type 2 physics reasoning is reserved for the dedicated physics pipeline. "
    "This placeholder keeps the API contract stable while the team implements "
    "paper-backed quantity extraction, formula selection, execution, and verification."
)


def run_type2_pipeline(request: PredictionRequest) -> PredictionResponse:
    """Stable API placeholder for future Type 2 physics work.

    A teammate can replace this function with the real Type 2 implementation.
    Keep the return type and fallback behavior so `/predict` and `/batch` never
    crash while research-driven modules are being developed.
    """
    logger = get_request_logger(
        __name__,
        request_id=request.id,
        task_type=TaskType.TYPE2_PHYSICS.value,
    )
    logger.info("Start Type 2 pipeline")
    logger.info("Type 2 placeholder response returned")

    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=QuestionType.NUMERICAL,
        answer="",
        explanation=TYPE2_NOT_IMPLEMENTED_MESSAGE,
        fol=None,
        cot=[
            "The request was routed to the Type 2 physics branch.",
            "The production physics solver has not been implemented in this repo yet.",
        ],
        premises=[
            "Type 2 receives only the question text.",
            "A future physics pipeline should derive answers from formulas, units, and verified computation.",
        ],
        confidence=0.0,
        error="type2_pipeline_not_implemented",
    )
