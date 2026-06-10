from __future__ import annotations

from fastapi import APIRouter, Request

from exact.common.schemas import PredictionRequest, PredictionResponse, TaskType
from exact.logger import get_request_logger
from exact.router.task_router import TaskRouter
from exact.logic.pipeline import run_type1_pipeline
from exact.type2.pipeline import run_type2_pipeline
from exact.config import TYPE1_PIPELINE_BUILD

api_router = APIRouter()
task_router = TaskRouter()


@api_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "build": TYPE1_PIPELINE_BUILD}


def _predict_internal(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """
    Internal prediction pipeline.

    Args:
        payload: Prediction request.
        request: FastAPI request.

    Returns:
        Prediction response.
    """
    route = task_router.route(payload)
    logger = get_request_logger(
        __name__,
        request_id=payload.id or request.headers.get("X-Request-ID"),
        task_type=route.task_type.value,
    )

    logger.info("Received request")
    logger.info("Route decision: %s", route.reason)

    try:
        if route.task_type == TaskType.TYPE1_LOGIC:
            return run_type1_pipeline(payload, question_type=route.question_type)
        return run_type2_pipeline(payload)
    except Exception as exc:
        logger.error("Prediction failed", exc_info=True)
        return PredictionResponse(
            id=payload.id,
            task_type=route.task_type,
            answer="",
            explanation=f"Prediction failed: {exc}",
            fol=None,
            cot=["The system attempted to process the request but failed."],
            premises=[],
            confidence=0.0,
            error=str(exc),
        )


@api_router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Run the full EXACT pipeline and return the merged prediction payload."""

    return _predict_internal(payload, request)