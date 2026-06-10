from __future__ import annotations

from fastapi import APIRouter, Request

from exact.common.schemas import PredictionRequest, PredictionResponse, TaskType
from exact.logger import get_request_logger
from exact.type2.pipeline import run_type2_pipeline

api_router = APIRouter()


@api_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "pipeline": "type2_physics"}


def _predict_internal(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """
    Internal prediction pipeline.

    Args:
        payload: Prediction request.
        request: FastAPI request.

    Returns:
        Prediction response.
    """
    logger = get_request_logger(
        __name__,
        request_id=payload.id or request.headers.get("X-Request-ID"),
        task_type=TaskType.TYPE2_PHYSICS.value,
    )

    logger.info("Received request")

    try:
        return run_type2_pipeline(payload)
    except Exception as exc:
        logger.error("Prediction failed", exc_info=True)
        return PredictionResponse(
            id=payload.id,
            task_type=TaskType.TYPE2_PHYSICS,
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
