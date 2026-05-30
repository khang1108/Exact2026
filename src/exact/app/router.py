from __future__ import annotations

from fastapi import APIRouter, Request

from exact.common.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    OfficialBatchPredictionResponse,
    OfficialPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    TaskType,
    to_official_response,
)
from exact.logger import get_request_logger
from exact.router.task_router import TaskRouter
from exact.logic.pipeline import run_type1_pipeline
from exact.type2.pipeline import run_type2_pipeline
from exact.version import TYPE1_PIPELINE_BUILD

api_router = APIRouter()
task_router = TaskRouter()


@api_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "build": TYPE1_PIPELINE_BUILD}


def _predict_internal(payload: PredictionRequest, request: Request) -> PredictionResponse:
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


@api_router.post("/predict", response_model=OfficialPredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> dict:
    """Run the full EXACT pipeline and return the official challenge shape."""

    return to_official_response(_predict_internal(payload, request))


@api_router.post("/debug/predict", response_model=PredictionResponse)
def debug_predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Run prediction while preserving local metadata useful during development."""

    return _predict_internal(payload, request)


@api_router.post("/batch", response_model=OfficialBatchPredictionResponse)
def batch_predict(
    payload: BatchPredictionRequest,
    request: Request,
) -> dict:
    predictions = [to_official_response(_predict_internal(item, request)) for item in payload.instances]
    return {"predictions": predictions}


@api_router.post("/debug/batch", response_model=BatchPredictionResponse)
def debug_batch_predict(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    predictions = [_predict_internal(item, request) for item in payload.instances]
    return BatchPredictionResponse(predictions=predictions)
