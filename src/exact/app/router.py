from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from exact.common.schemas import (
    PredictionRequest,
    PredictionResponse,
    TaskType,
    UnifiedPredictionRequest,
)
from exact.logger import get_request_logger
from exact.type1.pipeline import run_type1_pipeline
from exact.type2.pipeline import run_type2_pipeline

api_router = APIRouter()


@api_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "pipeline": "full_pipeline", "version": "0.1.0"}


@api_router.post("/predict", response_model=PredictionResponse)
async def predict(payload: UnifiedPredictionRequest, request: Request) -> PredictionResponse:
    """Route one unified request to the Type 1 or Type 2 pipeline."""
    task_type = _resolve_task_type(payload)
    logger = get_request_logger(
        name="api_router.predict",
        request_id=payload.query_id,
        task_type=task_type,
    )

    logger.info(f"Received prediction request: {payload}")

    if task_type == TaskType.TYPE2_PHYSICS:
        logger.info("Running Type 2 pipeline")
        return await asyncio.to_thread(run_type2_pipeline, payload)

    if not payload.premises:
        raise HTTPException(status_code=422, detail="Type 1 requests require non-empty premises")

    logger.info("Running Type 1 pipeline")
    parser = getattr(request.app.state, "type1_fol_parser", None)
    if parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    refiner = getattr(request.app.state, "type1_refiner", None)
    return await run_type1_pipeline(payload, parser, solver, refiner)


def _resolve_task_type(payload: PredictionRequest) -> TaskType:
    """Resolve explicit unified-schema types, with a legacy inference fallback."""

    if payload.type == "type1":
        return TaskType.TYPE1_LOGIC
    if payload.type == "type2":
        return TaskType.TYPE2_PHYSICS
    return TaskType.TYPE1_LOGIC if payload.premises else TaskType.TYPE2_PHYSICS


@api_router.post("/z3", response_model=PredictionResponse)
async def z3_predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Parse premises + question/options to FOL then answer via Z3 + self-refinement."""
    parser = getattr(request.app.state, "type1_fol_parser", None)
    if parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    refiner = getattr(request.app.state, "type1_refiner", None)
    return await run_type1_pipeline(payload, parser, solver, refiner)
