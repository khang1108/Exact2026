from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from exact.common.schemas import (
    PredictionRequest,
    PredictionResponse,
    TaskType,
)
from exact.logger import get_request_logger
from exact.type1.pipeline import run_type1_pipeline
from exact.type2.pipeline import run_type2_pipeline

api_router = APIRouter()


@api_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "pipeline": "full_pipeline", "version": "0.1.0"}


@api_router.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Run the full EXACT pipeline and return the merged prediction payload."""
    logger = get_request_logger(
        name="api_router.predict",
        request_id=payload.query_id,
        task_type=TaskType.TYPE1_LOGIC if payload.type == "type1" else TaskType.TYPE2_PHYSICS,
    )

    logger.info(f"Received prediction request: {payload}")

    if payload.type == "type2":
        logger.info("Running Type 2 pipeline")
        return await asyncio.to_thread(run_type2_pipeline, payload)

    logger.info("Running Type 1 pipeline")
    parser = getattr(request.app.state, "type1_fol_parser", None)
    if parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    llm_head = getattr(request.app.state, "type1_llm_head", None)
    return await run_type1_pipeline(payload, parser, solver, llm_head)


@api_router.post("/z3", response_model=PredictionResponse)
async def z3_predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Parse premises + question/options to FOL then answer via Z3 + LLM head."""
    parser = getattr(request.app.state, "type1_fol_parser", None)
    if parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    llm_head = getattr(request.app.state, "type1_llm_head", None)
    return await run_type1_pipeline(payload, parser, solver, llm_head)
