from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi import HTTPException

from exact.common.schemas import (
    ParsePremisesRequest,
    ParsePremisesResponse,
    ParsedPremise,
    PredictionRequest,
    PredictionResponse,
    TaskType,
)
from exact.logger import get_request_logger
from exact.type2.pipeline import run_type2_pipeline
from exact.type1.pipeline import fol_node_to_dict, run_type1_pipeline

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
    return await run_type1_pipeline(payload, parser)


@api_router.post("/parser", response_model=ParsePremisesResponse)
async def parse_premises(payload: ParsePremisesRequest, request: Request) -> ParsePremisesResponse:
    """Translate natural-language premises to FOL, without the full Type 1 pipeline."""
    parser = getattr(request.app.state, "type1_fol_parser", None)
    if parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")

    trees = await parser.parse_many(payload.premises)
    parsed = [
        ParsedPremise(
            id=f"premise-{index}",
            original_text=premise,
            fol=repr(tree),
            ast=fol_node_to_dict(tree),
        )
        for index, (premise, tree) in enumerate(zip(payload.premises, trees, strict=True), start=1)
    ]
    return ParsePremisesResponse(premises=parsed)
