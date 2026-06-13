from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from exact.common.schemas import (
    ParsePremisesRequest,
    ParsePremisesResponse,
    ParsedPremise,
    PredictionRequest,
    PredictionResponse,
    TaskType,
)
from exact.type1.pipeline import fol_node_to_dict
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
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    if premise_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    return await run_type1_pipeline(payload, premise_parser, solver)


@api_router.post("/z3", response_model=PredictionResponse)
async def z3_predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Parse premises + question/options to FOL then answer via Z3 entailment."""
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    if premise_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    return await run_type1_pipeline(payload, premise_parser, solver)


@api_router.post("/premises", response_model=ParsePremisesResponse)
async def parse_premises(payload: ParsePremisesRequest, request: Request) -> ParsePremisesResponse:
    """Translate raw NL premises to FOL without running the solver.

    Returns the canonicalized ASTs, predicate schema, and any predicate renames
    detected during schema construction.
    """
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    if premise_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")

    bundle = await premise_parser.parse_premises(payload.premises)

    return ParsePremisesResponse(
        premises=[
            ParsedPremise(
                id=f"premise-{i}",
                original_text=text,
                fol=repr(tree),
                ast=fol_node_to_dict(tree),
            )
            for i, (text, tree) in enumerate(zip(bundle.premises, bundle.trees), start=1)
        ],
    )
