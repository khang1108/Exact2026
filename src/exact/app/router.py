from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request

from exact.common.schemas import (
    ParsePremisesRequest,
    ParsePremisesResponse,
    ParsedPremise,
    ParserResponse,
    PredictionRequest,
    PredictionResponse,
    QParserRequest,
    QParserResponse,
    TaskType,
    UnifiedPredictionRequest,
)
from exact.config import get_settings
from exact.logger import get_request_logger
from exact.type1.parser.options import extract_mcq
from exact.type1.pipeline import (
    _normalize_options,
    _option_claim_to_dict,
    fol_node_to_dict,
    run_type1_pipeline,
)
from exact.type1.proof_connectivity import build_proof_connectivity_dashboard
from exact.type2.pipeline import run_type2_pipeline

api_router = APIRouter()


def _resolve_task_type(payload: PredictionRequest) -> TaskType:
    """Resolve explicit unified-schema types, with a legacy inference fallback."""
    if payload.type == "type1":
        return TaskType.TYPE1_LOGIC
    if payload.type == "type2":
        return TaskType.TYPE2_PHYSICS
    return TaskType.TYPE1_LOGIC if payload.premises else TaskType.TYPE2_PHYSICS


@api_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "pipeline": "full_pipeline", "version": "0.1.0"}


def _validate_models_payload(payload: Any, name: str) -> dict[str, Any]:
    """Validate the minimum OpenAI-compatible model-list contract."""

    if not isinstance(payload, dict) or payload.get("object") != "list":
        raise HTTPException(
            status_code=502,
            detail=f"{name} returned an invalid /v1/models response",
        )
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise HTTPException(
            status_code=502,
            detail=f"{name} returned no models from /v1/models",
        )
    if any(not isinstance(model, dict) or not model.get("id") for model in data):
        raise HTTPException(
            status_code=502,
            detail=f"{name} returned a model entry without an id",
        )
    return payload


async def _proxy_models(
    base_url: str | None,
    api_key: str | None,
    name: str,
) -> dict[str, Any]:
    """Forward a GET to a local vLLM ``/v1/models`` endpoint.

    The committee queries these public passthrough URLs during grading to verify
    the self-hosted models. The local vLLM servers are not exposed directly; this
    proxy injects the server API key so the caller needs no auth.
    """
    if not base_url:
        raise HTTPException(status_code=503, detail=f"{name} model server is not configured")
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return _validate_models_payload(response.json(), name)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"{name} model server unreachable: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"{name} returned invalid JSON: {exc}")


@api_router.get("/parser/v1/models")
async def parser_models() -> dict:
    """Passthrough to the Type 1 parser vLLM ``/v1/models``."""
    settings = get_settings()
    api_key = (
        settings.type1_parser_api_key.get_secret_value()
        if settings.type1_parser_api_key
        else None
    )
    return await _proxy_models(settings.type1_parser_base_url, api_key, "Type 1 parser")


@api_router.get("/v1/models")
# @api_router.get("/llm/v1/models")
async def llm_models() -> dict[str, Any]:
    """Passthrough to the main LLM vLLM ``/v1/models``."""
    settings = get_settings()
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return await _proxy_models(settings.llm_base_url, api_key, "Main LLM")


@api_router.post("/predict", response_model=PredictionResponse)
async def predict(payload: UnifiedPredictionRequest, request: Request) -> PredictionResponse:
    """Route one unified request to the Type 1 or Type 2 pipeline."""
    task_type = _resolve_task_type(payload)
    logger = get_request_logger(
        name="api_router.predict",
        request_id=payload.query_id,
        task_type=task_type,
    )
    logger.info(f"Received prediction request: type={payload.type!r} id={payload.query_id!r}")

    if task_type == TaskType.TYPE2_PHYSICS:
        logger.info("Running Type 2 pipeline")
        return await asyncio.to_thread(run_type2_pipeline, payload)

    if not payload.premises:
        raise HTTPException(status_code=422, detail="Type 1 requests require non-empty premises")

    logger.info("Running Type 1 pipeline")
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    question_parser = getattr(request.app.state, "type1_question_parser", None)
    if premise_parser is None or question_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    fallback_reasoner = getattr(request.app.state, "type1_fallback_reasoner", None)
    try:
        return await run_type1_pipeline(
            payload,
            premise_parser,
            question_parser,
            solver,
            fallback_reasoner,
        )
    except Exception as exc:
        logger.exception(f"Type 1 pipeline failed for {payload.query_id!r}: {exc}")
        return PredictionResponse(
            id=payload.query_id,
            task_type=TaskType.TYPE1_LOGIC,
            answer="Uncertain",
            explanation="Pipeline error — see server logs for details.",
            error=str(exc),
        )


@api_router.post("/z3", response_model=PredictionResponse)
async def z3_predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Parse premises + question/options to FOL then answer via Z3 entailment."""
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    question_parser = getattr(request.app.state, "type1_question_parser", None)
    if premise_parser is None or question_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    fallback_reasoner = getattr(request.app.state, "type1_fallback_reasoner", None)
    return await run_type1_pipeline(
        payload,
        premise_parser,
        question_parser,
        solver,
        fallback_reasoner,
    )


@api_router.post("/parser", response_model=ParserResponse)
async def parser(payload: ParsePremisesRequest, request: Request) -> ParserResponse:
    """Translate NL premises to FOL; returns ASTs, verification status, and schema renames."""
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    if premise_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")

    bundle = await premise_parser.parse_premises(payload.premises)

    return ParserResponse(
        premises=[
            ParsedPremise(
                id=f"premise-{i}",
                original_text=text,
                fol=repr(tree),
                ast=fol_node_to_dict(tree),
            )
            for i, (text, tree) in enumerate(zip(bundle.premises, bundle.trees), start=1)
        ],
        verified=bundle.verified,
        issues=list(bundle.verification_issues),
        renames=bundle.predicate_renames,
    )


@api_router.post("/qparser", response_model=QParserResponse)
async def qparser(payload: QParserRequest, request: Request) -> QParserResponse:
    """Classify a question into a QuerySpec (no solving).

    Parses the premises to build the shared schema, then runs the question-side
    parser and returns the QuerySpec for inspecting classification quality.
    """
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    question_parser = getattr(request.app.state, "type1_question_parser", None)
    if premise_parser is None or question_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")

    premise_bundle = await premise_parser.parse_premises(payload.premises)

    options_dict = _normalize_options(payload.options)
    mcq_extraction = None
    if not options_dict:
        mcq_extraction = extract_mcq(payload.question)
        options_dict = mcq_extraction.options

    q_bundle = await question_parser.parse_question(
        payload.question,
        options_dict or None,
        premise_bundle.schema,
        extraction=mcq_extraction,
    )
    spec = q_bundle.spec
    proof_connectivity = build_proof_connectivity_dashboard(
        q_bundle,
        premise_bundle.schema,
    )
    connectivity_by_id = {
        claim["claim_id"]: claim for claim in proof_connectivity["claims"]
    }

    return QParserResponse(
        question_format=spec.question_format,
        solver_mode=spec.solver_mode,
        can_interpretation=spec.can_interpretation,
        main_claim_text=spec.main_claim_text,
        main_claim_fol=(
            repr(q_bundle.main_claim_fol) if q_bundle.main_claim_fol is not None else None
        ),
        negate_claim=spec.negate_claim,
        supported=spec.supported,
        issues=list(spec.issues),
        marker_style=q_bundle.option_bundle.marker_style if q_bundle.option_bundle else None,
        role_distribution=(
            q_bundle.option_bundle.role_distribution if q_bundle.option_bundle else None
        ),
        extraction_diagnostics=(
            list(q_bundle.option_bundle.extraction_diagnostics) if q_bundle.option_bundle else []
        ),
        option_claims=[
            _option_claim_to_dict(c, connectivity_by_id.get(c.label))
            for c in spec.option_claims
        ],
        proof_connectivity=proof_connectivity,
    )


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
