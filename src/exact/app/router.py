from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from exact.common.schemas import (
    OfficialPredictionResponse,
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


def _render_predictions(results: list[PredictionResponse], debug: bool):
    """Official 6-field submission shape by default; full debug response when ``debug``.

    The full ``PredictionResponse`` already carries the FOL detail — ``fol`` plus
    ``routing_diagnostics.parsed_premises`` (premise → FOL) and
    ``routing_diagnostics.query_spec`` (translated question + per-option FOL).
    Returning a ``JSONResponse`` bypasses the route's ``OfficialPredictionResponse``
    filtering so grading keeps the stable 6-field contract by default.
    """
    if debug:
        return JSONResponse([r.model_dump(mode="json") for r in results])
    return [OfficialPredictionResponse.from_prediction(r) for r in results]


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
async def llm_models() -> dict[str, Any]:
    """Aggregate models from all vLLM servers (main LLM + Type 1 parser).

    The committee checks this endpoint to verify all self-hosted models are
    reachable. We merge both servers' model lists into one OpenAI-compatible
    response. If one server is down, its entries are omitted gracefully.
    """
    settings = get_settings()
    main_api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    parser_api_key = (
        settings.type1_parser_api_key.get_secret_value()
        if settings.type1_parser_api_key
        else None
    )

    # Fetch both servers concurrently; treat errors as empty results (soft degradation).
    async def _safe_proxy(base_url: str | None, api_key: str | None, name: str) -> list:
        try:
            payload = await _proxy_models(base_url, api_key, name)
            return payload.get("data", [])
        except HTTPException:
            return []

    main_models, parser_models_list = await asyncio.gather(
        _safe_proxy(settings.llm_base_url, main_api_key, "Main LLM"),
        _safe_proxy(settings.type1_parser_base_url, parser_api_key, "Type 1 parser"),
    )

    # Deduplicate by model id in case both servers serve the same model name.
    seen: set[str] = set()
    merged: list[dict] = []
    for model in [*main_models, *parser_models_list]:
        if model.get("id") not in seen:
            seen.add(model["id"])
            merged.append(model)

    if not merged:
        raise HTTPException(status_code=503, detail="No model servers are reachable")

    return {"object": "list", "data": merged}




@api_router.post("/predict", response_model=list[OfficialPredictionResponse])
async def predict(
    payload: UnifiedPredictionRequest, request: Request, debug: bool = False
):
    """Route one unified request to the Type 1 or Type 2 pipeline.

    ``?debug=true`` returns the full internal response (FOL translations of the
    premises, question, and options) instead of the 6-field submission shape.
    """
    task_type = _resolve_task_type(payload)
    logger = get_request_logger(
        name="api_router.predict",
        request_id=payload.query_id,
        task_type=task_type,
    )
    logger.info(f"Received prediction request: type={payload.type!r} id={payload.query_id!r}")

    if task_type == TaskType.TYPE2_PHYSICS:
        logger.info("Running Type 2 pipeline")
        try:
            result = await asyncio.to_thread(run_type2_pipeline, payload)
            return _render_predictions([result], debug)
        except Exception as exc:
            logger.exception(f"Type 2 pipeline failed for {payload.query_id!r}: {exc}")
            fallback = PredictionResponse(
                id=payload.query_id,
                task_type=TaskType.TYPE2_PHYSICS,
                answer="",
                explanation="Type 2 pipeline error — see server logs for details.",
                error=str(exc),
            )
            return _render_predictions([fallback], debug)

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
        result = await run_type1_pipeline(
            payload,
            premise_parser,
            question_parser,
            solver,
            fallback_reasoner,
        )
        return _render_predictions([result], debug)
    except Exception as exc:
        logger.exception(f"Type 1 pipeline failed for {payload.query_id!r}: {exc}")
        fallback = PredictionResponse(
            id=payload.query_id,
            task_type=TaskType.TYPE1_LOGIC,
            answer="Uncertain",
            explanation="Pipeline error — see server logs for details.",
            error=str(exc),
        )
        return _render_predictions([fallback], debug)


@api_router.post("/z3", response_model=list[OfficialPredictionResponse])
async def z3_predict(
    payload: PredictionRequest, request: Request
) -> list[OfficialPredictionResponse]:
    """Parse premises + question/options to FOL then answer via Z3 entailment."""
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    question_parser = getattr(request.app.state, "type1_question_parser", None)
    if premise_parser is None or question_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    fallback_reasoner = getattr(request.app.state, "type1_fallback_reasoner", None)
    result = await run_type1_pipeline(
        payload,
        premise_parser,
        question_parser,
        solver,
        fallback_reasoner,
    )
    return [OfficialPredictionResponse.from_prediction(result)]


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
