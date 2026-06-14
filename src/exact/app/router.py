from __future__ import annotations

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
)
from exact.logger import get_request_logger
from exact.type1.parser.options import parse_mcq_options
from exact.type1.pipeline import (
    _normalize_options,
    _option_claim_to_dict,
    fol_node_to_dict,
    run_type1_pipeline,
)

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
        task_type=TaskType.TYPE1_LOGIC,
    )
    logger.info(f"Received prediction request: {payload}")
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    question_parser = getattr(request.app.state, "type1_question_parser", None)
    if premise_parser is None or question_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    return await run_type1_pipeline(payload, premise_parser, question_parser, solver)


@api_router.post("/z3", response_model=PredictionResponse)
async def z3_predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Parse premises + question/options to FOL then answer via Z3 entailment."""
    premise_parser = getattr(request.app.state, "type1_premise_parser", None)
    question_parser = getattr(request.app.state, "type1_question_parser", None)
    if premise_parser is None or question_parser is None:
        raise HTTPException(status_code=503, detail="Type 1 parser model service is not configured")
    solver = getattr(request.app.state, "type1_solver", None)
    return await run_type1_pipeline(payload, premise_parser, question_parser, solver)


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
    if not options_dict:
        _, embedded = parse_mcq_options(payload.question)
        options_dict = embedded

    q_bundle = await question_parser.parse_question(
        payload.question,
        options_dict or None,
        premise_bundle.schema,
    )
    spec = q_bundle.spec

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
        option_claims=[_option_claim_to_dict(c) for c in spec.option_claims],
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
