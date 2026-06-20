from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

from exact.common.schemas import PredictionRequest, PredictionResponse, TaskType
from exact.config import Settings, get_settings
from exact.logger import get_request_logger
from exact.type2.extraction.verifier import verify_type2_extraction
from exact.type2.formulas.knowledge import retrieve_formula_context
from exact.type2.schemas import Verification
from exact.type2.solving.pot_solver import solve_with_pot

from exact.type2.pipeline import (
    _build_solver_extraction,
    _to_prediction_response,
    _GENERATE_FINAL_EXPLANATION_OVERRIDE,
)
from exact.type2.deterministic import run_deterministic_stage, merge_routing_diagnostics
from exact.type2.routing import build_routing_diagnostics, mark_current_solver_used


def run_ld_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Run the LD (electrostatics/vector) specific pipeline."""
    settings = settings or get_settings()
    logger = get_request_logger(
        __name__,
        request_id=request.id,
        task_type=TaskType.TYPE2_PHYSICS.value,
    )
    logger.info("Start Type 2 LD/DT deterministic-first pipeline")

    extraction = _build_solver_extraction(request.question, settings=settings)
    if settings.type2_use_extraction_verifier:
        review = verify_type2_extraction(extraction)
    else:
        review = SimpleNamespace(
            verification=Verification(True, "Extraction verifier disabled by Type 2 config.")
        )

    formula_limit = settings.type2_formula_limit if settings else 24
    formula_context = retrieve_formula_context(request.question, extraction, limit=formula_limit, settings=settings)
    request_extra = getattr(request, "model_extra", None) or {}
    routing_diagnostics = build_routing_diagnostics(
        request.question,
        extraction,
        formula_context,
        request_id=request.id,
        gold_or_dataset_method=request_extra.get("gold_or_dataset_method"),
        gold_solver_family=request_extra.get("gold_solver_family"),
        gold_formula_family=request_extra.get("gold_formula_family"),
    )

    logger.info(
        "Type 2 extraction (LD): kind=%s target=%s quantities=%s extraction_ok=%s formulas=%s route=%s eligible=%s",
        extraction.kind.value,
        extraction.target,
        sorted(extraction.quantities),
        review.verification.ok,
        formula_context.formula_ids,
        routing_diagnostics["predicted_method"],
        routing_diagnostics["eligible_solvers"],
    )

    deterministic = run_deterministic_stage(extraction)
    if deterministic.result is not None:
        result = deterministic.result
        if result.cot is not None:
            result.cot.insert(0, review.verification.message)
        return _to_prediction_response(request, result)

    # Legacy formula/graph fallback second
    from exact.type2.solving.pot_solver import _try_executable_formula_fallback
    fallback = _try_executable_formula_fallback(
        extraction,
        formula_context,
        "LD/DT domain prefers formula/graph fallback before LLM PoT.",
        settings,
    )
    if fallback is not None:
        fallback_diagnostics = mark_current_solver_used(
            routing_diagnostics,
            error=fallback.error,
        )
        fallback = replace(
            fallback,
            routing_diagnostics=merge_routing_diagnostics(
                deterministic.diagnostics,
                fallback_diagnostics,
            ),
        )
        if fallback.cot is not None:
            fallback.cot.insert(0, review.verification.message)
        return _to_prediction_response(request, fallback)

    generate_explanation = (
        settings.type2_generate_explanation
        if _GENERATE_FINAL_EXPLANATION_OVERRIDE is None
        else _GENERATE_FINAL_EXPLANATION_OVERRIDE
    )
    result = solve_with_pot(
        extraction,
        formula_context,
        settings=settings,
        generate_explanation=generate_explanation,
    )
    fallback_diagnostics = mark_current_solver_used(
        routing_diagnostics,
        error=result.error,
    )
    result = replace(
        result,
        routing_diagnostics=merge_routing_diagnostics(
            deterministic.diagnostics,
            fallback_diagnostics,
        ),
    )
    if result.cot is not None:
        result.cot.insert(0, review.verification.message)
    return _to_prediction_response(request, result)
