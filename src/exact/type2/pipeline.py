from __future__ import annotations

from dataclasses import replace
from typing import Any
from types import SimpleNamespace

from exact.config import Settings, get_settings
from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.logger import get_request_logger
from exact.type2.extraction.extractor import extract_type2
from exact.type2.extraction.llm_structured import (
    classify_question_kind_with_llm,
    generate_direct_answer,
    parse_with_llm,
)
from exact.type2.extraction.verifier import verify_type2_extraction
from exact.type2.deterministic import merge_routing_diagnostics, run_deterministic_stage
from exact.type2.formulas.knowledge import retrieve_formula_context
from exact.type2.routing import build_routing_diagnostics, mark_current_solver_used
from exact.type2.schemas import (
    Extraction,
    Quantity,
    Type2QuestionKind,
    Type2SolveResult,
    Verification,
)
from exact.type2.solving.pot_solver import solve_with_pot
from exact.type2.solving.units import parse_quantity


_GENERATE_FINAL_EXPLANATION_OVERRIDE: bool | None = None


def run_type2_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Route the request to a domain-specific pipeline or fallback to generic."""
    from exact.type2.domains.router import route_domain_with_metadata

    settings = settings or get_settings()
    try:
        domain_route = route_domain_with_metadata(request.id, request.question, settings=settings)
        domain = domain_route.domain

        if domain == "LD":
            from exact.type2.domains.ld.pipeline import run_ld_pipeline
            response = run_ld_pipeline(request, settings)
        elif domain == "TD":
            from exact.type2.domains.td.pipeline import run_td_pipeline
            response = run_td_pipeline(request, settings)
        elif domain == "NL_ENERGY":
            from exact.type2.domains.nl_energy.pipeline import run_nl_energy_pipeline
            nl_result, fallback = run_nl_energy_pipeline(request, settings)
            response = run_generic_pipeline(request, settings) if fallback or nl_result is None else nl_result
        else:
            response = run_generic_pipeline(request, settings)

        response = _with_domain_route_diagnostics(response, domain_route.to_dict())
        return _with_agent_loop_if_needed(request, response, settings, trigger=response.error)
    except Exception as exc:
        return _agent_loop_response(
            request,
            settings,
            trigger=f"pipeline_exception: {exc}",
            diagnostics={"exception_type": type(exc).__name__},
        )


def run_generic_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Run the Type 2 physics pipeline with deterministic routing before PoT.

    Flow:
    query -> extraction -> formal contract routing/validation -> deterministic
    solver when eligible -> PoT fallback when deterministic routing is unsolved.
    """
    from exact.config import get_settings
    settings = settings or get_settings()
    logger = get_request_logger(
        __name__,
        request_id=request.id,
        task_type=TaskType.TYPE2_PHYSICS.value,
    )
    logger.info("Start Type 2 deterministic-first pipeline")

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
        "Type 2 extraction: kind=%s target=%s quantities=%s extraction_ok=%s formulas=%s route=%s eligible=%s",
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


def set_generate_final_explanation(enabled: bool) -> None:
    global _GENERATE_FINAL_EXPLANATION_OVERRIDE
    _GENERATE_FINAL_EXPLANATION_OVERRIDE = enabled


def _build_solver_extraction(
    question: str,
    settings: Settings | None = None,
) -> Extraction:
    heuristic = extract_type2(question)
    mode = settings.type2_extraction_mode if settings else "merge"
    if mode == "heuristic_only":
        return _with_llm_question_kind(
            _append_extraction_note(heuristic, "structured_extraction_source=heuristic_only"),
            question,
            settings=settings,
        )

    llm_extraction = _try_llm_extraction(question, settings=settings)
    if llm_extraction is None:
        return _with_llm_question_kind(
            _append_extraction_note(heuristic, "structured_extraction_source=heuristic; llm_extraction=unavailable"),
            question,
            settings=settings,
        )
    if mode == "llm_only":
        return _with_llm_question_kind(
            _append_extraction_note(llm_extraction, "structured_extraction_source=llm"),
            question,
            settings=settings,
        )
    return _with_llm_question_kind(
        _append_extraction_note(
            _merge_extractions(heuristic, llm_extraction),
            "structured_extraction_source=merged",
        ),
        question,
        settings=settings,
    )


def _append_extraction_note(extraction: Extraction, note: str) -> Extraction:
    if note in extraction.notes:
        return extraction
    return replace(extraction, notes=(*extraction.notes, note))


def _to_prediction_response(
    request: PredictionRequest,
    result: Type2SolveResult,
) -> PredictionResponse:
    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=_question_type(result),
        answer=result.answer,
        explanation=_build_explanation(result),
        fol=None,
        cot=result.cot,
        premises=result.premises,
        confidence=result.confidence,
        unit=result.unit,
        error=result.error,
        routing_diagnostics=result.routing_diagnostics,
    )


def _with_domain_route_diagnostics(
    response: PredictionResponse,
    domain_route: dict[str, Any],
) -> PredictionResponse:
    diagnostics = dict(response.routing_diagnostics or {})
    diagnostics["type2_domain_route"] = domain_route
    return response.model_copy(update={"routing_diagnostics": diagnostics})


def _with_agent_loop_if_needed(
    request: PredictionRequest,
    response: PredictionResponse,
    settings: Settings,
    trigger: str | None,
) -> PredictionResponse:
    diagnostics = dict(response.routing_diagnostics or {})
    diagnostics["type2_llm_configured"] = bool(settings.llm_base_url)
    diagnostics["type2_llm_domain_routing_enabled"] = settings.type2_use_llm_domain_routing
    diagnostics["type2_llm_question_kind_routing_enabled"] = settings.type2_use_llm_question_kind_routing
    diagnostics["type2_agent_loop_enabled"] = settings.type2_use_agent_loop
    diagnostics["type2_agent_loop_max_attempts"] = settings.type2_agent_loop_max_attempts
    diagnostics["type2_pot_solver_enabled"] = settings.type2_use_pot_solver
    response = response.model_copy(update={"routing_diagnostics": diagnostics})
    if not trigger:
        diagnostics["type2_agent_loop"] = {
            "used": False,
            "trigger": None,
            "reason": "no pipeline error",
        }
        response = response.model_copy(update={"routing_diagnostics": diagnostics})
        return response
    return _agent_loop_response(
        request,
        settings,
        trigger=trigger,
        diagnostics=diagnostics,
        previous_response=response,
    )


def _agent_loop_response(
    request: PredictionRequest,
    settings: Settings,
    *,
    trigger: str,
    diagnostics: dict[str, Any] | None = None,
    previous_response: PredictionResponse | None = None,
) -> PredictionResponse:
    diagnostics = dict(diagnostics or {})
    diagnostics["type2_llm_configured"] = bool(settings.llm_base_url)
    diagnostics["type2_llm_domain_routing_enabled"] = settings.type2_use_llm_domain_routing
    diagnostics["type2_llm_question_kind_routing_enabled"] = settings.type2_use_llm_question_kind_routing
    diagnostics["type2_agent_loop_enabled"] = settings.type2_use_agent_loop
    diagnostics["type2_agent_loop_max_attempts"] = settings.type2_agent_loop_max_attempts
    diagnostics["type2_pot_solver_enabled"] = settings.type2_use_pot_solver
    if not settings.type2_use_agent_loop:
        diagnostics["type2_agent_loop"] = {
            "used": False,
            "trigger": trigger,
            "reason": "agent loop disabled",
        }
        if previous_response is not None:
            return previous_response.model_copy(update={"routing_diagnostics": diagnostics})
        return _guardrail_response(request, trigger, diagnostics)

    attempts: list[dict[str, Any]] = []
    for attempt in range(1, settings.type2_agent_loop_max_attempts + 1):
        try:
            spec = generate_direct_answer(
                request.question,
                _agent_loop_failure_context(trigger, previous_response),
                settings=settings,
            )
        except Exception as exc:
            attempts.append({"attempt": attempt, "status": "error", "reason": str(exc)})
            continue
        if spec is None:
            attempts.append({"attempt": attempt, "status": "unavailable"})
            continue
        if not spec.answer.strip():
            attempts.append({"attempt": attempt, "status": "empty"})
            continue
        attempts.append({"attempt": attempt, "status": "answered"})
        diagnostics["type2_agent_loop"] = {
            "used": True,
            "enabled": True,
            "trigger": trigger,
            "attempts": attempts,
        }
        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE2_PHYSICS,
            question_type=QuestionType.OPEN_ENDED if spec.unit is None else QuestionType.NUMERICAL,
            answer=spec.answer.strip(),
            explanation=spec.explanation,
            fol=None,
            cot=["Type 2 safety agent loop generated a direct fallback answer."],
            premises=spec.premises,
            confidence=spec.confidence or 0.35,
            unit=spec.unit,
            error=None,
            routing_diagnostics=diagnostics,
        )

    diagnostics["type2_agent_loop"] = {
        "used": True,
        "enabled": True,
        "trigger": trigger,
        "attempts": attempts,
        "fallback": "guardrail_response",
    }
    return _guardrail_response(request, trigger, diagnostics)


def _guardrail_response(
    request: PredictionRequest,
    trigger: str,
    diagnostics: dict[str, Any],
) -> PredictionResponse:
    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=QuestionType.UNKNOWN,
        answer="",
        explanation="No verified Type 2 answer could be generated.",
        fol=None,
        cot=["Type 2 pipeline guardrail returned a structured response instead of raising."],
        premises=[],
        confidence=0.0,
        unit=None,
        error=None,
        routing_diagnostics=diagnostics | {"guardrail_trigger": trigger},
    )


def _agent_loop_failure_context(
    trigger: str,
    previous_response: PredictionResponse | None,
) -> str:
    if previous_response is None:
        return trigger
    payload = {
        "trigger": trigger,
        "previous_answer": previous_response.answer,
        "previous_unit": previous_response.unit,
        "previous_explanation": previous_response.explanation,
    }
    return str(payload)


def _question_type(result: Type2SolveResult) -> QuestionType:
    if result.extraction.kind == Type2QuestionKind.CONCEPTUAL:
        return QuestionType.OPEN_ENDED
    return QuestionType.NUMERICAL


def _build_explanation(result: Type2SolveResult) -> str:
    if result.error is not None:
        return result.verification.message
    return result.premises[0] if result.premises else result.verification.message


def _try_llm_extraction(
    question: str,
    settings: Settings | None = None,
) -> Extraction | None:
    try:
        spec = parse_with_llm(question, settings=settings)
    except Exception:
        return None
    if spec is None:
        return None

    quantities: dict[str, Quantity] = {}
    for item in spec.quantities:
        try:
            value = parse_quantity(item.value, item.unit)
        except Exception:
            continue
        key = _canonical_quantity_name(item.name)
        if key in quantities:
            suffix = 2
            while f"{key}_{suffix}" in quantities:
                suffix += 1
            key = f"{key}_{suffix}"
        quantities[key] = Quantity(
            name=key,
            value=value,
            evidence=item.evidence or f"{item.name} = {item.value} {item.unit}",
            confidence=0.7,
        )

    return Extraction(
        kind=Type2QuestionKind.NUMERICAL,
        normalized_question=question,
        target=spec.target,
        quantities=quantities,
        notes=tuple(spec.notes),
    )


def _with_llm_question_kind(
    extraction: Extraction,
    question: str,
    settings: Settings | None = None,
) -> Extraction:
    settings = settings or get_settings()
    notes = list(extraction.notes)
    if not settings.type2_use_llm_question_kind_routing:
        notes.append("question_kind_source=disabled; fail_safe_kind=numerical")
        return replace(extraction, kind=Type2QuestionKind.NUMERICAL, notes=tuple(notes))

    try:
        spec = classify_question_kind_with_llm(question, settings=settings)
    except Exception as exc:
        spec = None
        notes.append(f"question_kind_source=llm_error; reason={exc}")

    if spec is None:
        notes.append("question_kind_source=llm_unavailable; fail_safe_kind=numerical")
        return replace(extraction, kind=Type2QuestionKind.NUMERICAL, notes=tuple(notes))

    notes.append(
        "question_kind_source=llm; "
        f"confidence={spec.confidence}; reason={spec.reason or ''}".rstrip()
    )
    return replace(extraction, kind=Type2QuestionKind(spec.kind), notes=tuple(notes))


def _merge_extractions(heuristic: Extraction, llm_extraction: Extraction) -> Extraction:
    quantities = dict(heuristic.quantities)
    notes = [*heuristic.notes]

    for note in llm_extraction.notes:
        if note and note not in notes:
            notes.append(note)

    for key, quantity in llm_extraction.quantities.items():
        canonical_key = _canonical_quantity_name(key)
        canonical_quantity = Quantity(
            name=canonical_key,
            value=quantity.value,
            evidence=quantity.evidence,
            confidence=quantity.confidence,
        )
        if _has_equivalent_quantity(quantities, canonical_key, canonical_quantity):
            continue
        target_key = canonical_key
        if target_key in quantities:
            suffix = 2
            while f"{target_key}_{suffix}" in quantities:
                suffix += 1
            target_key = f"{target_key}_{suffix}"
        quantities[target_key] = canonical_quantity

    target = llm_extraction.target or heuristic.target
    normalized_question = heuristic.normalized_question or llm_extraction.normalized_question

    return Extraction(
        kind=Type2QuestionKind.NUMERICAL,
        normalized_question=normalized_question,
        target=target,
        quantities=quantities,
        notes=tuple(notes),
    )


def _has_equivalent_quantity(
    quantities: dict[str, Quantity],
    key: str,
    candidate: Quantity,
) -> bool:
    existing = quantities.get(key)
    if existing is None:
        return False
    try:
        converted = candidate.value.to(existing.value.units)
        return abs(float(converted.magnitude) - float(existing.value.magnitude)) <= 1e-9
    except Exception:
        return False


def _canonical_quantity_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "potential_difference": "voltage",
        "electric_potential": "voltage",
        "pd": "voltage",
        "i": "current",
        "u": "voltage",
        "v": "voltage",
        "r": "resistance",
        "distance": "length",
        "separation": "length",
        "radius": "length",
        "height": "length",
        "width": "length",
        "side": "length",
        "electric_field_strength": "electric_field",
        "field_strength": "electric_field",
        "electric_force": "force",
        "net_force": "force",
        "heat": "energy",
        "work": "energy",
    }
    return aliases.get(normalized, normalized)
