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
from exact.type2.geometry_context import build_geometry_prompt_context
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


def _accept_specialized_response(
    response: PredictionResponse | None,
    *,
    expected_domain: str,
) -> tuple[bool, str]:
    if response is None:
        return False, "response_missing"
    if response.error:
        return False, "response_has_error"
    if not str(response.answer or "").strip():
        return False, "answer_missing"

    diagnostics = response.routing_diagnostics or {}
    if diagnostics.get("domain") != expected_domain:
        return False, "domain_mismatch"
    if diagnostics.get("fallback_used") is True:
        return False, "specialized_fallback_used"

    solver = str(diagnostics.get("solver") or "").strip()
    if not solver:
        return False, "solver_missing"

    if expected_domain in {"THCB", "DDT", "NL_ENERGY"}:
        if not str(diagnostics.get("family") or "").strip():
            return False, "family_missing"

    if expected_domain == "NL_ENERGY" and solver != "nl_energy_solver":
        return False, "unexpected_solver"
    if expected_domain == "THCB" and solver != "thcb_deterministic_solver":
        return False, "unexpected_solver"
    if expected_domain == "DDT" and solver != "ddt_deterministic_solver":
        return False, "unexpected_solver"

    return True, "accepted"


def _accept_routed_response(response: PredictionResponse | None) -> tuple[bool, str]:
    if response is None:
        return False, "response_missing"
    if response.error:
        return False, "response_has_error"
    if not str(response.answer or "").strip():
        return False, "answer_missing"
    return True, "accepted"


def run_type2_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Route a Type 2 question to one heuristic domain pipeline before fallback."""
    from exact.type2.domains.ch.pipeline import run_ch_pipeline
    from exact.type2.domains.ddt.pipeline import run_ddt_pipeline
    from exact.type2.domains.dt.pipeline import run_dt_pipeline
    from exact.type2.domains.ld.pipeline import run_ld_pipeline
    from exact.type2.domains.nl_energy.pipeline import run_nl_energy_pipeline
    from exact.type2.domains.router import route_domain
    from exact.type2.domains.td.pipeline import run_td_pipeline
    from exact.type2.domains.thcb.pipeline import try_thcb_pipeline

    settings = settings or get_settings()
    try:
        response: PredictionResponse | None
        routed_domain = route_domain(request.id, request.question)
        attempts: list[str] = []

        if routed_domain == "THCB":
            response, fallback = try_thcb_pipeline(request, settings)
            accepted, reason = _accept_specialized_response(response, expected_domain="THCB")
            attempts.append(_domain_attempt("THCB", fallback=fallback, accepted=accepted, reason=reason))
            if not fallback and accepted and response is not None:
                return _finish_domain_response(
                    request,
                    response,
                    settings,
                    domain="THCB",
                    routed_domain=routed_domain,
                    attempts=attempts,
                    question_kind_route=_question_kind_route_metadata(request, settings, allow_llm=False),
                )

        elif routed_domain == "DDT":
            response, fallback = run_ddt_pipeline(request, settings)
            accepted, reason = _accept_specialized_response(response, expected_domain="DDT")
            attempts.append(_domain_attempt("DDT", fallback=fallback, accepted=accepted, reason=reason))
            if not fallback and accepted and response is not None:
                return _finish_domain_response(
                    request,
                    response,
                    settings,
                    domain="DDT",
                    routed_domain=routed_domain,
                    attempts=attempts,
                    question_kind_route=_question_kind_route_metadata(request, settings, allow_llm=False),
                )

        elif routed_domain == "NL_ENERGY":
            response, fallback = run_nl_energy_pipeline(request, settings)
            accepted, reason = _accept_specialized_response(response, expected_domain="NL_ENERGY")
            attempts.append(_domain_attempt("NL_ENERGY", fallback=fallback, accepted=accepted, reason=reason))
            if not fallback and accepted and response is not None:
                return _finish_domain_response(
                    request,
                    response,
                    settings,
                    domain="NL_ENERGY",
                    routed_domain=routed_domain,
                    attempts=attempts,
                    question_kind_route=_question_kind_route_metadata(request, settings, allow_llm=False),
                )

        elif routed_domain == "LD":
            response = run_ld_pipeline(request, settings)
            accepted, reason = _accept_routed_response(response)
            attempts.append(f"LD:{'accepted' if accepted else reason}")
            if accepted:
                return _finish_domain_response(
                    request,
                    response,
                    settings,
                    domain="LD",
                    routed_domain=routed_domain,
                    attempts=attempts,
                    question_kind_route=_question_kind_route_metadata(request, settings, allow_llm=False),
                )

        elif routed_domain == "CH":
            response = run_ch_pipeline(request, settings)
            accepted, reason = _accept_routed_response(response)
            attempts.append(f"CH:{'accepted' if accepted else reason}")
            if accepted:
                return _finish_domain_response(
                    request,
                    response,
                    settings,
                    domain="CH",
                    routed_domain=routed_domain,
                    attempts=attempts,
                    question_kind_route=_question_kind_route_metadata(request, settings, allow_llm=False),
                )

        elif routed_domain == "DT":
            response = run_dt_pipeline(request, settings)
            accepted, reason = _accept_routed_response(response)
            attempts.append(f"DT:{'accepted' if accepted else reason}")
            if accepted:
                return _finish_domain_response(
                    request,
                    response,
                    settings,
                    domain="DT",
                    routed_domain=routed_domain,
                    attempts=attempts,
                    question_kind_route=_question_kind_route_metadata(request, settings, allow_llm=False),
                )

        elif routed_domain == "TD":
            response = run_td_pipeline(request, settings)
            accepted, reason = _accept_routed_response(response)
            attempts.append(f"TD:{'accepted' if accepted else reason}")
            if accepted or response is not None:
                return _finish_domain_response(
                    request,
                    response,
                    settings,
                    domain="TD",
                    routed_domain=routed_domain,
                    attempts=attempts,
                    question_kind_route=_question_kind_route_metadata(request, settings, allow_llm=False),
                )

        question_kind_route = _question_kind_route_metadata(request, settings, allow_llm=True)
        domain_hint = str(question_kind_route.get("domain") or "")
        response = run_generic_pipeline(request, settings, domain_hint=domain_hint)
        return _finish_domain_response(
            request,
            response,
            settings,
            domain="GENERIC",
            routed_domain=routed_domain,
            attempts=attempts,
            question_kind_route=question_kind_route,
        )
    except Exception as exc:
        return _recovery_loop_response(
            request,
            settings,
            trigger=f"pipeline_exception: {exc}",
            diagnostics={"exception_type": type(exc).__name__},
        )


def run_type2_internal_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Backward-compatible alias for older smoke tests and scripts.

    The rebuilt Type 2 reference pipeline now returns the shared
    ``PredictionResponse`` directly instead of the older internal result model.
    """

    return run_type2_pipeline(request, settings=settings)


def _domain_attempt(
    domain: str,
    *,
    fallback: bool,
    accepted: bool,
    reason: str,
) -> str:
    if fallback:
        return f"{domain}:fallback_requested"
    return f"{domain}:{'accepted' if accepted else reason}"


def _finish_domain_response(
    request: PredictionRequest,
    response: PredictionResponse,
    settings: Settings,
    *,
    domain: str,
    routed_domain: str,
    attempts: list[str],
    question_kind_route: dict[str, Any],
) -> PredictionResponse:
    response = _with_domain_route_diagnostics(
        response,
        {
            "domain": domain,
            "routed_domain": routed_domain,
            "source": "type2_domain_heuristic",
            "attempts": attempts,
            "question_kind_route": question_kind_route,
        },
    )
    return _with_recovery_loop_if_needed(
        request,
        response,
        settings,
        trigger=_recovery_loop_trigger(response),
    )


def _question_kind_route_metadata(
    request: PredictionRequest,
    settings: Settings,
    *,
    allow_llm: bool,
) -> dict[str, Any]:
    from exact.type2.domains.router import route_domain_with_metadata, route_question_kind_heuristic

    if allow_llm:
        return route_domain_with_metadata(request.id, request.question, settings=settings).to_dict()

    return {
        "domain": route_question_kind_heuristic(request.id, request.question),
        "source": "heuristic",
        "confidence": None,
        "reason": None,
        "fallback_reason": "LLM question-kind route skipped for accepted specialized pipeline route",
    }


def run_generic_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
    domain_hint: str | None = None,
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
    if domain_hint in {"numerical", "conceptual", "mixed"}:
        extraction = replace(extraction, kind=Type2QuestionKind(domain_hint))

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

    from exact.type2.solving.pot_solver import _try_executable_formula_fallback

    fallback = _try_executable_formula_fallback(
        extraction,
        formula_context,
        "Generic Type 2 route uses formula execution before LLM PoT.",
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


def _recovery_loop_trigger(response: PredictionResponse) -> str | None:
    if response.error:
        return response.error
    if not str(response.answer or "").strip():
        return "empty_answer"
    return None


def _with_recovery_loop_if_needed(
    request: PredictionRequest,
    response: PredictionResponse,
    settings: Settings,
    trigger: str | None,
) -> PredictionResponse:
    diagnostics = dict(response.routing_diagnostics or {})
    diagnostics["type2_llm_configured"] = bool(settings.llm_base_url)
    diagnostics["type2_llm_domain_routing_enabled"] = settings.type2_use_llm_domain_routing
    diagnostics["type2_llm_question_kind_routing_enabled"] = settings.type2_use_llm_question_kind_routing
    diagnostics["type2_recovery_loop_enabled"] = settings.type2_use_recovery_loop
    diagnostics["type2_recovery_loop_max_attempts"] = settings.type2_recovery_loop_max_attempts
    diagnostics["type2_pot_solver_enabled"] = settings.type2_use_pot_solver
    response = response.model_copy(update={"routing_diagnostics": diagnostics})
    if not trigger:
        diagnostics["type2_recovery_loop"] = {
            "used": False,
            "trigger": None,
            "reason": "no pipeline error",
        }
        response = response.model_copy(update={"routing_diagnostics": diagnostics})
        return response
    return _recovery_loop_response(
        request,
        settings,
        trigger=trigger,
        diagnostics=diagnostics,
        previous_response=response,
    )


def _recovery_loop_response(
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
    diagnostics["type2_recovery_loop_enabled"] = settings.type2_use_recovery_loop
    diagnostics["type2_recovery_loop_max_attempts"] = settings.type2_recovery_loop_max_attempts
    diagnostics["type2_pot_solver_enabled"] = settings.type2_use_pot_solver
    if not settings.type2_use_recovery_loop:
        diagnostics["type2_recovery_loop"] = {
            "used": False,
            "trigger": trigger,
            "reason": "recovery loop disabled",
        }
        if previous_response is not None:
            return previous_response.model_copy(update={"routing_diagnostics": diagnostics})
        return _guardrail_response(request, trigger, diagnostics)

    attempts: list[dict[str, Any]] = []
    for attempt in range(1, settings.type2_recovery_loop_max_attempts + 1):
        extraction = _build_solver_extraction(request.question, settings=settings)
        formula_context = retrieve_formula_context(
            request.question,
            extraction,
            limit=settings.type2_formula_limit,
            settings=settings,
        )
        geometry_prompt_context = build_geometry_prompt_context(extraction)

        result = solve_with_pot(
            extraction,
            formula_context,
            settings=settings,
            generate_explanation=settings.type2_generate_explanation,
        )
        if _is_usable_agent_result(result):
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "solved_with_pot",
                    "answer": result.answer,
                    "unit": result.unit,
                }
            )
            diagnostics["type2_recovery_loop"] = {
                "used": True,
                "enabled": True,
                "trigger": trigger,
                "attempts": attempts,
                "strategy": "re_extract_then_pot_then_direct_answer",
            }
            result = replace(result, routing_diagnostics=diagnostics)
            return _to_prediction_response(request, result)

        try:
            spec = generate_direct_answer(
                request.question,
                _recovery_loop_failure_context(
                    trigger,
                    previous_response,
                    attempt,
                    attempts,
                    result.error if result.error else None,
                    geometry_prompt_context=geometry_prompt_context,
                ),
                settings=settings,
                temperature=_recovery_loop_temperature(settings, attempt),
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
        attempts.append(
            {
                "attempt": attempt,
                "status": "answered_directly",
                "answer": spec.answer.strip(),
                "unit": spec.unit,
            }
        )
        diagnostics["type2_recovery_loop"] = {
            "used": True,
            "enabled": True,
            "trigger": trigger,
            "attempts": attempts,
            "strategy": "re_extract_then_pot_then_direct_answer",
        }
        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE2_PHYSICS,
            question_type=QuestionType.OPEN_ENDED if spec.unit is None else QuestionType.NUMERICAL,
            answer=spec.answer.strip(),
            explanation=spec.explanation,
            fol=None,
            cot=["Type 2 recovery loop generated a direct answer after structured solving did not finish cleanly."],
            premises=spec.premises,
            confidence=spec.confidence or 0.35,
            unit=spec.unit,
            error=None,
            routing_diagnostics=diagnostics,
        )

    diagnostics["type2_recovery_loop"] = {
        "used": True,
        "enabled": True,
        "trigger": trigger,
        "attempts": attempts,
        "fallback": "best_previous_response_or_guardrail",
        "strategy": "re_extract_then_pot_then_direct_answer",
    }
    if previous_response is not None and _response_has_answer(previous_response):
        return previous_response.model_copy(update={"routing_diagnostics": diagnostics})
    return _guardrail_response(request, trigger, diagnostics)


def _recovery_loop_temperature(settings: Settings, attempt: int) -> float:
    if attempt <= 1:
        return settings.llm_temperature
    return max(settings.llm_temperature, min(settings.type2_pot_batch_temperature, 0.7))


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


def _recovery_loop_failure_context(
    trigger: str,
    previous_response: PredictionResponse | None,
    attempt: int,
    attempts: list[dict[str, Any]],
    latest_solver_error: str | None = None,
    geometry_prompt_context: str | None = None,
) -> str:
    guidance = (
        "If the structured pipeline context is incomplete or wrong, solve directly from the "
        "question using your physics knowledge. Do not refuse just because the pipeline failed. "
        "Infer the relevant physics principle, compute the answer when possible, and return the "
        "best direct answer. Prefer a concise numeric answer with a unit when the question asks "
        "for a calculation."
    )
    geometry_payload = (
        {"geometry_prompt_context": geometry_prompt_context}
        if geometry_prompt_context
        else {}
    )
    if previous_response is None:
        return str(
            {
                "trigger": trigger,
                "attempt": attempt,
                "previous_attempts": attempts,
                "latest_solver_error": latest_solver_error,
                "guidance": guidance,
                **geometry_payload,
            }
        )
    payload = {
        "trigger": trigger,
        "attempt": attempt,
        "previous_attempts": attempts,
        "previous_answer": previous_response.answer,
        "previous_unit": previous_response.unit,
        "previous_explanation": previous_response.explanation,
        "latest_solver_error": latest_solver_error,
        "guidance": guidance,
        **geometry_payload,
    }
    return str(payload)


def _is_usable_agent_result(result: Type2SolveResult) -> bool:
    return result.error is None and bool(str(result.answer).strip())


def _response_has_answer(response: PredictionResponse) -> bool:
    return bool(str(response.answer or "").strip())


def _question_type(result: Type2SolveResult) -> QuestionType:
    if result.extraction.kind == Type2QuestionKind.CONCEPTUAL:
        return QuestionType.OPEN_ENDED
    return QuestionType.NUMERICAL


def _build_explanation(result: Type2SolveResult) -> str:
    """Build a clean human-readable explanation for Type 2 responses.

    Prefers CoT steps from the solver. Falls back to the verification message
    or a generic explanation to ensure the field is never empty.
    """
    if result.error is not None:
        return result.verification.message or "An error occurred during solving."
    # Use CoT steps joined as sentences if available
    if result.cot:
        # Filter out internal plumbing notes; keep domain-logic steps
        steps = [
            s for s in result.cot
            if not any(
                skip in s.lower()
                for skip in ("extraction checklist", "extracted a formal contract",
                             "validated the contract", "validated the deterministic")
            )
        ]
        if steps:
            return " ".join(steps)
    if result.premises:
        # If premises[0] looks like a diagnostic dict string, don't expose it
        first = result.premises[0]
        if not first.startswith("diagnostics="):
            return first
    return result.verification.message or "Solved via physics pipeline."


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

    notes = list(spec.notes)
    try:
        kind = Type2QuestionKind(spec.kind.strip().lower())
        notes.append(f"question_kind_source=llm_extraction; kind={kind.value}")
    except Exception:
        kind = Type2QuestionKind.NUMERICAL
        notes.append(f"question_kind_source=llm_extraction_invalid; raw_kind={spec.kind}")

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
        kind=kind,
        normalized_question=question,
        target=spec.target,
        quantities=quantities,
        notes=tuple(notes),
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
    if any(note.startswith("question_kind_source=llm_extraction") for note in notes):
        return extraction

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
        if canonical_key in quantities:
            notes.append(f"merge_conflict_skipped={canonical_key}")
            continue
        quantities[canonical_key] = canonical_quantity

    target = heuristic.target or llm_extraction.target
    normalized_question = heuristic.normalized_question or llm_extraction.normalized_question

    return Extraction(
        kind=heuristic.kind,
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
