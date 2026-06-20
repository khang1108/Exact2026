from __future__ import annotations

from dataclasses import replace

from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.config import Settings, get_settings
from exact.type2.domains.thcb.extraction import extract_thcb_heuristic, extract_thcb_with_llm
from exact.type2.domains.thcb.schemas import ThcbAnswer, ThcbContract
from exact.type2.domains.thcb.solver import solve_thcb_contract
from exact.type2.extraction.extractor import extract_type2
from exact.type2.formulas.knowledge import retrieve_formula_context
from exact.type2.pipeline import _to_prediction_response
from exact.type2.routing import build_routing_diagnostics, mark_current_solver_used
from exact.type2.solving.pot_solver import solve_with_pot


def try_thcb_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> tuple[PredictionResponse | None, bool]:
    settings = settings or get_settings()
    contract, answer, source = _solve_with_contract_order(request.question, settings)
    if answer is not None:
        return _to_response(request, contract, answer, source), False
    return None, True


def run_thcb_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    settings = settings or get_settings()
    fast_response, fallback = try_thcb_pipeline(request, settings)
    if not fallback and fast_response is not None:
        return fast_response

    contract, _answer_unused, source = _solve_with_contract_order(request.question, settings)

    extraction = extract_type2(request.question)
    formula_context = retrieve_formula_context(
        _thcb_formula_query(request.question, contract),
        extraction,
        limit=min(settings.type2_formula_limit, 16),
        settings=settings,
    )
    diagnostics = build_routing_diagnostics(
        request.question,
        extraction,
        formula_context,
        request_id=request.id,
    )
    result = solve_with_pot(
        extraction,
        formula_context,
        settings=settings,
        generate_explanation=True,
    )
    result = replace(
        result,
        routing_diagnostics={
            **mark_current_solver_used(diagnostics, error=result.error),
            "domain": "THCB",
            "extraction_source": source,
            "formula_scope": "thcb_measurement_error_and_basic_circuits",
        },
    )
    return _to_prediction_response(request, result)


def _solve_with_contract_order(
    question: str,
    settings: Settings,
) -> tuple[ThcbContract, ThcbAnswer | None, str]:
    llm_contract = _try_llm_contract(question, settings)
    if llm_contract is not None:
        answer = solve_thcb_contract(llm_contract)
        if answer is not None:
            return llm_contract, answer, "llm_thcb_contract"

    heuristic = extract_thcb_heuristic(question)
    answer = solve_thcb_contract(heuristic)
    if answer is not None:
        return heuristic, answer, "heuristic_thcb_contract"
    return heuristic, None, "heuristic_thcb_contract"


def _try_llm_contract(question: str, settings: Settings) -> ThcbContract | None:
    try:
        contract = extract_thcb_with_llm(question, settings)
    except Exception:
        return None
    if contract is None or contract.family == "UNKNOWN":
        return None
    return contract


def _to_response(
    request: PredictionRequest,
    contract: ThcbContract,
    answer: ThcbAnswer,
    source: str,
) -> PredictionResponse:
    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=QuestionType.OPEN_ENDED if contract.target == "conceptual" else QuestionType.NUMERICAL,
        answer=answer.answer,
        explanation=answer.explanation,
        fol=None,
        cot=[f"Built THCB contract using {source}.", *answer.cot],
        premises=[answer.explanation],
        confidence=answer.confidence,
        unit=answer.unit,
        error=None,
        routing_diagnostics={
            "domain": "THCB",
            "family": contract.family,
            "target": contract.target,
            "solver": "thcb_deterministic_solver",
            "extraction_source": source,
            "fallback_used": False,
        },
    )


def _thcb_formula_query(question: str, contract: ThcbContract) -> str:
    scope_terms = {
        "MEASUREMENT_ERROR": "absolute error relative error mean absolute error measurement uncertainty",
        "ERROR_PROPAGATION": "error propagation relative error voltage current resistance power",
        "PARALLEL_CIRCUIT": "parallel circuit ohm law equivalent resistance branch current total current power",
        "SIMPLE_CIRCUIT": "ohm law current voltage resistance power",
    }
    return f"{question}\nTHCB scope: {scope_terms.get(contract.family, 'measurement error basic circuit')}"
