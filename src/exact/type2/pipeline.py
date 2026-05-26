from __future__ import annotations

from exact.config import Settings
from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.logger import get_request_logger
from exact.type2.extraction.extractor import extract_type2
from exact.type2.extraction.llm_structured import parse_with_llm
from exact.type2.extraction.verifier import verify_type2_extraction
from exact.type2.formulas.knowledge import retrieve_formula_context
from exact.type2.schemas import Extraction, Quantity, Type2QuestionKind, Type2SolveResult
from exact.type2.solving.pot_solver import solve_with_pot
from exact.type2.solving.units import parse_quantity


_GENERATE_FINAL_EXPLANATION = True


def run_type2_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Run the PoT-first Type 2 physics pipeline.

    Flow:
    query -> formula retrieval -> LLM Pint code solver -> verifier ->
    LLM/evidence explanation -> PredictionResponse.
    """
    logger = get_request_logger(
        __name__,
        request_id=request.id,
        task_type=TaskType.TYPE2_PHYSICS.value,
    )
    logger.info("Start Type 2 PoT-first pipeline")

    extraction = _build_solver_extraction(request.question, settings=settings)
    review = verify_type2_extraction(extraction)
    formula_context = retrieve_formula_context(request.question, extraction)

    logger.info(
        "PoT extraction: kind=%s target=%s quantities=%s extraction_ok=%s formulas=%s",
        extraction.kind.value,
        extraction.target,
        sorted(extraction.quantities),
        review.verification.ok,
        formula_context.formula_ids,
    )

    result = solve_with_pot(
        extraction,
        formula_context,
        settings=settings,
        generate_explanation=_GENERATE_FINAL_EXPLANATION,
    )
    if result.cot is not None:
        result.cot.insert(0, review.verification.message)
    return _to_prediction_response(request, result)


def set_generate_final_explanation(enabled: bool) -> None:
    global _GENERATE_FINAL_EXPLANATION
    _GENERATE_FINAL_EXPLANATION = enabled


def _build_solver_extraction(
    question: str,
    settings: Settings | None = None,
) -> Extraction:
    llm_extraction = _try_llm_extraction(question, settings=settings)
    if llm_extraction is not None:
        return llm_extraction
    return extract_type2(question)


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
    )


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
        key = item.name
        if key in quantities:
            suffix = 2
            while f"{key}_{suffix}" in quantities:
                suffix += 1
            key = f"{key}_{suffix}"
        quantities[key] = Quantity(
            name=item.name,
            value=value,
            evidence=item.evidence or f"{item.name} = {item.value} {item.unit}",
            confidence=0.7,
        )

    kind = spec.kind if spec.kind in {"numerical", "conceptual", "mixed"} else "numerical"
    return Extraction(
        kind=Type2QuestionKind(kind),
        normalized_question=question,
        target=spec.target,
        quantities=quantities,
        notes=tuple(spec.notes),
    )
