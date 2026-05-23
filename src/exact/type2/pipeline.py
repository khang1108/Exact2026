from __future__ import annotations

from exact.config import Settings
from exact.datasets.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.type2.extraction.llm_structured import parse_with_llm
from exact.logger import get_request_logger
from exact.type2.extraction.extractor import extract_type2
from exact.type2.fallback.pot import run_pot_fallback
from exact.type2.schemas import Extraction, Quantity, Type2QuestionKind, Type2SolveResult
from exact.type2.solving.solver import answer_conceptual, solve_extraction
from exact.type2.solving.units import parse_quantity


def run_type2_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Run a formula-grounded Type 2 physics pipeline."""
    logger = get_request_logger(
        __name__,
        request_id=request.id,
        task_type=TaskType.TYPE2_PHYSICS.value,
    )
    logger.info("Start Type 2 pipeline")

    extraction = extract_type2(request.question)
    logger.info(
        "Extraction complete: kind=%s target=%s quantities=%s",
        extraction.kind.value,
        extraction.target,
        sorted(extraction.quantities),
    )

    if extraction.kind == Type2QuestionKind.CONCEPTUAL:
        result = answer_conceptual(extraction)
    else:
        result = solve_extraction(extraction)
        if result.error is not None:
            llm_extraction = _try_llm_extraction(request.question, settings=settings)
            if llm_extraction is not None:
                logger.info(
                    "LLM extraction complete: kind=%s target=%s quantities=%s",
                    llm_extraction.kind.value,
                    llm_extraction.target,
                    sorted(llm_extraction.quantities),
                )
                if llm_extraction.kind == Type2QuestionKind.CONCEPTUAL:
                    result = answer_conceptual(llm_extraction)
                else:
                    result = solve_extraction(llm_extraction)
                    extraction = llm_extraction
            if result.error is not None and extraction.kind == Type2QuestionKind.MIXED:
                conceptual_result = answer_conceptual(extraction)
                if conceptual_result.error is None:
                    result = conceptual_result

    if result.error is not None:
        result = run_pot_fallback(extraction, result.verification.message, settings=settings)

    return _to_prediction_response(request, result)


def _to_prediction_response(
    request: PredictionRequest,
    result: Type2SolveResult,
) -> PredictionResponse:
    formula_text = result.formula.expression if result.formula else None
    explanation = _build_explanation(result)
    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=_question_type(result),
        answer=result.answer,
        explanation=explanation,
        fol=formula_text,
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

    if result.formula is None:
        return result.premises[0] if result.premises else result.verification.message

    unit_suffix = f" {result.unit}" if result.unit else ""
    return (
        f"{result.formula.explanation_template} Substituting the extracted values gives "
        f"{result.formula.target} = {result.answer}{unit_suffix}."
    )


def _try_llm_extraction(
    question: str,
    settings: Settings | None = None,
) -> Extraction | None:
    spec = parse_with_llm(question, settings=settings)
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
