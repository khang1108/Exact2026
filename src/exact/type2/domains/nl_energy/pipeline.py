from __future__ import annotations

from exact.common.schemas import PredictionRequest, PredictionResponse, TaskType, QuestionType
from exact.config import Settings, get_settings
from exact.type2.domains.nl_energy.classifier import classify_nl_energy_family
from exact.type2.domains.nl_energy.extraction import extract_nl_energy_quantities, run_llm_extraction_repair
from exact.type2.domains.nl_energy.solvers import format_scalar_answer, solve_nl_energy
from exact.type2.extraction.llm_structured import classify_question_kind_with_llm

def run_nl_energy_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> tuple[PredictionResponse | None, bool]:
    """Run the specialized NL energy deterministic pipeline.
    
    Returns (response, fallback_needed).
    """
    ext = extract_nl_energy_quantities(request.question)
    ext.family = classify_nl_energy_family(request.question)
    
    if ext.family == "UNKNOWN":
        return None, True
        
    ans_tuple = None
    try:
        ans_tuple = solve_nl_energy(ext, request.question)
    except Exception:
        pass
    
    if ans_tuple is None:
        # Retry with LLM repair
        try:
            ext = run_llm_extraction_repair(request.question, ext, settings)
            ans_tuple = solve_nl_energy(ext, request.question)
        except Exception:
            pass
        
    if ans_tuple is None:
        # Still failed, fallback to generic pipeline
        return None, True
        
    value, unit = ans_tuple
    
    if isinstance(value, float):
        answer_str = format_scalar_answer(value)
    else:
        answer_str = str(value)
        
    q_type, question_kind_route = _llm_question_type(request.question, settings)
    diagnostics = {
        "domain": "NL_ENERGY",
        "family": ext.family,
        "solver": "nl_energy_solver",
        "fallback_used": False,
        "reason": "solved_by_deterministic_energy_formula",
        "llm_question_kind_route": question_kind_route,
    }

    response = PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=q_type,
        answer=answer_str,
        explanation="Solved deterministically using the NL energy domain solver.",
        fol=None,
        cot=["Extracted quantities", f"Identified family: {ext.family}", "Applied deterministic energy formula."],
        premises=["Deterministic energy equations."],
        confidence=1.0,
        unit=unit,
        error=None,
        routing_diagnostics=diagnostics,
    )
    
    return response, False


def _llm_question_type(
    question: str,
    settings: Settings | None,
) -> tuple[QuestionType, dict[str, object]]:
    settings = settings or get_settings()
    if not settings.type2_use_llm_question_kind_routing:
        return QuestionType.NUMERICAL, {
            "source": "disabled",
            "kind": "numerical",
            "fail_safe_kind": "numerical",
        }

    try:
        spec = classify_question_kind_with_llm(question, settings=settings)
    except Exception as exc:
        return QuestionType.NUMERICAL, {
            "source": "llm_error",
            "kind": "numerical",
            "fail_safe_kind": "numerical",
            "reason": str(exc),
        }

    if spec is None:
        return QuestionType.NUMERICAL, {
            "source": "llm_unavailable",
            "kind": "numerical",
            "fail_safe_kind": "numerical",
        }

    q_type = QuestionType.OPEN_ENDED if spec.kind == "conceptual" else QuestionType.NUMERICAL
    return q_type, {
        "source": "llm",
        "kind": spec.kind,
        "confidence": spec.confidence,
        "reason": spec.reason,
    }
