from __future__ import annotations

from exact.common.schemas import PredictionRequest, PredictionResponse, TaskType, QuestionType
from exact.config import Settings
from exact.type2.domains.nl_energy.classifier import classify_nl_energy_family
from exact.type2.domains.nl_energy.extraction import extract_nl_energy_quantities, run_llm_extraction_repair
from exact.type2.domains.nl_energy.solvers import solve_nl_energy

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
        # Format scalar nicely
        answer_str = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        answer_str = str(value)
        
    diagnostics = {
        "domain": "NL_ENERGY",
        "family": ext.family,
        "solver": "nl_energy_solver",
        "fallback_used": False,
        "reason": "solved_by_deterministic_energy_formula"
    }
    
    q_type = QuestionType.OPEN_ENDED if ext.family in ("GRAPH_SHAPE", "CONCEPTUAL_UNIT") else QuestionType.NUMERICAL
    
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
