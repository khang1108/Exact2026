from __future__ import annotations

from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.config import Settings, get_settings
from exact.type2.domains.ddt.extraction import build_ddt_contract
from exact.type2.domains.ddt.schemas import DdtAnswer, DdtContract
from exact.type2.domains.ddt.solver import solve_ddt_contract



def run_ddt_pipeline(
    request: PredictionRequest, 
    settings: Settings | None = None
) -> tuple[PredictionResponse | None, bool]:
    settings = settings or get_settings()
    contract = build_ddt_contract(request.question, settings)
    
    if contract.target == "conceptual" or contract.family == "CONCEPTUAL_SOLENOID":
        return None, True

    answer = solve_ddt_contract(contract)
    if answer is not None:
        return _to_response(request, contract, answer), False

    return None, True



def _to_response(request: PredictionRequest, contract: DdtContract, answer: DdtAnswer) -> PredictionResponse:
    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=QuestionType.OPEN_ENDED if contract.target == "conceptual" else QuestionType.NUMERICAL,
        answer=answer.answer,
        explanation=answer.explanation,
        fol=None,
        cot=[f"Built reconciled DDT contract using {contract.source}.", *answer.cot],
        premises=[answer.explanation],
        confidence=answer.confidence,
        unit=answer.unit,
        error=None,
        routing_diagnostics={
            "domain": "DDT",
            "family": contract.family,
            "target": contract.target,
            "solver": "ddt_deterministic_solver",
            "extraction_source": contract.source,
            "fallback_used": False,
        },
    )


def _ddt_formula_query(question: str, contract: DdtContract) -> str:
    return (
        f"{question}\nDDT scope: solenoid magnetic field inductance induced emf magnetic flux "
        f"RLC reactance impedance resonance LC energy family={contract.family} target={contract.target}"
    )
