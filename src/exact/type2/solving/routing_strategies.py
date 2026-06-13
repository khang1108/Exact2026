from typing import Any
from exact.type2.schemas import Extraction, Type2SolveResult, Verification
from exact.type2.deterministic import run_deterministic_stage, DeterministicStageResult

from exact.type2.solver_contract.models import SolverContract
from exact.type2.execution_policy import ExecutionPolicy

def run_scalar_numeric_strategy(extraction: Extraction, solver_contract: SolverContract | None = None) -> DeterministicStageResult:
    """Best first solver: deterministic solver
    Second solver: LLM semantic parser -> deterministic"""
    
    stage_result = run_deterministic_stage(extraction)
    if stage_result.result is not None:
        stage_result.diagnostics["strategy"] = "scalar_numeric_strategy"
        return stage_result

    diagnostics = dict(stage_result.diagnostics)
    diagnostics["strategy"] = "scalar_numeric_strategy"
    
    if solver_contract and getattr(solver_contract, "semantic_contract", None):
        from exact.type2.solver_contract.coordinate_builder import build_coordinates
        from exact.type2.solver_contract.solver_adapter import run_electrostatics_vector_solver
        from exact.type2.schemas import Type2SolveResult
        
        errors = getattr(solver_contract, "errors", [])
        
        if not errors:
            coords = build_coordinates(solver_contract)
            if not solver_contract.has_unresolved() and solver_contract.domain == "electrostatics":
                result = run_electrostatics_vector_solver(solver_contract, coords)
                if result.status == "solved" and result.value is not None:
                    diagnostics["semantic_contract"] = solver_contract.semantic_contract.model_dump()
                    diagnostics["solver_contract_errors"] = []
                    return DeterministicStageResult(
                        result=Type2SolveResult(
                            answer=result.answer,
                            unit=result.unit,
                            value=result.value,
                            formula=None,
                            extraction=extraction,
                            verification=Verification(True, "Solved by semantic contract electrostatics solver."),
                            cot=["Built coordinate system from contract geometry", "Computed Coulomb superposition"],
                            premises=[],
                            confidence=0.9,
                            error=None,
                            routing_diagnostics=diagnostics | result.diagnostics,
                        ),
                        diagnostics=diagnostics | result.diagnostics,
                    )
        
        diagnostics["solver_contract_errors"] = errors
        diagnostics["unresolved"] = list(solver_contract.unresolved)
        diagnostics["semantic_contract"] = solver_contract.semantic_contract.model_dump()

    diagnostics["status"] = "stub_unimplemented"
    diagnostics["reason"] = "Semantic contract extracted but no deterministic solver could solve it."
    diagnostics["next_fallback"] = "pot"
    
    return DeterministicStageResult(result=None, diagnostics=diagnostics)


def _infer_vector_input_type(contract: SolverContract) -> str:
    has_vectors = False
    has_sources = False
    
    for b in contract.bodies:
        if b.body_type in {"electric_force", "force", "magnetic_force", "electric_field", "magnetic_field", "vector"}:
            has_vectors = True
        elif b.body_type in {"charge", "mass", "current", "wire", "particle"}:
            has_sources = True
            
    if has_vectors and not has_sources:
        return "given_vectors"
    if has_sources:
        return "physical_sources"
    return "unknown"

def _run_vector_solver(extraction: Extraction, contract: SolverContract, policy: ExecutionPolicy) -> DeterministicStageResult:
    from exact.type2.solver_contract.coordinate_builder import build_coordinates
    from exact.type2.solver_contract.solver_adapter import run_electrostatics_vector_solver
    from exact.type2.physics.general_vector_resultant_solver import run_general_vector_resultant_solver
    
    vector_input_type = _infer_vector_input_type(contract)
    
    if vector_input_type == "given_vectors":
        return run_general_vector_resultant_solver(contract, extraction)
        
    diagnostics: dict[str, Any] = {
        "strategy": "vector_solver",
        "vector_input_type": vector_input_type,
        "semantic_contract": contract.semantic_contract.model_dump() if contract.semantic_contract else None,
        "solver_contract_errors": contract.errors
    }

    if contract.errors:
        diagnostics["status"] = "unsolved"
        diagnostics["reason"] = "solver_contract_validation_failed"
        diagnostics["next_fallback"] = policy.pot_mode.value
        return DeterministicStageResult(result=None, diagnostics=diagnostics)
    
    if vector_input_type == "physical_sources":
        try:
            if not contract.has_unresolved() and contract.target.quantity in ("electric_force", "electric_field"):
                coords = build_coordinates(contract)
                result = run_electrostatics_vector_solver(contract, coords)
                
                if result.status == "solved" and result.value is not None:
                    solve_res = Type2SolveResult(
                        answer=result.answer,
                        unit=result.unit,
                        value=result.value,
                        routing_diagnostics=diagnostics | result.diagnostics,
                        verification=Verification(True, "Solved by vector_solver engine."),
                        extraction=extraction,
                        formula=None,
                        cot=["Built coordinate system from contract geometry", "Computed Coulomb superposition"],
                        premises=[],
                        confidence=0.9,
                        error=None
                    )
                    return DeterministicStageResult(result=_apply_answer_mode(solve_res, policy), diagnostics=diagnostics | result.diagnostics)
        except Exception as e:
            diagnostics["error"] = str(e)

    diagnostics["status"] = "unsolved"
    diagnostics["reason"] = f"vector_solver failed to resolve contract for input_type={vector_input_type}"
    diagnostics["next_fallback"] = policy.pot_mode.value
    
    return DeterministicStageResult(result=None, diagnostics=diagnostics)


def run_multi_value_strategy(extraction: Extraction) -> DeterministicStageResult:
    """Best first solver: decomposer + deterministic per subtask
    Second solver: PoT with strict JSON output"""
    return DeterministicStageResult(
        result=None,
        diagnostics={
            "strategy": "multi_value_strategy",
            "status": "stub_unimplemented",
            "reason": "Decomposer + deterministic per subtask not implemented yet.",
            "next_fallback": "pot_structured_json"
        }
    )


def run_symbolic_strategy(extraction: Extraction) -> DeterministicStageResult:
    """Best first solver: symbolic template / SymPy
    Second solver: LLM semantic parser + SymPy"""
    return DeterministicStageResult(
        result=None,
        diagnostics={
            "strategy": "symbolic_strategy",
            "status": "stub_unimplemented",
            "reason": "Symbolic template / SymPy solver not implemented yet.",
            "next_fallback": "llm_semantic_sympy_or_pot"
        }
    )


def run_location_geometry_strategy(extraction: Extraction, solver_contract: SolverContract | None = None) -> DeterministicStageResult:
    """Best first solver: equation solver / geometry solver
    Second solver: LLM contract extractor + deterministic"""
    
    stage_result = run_deterministic_stage(extraction)
    if stage_result.result is not None:
        stage_result.diagnostics["strategy"] = "location_geometry_strategy"
        return stage_result

    diagnostics = dict(stage_result.diagnostics)
    diagnostics["strategy"] = "location_geometry_strategy"

    if solver_contract and getattr(solver_contract, "semantic_contract", None):
        from exact.type2.solver_contract.coordinate_builder import build_coordinates
        from exact.type2.solver_contract.solver_adapter import run_electrostatics_vector_solver
        from exact.type2.schemas import Type2SolveResult
        
        errors = getattr(solver_contract, "errors", [])
        
        if not errors:
            coords = build_coordinates(solver_contract)
            if not solver_contract.has_unresolved() and solver_contract.domain == "electrostatics":
                result = run_electrostatics_vector_solver(solver_contract, coords)
                if result.status == "solved" and result.value is not None:
                    diagnostics["semantic_contract"] = solver_contract.semantic_contract.model_dump()
                    diagnostics["solver_contract_errors"] = []
                    return DeterministicStageResult(
                        result=Type2SolveResult(
                            answer=result.answer,
                            unit=result.unit,
                            value=result.value,
                            formula=None,
                            extraction=extraction,
                            verification=Verification(True, "Solved by semantic contract electrostatics solver."),
                            cot=["Built coordinate system from contract geometry", "Computed Coulomb superposition"],
                            premises=[],
                            confidence=0.9,
                            error=None,
                            routing_diagnostics=diagnostics | result.diagnostics,
                        ),
                        diagnostics=diagnostics | result.diagnostics,
                    )

        diagnostics["solver_contract_errors"] = errors
        diagnostics["unresolved"] = list(solver_contract.unresolved)
        diagnostics["semantic_contract"] = solver_contract.semantic_contract.model_dump()

    diagnostics["status"] = "stub_unimplemented"
    diagnostics["reason"] = "Semantic contract extracted but no deterministic solver could solve it."
    diagnostics["next_fallback"] = "pot"
    
    return DeterministicStageResult(result=None, diagnostics=diagnostics)


def run_directional_strategy(extraction: Extraction) -> DeterministicStageResult:
    """Best first solver: rule-based vector/sign classifier
    Second solver: small LLM classifier"""
    return DeterministicStageResult(
        result=None,
        diagnostics={
            "strategy": "directional_strategy",
            "status": "stub_unimplemented",
            "reason": "Rule-based vector/sign classifier not implemented.",
            "next_fallback": "pot_or_small_llm_classifier"
        }
    )


def run_conceptual_strategy(extraction: Extraction) -> DeterministicStageResult:
    """Best first solver: retrieval/rule QA
    Second solver: small LLM answer classifier"""
    return DeterministicStageResult(
        result=None,
        diagnostics={
            "strategy": "conceptual_strategy",
            "status": "stub_unimplemented",
            "reason": "Retrieval/rule QA not unified in deterministic stage.",
            "next_fallback": "pot_conceptual"
        }
    )

def _apply_answer_mode(result: Type2SolveResult, policy: ExecutionPolicy) -> Type2SolveResult:
    import json
    from dataclasses import replace
    from exact.type2.execution_policy import PotMode
    
    if policy.pot_mode in {PotMode.NUMERIC_MULTI_JSON, PotMode.SYMBOLIC_EXPR_JSON} and not result.answer.startswith("{"):
        # Wrap simple outputs if policy demands JSON format
        wrapped = {"result": result.answer}
        return replace(result, answer=json.dumps(wrapped))
        
    return result

def run_deterministic_solver(extraction: Extraction, contract: SolverContract | None, policy: ExecutionPolicy) -> DeterministicStageResult:
    if contract is None:
        return DeterministicStageResult(
            result=None,
            diagnostics={
                "strategy": "run_deterministic_solver",
                "status": "unsolved",
                "reason": "No solver contract generated.",
                "next_fallback": policy.pot_mode.value
            }
        )
        
    family = policy.solver_family or contract.domain
    
    if family == "vector_solver":
        return _run_vector_solver(extraction, contract, policy)
        
    # Other families fallback
    diagnostics = {
        "strategy": "run_deterministic_solver",
        "status": "unsolved",
        "reason": f"No deterministic solver implemented yet for family: {family}",
        "next_fallback": policy.pot_mode.value
    }
    return DeterministicStageResult(result=None, diagnostics=diagnostics)
