from __future__ import annotations

import math
from typing import Any

from exact.type2.solver_contract.models import SolverContract
from exact.type2.deterministic import DeterministicStageResult
from exact.type2.schemas import Type2SolveResult, Verification, Extraction
from exact.type2.solving.units import ureg


def run_general_vector_resultant_solver(contract: SolverContract, extraction: Extraction) -> DeterministicStageResult:
    diagnostics: dict[str, Any] = {
        "strategy": "general_vector_resultant_solver",
        "status": "unsolved",
    }
    
    if contract.has_unresolved():
        diagnostics["reason"] = "Contract has unresolved parsing errors."
        return DeterministicStageResult(result=None, diagnostics=diagnostics)

    try:
        # 1. Identify input vectors
        vectors = []
        resultant_body = None
        for b in contract.bodies:
            # We treat bodies with known values as vectors if they have same unit types
            if b.value is not None:
                if b.role in {"resultant", "net_force"}:
                    resultant_body = b
                else:
                    vectors.append(b)

        if len(vectors) < 2:
            diagnostics["reason"] = f"Expected at least 2 input vectors, got {len(vectors)}"
            return DeterministicStageResult(result=None, diagnostics=diagnostics)
            
        v1 = vectors[0].value
        v2 = vectors[1].value
        v_unit = v1.units

        # Target logic
        is_solving_angle = "angle" in contract.target.quantity.lower()
        is_solving_resultant = not is_solving_angle
        
        # 2. Check geometry relations for angle or layout
        angle_rad = None
        is_parallel = False
        is_opposite = False
        is_perpendicular = False
        
        for rel in contract.geometry.relations:
            rel_type = rel.type.lower()
            if rel_type in {"parallel", "same_direction"}:
                is_parallel = True
            elif rel_type in {"opposite", "opposite_direction"}:
                is_opposite = True
            elif rel_type == "perpendicular":
                is_perpendicular = True
            elif rel_type == "angle" and rel.value is not None:
                try:
                    angle_rad = float(rel.value.quantity.to("radian").magnitude)
                except Exception:
                    pass

        # 3. Solve Resultant
        if is_solving_resultant:
            v1_mag = float(v1.magnitude)
            v2_mag = float(v2.to(v_unit).magnitude)
            
            if is_parallel:
                r_mag = v1_mag + v2_mag
                diagnostics["math"] = "R = F1 + F2"
            elif is_opposite:
                r_mag = abs(v1_mag - v2_mag)
                diagnostics["math"] = "R = |F1 - F2|"
            elif is_perpendicular:
                r_mag = math.sqrt(v1_mag**2 + v2_mag**2)
                diagnostics["math"] = "R = sqrt(F1^2 + F2^2)"
            elif angle_rad is not None:
                r_mag = math.sqrt(v1_mag**2 + v2_mag**2 + 2 * v1_mag * v2_mag * math.cos(angle_rad))
                diagnostics["math"] = "R = sqrt(F1^2 + F2^2 + 2*F1*F2*cos(theta))"
            else:
                diagnostics["reason"] = "Could not identify vector relationship (parallel, opposite, perpendicular, or explicit angle)."
                return DeterministicStageResult(result=None, diagnostics=diagnostics)
                
            formatted_answer = f"{r_mag:.6f}".rstrip("0").rstrip(".") or "0"
            if abs(r_mag) >= 1e4 or (0 < abs(r_mag) < 1e-3):
                formatted_answer = f"{r_mag:.6g}"
                
            solve_res = Type2SolveResult(
                answer=formatted_answer,
                unit=str(v_unit),
                value=r_mag * v_unit,
                routing_diagnostics=diagnostics | {"status": "solved"},
                verification=Verification(True, "Solved by general vector resultant solver."),
                extraction=extraction,
                formula=None,
                cot=["Identified algebraic vector sum relationship.", f"Applied formula: {diagnostics['math']}"],
                premises=[],
                confidence=0.9,
                error=None
            )
            return DeterministicStageResult(result=solve_res, diagnostics=diagnostics | {"status": "solved"})
            
        # 4. Solve Angle given Resultant
        elif is_solving_angle:
            if resultant_body is None or resultant_body.value is None:
                diagnostics["reason"] = "Solving for angle requires known resultant magnitude."
                return DeterministicStageResult(result=None, diagnostics=diagnostics)
                
            v1_mag = float(v1.magnitude)
            v2_mag = float(v2.to(v_unit).magnitude)
            r_mag = float(resultant_body.value.to(v_unit).magnitude)
            
            # cos(theta) = (R^2 - F1^2 - F2^2) / (2 F1 F2)
            cos_theta = (r_mag**2 - v1_mag**2 - v2_mag**2) / (2 * v1_mag * v2_mag)
            
            # Clamp for floating point errors
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta_rad = math.acos(cos_theta)
            theta_deg = math.degrees(theta_rad)
            
            diagnostics["math"] = "theta = acos((R^2 - F1^2 - F2^2) / (2*F1*F2))"
            
            formatted_answer = f"{theta_deg:.6f}".rstrip("0").rstrip(".") or "0"
            
            solve_res = Type2SolveResult(
                answer=formatted_answer,
                unit="degrees",
                value=theta_deg * ureg.degree,
                routing_diagnostics=diagnostics | {"status": "solved"},
                verification=Verification(True, "Solved by general vector resultant solver."),
                extraction=extraction,
                formula=None,
                cot=["Identified vector sum angle problem.", f"Applied formula: {diagnostics['math']}"],
                premises=[],
                confidence=0.9,
                error=None
            )
            return DeterministicStageResult(result=solve_res, diagnostics=diagnostics | {"status": "solved"})

    except Exception as e:
        diagnostics["error"] = str(e)
        
    return DeterministicStageResult(result=None, diagnostics=diagnostics)
