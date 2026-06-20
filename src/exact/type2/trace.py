from __future__ import annotations

from exact.type2.schemas import SolutionStep, SolutionTrace, SolveContext, SolveResult, VerificationResult


def build_solution_trace(result: SolveResult, verification: VerificationResult, ctx: SolveContext) -> SolutionTrace:
    steps = list(result.solution_steps)
    steps.append(
        SolutionStep(
            step_id=f"s{len(steps) + 1}",
            kind="verification",
            description="Verification completed on the final solver output.",
            result_value=result.answer,
            result_unit=result.unit,
        )
    )
    return SolutionTrace(
        query_id=ctx.request.query_id,
        question_text=ctx.request.problem_text,
        solver_used=result.solver_name,
        target_dimension=ctx.ir.target.dimension,
        known_quantities=list(ctx.ir.knowns.values()),
        formula_ids_used=list(result.formula_ids_used),
        steps=steps,
        final_answer=result.answer or "",
        final_unit=result.unit or "",
        verification_accepted=verification.accepted,
        verification_warnings=list(verification.warnings),
        assumptions=list(ctx.extraction.assumptions),
        warnings=list(result.warnings),
    )
