from __future__ import annotations

from exact.type2.schemas import SolveContext, SolveResult
from exact.type2.solvers import ConceptualSolver, FormulaGraphSolver, NetworkCircuitSolver, PoTSolver, VectorGeometrySolver


class SolverManager:
    def __init__(self, settings=None) -> None:
        self.settings = settings

    def solve(self, ctx: SolveContext) -> SolveResult:
        policy = self.build_policy(ctx)
        ctx.diagnostics.solver_policy = [solver.name for solver in policy]
        for solver in policy:
            eligibility = solver.can_solve(ctx)
            ctx.diagnostics.record_eligibility(solver.name, eligibility)
            if not eligibility.eligible:
                continue
            result = solver.solve(ctx)
            ctx.diagnostics.record_result(result)
            if result.success:
                return result
        return SolveResult(success=False, error="NO_SOLVER_SUCCEEDED", confidence=0.0)

    def build_policy(self, ctx: SolveContext):
        if ctx.ir.question_kind == "conceptual":
            return [ConceptualSolver(), PoTSolver()]
        if ctx.ir.flags.get("requires_vector"):
            return [VectorGeometrySolver(), PoTSolver(), FormulaGraphSolver()]
        if ctx.ir.flags.get("requires_network"):
            return [NetworkCircuitSolver(), FormulaGraphSolver(), PoTSolver()]
        if ctx.ir.question_kind == "mixed":
            return [FormulaGraphSolver(), VectorGeometrySolver(), ConceptualSolver(), PoTSolver()]
        return [FormulaGraphSolver(), PoTSolver()]
