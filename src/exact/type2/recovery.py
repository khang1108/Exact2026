from __future__ import annotations

from exact.type2.schemas import SolveContext, SolveResult, VerificationResult
from exact.type2.solvers import PoTSolver


class RecoveryLoop:
    def __init__(self, settings=None) -> None:
        self.settings = settings

    def recover(self, ctx: SolveContext, solve_result: SolveResult, verification: VerificationResult) -> SolveResult:
        if verification.accepted:
            return solve_result
        ctx.diagnostics.notes.append("RecoveryLoop invoked")
        return PoTSolver().solve(ctx)
