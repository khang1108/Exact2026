"""Z3 backend for symbolic verification.

This initial backend supports the same Horn-like IR as the forward chainer, but
executes entailment checks through Z3. Later we can extend the encoder with
integer credits, GPA ranges, dates, and richer FOL fragments.
"""

from __future__ import annotations

from dataclasses import dataclass

from exact.logic.ir import Atom, SolveResult
from exact.logic.kb import KnowledgeBase
from exact.symbolic_solvers.forward_chain import ForwardChainSolver
from exact.symbolic_solvers.z3_solver.encoder import encode_kb


@dataclass(frozen=True)
class Z3Solver:
    """SMT-backed solver using entailment checks over encoded constraints."""

    name: str = "z3_bool_horn"
    timeout_ms: int = 5000
    explain_with_forward_chain: bool = True

    def solve(self, kb: KnowledgeBase, claim: Atom) -> SolveResult:
        try:
            judgment = self._judge(kb, claim)
        except ImportError as exc:
            return SolveResult(
                label="Unknown",
                claim=claim,
                mode=self.name,
                warnings=(f"z3-solver is not installed: {exc}", *kb.warnings),
            )
        except Exception as exc:
            return SolveResult(
                label="Unknown",
                claim=claim,
                mode=self.name,
                warnings=(f"Z3 solver failed: {exc}", *kb.warnings),
            )

        if self.explain_with_forward_chain and judgment in {"Yes", "No"}:
            proof_result = ForwardChainSolver(name=self.name).solve(kb, claim)
            if proof_result.label == judgment:
                return proof_result

        return SolveResult(
            label=judgment,
            claim=claim,
            mode=self.name,
            warnings=kb.warnings,
        )

    def _judge(self, kb: KnowledgeBase, claim: Atom) -> str:
        from z3 import Not, Solver, unsat

        constraints, symbols = encode_kb(kb, claim)
        claim_symbol = symbols[claim]
        negated_claim_symbol = symbols[claim.negation()]

        if _is_unsat(constraints, Not(claim_symbol), self.timeout_ms):
            return "Yes"
        if _is_unsat(constraints, Not(negated_claim_symbol), self.timeout_ms):
            return "No"
        return "Unknown"


def _is_unsat(constraints: list[object], extra_constraint: object, timeout_ms: int) -> bool:
    from z3 import Solver, unsat

    solver = Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*constraints)
    solver.add(extra_constraint)
    return solver.check() == unsat
