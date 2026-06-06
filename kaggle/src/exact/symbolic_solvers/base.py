"""Common interfaces for symbolic solver backends."""

from __future__ import annotations

from typing import Protocol

from exact.logic.ir import Atom, SolveResult
from exact.logic.kb import KnowledgeBase


class SymbolicSolver(Protocol):
    """Solver interface consumed by Type 1 pipelines."""

    name: str

    def solve(self, kb: KnowledgeBase, claim: Atom) -> SolveResult:
        """Return a symbolic judgment and proof trace for `claim`."""
        ...