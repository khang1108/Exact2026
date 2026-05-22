"""Forward-chaining Horn solver backend."""

from __future__ import annotations

from exact.symbolic_solvers.forward_chain.solver import (
    ForwardChainSolver,
    derive_closure,
    solve_query,
)

__all__ = ["ForwardChainSolver", "derive_closure", "solve_query"]
