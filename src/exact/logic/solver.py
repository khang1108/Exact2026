"""Compatibility wrapper for symbolic solvers.

New code should import from `exact.symbolic_solvers`. This module stays as a
stable bridge for older imports while the codebase is reorganized.
"""

from __future__ import annotations

from exact.symbolic_solvers.forward_chain import ForwardChainSolver, derive_closure, solve_query
from exact.symbolic_solvers.z3_prop import Z3PropSolver
from exact.symbolic_solvers.z3_solver import Z3Solver

__all__ = ["ForwardChainSolver", "Z3PropSolver", "Z3Solver", "derive_closure", "solve_query"]
