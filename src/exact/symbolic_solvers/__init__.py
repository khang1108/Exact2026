"""Symbolic solver backends for EXACT.

This package follows the Logic-LM separation: parsers produce symbolic
representations, solvers execute them deterministically, and pipelines decide
how to fall back or refine when execution fails.
"""

from __future__ import annotations

from exact.symbolic_solvers.base import SymbolicSolver
from exact.symbolic_solvers.forward_chain import ForwardChainSolver, derive_closure, solve_query
from exact.symbolic_solvers.z3_solver import Z3Solver
from exact.symbolic_solvers.z3_prop import Z3PropSolver

__all__ = [
    "ForwardChainSolver",
    "SymbolicSolver",
    "Z3Solver",
    "Z3PropSolver",
    "derive_closure",
    "solve_query",
]
