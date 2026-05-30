"""Typed Z3 SMT solver for formula-level Type 1 IR."""

from __future__ import annotations

from exact.symbolic_solvers.z3_prop.encoder import (
    GENERIC_CONSTANT,
    AtomKey,
    EncodingContext,
    build_theory,
    collect_domain,
    encode_goal,
    formula_variables,
    ground_formula,
    is_variable,
)
from exact.symbolic_solvers.z3_prop.solver import (
    EncodedOption,
    FormulaZ3Result,
    InconsistentTheoryError,
    Z3PropSolver,
    count_implied_siblings,
    entails,
    judge_mcq,
    judge_query,
    theory_status,
    unsat_core_indices,
    unsat_core_size,
)

__all__ = [
    "AtomKey",
    "EncodedOption",
    "EncodingContext",
    "FormulaZ3Result",
    "GENERIC_CONSTANT",
    "InconsistentTheoryError",
    "Z3PropSolver",
    "build_theory",
    "collect_domain",
    "count_implied_siblings",
    "encode_goal",
    "entails",
    "formula_variables",
    "ground_formula",
    "is_variable",
    "judge_mcq",
    "judge_query",
    "theory_status",
    "unsat_core_indices",
    "unsat_core_size",
]
