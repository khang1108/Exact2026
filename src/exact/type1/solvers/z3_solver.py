"""Z3-backed first-order logic entailment checker.

Convert FOL AST nodes (AtomicNode / LogicalNode / QuantifiedNode) produced by
the Type 1 parser into Z3 formulas and decide entailment via SMT.

Typical use:

    solver = FOLSolver()
    answer = solver.check_ynu(premises_fol, conclusion_fol)   # "Yes"/"No"/"Uncertain"
    label  = solver.check_mcq(premises_fol, option_fols)      # "A"/"B"/"C"/"D"/"Uncertain"
"""

from __future__ import annotations

from typing import Literal

import z3

from exact.type1.ast.nodes import AtomicNode, FOLNode, LogicalNode, QuantifiedNode

# Shared sort for every entity in the domain.  Must be a module-level singleton:
# calling DeclareSort('Entity') twice creates two *distinct* sorts in Z3.
_ENTITY = z3.DeclareSort("Entity")

Answer = Literal["Yes", "No", "Uncertain"]


class FOLSolver:
    """Translate FOL AST trees to Z3 and check semantic entailment.

    One instance can be reused across multiple checks.  The predicate cache
    avoids re-declaring the same Z3 function within an instance's lifetime.
    """

    def __init__(self, *, timeout_ms: int = 5_000) -> None:
        self._timeout_ms = timeout_ms
        # (predicate_name, arity) → Z3 function declaration
        self._func_cache: dict[tuple[str, int], z3.FuncDeclRef] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_ynu(self, premises: list[FOLNode], conclusion: FOLNode) -> Answer:
        """Return Yes / No / Uncertain for a yes-no-uncertain question."""
        p = [self._to_z3(n) for n in premises]
        c = self._to_z3(conclusion)
        return self._entails(p, c)

    def check_mcq(
        self,
        premises: list[FOLNode],
        options: dict[str, FOLNode],
    ) -> str:
        """Return the label (A/B/C/D/…) of the entailed option, or 'Uncertain'.

        options: {"A": fol_node, "B": fol_node, ...}
        """
        p = [self._to_z3(n) for n in premises]
        entailed = [
            label
            for label, node in options.items()
            if self._entails(p, self._to_z3(node)) == "Yes"
        ]
        # Exactly one entailed option → confident answer; otherwise uncertain.
        return entailed[0] if len(entailed) == 1 else "Uncertain"

    # ------------------------------------------------------------------
    # FOL AST → Z3
    # ------------------------------------------------------------------

    def _to_z3(
        self,
        node: FOLNode,
        var_map: dict[str, z3.ExprRef] | None = None,
    ) -> z3.ExprRef:  # BoolRef at runtime; ExprRef avoids imprecise z3 stubs
        """Recursively convert a FOL AST node to a Z3 boolean expression."""
        if var_map is None:
            var_map = {}

        if isinstance(node, AtomicNode):
            fn = self._predicate(node.predicate.name, len(node.arguments))
            args = [self._term(a, var_map) for a in node.arguments]
            return fn(*args)

        if isinstance(node, LogicalNode):
            left = self._to_z3(node.left, var_map)
            if node.operator == "NOT":
                return z3.Not(left)  # type: ignore[return-value]
            assert node.right is not None, f"{node.operator} requires a right operand"
            right = self._to_z3(node.right, var_map)
            if node.operator == "AND":
                return z3.And(left, right)  # type: ignore[return-value]
            if node.operator == "OR":
                return z3.Or(left, right)  # type: ignore[return-value]
            if node.operator == "IMPLIES":
                return z3.Implies(left, right)  # type: ignore[return-value]
            if node.operator == "IFF":
                return z3.And(z3.Implies(left, right), z3.Implies(right, left))  # type: ignore[return-value]
            raise ValueError(f"Unknown operator: {node.operator!r}")

        # if isinstance(node, QuantifiedNode):
        #     var = z3.Const(node.variable, _ENTITY)
        #     body = self._to_z3(node.body, {**var_map, node.variable: var})
        #     if node.quantifier == "FORALL":
        #         return z3.ForAll([var], body)
        #     return z3.Exists([var], body)
        """That makes the parser lose the noun class:
            - student, curriculum, course, faculty, applicant, etc.

        For EXACT, that is dangerous because educational regulations are full of restricted groups:

        All students who miss the lab exam fail the course.
        Students with GPA above 8.0 qualify for scholarship.
        A course with a final exam requires attendance."""
        if isinstance(node, QuantifiedNode):
            var = z3.Const(node.variable, _ENTITY)
            local_vars = {**var_map, node.variable: var}

            body = self._to_z3(node.body, local_vars)

            if node.restrictor is not None:
                restrictor = self._to_z3(node.restrictor, local_vars)

                if node.quantifier == "FORALL":
                    return z3.ForAll([var], z3.Implies(restrictor, body))

                return z3.Exists([var], z3.And(restrictor, body))

            if node.quantifier == "FORALL":
                return z3.ForAll([var], body)

            return z3.Exists([var], body)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predicate(self, name: str, arity: int) -> z3.FuncDeclRef:
        key = (name, arity)
        if key not in self._func_cache:
            self._func_cache[key] = z3.Function(
                name, *([_ENTITY] * arity), z3.BoolSort()
            )
        return self._func_cache[key]

    def _term(self, name: str, var_map: dict[str, z3.ExprRef]) -> z3.ExprRef:
        """Return the Z3 expression for a variable (bound) or constant (free)."""
        return var_map.get(name, z3.Const(name, _ENTITY))

    def _solver(self) -> z3.Solver:
        s = z3.Solver()
        s.set("timeout", self._timeout_ms)
        return s

    def _entails(self, premises_z3: list[z3.ExprRef], conclusion_z3: z3.ExprRef) -> Answer:
        # Yes  → premises ∧ ¬conclusion is UNSAT (negation of conclusion impossible)
        s = self._solver()
        s.add(*premises_z3, z3.Not(conclusion_z3))
        if s.check() == z3.unsat:
            return "Yes"

        # No   → premises ∧ conclusion is UNSAT (conclusion contradicts premises)
        s = self._solver()
        s.add(*premises_z3, conclusion_z3)
        if s.check() == z3.unsat:
            return "No"

        # unknown / sat → not enough information to decide
        return "Uncertain"
