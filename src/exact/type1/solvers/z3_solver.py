"""Z3-backed first-order logic entailment checker.

Convert FOL AST nodes (AtomicNode / LogicalNode / QuantifiedNode) produced by
the Type 1 parser into Z3 formulas and decide entailment via SMT.

Typical use:

    solver = FOLSolver()
    answer = solver.check_ynu(premises_fol, conclusion_fol)   # "Yes"/"No"/"Uncertain"
    label  = solver.check_mcq(premises_fol, option_fols)      # "A"/"B"/"C"/"D"/"Uncertain"
"""

from __future__ import annotations

from typing import Callable, Literal

import z3

from exact.type1.ast.nodes import (
    AtomicNode,
    ComparisonNode,
    DateTerm,
    FOLNode,
    FunctionTerm,
    LogicalNode,
    NumericTerm,
    QuantifiedNode,
)

# Shared sort for every entity in the domain.  Must be a module-level singleton:
# calling DeclareSort('Entity') twice creates two *distinct* sorts in Z3.
_ENTITY = z3.DeclareSort("Entity")

Answer = Literal["Yes", "No", "Uncertain"]

_COMPARISON_OPS: dict[str, Callable[[z3.ExprRef, z3.ExprRef], z3.BoolRef]] = {
    "=": lambda a, b: a == b,  # type: ignore[return-value]
    ">=": lambda a, b: a >= b,  # type: ignore[return-value]
    ">": lambda a, b: a > b,  # type: ignore[return-value]
    "<=": lambda a, b: a <= b,  # type: ignore[return-value]
    "<": lambda a, b: a < b,  # type: ignore[return-value]
    "!=": lambda a, b: a != b,  # type: ignore[return-value]
}


class FOLSolver:
    """Translate FOL AST trees to Z3 and check semantic entailment.

    One instance can be reused across multiple checks.  The predicate cache
    avoids re-declaring the same Z3 function within an instance's lifetime.
    """

    def __init__(self, *, timeout_ms: int = 5_000) -> None:
        self._timeout_ms = timeout_ms
        # (predicate_name, arity) → Z3 function Entity*→Bool
        self._func_cache: dict[tuple[str, int], z3.FuncDeclRef] = {}
        # (function_name, arity) → Z3 function Entity*→Real (numeric attributes)
        self._numeric_cache: dict[tuple[str, int], z3.FuncDeclRef] = {}
        # date_identifier → Z3 Int constant (ordinal, enables ≤/≥ ordering)
        self._date_cache: dict[str, z3.ExprRef] = {}

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
        none_of_above_label: str | None = None,
    ) -> str:
        """Return the label (A/B/C/D/…) of the entailed option, or 'Uncertain'.

        ``options`` holds only the ordinary (selectable) options. When a
        none-of-above option exists, pass its label: if exactly one ordinary
        option is entailed return that label; if none are entailed return the
        none-of-above label; otherwise 'Uncertain' (B10 §16.5).
        """
        p = [self._to_z3(n) for n in premises]
        entailed = [
            label
            for label, node in options.items()
            if self._entails(p, self._to_z3(node)) == "Yes"
        ]
        if len(entailed) == 1:
            return entailed[0]
        if not entailed and none_of_above_label is not None:
            return none_of_above_label
        return "Uncertain"

    def check_mcq_refutation(
        self,
        premises: list[FOLNode],
        options: dict[str, FOLNode],
    ) -> str:
        """Return the label of the single FALSE option, or 'Uncertain'.

        For "which statement is NOT true / cannot follow", the answer is the
        option that contradicts the premises (its negation is entailed).
        """
        p = [self._to_z3(n) for n in premises]
        refuted = [
            label
            for label, node in options.items()
            if self._entails(p, self._to_z3(node)) == "No"
        ]
        return refuted[0] if len(refuted) == 1 else "Uncertain"

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

        if isinstance(node, ComparisonNode):
            left_fn = self._numeric_predicate(node.left.name, len(node.left.arguments))
            left_z3 = left_fn(*[self._term(a, var_map) for a in node.left.arguments])
            right_z3 = self._comparison_right(node.right, var_map)
            op = _COMPARISON_OPS.get(node.operator)
            if op is None:
                raise ValueError(f"Unknown comparison operator: {node.operator!r}")
            return op(left_z3, right_z3)  # type: ignore[return-value]

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

        # QuantifiedNode — restrictor encodes the noun class domain restriction.
        # ∀x[R].B → ForAll(x, R → B)   ∃x[R].B → Exists(x, R ∧ B)
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

        raise ValueError(f"Unknown FOL node type: {type(node).__name__}")  # type: ignore[return-value]

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

    def _numeric_predicate(self, name: str, arity: int) -> z3.FuncDeclRef:
        key = (name, arity)
        if key not in self._numeric_cache:
            self._numeric_cache[key] = z3.Function(
                name, *([_ENTITY] * arity), z3.RealSort()
            )
        return self._numeric_cache[key]

    def _date_const(self, value: str) -> z3.ExprRef:
        if value not in self._date_cache:
            self._date_cache[value] = z3.Int(value)
        return self._date_cache[value]

    def _comparison_right(
        self,
        right: NumericTerm | FunctionTerm | DateTerm,
        var_map: dict[str, z3.ExprRef],
    ) -> z3.ExprRef:
        if isinstance(right, NumericTerm):
            return z3.RealVal(right.value)
        if isinstance(right, DateTerm):
            return self._date_const(right.value)
        # FunctionTerm
        fn = self._numeric_predicate(right.name, len(right.arguments))
        return fn(*[self._term(a, var_map) for a in right.arguments])

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
