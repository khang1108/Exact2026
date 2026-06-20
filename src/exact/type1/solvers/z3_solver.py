"""Z3-backed first-order logic entailment checker.

Convert FOL AST nodes (AtomicNode / LogicalNode / QuantifiedNode / ComparisonNode)
produced by the Type 1 parser into Z3 formulas and decide entailment via SMT.

Typical use:

    solver = FOLSolver()
    answer = solver.check_ynu(premises_fol, conclusion_fol)   # "Yes"/"No"/"Uncertain"
    label  = solver.check_mcq(premises_fol, option_fols)      # "A"/"B"/"C"/"D"/"Uncertain"

Design notes:
- Predicate caches are per-call (not per-instance) so sort declarations never
  bleed across requests.  Two functions with the same name but different sorts
  (Entity→Bool vs Entity→Real) are detected early and raise ValueError.
- _ENTITY is a module-level singleton: calling DeclareSort('Entity') twice
  creates two *distinct* sorts in Z3, breaking cross-request compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
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

# Shared sort for every entity in the domain — module-level singleton.
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


@dataclass
class _CallCtx:
    """Per-call sort registry.  Isolates predicate declarations across requests."""

    # (name, arity) → Entity*→Bool
    bool_fns: dict[tuple[str, int], z3.FuncDeclRef] = field(default_factory=dict)
    # (name, arity) → Entity*→Real
    real_fns: dict[tuple[str, int], z3.FuncDeclRef] = field(default_factory=dict)
    # date_identifier → Z3 Int constant
    dates: dict[str, z3.ExprRef] = field(default_factory=dict)

    def bool_fn(self, name: str, arity: int) -> z3.FuncDeclRef:
        key = (name, arity)
        if key in self.real_fns:
            raise ValueError(
                f"SORT_CONFLICT: predicate '{name}/{arity}' already declared as "
                f"Entity→Real (numeric); cannot also use as Entity→Bool (atomic). "
                f"Check that the same name is not used for both a ComparisonNode "
                f"function and an AtomicNode predicate."
            )
        if key not in self.bool_fns:
            self.bool_fns[key] = z3.Function(name, *([_ENTITY] * arity), z3.BoolSort())
        return self.bool_fns[key]

    def real_fn(self, name: str, arity: int) -> z3.FuncDeclRef:
        key = (name, arity)
        if key in self.bool_fns:
            raise ValueError(
                f"SORT_CONFLICT: predicate '{name}/{arity}' already declared as "
                f"Entity→Bool (atomic); cannot also use as Entity→Real (numeric). "
                f"Check that the same name is not used for both an AtomicNode "
                f"predicate and a ComparisonNode function."
            )
        if key not in self.real_fns:
            self.real_fns[key] = z3.Function(
                name, *([_ENTITY] * arity), z3.RealSort()
            )
        return self.real_fns[key]

    def date_const(self, value: str) -> z3.ExprRef:
        if value not in self.dates:
            self.dates[value] = z3.Int(value)
        return self.dates[value]


class FOLSolver:
    """Translate FOL AST trees to Z3 and check semantic entailment.

    One instance can be reused across multiple checks.  Predicate sorts are
    resolved fresh per call via _CallCtx — no state bleeds between requests.
    """

    def __init__(self, *, timeout_ms: int = 5_000) -> None:
        self._timeout_ms = timeout_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_ynu(self, premises: list[FOLNode], conclusion: FOLNode) -> Answer:
        """Return Yes / No / Uncertain for a yes-no-uncertain question."""
        ctx = _CallCtx()
        p = [self._to_z3(n, {}, ctx) for n in premises]
        c = self._to_z3(conclusion, {}, ctx)
        return self._entails(p, c)

    def check_ynu_with_used(
        self,
        premises: list[FOLNode],
        conclusion: FOLNode,
    ) -> tuple[Answer, list[int]]:
        """Like check_ynu but also returns 0-based indices of premises used.

        Uses Z3 unsat-core tracking to identify the minimal set of premises
        that contributed to a Yes or No answer.  Returns an empty list when
        the answer is Uncertain or the solver times out.
        """
        ctx = _CallCtx()
        p_z3 = [self._to_z3(n, {}, ctx) for n in premises]
        c_z3 = self._to_z3(conclusion, {}, ctx)

        # Yes: premises ∧ ¬conclusion is UNSAT
        answer, used = self._entails_with_core(p_z3, z3.Not(c_z3))
        if answer == z3.unsat:
            return "Yes", used

        # No: premises ∧ conclusion is UNSAT
        answer, used = self._entails_with_core(p_z3, c_z3)
        if answer == z3.unsat:
            return "No", used

        return "Uncertain", []

    def check_mcq(
        self,
        premises: list[FOLNode],
        options: dict[str, FOLNode],
        none_of_above_label: str | None = None,
    ) -> str:
        """Return the label of the entailed option, or 'Uncertain'.

        A shared _CallCtx ensures all premises and options use consistent
        sort declarations within one MCQ call.

        When a none-of-above option exists, pass its label: if exactly one
        ordinary option is entailed return that label; if none are entailed
        return the none-of-above label; otherwise 'Uncertain' (B10 §16.5).
        """
        ctx = _CallCtx()
        p = [self._to_z3(n, {}, ctx) for n in premises]
        entailed = [
            label
            for label, node in options.items()
            if self._entails(p, self._to_z3(node, {}, ctx)) == "Yes"
        ]
        if len(entailed) == 1:
            return entailed[0]
        if not entailed and none_of_above_label is not None:
            return none_of_above_label
        return "Uncertain"

    def check_mcq_with_used(
        self,
        premises: list[FOLNode],
        options: dict[str, FOLNode],
        none_of_above_label: str | None = None,
    ) -> tuple[str, list[int]]:
        """Like check_mcq but also returns 0-based indices of premises used.

        Uses unsat-core tracking on the entailed option so the proof's premise
        set is reported (not an empty list). Returns [] for none-of-above or
        Uncertain, mirroring check_mcq's label rules.
        """
        ctx = _CallCtx()
        p = [self._to_z3(n, {}, ctx) for n in premises]

        # Closed-world evaluation: EXACT options use "established by the premises"
        # framing, so an atom that is not provable is treated as false (and its
        # negation as true). Evaluate each option formula under that assumption.
        cache: dict[str, tuple[bool, list[int]]] = {}

        def provable(node: FOLNode) -> tuple[bool, list[int]]:
            z = self._to_z3(node, {}, ctx)
            key = str(z)
            if key not in cache:
                result, used = self._entails_with_core(p, z3.Not(z))
                cache[key] = (result == z3.unsat, used)
            return cache[key]

        def evaluate(node: FOLNode) -> tuple[bool, list[int]]:
            if isinstance(node, LogicalNode):
                if node.operator == "NOT":
                    val, _ = evaluate(node.left)
                    return (not val, [])  # CWA: ¬X holds iff X is not provable
                left_val, left_used = evaluate(node.left)
                right = node.right
                if right is None:
                    return (left_val, left_used)
                right_val, right_used = evaluate(right)
                if node.operator == "AND":
                    return (left_val and right_val, left_used + right_used)
                if node.operator == "OR":
                    if left_val:
                        return (True, left_used)
                    return (right_val, right_used if right_val else [])
                if node.operator == "IMPLIES":
                    if not left_val:
                        return (True, [])
                    return (right_val, right_used)
                if node.operator == "IFF":
                    return (left_val == right_val, left_used + right_used)
            # AtomicNode / ComparisonNode / QuantifiedNode → provability check
            return provable(node)

        candidates: list[tuple[str, list[int]]] = []
        for label, node in options.items():
            value, used = evaluate(node)
            if value:
                candidates.append((label, sorted(set(used))))

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Several options hold (e.g. chain intermediates + final). Pick the
            # strongest = deepest derivation (largest support); tie → Uncertain.
            candidates.sort(key=lambda c: len(c[1]), reverse=True)
            if len(candidates[0][1]) > len(candidates[1][1]):
                return candidates[0]
            return "Uncertain", []
        if none_of_above_label is not None:
            return none_of_above_label, []
        return "Uncertain", []

    def check_mcq_refutation(
        self,
        premises: list[FOLNode],
        options: dict[str, FOLNode],
    ) -> str:
        """Return the label of the single FALSE option, or 'Uncertain'.

        For "which statement is NOT true / cannot follow", the answer is the
        option whose negation is entailed by the premises.
        """
        ctx = _CallCtx()
        p = [self._to_z3(n, {}, ctx) for n in premises]
        refuted = [
            label
            for label, node in options.items()
            if self._entails(p, self._to_z3(node, {}, ctx)) == "No"
        ]
        return refuted[0] if len(refuted) == 1 else "Uncertain"

    def check_mcq_fewest_premises(
        self,
        premises: list[FOLNode],
        options: dict[str, FOLNode],
    ) -> str:
        """Return the uniquely entailed option requiring the fewest premises."""

        minimum_sizes: dict[str, int] = {}
        for label, option in options.items():
            # Entailment is monotonic: if the full premise set cannot prove the
            # option, no subset can. This avoids exponential work for distractors.
            if self.check_ynu(premises, option) != "Yes":
                continue
            for size in range(1, len(premises) + 1):
                if any(
                    self.check_ynu(list(subset), option) == "Yes"
                    for subset in combinations(premises, size)
                ):
                    minimum_sizes[label] = size
                    break

        if not minimum_sizes:
            return "Uncertain"
        minimum = min(minimum_sizes.values())
        winners = [
            label for label, required in minimum_sizes.items() if required == minimum
        ]
        return winners[0] if len(winners) == 1 else "Uncertain"

    # ------------------------------------------------------------------
    # FOL AST → Z3
    # ------------------------------------------------------------------

    def _to_z3(
        self,
        node: FOLNode,
        var_map: dict[str, z3.ExprRef],
        ctx: _CallCtx,
    ) -> z3.ExprRef:
        """Recursively convert a FOL AST node to a Z3 boolean expression."""

        if isinstance(node, AtomicNode):
            fn = ctx.bool_fn(node.predicate.name, len(node.arguments))
            args = [self._term(a, var_map) for a in node.arguments]
            return fn(*args)

        if isinstance(node, ComparisonNode):
            left_fn = ctx.real_fn(node.left.name, len(node.left.arguments))
            left_z3 = left_fn(*[self._term(a, var_map) for a in node.left.arguments])
            right_z3 = self._comparison_right(node.right, var_map, ctx)
            op = _COMPARISON_OPS.get(node.operator)
            if op is None:
                raise ValueError(f"Unknown comparison operator: {node.operator!r}")
            return op(left_z3, right_z3)  # type: ignore[return-value]

        if isinstance(node, LogicalNode):
            left = self._to_z3(node.left, var_map, ctx)
            if node.operator == "NOT":
                return z3.Not(left)  # type: ignore[return-value]
            assert node.right is not None, f"{node.operator} requires a right operand"
            right = self._to_z3(node.right, var_map, ctx)
            if node.operator == "AND":
                return z3.And(left, right)  # type: ignore[return-value]
            if node.operator == "OR":
                return z3.Or(left, right)  # type: ignore[return-value]
            if node.operator == "IMPLIES":
                return z3.Implies(left, right)  # type: ignore[return-value]
            if node.operator == "IFF":
                return z3.And(  # type: ignore[return-value]
                    z3.Implies(left, right), z3.Implies(right, left)
                )
            raise ValueError(f"Unknown operator: {node.operator!r}")

        # QuantifiedNode — restrictor encodes the noun class domain restriction.
        # ∀x[R].B → ForAll(x, R → B)   ∃x[R].B → Exists(x, R ∧ B)
        if isinstance(node, QuantifiedNode):
            var = z3.Const(node.variable, _ENTITY)
            local_vars = {**var_map, node.variable: var}
            body = self._to_z3(node.body, local_vars, ctx)

            if node.restrictor is not None:
                restrictor = self._to_z3(node.restrictor, local_vars, ctx)
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

    def _comparison_right(
        self,
        right: NumericTerm | FunctionTerm | DateTerm,
        var_map: dict[str, z3.ExprRef],
        ctx: _CallCtx,
    ) -> z3.ExprRef:
        if isinstance(right, NumericTerm):
            return z3.RealVal(right.value)
        if isinstance(right, DateTerm):
            return ctx.date_const(right.value)
        fn = ctx.real_fn(right.name, len(right.arguments))
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

    def _entails_with_core(
        self,
        premises_z3: list[z3.ExprRef],
        extra: z3.ExprRef,
    ) -> tuple[z3.CheckSatResult, list[int]]:
        """Add premises + extra assertion, return (result, used_premise_indices).

        Uses assert_and_track so Z3 can return an unsat core.  Each premise
        is tracked with a fresh Boolean constant ``pN`` (N = 0-based index).
        The extra assertion (negated conclusion or conclusion itself) is added
        directly — it is always part of the check and not tracked.
        """
        s = self._solver()
        s.set("unsat_core", True)
        trackers: list[z3.BoolRef] = []
        for i, p in enumerate(premises_z3):
            tracker = z3.Bool(f"__p{i}")
            trackers.append(tracker)
            s.assert_and_track(p, tracker)
        s.add(extra)
        result = s.check()
        if result != z3.unsat:
            return result, []
        core_names = {str(expr) for expr in s.unsat_core()}
        used = [
            i for i, t in enumerate(trackers) if str(t) in core_names
        ]
        return result, sorted(used)
