"""Z3-backed first-order logic entailment checker.

Convert FOL AST nodes (AtomicNode / LogicalNode / QuantifiedNode) produced by
the Type 1 parser into Z3 formulas and decide entailment via SMT.

Typical use:

    solver = FOLSolver()
    answer = solver.check_ynu(premises_fol, conclusion_fol)   # "Yes"/"No"/"Uncertain"
    label  = solver.check_mcq(premises_fol, option_fols)      # "A"/"B"/"C"/"D"/"Uncertain"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import z3

from exact.type1.ast.nodes import AtomicNode, FOLNode, LogicalNode, QuantifiedNode


def _sanitize_name(name: str) -> str:
    """Strip characters Z3 rejects in symbol names; ensure non-empty."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = "P_" + sanitized
    return sanitized


# A logic variable is a single lowercase letter (+ optional digits): x, y, z, x1.
# Constants are Capitalized/CamelCase after the parser's normalization, so this
# never matches an entity name like "Curriculum" or "John".
_VAR_RE = re.compile(r"[a-z]\d*")


def _free_vars(node: FOLNode, bound: frozenset[str]) -> set[str]:
    """Collect variable arguments not bound by an enclosing quantifier."""
    if isinstance(node, AtomicNode):
        return {a for a in node.arguments if a not in bound and _VAR_RE.fullmatch(a)}
    if isinstance(node, QuantifiedNode):
        return _free_vars(node.body, bound | {node.variable})
    free = _free_vars(node.left, bound)
    if node.right is not None:
        free |= _free_vars(node.right, bound)
    return free


def _close_universally(node: FOLNode) -> FOLNode:
    """Universally quantify a premise's free variables.

    The parser emits rules with implicitly universally quantified variables, e.g.
    ``(WellStructured(x) AND HasExercises(x)) IMPLIES Engaging(x)`` with no FORALL.
    Z3 treats a free ``x`` as a single constant it may pick distinct from the
    entity in the facts, so the rule never fires and every chain collapses to
    Uncertain. Wrapping each premise's free variables in FORALL restores the
    intended reading. A no-op when the premise is already ground or quantified.
    """
    for var in sorted(_free_vars(node, frozenset())):
        node = QuantifiedNode(quantifier="FORALL", variable=var, body=node)
    return node

# Shared sort for every entity in the domain.  Must be a module-level singleton:
# calling DeclareSort('Entity') twice creates two *distinct* sorts in Z3.
_ENTITY = z3.DeclareSort("Entity")

Answer = Literal["Yes", "No", "Uncertain"]


@dataclass(frozen=True)
class SolverResult:
    answer: str
    premises_used: list[str] | None = None


class FOLSolver:
    """Translate FOL AST trees to Z3 and check semantic entailment.

    One instance can be reused across multiple checks.  The predicate cache
    avoids re-declaring the same Z3 function within an instance's lifetime.
    """

    def __init__(self, *, timeout_ms: int = 2_000) -> None:
        # Per-check cap. An MCQ runs up to 8 checks (4 options × 2), so the
        # synchronous worst case is ~16s — well inside the request deadline.
        self._timeout_ms = timeout_ms
        # (predicate_name, arity) → Z3 function declaration
        self._func_cache: dict[tuple[str, int], z3.FuncDeclRef] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_ynu(self, premises: list[FOLNode], conclusion: FOLNode) -> SolverResult:
        """Return Yes / No / Uncertain for a yes-no-uncertain question."""
        p = [self._to_z3(_close_universally(n)) for n in premises]
        if not self._premises_are_consistent(p):
            return SolverResult(answer="Uncertain")
        c = self._to_z3(conclusion)
        return self._entails(p, c)

    def check_mcq(
        self,
        premises: list[FOLNode],
        options: dict[str, FOLNode],
    ) -> SolverResult:
        """Return the best-supported entailed option, or ``Uncertain``.

        Dataset MCQs commonly contain multiple true conclusions and ask for the
        one following with the fewest premises. Ignore premise-independent
        tautologies, then rank entailed options by a minimized Z3 unsat core.
        """
        p = [self._to_z3(_close_universally(n)) for n in premises]
        if not self._premises_are_consistent(p):
            return SolverResult(answer="Uncertain")

        support_sizes: dict[str, tuple[int, list[str]]] = {}
        for label, node in options.items():
            conclusion = self._to_z3(node)
            if self._entails([], conclusion).answer == "Yes":
                continue
            support = self._minimum_support(p, conclusion)
            if support is not None:
                support_sizes[label] = (len(support), support)

        if not support_sizes:
            return SolverResult(answer="Uncertain")
        smallest = min(size for size, _ in support_sizes.values())
        winners = [
            (label, support)
            for label, (size, support) in support_sizes.items()
            if size == smallest
        ]
        if len(winners) != 1:
            return SolverResult(answer="Uncertain")
        label, premises_used = winners[0]
        return SolverResult(answer=label, premises_used=premises_used)

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

        if isinstance(node, QuantifiedNode):
            var = z3.Const(node.variable, _ENTITY)
            body = self._to_z3(node.body, {**var_map, node.variable: var})
            if node.quantifier == "FORALL":
                return z3.ForAll([var], body)
            return z3.Exists([var], body)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predicate(self, name: str, arity: int) -> z3.FuncDeclRef:
        # Z3 symbol names must be valid identifiers; strip anything unsafe.
        safe_name = _sanitize_name(name)
        key = (safe_name, arity)
        if key not in self._func_cache:
            if arity == 0:
                # 0-arity "function" → Z3 boolean constant
                self._func_cache[key] = z3.Function(safe_name, z3.BoolSort())
            else:
                self._func_cache[key] = z3.Function(
                    safe_name, *([_ENTITY] * arity), z3.BoolSort()
                )
        return self._func_cache[key]

    def _term(self, name: str, var_map: dict[str, z3.ExprRef]) -> z3.ExprRef:
        """Return the Z3 expression for a variable (bound) or constant (free)."""
        return var_map.get(name, z3.Const(name, _ENTITY))

    def _solver(self) -> z3.Solver:
        s = z3.Solver()
        s.set("timeout", self._timeout_ms)
        return s

    def _premises_are_consistent(self, premises_z3: list[z3.ExprRef]) -> bool:
        """Trust entailment only when the premise set is satisfiable."""

        s = self._solver()
        s.add(*premises_z3)
        return s.check() == z3.sat

    def _minimum_support(
        self,
        premises_z3: list[z3.ExprRef],
        conclusion_z3: z3.ExprRef,
    ) -> list[str] | None:
        """Return a subset-minimal premise list proving ``conclusion_z3``."""

        assumptions = [z3.FreshBool("premise") for _ in premises_z3]
        s = self._solver()
        s.add(
            *(
                z3.Implies(assumption, premise)
                for assumption, premise in zip(assumptions, premises_z3)
            ),
            z3.Not(conclusion_z3),
        )
        if s.check(*assumptions) != z3.unsat:
            return None

        core = list(s.unsat_core())
        index = 0
        while index < len(core):
            candidate = core[:index] + core[index + 1 :]
            if s.check(*candidate) == z3.unsat:
                core = candidate
            else:
                index += 1
        return self._core_to_premise_ids(assumptions, core)

    def _core_to_premise_ids(
        self,
        assumptions: list[z3.BoolRef],
        core: list[z3.BoolRef],
    ) -> list[str]:
        premise_ids = [
            f"premise-{index}"
            for index, assumption in enumerate(assumptions, start=1)
            if assumption in core
        ]
        return premise_ids

    def _entails(self, premises_z3: list[z3.ExprRef], conclusion_z3: z3.ExprRef) -> SolverResult:
        # Yes  → premises ∧ ¬conclusion is UNSAT (negation of conclusion impossible)
        support = self._minimum_support(premises_z3, conclusion_z3)
        if support is not None:
            return SolverResult(answer="Yes", premises_used=support)

        # No   → premises ∧ conclusion is UNSAT (conclusion contradicts premises)
        assumptions = [z3.FreshBool("premise") for _ in premises_z3]
        s = self._solver()
        s.add(
            *(
                z3.Implies(assumption, premise)
                for assumption, premise in zip(assumptions, premises_z3)
            ),
            conclusion_z3,
        )
        if s.check(*assumptions) == z3.unsat:
            return SolverResult(
                answer="No",
                premises_used=self._core_to_premise_ids(assumptions, list(s.unsat_core())),
            )

        # unknown / sat → not enough information to decide
        return SolverResult(answer="Uncertain")
