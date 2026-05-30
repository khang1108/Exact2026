"""Encode typed Type 1 formula IR into Z3 SMT constraints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from exact.logic.ir import (
    And as FormulaAnd,
    Arithmetic,
    Atom,
    Compare,
    Exists as FormulaExists,
    ForAll as FormulaForAll,
    Formula,
    FormulaItem,
    Function as TermFunction,
    Iff as FormulaIff,
    Implies as FormulaImplies,
    InSet,
    Not as FormulaNot,
    Number,
    Or as FormulaOr,
    Term,
    TranslatedProblem,
    term_to_text,
)

GENERIC_CONSTANT = "g"
AtomKey = tuple[str, tuple[str, ...]]
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass
class EncodingContext:
    """Shared Z3 declarations for one translated problem."""

    boolean_declarations: dict[tuple[str, int], Any] = field(default_factory=dict)
    value_declarations: dict[tuple[str, int], Any] = field(default_factory=dict)
    constants: dict[str, Any] = field(default_factory=dict)
    fresh_index: int = 0

    def boolean(self, name: str, args: tuple[Any, ...]) -> Any:
        from z3 import Bool, BoolSort, Function, RealSort

        if not args:
            return Bool(_symbol_name("pred", name))
        key = (name, len(args))
        if key not in self.boolean_declarations:
            self.boolean_declarations[key] = Function(
                _symbol_name("pred", name, str(len(args))),
                *([RealSort()] * len(args)),
                BoolSort(),
            )
        return self.boolean_declarations[key](*args)

    def value(self, name: str, args: tuple[Any, ...] = ()) -> Any:
        from z3 import Function, Real, RealSort

        if not args:
            if name not in self.constants:
                self.constants[name] = Real(_symbol_name("value", name))
            return self.constants[name]
        key = (name, len(args))
        if key not in self.value_declarations:
            self.value_declarations[key] = Function(
                _symbol_name("value", name, str(len(args))),
                *([RealSort()] * len(args)),
                RealSort(),
            )
        return self.value_declarations[key](*args)

    def fresh_bound(self, name: str) -> Any:
        from z3 import Real

        self.fresh_index += 1
        return Real(_symbol_name("bound", name, str(self.fresh_index)))


def is_variable(term: Term) -> bool:
    """Return True for the compact variable convention used by translators."""

    if not isinstance(term, str):
        return False
    term = term.strip()
    return term.startswith("?") or term in {"x", "y", "z"}


def collect_domain(problem: TranslatedProblem) -> tuple[str, ...]:
    """Collect symbolic constants for traces and compatibility diagnostics."""

    constants: set[str] = {GENERIC_CONSTANT}
    for item in (*problem.premises, *problem.goals):
        _collect_formula_constants(item.formula, constants, bound=frozenset())
    return tuple(sorted(constants))


def build_theory(problem: TranslatedProblem) -> tuple[list[Any], EncodingContext]:
    """Return Z3 premise constraints and shared typed declarations."""

    context = EncodingContext()
    constraints = [
        ground_formula(item.formula, collect_domain(problem), context)
        for item in problem.premises
    ]
    return constraints, context


def encode_goal(
    item: FormulaItem,
    domain: tuple[str, ...],
    symbol_table: EncodingContext,
) -> Any:
    """Encode a translated query or option using the shared declarations."""

    return ground_formula(item.formula, domain, symbol_table)


def ground_formula(
    formula: Formula,
    domain: tuple[str, ...],
    symbol_table: EncodingContext | None = None,
) -> Any:
    """Encode a formula, universally closing any remaining free variables.

    ``domain`` remains in the public signature for compatibility with the
    previous finite-grounding backend. Native Z3 quantifiers avoid exponential
    enumeration while preserving the intended open-domain formula structure.
    """

    del domain
    context = symbol_table if isinstance(symbol_table, EncodingContext) else EncodingContext()
    free_variables = formula_variables(formula)
    if not free_variables:
        return _encode_formula(formula, context, {})

    quantified = tuple(context.fresh_bound(variable) for variable in free_variables)
    subst = dict(zip(free_variables, quantified, strict=True))
    from z3 import ForAll

    return ForAll(quantified, _encode_formula(formula, context, subst))


def formula_variables(formula: Formula) -> tuple[str, ...]:
    """Return free variables used by a formula."""

    variables: set[str] = set()
    _collect_formula_variables(formula, variables, bound=frozenset())
    return tuple(sorted(variables))


def _encode_formula(
    formula: Formula,
    context: EncodingContext,
    subst: dict[str, Any],
) -> Any:
    from z3 import And, Exists, ForAll, Implies, Not, Or

    if isinstance(formula, Atom):
        atom_expr = context.boolean(
            formula.pred,
            tuple(_encode_term(arg, context, subst) for arg in formula.args),
        )
        return Not(atom_expr) if formula.negated else atom_expr
    if isinstance(formula, FormulaNot):
        return Not(_encode_formula(formula.arg, context, subst))
    if isinstance(formula, FormulaAnd):
        return And(*(_encode_formula(child, context, subst) for child in formula.args))
    if isinstance(formula, FormulaOr):
        return Or(*(_encode_formula(child, context, subst) for child in formula.args))
    if isinstance(formula, FormulaImplies):
        return Implies(
            _encode_formula(formula.antecedent, context, subst),
            _encode_formula(formula.consequent, context, subst),
        )
    if isinstance(formula, FormulaIff):
        return _encode_formula(formula.left, context, subst) == _encode_formula(
            formula.right, context, subst
        )
    if isinstance(formula, FormulaForAll):
        quantified, nested_subst = _bind_variables(formula.variables, context, subst)
        return ForAll(quantified, _encode_formula(formula.body, context, nested_subst))
    if isinstance(formula, FormulaExists):
        quantified, nested_subst = _bind_variables(formula.variables, context, subst)
        return Exists(quantified, _encode_formula(formula.body, context, nested_subst))
    if isinstance(formula, Compare):
        left = _encode_term(formula.left, context, subst)
        right = _encode_term(formula.right, context, subst)
        return _compare(formula.op, left, right)
    if isinstance(formula, InSet):
        member = _encode_term(formula.member, context, subst)
        return Or(*(member == _encode_term(option, context, subst) for option in formula.options))
    raise TypeError(f"unsupported formula node: {type(formula).__name__}")


def _bind_variables(
    variables: tuple[str, ...],
    context: EncodingContext,
    subst: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    quantified = tuple(context.fresh_bound(variable) for variable in variables)
    nested_subst = dict(subst)
    for variable, bound in zip(variables, quantified, strict=True):
        nested_subst[_normalize_variable(variable)] = bound
        nested_subst[variable.lstrip("?")] = bound
    return quantified, nested_subst


def _encode_term(term: Term, context: EncodingContext, subst: dict[str, Any]) -> Any:
    from z3 import RealVal

    if isinstance(term, Number):
        return RealVal(term.value)
    if isinstance(term, str):
        if term in subst:
            return subst[term]
        if is_variable(term):
            variable = _normalize_variable(term)
            if variable not in subst:
                raise ValueError(f"unbound variable in formula encoding: {term!r}")
            return subst[variable]
        if _NUMBER_RE.fullmatch(term):
            return RealVal(term)
        return context.value(term)
    if isinstance(term, TermFunction):
        return context.value(
            term.name,
            tuple(_encode_term(arg, context, subst) for arg in term.args),
        )
    if isinstance(term, Arithmetic):
        args = tuple(_encode_term(arg, context, subst) for arg in term.args)
        return _encode_arithmetic(term.op, args, context)
    raise TypeError(f"unsupported term node: {type(term).__name__}")


def _encode_arithmetic(op: str, args: tuple[Any, ...], context: EncodingContext) -> Any:
    if op == "+":
        return sum(args)
    if op == "-":
        if len(args) == 1:
            return -args[0]
        result = args[0]
        for arg in args[1:]:
            result -= arg
        return result
    if op == "*":
        result = args[0]
        for arg in args[1:]:
            result *= arg
        return result
    if op == "/":
        result = args[0]
        for arg in args[1:]:
            result /= arg
        return result
    # Symbolic powers and threshold-like terms are kept inspectable without
    # forcing nonlinear arithmetic support from the main solver.
    return context.value(f"arith_{op}", args)


def _compare(op: str, left: Any, right: Any) -> Any:
    if op == "=":
        return left == right
    if op == "!=":
        return left != right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    raise ValueError(f"unsupported comparison operator: {op!r}")


def _collect_formula_variables(formula: Formula, out: set[str], bound: frozenset[str]) -> None:
    if isinstance(formula, Atom):
        for arg in formula.args:
            _collect_term_variables(arg, out, bound)
    elif isinstance(formula, FormulaNot):
        _collect_formula_variables(formula.arg, out, bound)
    elif isinstance(formula, (FormulaAnd, FormulaOr)):
        for child in formula.args:
            _collect_formula_variables(child, out, bound)
    elif isinstance(formula, FormulaImplies):
        _collect_formula_variables(formula.antecedent, out, bound)
        _collect_formula_variables(formula.consequent, out, bound)
    elif isinstance(formula, FormulaIff):
        _collect_formula_variables(formula.left, out, bound)
        _collect_formula_variables(formula.right, out, bound)
    elif isinstance(formula, (FormulaForAll, FormulaExists)):
        nested_bound = bound | frozenset(_normalize_variable(v) for v in formula.variables)
        _collect_formula_variables(formula.body, out, nested_bound)
    elif isinstance(formula, Compare):
        _collect_term_variables(formula.left, out, bound)
        _collect_term_variables(formula.right, out, bound)
    elif isinstance(formula, InSet):
        _collect_term_variables(formula.member, out, bound)
        for option in formula.options:
            _collect_term_variables(option, out, bound)


def _collect_term_variables(term: Term, out: set[str], bound: frozenset[str]) -> None:
    if isinstance(term, str):
        if _normalize_variable(term) in bound:
            return
        if is_variable(term):
            variable = _normalize_variable(term)
            if variable not in bound:
                out.add(variable)
    elif isinstance(term, (TermFunction, Arithmetic)):
        for arg in term.args:
            _collect_term_variables(arg, out, bound)


def _collect_formula_constants(formula: Formula, out: set[str], bound: frozenset[str]) -> None:
    if isinstance(formula, Atom):
        for arg in formula.args:
            _collect_term_constants(arg, out, bound)
    elif isinstance(formula, FormulaNot):
        _collect_formula_constants(formula.arg, out, bound)
    elif isinstance(formula, (FormulaAnd, FormulaOr)):
        for child in formula.args:
            _collect_formula_constants(child, out, bound)
    elif isinstance(formula, FormulaImplies):
        _collect_formula_constants(formula.antecedent, out, bound)
        _collect_formula_constants(formula.consequent, out, bound)
    elif isinstance(formula, FormulaIff):
        _collect_formula_constants(formula.left, out, bound)
        _collect_formula_constants(formula.right, out, bound)
    elif isinstance(formula, (FormulaForAll, FormulaExists)):
        nested_bound = bound | frozenset(_normalize_variable(v) for v in formula.variables)
        _collect_formula_constants(formula.body, out, nested_bound)
    elif isinstance(formula, Compare):
        _collect_term_constants(formula.left, out, bound)
        _collect_term_constants(formula.right, out, bound)
    elif isinstance(formula, InSet):
        _collect_term_constants(formula.member, out, bound)
        for option in formula.options:
            _collect_term_constants(option, out, bound)


def _collect_term_constants(term: Term, out: set[str], bound: frozenset[str]) -> None:
    if isinstance(term, str):
        if _normalize_variable(term) in bound:
            return
        if not is_variable(term) and not _NUMBER_RE.fullmatch(term):
            out.add(term)
    elif isinstance(term, (TermFunction, Arithmetic)):
        for arg in term.args:
            _collect_term_constants(arg, out, bound)


def _normalize_variable(term: str) -> str:
    term = term.strip()
    return term if term.startswith("?") else f"?{term}"


def _symbol_name(*parts: str) -> str:
    return "__".join(_sanitize_symbol_part(part) for part in parts if part)


def _sanitize_symbol_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return value or "empty"
