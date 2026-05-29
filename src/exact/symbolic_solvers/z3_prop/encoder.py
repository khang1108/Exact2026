"""Encode formula-level Type 1 IR into finite-domain propositional Z3."""

from __future__ import annotations

import re
from itertools import product
from typing import Any

from exact.logic.ir import (
    And as FormulaAnd,
    Atom,
    Formula,
    FormulaItem,
    Implies as FormulaImplies,
    Not as FormulaNot,
    Or as FormulaOr,
    TranslatedProblem,
)

GENERIC_CONSTANT = "g"
AtomKey = tuple[str, tuple[str, ...]]


def is_variable(term: str) -> bool:
    """Return True for variables emitted by the translator."""

    term = term.strip()
    return term.startswith("?") or term in {"x", "y", "z"}


def collect_domain(problem: TranslatedProblem) -> tuple[str, ...]:
    """Collect the finite grounding domain from premises/goals plus generic g."""

    constants: set[str] = {GENERIC_CONSTANT}
    for item in (*problem.premises, *problem.goals):
        for atom in _iter_atoms(item.formula):
            for arg in atom.args:
                arg = arg.strip()
                if arg and not is_variable(arg):
                    constants.add(arg)
    return tuple(sorted(constants))


def build_theory(problem: TranslatedProblem) -> tuple[list[Any], dict[AtomKey, Any]]:
    """Return Z3 premise constraints and the shared atom symbol table."""

    domain = collect_domain(problem)
    symbol_table: dict[AtomKey, Any] = {}
    constraints = [
        ground_formula(item.formula, domain, symbol_table)
        for item in problem.premises
    ]
    return constraints, symbol_table


def encode_goal(
    item: FormulaItem,
    domain: tuple[str, ...],
    symbol_table: dict[AtomKey, Any],
) -> Any:
    """Encode a translated query/option formula using the shared symbol table."""

    return ground_formula(item.formula, domain, symbol_table)


def ground_formula(
    formula: Formula,
    domain: tuple[str, ...],
    symbol_table: dict[AtomKey, Any] | None = None,
) -> Any:
    """Ground free variables universally over the finite domain and encode to Z3."""

    symbol_table = symbol_table if symbol_table is not None else {}
    variables = formula_variables(formula)
    if not variables:
        return _encode_formula(formula, domain, symbol_table, {})

    assignments = product(domain, repeat=len(variables))
    grounded = [
        _encode_formula(formula, domain, symbol_table, dict(zip(variables, values, strict=True)))
        for values in assignments
    ]
    return _z3_and(grounded)


def formula_variables(formula: Formula) -> tuple[str, ...]:
    variables: set[str] = set()
    for atom in _iter_atoms(formula):
        for arg in atom.args:
            if is_variable(arg):
                variables.add(_normalize_variable(arg))
    return tuple(sorted(variables))


def _encode_formula(
    formula: Formula,
    domain: tuple[str, ...],
    symbol_table: dict[AtomKey, Any],
    subst: dict[str, str],
) -> Any:
    from z3 import Implies as z3_implies
    from z3 import Not as z3_not

    if isinstance(formula, Atom):
        atom_expr = _encode_atom(formula, symbol_table, subst)
        return z3_not(atom_expr) if formula.negated else atom_expr

    if isinstance(formula, FormulaNot):
        return z3_not(_encode_formula(formula.arg, domain, symbol_table, subst))

    if isinstance(formula, FormulaAnd):
        return _z3_and(
            _encode_formula(child, domain, symbol_table, subst)
            for child in formula.args
        )

    if isinstance(formula, FormulaOr):
        return _z3_or(
            _encode_formula(child, domain, symbol_table, subst)
            for child in formula.args
        )

    if isinstance(formula, FormulaImplies):
        return z3_implies(
            _encode_formula(formula.antecedent, domain, symbol_table, subst),
            _encode_formula(formula.consequent, domain, symbol_table, subst),
        )

    raise TypeError(f"unsupported formula node: {type(formula).__name__}")


def _encode_atom(
    atom: Atom,
    symbol_table: dict[AtomKey, Any],
    subst: dict[str, str],
) -> Any:
    from z3 import Bool

    args = tuple(_ground_arg(arg, subst) for arg in atom.args)
    key = (atom.pred, args)
    if key not in symbol_table:
        symbol_table[key] = Bool(_atom_symbol_name(key))
    return symbol_table[key]


def _ground_arg(arg: str, subst: dict[str, str]) -> str:
    if is_variable(arg):
        variable = _normalize_variable(arg)
        if variable not in subst:
            raise ValueError(f"unbound variable in formula grounding: {arg!r}")
        return subst[variable]
    return arg


def _normalize_variable(term: str) -> str:
    term = term.strip()
    return term if term.startswith("?") else f"?{term}"


def _z3_and(values) -> Any:
    from z3 import And as z3_and
    from z3 import BoolVal

    values = list(values)
    if not values:
        return BoolVal(True)
    if len(values) == 1:
        return values[0]
    return z3_and(*values)


def _z3_or(values) -> Any:
    from z3 import BoolVal
    from z3 import Or as z3_or

    values = list(values)
    if not values:
        return BoolVal(False)
    if len(values) == 1:
        return values[0]
    return z3_or(*values)


def _iter_atoms(formula: Formula):
    if isinstance(formula, Atom):
        yield formula
        return
    if isinstance(formula, FormulaNot):
        yield from _iter_atoms(formula.arg)
        return
    if isinstance(formula, (FormulaAnd, FormulaOr)):
        for child in formula.args:
            yield from _iter_atoms(child)
        return
    if isinstance(formula, FormulaImplies):
        yield from _iter_atoms(formula.antecedent)
        yield from _iter_atoms(formula.consequent)
        return
    raise TypeError(f"unsupported formula node: {type(formula).__name__}")


def _atom_symbol_name(key: AtomKey) -> str:
    pred, args = key
    parts = ["atom", pred, *args]
    return "__".join(_sanitize_symbol_part(part) for part in parts if part)


def _sanitize_symbol_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return value or "empty"
