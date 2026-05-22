"""Encode the current EXACT Horn IR into Z3 Boolean formulas."""

from __future__ import annotations

from typing import Any

from exact.logic.ir import Atom
from exact.logic.kb import KnowledgeBase


def atom_symbol_name(atom: Atom) -> str:
    prefix = "not__" if atom.negated else ""
    args = "__".join(atom.args)
    suffix = f"__{args}" if args else ""
    return f"{prefix}{atom.pred}{suffix}"


def collect_atoms(kb: KnowledgeBase, claim: Atom) -> set[Atom]:
    atoms = {claim, claim.negation()}
    for fact in kb.facts:
        atoms.add(fact.atom)
    for rule in kb.rules:
        atoms.add(rule.conclusion)
        atoms.update(rule.conditions)
    return atoms


def encode_kb(kb: KnowledgeBase, claim: Atom) -> tuple[list[Any], dict[Atom, Any]]:
    """Return Z3 constraints and atom -> Bool mapping.

    Z3 is imported lazily so the rest of the package can still import without
    `z3-solver` installed.
    """

    from z3 import And, Bool, Implies, Not

    symbols = {atom: Bool(atom_symbol_name(atom)) for atom in collect_atoms(kb, claim)}
    constraints: list[Any] = []

    for atom in list(symbols):
        positive = atom.positive()
        negative = positive.negation()
        if positive in symbols and negative in symbols:
            constraints.append(Implies(symbols[negative], Not(symbols[positive])))
            constraints.append(Implies(symbols[positive], Not(symbols[negative])))

    for fact in kb.facts:
        constraints.append(symbols[fact.atom])

    for rule in kb.rules:
        antecedent = And(*[symbols[condition] for condition in rule.conditions])
        constraints.append(Implies(antecedent, symbols[rule.conclusion]))

    return constraints, symbols
