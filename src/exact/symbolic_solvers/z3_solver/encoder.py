"""Encode the current EXACT Horn IR into Z3 Boolean formulas."""

from __future__ import annotations

from typing import Any

from exact.logic.ir import Atom, Fact, Rule
from exact.logic.kb import KnowledgeBase


def atom_symbol_name(atom: Atom) -> str:
    prefix = "not__" if atom.negated else ""
    args = "__".join(atom.args)
    suffix = f"__{args}" if args else ""
    return f"{prefix}{atom.pred}{suffix}"


def collect_atoms(facts: list[Fact], rules: list[Rule], claim: Atom) -> set[Atom]:
    atoms = {claim, claim.negation()}
    for fact in facts:
        atoms.add(fact.atom)
    for rule in rules:
        atoms.add(rule.conclusion)
        atoms.update(rule.conditions)
    return atoms


def _ground_kb_for_z3(kb: KnowledgeBase) -> tuple[list[Fact], list[Rule]]:
    """Instantiate variable rules against the forward-chain closure.

    The Z3 encoder maps each atom to a unique Boolean symbol, so variables like
    '?x' must be resolved to concrete constants before encoding.  Running
    forward chaining first gives us the full set of reachable ground atoms;
    we then instantiate every variable rule against that closure.
    """
    from exact.logic.ir import Fact as _Fact, Rule as _Rule
    from exact.symbolic_solvers.forward_chain.solver import (
        _is_ground,
        _match_conditions,
        apply_subst,
        derive_closure,
    )

    known, _ = derive_closure(kb)
    known_tuple = tuple(known)

    existing_fact_atoms = {f.atom for f in kb.facts}
    extra_facts = [
        _Fact(atom=atom, source_idx=0, text=atom.display())
        for atom in known
        if atom not in existing_fact_atoms
    ]
    ground_facts: list[Fact] = list(kb.facts) + extra_facts

    ground_rules: list[Rule] = []
    for rule in kb.rules:
        has_vars = any(
            a.startswith("?") for c in rule.conditions for a in c.args
        ) or any(a.startswith("?") for a in rule.conclusion.args)

        if not has_vars:
            ground_rules.append(rule)
            continue

        for binding, _ in _match_conditions(rule.conditions, known_tuple):
            grnd_conds = tuple(apply_subst(c, binding) for c in rule.conditions)
            grnd_conc = apply_subst(rule.conclusion, binding)
            if _is_ground(grnd_conc) and all(_is_ground(c) for c in grnd_conds):
                ground_rules.append(
                    _Rule(
                        conditions=grnd_conds,
                        conclusion=grnd_conc,
                        source_idx=rule.source_idx,
                        text=rule.text,
                    )
                )

    return ground_facts, ground_rules


def encode_kb(kb: KnowledgeBase, claim: Atom) -> tuple[list[Any], dict[Atom, Any]]:
    """Return Z3 constraints and atom→Bool mapping.

    Variable rules are grounded against the forward-chain closure so that Z3
    constraints fire correctly for queries about specific entities.
    Z3 is imported lazily so the rest of the package can still import without
    ``z3-solver`` installed.
    """

    from z3 import And, Bool, Implies, Not

    facts, rules = _ground_kb_for_z3(kb)
    atoms = collect_atoms(facts, rules, claim)
    symbols = {atom: Bool(atom_symbol_name(atom)) for atom in atoms}
    constraints: list[Any] = []

    # Mutual exclusion: positive and negative cannot both be true
    for atom in list(symbols):
        positive = atom.positive()
        negative = positive.negation()
        if positive in symbols and negative in symbols:
            constraints.append(Implies(symbols[negative], Not(symbols[positive])))
            constraints.append(Implies(symbols[positive], Not(symbols[negative])))

    for fact in facts:
        if fact.atom in symbols:
            constraints.append(symbols[fact.atom])

    for rule in rules:
        if rule.conclusion not in symbols:
            continue
        cond_syms = [symbols[c] for c in rule.conditions if c in symbols]
        if len(cond_syms) != len(rule.conditions):
            continue  # some conditions not in symbol table — skip
        antecedent = And(*cond_syms) if len(cond_syms) > 1 else cond_syms[0]
        constraints.append(Implies(antecedent, symbols[rule.conclusion]))

    return constraints, symbols
