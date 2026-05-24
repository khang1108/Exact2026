"""Forward-chaining Horn solver.

This backend is the first deterministic executor in the Logic-LM style stack.
It is deliberately small, explainable, and source-preserving; richer engines
such as Z3 can be added beside it without changing the pipeline contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from exact.logic.ir import Atom, ProofStep, SolveResult
from exact.logic.kb import KnowledgeBase


@dataclass(frozen=True)
class ForwardChainSolver:
    """Derive all Horn consequences and answer by proof lookup."""

    name: str = "forward_chain_horn"

    def solve(self, kb: KnowledgeBase, claim: Atom) -> SolveResult:
        return solve_query(kb, claim, mode=self.name)


def solve_query(
    kb: KnowledgeBase,
    claim: Atom,
    mode: str = "forward_chain_horn",
) -> SolveResult:
    """Prove claim, prove its negation, or return Unknown."""

    known, proofs = derive_closure(kb)

    if claim in known:
        proof = _trace_proof(claim, proofs)
        return SolveResult(
            label="Yes",
            claim=claim,
            proof=tuple(proof),
            supporting_premises=_support_from_proof(proof),
            mode=mode,
            warnings=kb.warnings,
        )

    negated_claim = claim.negation()
    if negated_claim in known:
        proof = _trace_proof(negated_claim, proofs)
        return SolveResult(
            label="No",
            claim=claim,
            proof=tuple(proof),
            supporting_premises=_support_from_proof(proof),
            mode=mode,
            warnings=kb.warnings,
        )

    return SolveResult(
        label="Unknown",
        claim=claim,
        proof=(),
        supporting_premises=(),
        mode=mode,
        warnings=kb.warnings,
    )


def derive_closure(kb: KnowledgeBase) -> tuple[set[Atom], dict[Atom, ProofStep]]:
    """Derive all reachable atoms and keep the first proof for each atom."""

    known: set[Atom] = set()
    proofs: dict[Atom, ProofStep] = {}

    for fact in kb.facts:
        if fact.atom not in known:
            known.add(fact.atom)
            proofs[fact.atom] = ProofStep(
                derived=fact.atom,
                used_premises=(fact.source_idx,),
                rule_idx=None,
                parents=(),
                natural_language=f"Premise {fact.source_idx + 1} states {fact.atom.display()}.",
            )

    changed = True
    while changed:
        changed = False
        for rule in kb.rules:
            for binding, parents in _match_conditions(rule.conditions, tuple(known)):
                conclusion = apply_subst(rule.conclusion, binding)
                if not _is_ground(conclusion) or conclusion in known:
                    continue

                known.add(conclusion)
                parent_premises: list[int] = [rule.source_idx]
                for parent in parents:
                    parent_premises.extend(proofs[parent].used_premises)
                proofs[conclusion] = ProofStep(
                    derived=conclusion,
                    used_premises=tuple(sorted(set(parent_premises))),
                    rule_idx=rule.source_idx,
                    parents=parents,
                    natural_language=(
                        f"Premise {rule.source_idx + 1} derives {conclusion.display()} "
                        f"when {', '.join(parent.display() for parent in parents)} hold."
                    ),
                )
                changed = True

    return known, proofs


def is_variable(term: str) -> bool:
    return term.startswith("?")


def unify(
    pattern: Atom,
    ground_fact: Atom,
    subst: dict[str, str] | None = None,
) -> dict[str, str] | None:
    if (
        pattern.pred != ground_fact.pred
        or pattern.negated != ground_fact.negated
        or len(pattern.args) != len(ground_fact.args)
    ):
        return None

    bindings = dict(subst or {})
    for pattern_term, fact_term in zip(pattern.args, ground_fact.args, strict=True):
        if is_variable(pattern_term):
            resolved = _resolve_term(pattern_term, bindings)
            if resolved != pattern_term:
                if resolved != fact_term:
                    return None
            else:
                bindings[pattern_term] = fact_term
        elif pattern_term != fact_term:
            return None

    return bindings


def apply_subst(atom: Atom, subst: dict[str, str]) -> Atom:
    return Atom(
        pred=atom.pred,
        args=tuple(_resolve_term(arg, subst) for arg in atom.args),
        negated=atom.negated,
        text=atom.text,
    )


def _match_conditions(
    conditions: tuple[Atom, ...],
    known: tuple[Atom, ...],
    subst: dict[str, str] | None = None,
    parents: tuple[Atom, ...] = (),
):
    if not conditions:
        yield dict(subst or {}), parents
        return

    current, *remaining = conditions
    for candidate in known:
        next_subst = unify(current, candidate, subst)
        if next_subst is None:
            continue
        grounded_parent = apply_subst(current, next_subst)
        if grounded_parent != candidate:
            continue
        yield from _match_conditions(tuple(remaining), known, next_subst, (*parents, candidate))


def _is_ground(atom: Atom) -> bool:
    return all(not is_variable(arg) for arg in atom.args)


def _resolve_term(term: str, subst: dict[str, str]) -> str:
    seen: set[str] = set()
    while is_variable(term) and term in subst and term not in seen:
        seen.add(term)
        next_term = subst[term]
        if next_term == term:
            break
        term = next_term
    return term


def _trace_proof(target: Atom, proofs: dict[Atom, ProofStep]) -> list[ProofStep]:
    ordered: list[ProofStep] = []
    visited: set[Atom] = set()

    def visit(atom: Atom) -> None:
        if atom in visited or atom not in proofs:
            return
        visited.add(atom)
        for parent in proofs[atom].parents:
            visit(parent)
        ordered.append(proofs[atom])

    visit(target)
    return ordered


def _support_from_proof(proof: list[ProofStep]) -> tuple[int, ...]:
    support: set[int] = set()
    for step in proof:
        support.update(step.used_premises)
    return tuple(sorted(support))
