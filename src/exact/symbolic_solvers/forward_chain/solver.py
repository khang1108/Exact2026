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
            if rule.conclusion in known:
                continue
            if all(condition in known for condition in rule.conditions):
                known.add(rule.conclusion)
                parent_premises: list[int] = [rule.source_idx]
                for condition in rule.conditions:
                    parent_premises.extend(proofs[condition].used_premises)
                proofs[rule.conclusion] = ProofStep(
                    derived=rule.conclusion,
                    used_premises=tuple(sorted(set(parent_premises))),
                    rule_idx=rule.source_idx,
                    parents=rule.conditions,
                    natural_language=(
                        f"Premise {rule.source_idx + 1} derives {rule.conclusion.display()} "
                        f"when {', '.join(parent.display() for parent in rule.conditions)} hold."
                    ),
                )
                changed = True

    return known, proofs


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
