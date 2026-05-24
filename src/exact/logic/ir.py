"""
Internal representations for the Type 1 logic branch.

The first production target is a small, inspectable Horn-style core:
LLM/CLOVER-style parsers can emit these objects, VERUS-style KB caching can
reuse them, and a Logic-LM/LINC-style solver/verifier can reason over them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class Atom:
    """A normalized propositional or predicate-like logical statement."""

    pred: str
    args: tuple[str, ...] = ()
    negated: bool = False
    text: str | None = field(default=None, compare=False)

    def positive(self) -> "Atom":
        return Atom(pred=self.pred, args=self.args, negated=False, text=self.text)

    def negation(self) -> "Atom":
        return Atom(pred=self.pred, args=self.args, negated=not self.negated, text=self.text)

    def display(self) -> str:
        label = self.text or (
            f"{self.pred}({', '.join(self.args)})" if self.args else self.pred.replace("_", " ")
        )
        return f"not {label}" if self.negated else label


@dataclass(frozen=True)
class Rule:
    """A Horn-style rule: all conditions must hold to derive conclusion."""

    conditions: tuple[Atom, ...]
    conclusion: Atom
    source_idx: int
    text: str


@dataclass(frozen=True)
class Fact:
    """A directly stated premise fact."""

    atom: Atom
    source_idx: int
    text: str


@dataclass(frozen=True)
class ProofStep:
    """One derivation step with provenance for explanation and scoring depth."""

    derived: Atom
    used_premises: tuple[int, ...]
    rule_idx: int | None
    parents: tuple[Atom, ...] = ()
    natural_language: str | None = None


@dataclass(frozen=True)
class ParsedPremise:
    """Parser output for one source premise."""

    facts: tuple[Fact, ...] = ()
    rules: tuple[Rule, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Query:
    """Normalized target claim for yes/no/unknown reasoning."""

    claim: Atom
    raw_question: str
    expects_negation: bool = False


@dataclass(frozen=True)
class SolveResult:
    """Result returned by a symbolic solver."""

    label: str
    claim: Atom
    proof: tuple[ProofStep, ...] = ()
    supporting_premises: tuple[int, ...] = ()
    mode: str = "symbolic_forward_chain"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Theory:
    """Future extension point for richer typed FOL/Z3 encodings."""

    sorts: dict[str, list[str]] = field(default_factory=dict)
    predicates: dict[str, tuple[str, ...]] = field(default_factory=dict)
    functions: dict[str, tuple[tuple[str, ...], str]] = field(default_factory=dict)
    constants: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PremiseItem:
    """Unified representation of an NL/FOL premise pair from training data."""

    id: str
    nl: str | None
    fol: str | None
    source: str


@dataclass(frozen=True)
class SymbolEvidence:
    """Traceable mapping from a symbol back to its source premise."""

    premise_id: str
    source: str
    symbol: str
    confidence: float = 1.0
