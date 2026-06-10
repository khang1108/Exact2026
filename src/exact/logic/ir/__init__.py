"""
Internal representations for the Type 1 logic branch.

The legacy Horn core remains available for fast forward chaining. Formula IR
adds typed terms and first-order structure for the Type 1 SMT path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass(frozen=True, order=True)
class Number:
    """Exact numeric literal preserved as text for solver conversion."""

    value: str


@dataclass(frozen=True, order=True)
class Function:
    """Term-valued function application such as ``gpa(sophia)``."""

    name: str
    args: tuple["Term", ...] = ()


@dataclass(frozen=True, order=True)
class Arithmetic:
    """Arithmetic term used inside comparisons or nested function arguments."""

    op: str
    args: tuple["Term", ...]


# Strings intentionally remain valid terms so the Horn fallback keeps its
# compact variable/constant convention without a migration.
Term = Union[str, Number, Function, Arithmetic]


@dataclass(frozen=True, order=True)
class Atom:
    """A normalized propositional or predicate-like logical statement."""

    pred: str
    args: tuple[Term, ...] = ()
    negated: bool = False
    text: str | None = field(default=None, compare=False)

    def positive(self) -> "Atom":
        return Atom(pred=self.pred, args=self.args, negated=False, text=self.text)

    def negation(self) -> "Atom":
        return Atom(pred=self.pred, args=self.args, negated=not self.negated, text=self.text)

    def display(self) -> str:
        label = self.text or (
            f"{self.pred}({', '.join(term_to_text(arg) for arg in self.args)})"
            if self.args
            else self.pred.replace("_", " ")
        )
        return f"not {label}" if self.negated else label


@dataclass(frozen=True)
class Not:
    """Formula-level negation around an atom or compound formula."""

    arg: "Formula"


@dataclass(frozen=True)
class And:
    """Formula-level conjunction."""

    args: tuple["Formula", ...]


@dataclass(frozen=True)
class Or:
    """Formula-level disjunction."""

    args: tuple["Formula", ...]


@dataclass(frozen=True)
class Implies:
    """Formula-level implication."""

    antecedent: "Formula"
    consequent: "Formula"


@dataclass(frozen=True)
class Iff:
    """Logical biconditional."""

    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class ForAll:
    """Universal quantifier over one or more variables."""

    variables: tuple[str, ...]
    body: "Formula"


@dataclass(frozen=True)
class Exists:
    """Existential quantifier over one or more variables."""

    variables: tuple[str, ...]
    body: "Formula"


@dataclass(frozen=True)
class Compare:
    """Comparison between symbolic, functional, or arithmetic terms."""

    op: Literal["=", "!=", ">", ">=", "<", "<="]
    left: Term
    right: Term


@dataclass(frozen=True)
class InSet:
    """Finite set-membership test."""

    member: Term
    options: tuple[Term, ...]


# Atom remains the literal leaf so existing Horn IR users keep working.
Formula = Union[Atom, Not, And, Or, Implies, Iff, ForAll, Exists, Compare, InSet]


def term_to_text(term: Term) -> str:
    """Render a typed term deterministically for traces and symbolic names."""

    if isinstance(term, str):
        return term
    if isinstance(term, Number):
        return term.value
    if isinstance(term, Function):
        args = ", ".join(term_to_text(arg) for arg in term.args)
        return f"{term.name}({args})"
    if isinstance(term, Arithmetic):
        if len(term.args) == 1:
            return f"{term.op}{term_to_text(term.args[0])}"
        return "(" + f" {term.op} ".join(term_to_text(arg) for arg in term.args) + ")"
    raise TypeError(f"unsupported term node: {type(term).__name__}")


@dataclass(frozen=True)
class FormulaItem:
    """Translated premise, query, or option formula with source provenance."""

    formula: Formula
    source_idx: int
    text: str
    role: Literal["premise", "query", "option"]
    label: str | None = None


@dataclass(frozen=True)
class TranslatedProblem:
    """One-shot Type 1 translation containing premise and goal formulas."""

    predicates: dict[str, int]
    premises: tuple[FormulaItem, ...]
    goals: tuple[FormulaItem, ...]


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
