"""Structured response schemas produced by Type 1 parser operations.

Each parser operation has a narrow output contract. These Pydantic models are
sent to vLLM as JSON schemas and then validate the returned model content.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass
from exact.type1.ast import AtomicNode, FOLNode, LogicalNode, QuantifiedNode
from exact.type1.models.schemas import Predicate

import re

# _ALIGN_THRESHOLD = 0.6 controls when two differently named predicates are considered equivalent
_ALIGN_THRESHOLD = 0.6
_INTERROGATIVE_START = re.compile(
    r"^(?:can|could|did|do|does|how|is|may|should|was|were|what|when|where|"
    r"which|who|whom|whose|why|will|would)\b",
    re.IGNORECASE,
)
_OPTION_LINE = re.compile(r"^[A-E]\.\s+", re.IGNORECASE)



class ParserResult(BaseModel):
    """Base schema that rejects fields not requested by the parser operation."""

    model_config = ConfigDict(extra="forbid")


class RephraseResult(ParserResult):
    """Result of minimally rewriting a sentence for clearer FOL parsing."""

    rephrased: str


class QuantifiedResult(ParserResult):
    """Result of extracting one outer quantifier and its remaining sentence."""

    quantifier: Literal["ForAll", "ThereExists"]
    variable: str
    restrictor_sentence: str | None = None
    scope_sentence: str


class LogicalResult(ParserResult):
    """Result of splitting a sentence at its outermost logical operator."""

    operator: Literal["AND", "OR", "IMPLIES", "IFF", "NOT"]
    left_operand: str
    right_operand: str | None


class AtomicResult(ParserResult):
    """Result of extracting a predicate and arguments from an atomic sentence."""

    predicate: str
    arguments: list[str]
    negated: bool


class CoreferenceResult(ParserResult):
    """Result of resolving pronouns in a right-hand clause."""

    resolved_right: str


class PremiseFrameResult(ParserResult):
    """Structural decomposition of one premise before FOL generation.

    The frame identifies the logical role of each text fragment so the
    compiler can build the correct quantifier / operator skeleton deterministically.
    All text fragments must reference the entity through ``variable``.
    """

    kind: Literal[
        "fact",
        "universal_rule",
        "existential_fact",
        "equivalence",
        "numeric_fact",
        "numeric_rule",
        "deontic_rule",
        "permission_rule",
        "prohibition_rule",
        "temporal_rule",
        "meta_rule",
        "unsupported",
    ]

    variable: str | None = None
    restrictor_text: str | None = None

    condition_texts: list[str] = []
    conclusion_texts: list[str] = []
    fact_texts: list[str] = []
    numeric_constraints: list[str] = []
    temporal_constraints: list[str] = []

    modality: Literal[
        "none",
        "must",
        "can",
        "may",
        "allowed",
        "required",
        "prohibited",
        "not_necessarily",
    ] = "none"

    confidence: float = 1.0


# ------------------------------------------------------------------
# Schemas for Premise Parser
# ------------------------------------------------------------------
@dataclass(frozen=True)
class PredicateSignature:
    """One canonical predicate available to a Type 1 problem."""

    name: str
    arity: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PremiseSchema:
    """Canonical predicate vocabulary derived from premise ASTs."""

    predicates: tuple[PredicateSignature, ...]

    @classmethod
    def from_trees(cls, trees: list[FOLNode]) -> PremiseSchema:
        """Build a stable schema, merging similarly named predicates by arity."""

        canonical: list[PredicateSignature] = []
        for tree in trees:
            for name, arity in _collect_predicates_in_order(tree):
                match = _best_schema_match(name, arity, canonical)
                if match is None:
                    canonical.append(PredicateSignature(name=name, arity=arity))
                    continue
                if name != match.name and name not in match.aliases:
                    index = canonical.index(match)
                    canonical[index] = PredicateSignature(
                        name=match.name,
                        arity=match.arity,
                        aliases=(*match.aliases, name),
                    )
        return cls(predicates=tuple(canonical))

    def canonicalize(self, trees: list[FOLNode]) -> tuple[list[FOLNode], list[dict[str, object]]]:
        """Rename predicates in ``trees`` to matching canonical schema names."""

        remap: dict[tuple[str, int], str] = {}
        for tree in trees:
            for name, arity in _collect_predicates_in_order(tree):
                match = _best_schema_match(name, arity, list(self.predicates))
                if match is not None and match.name != name:
                    remap[(name, arity)] = match.name

        renames = [
            {"from": name, "arity": arity, "to": canonical}
            for (name, arity), canonical in remap.items()
        ]
        if not remap:
            return trees, renames
        return [_rename_in_node(tree, remap) for tree in trees], renames

    def contains(self, name: str, arity: int) -> bool:
        """Return whether the exact canonical predicate is in the schema."""

        return any(
            predicate.name == name and predicate.arity == arity
            for predicate in self.predicates
        )


@dataclass(frozen=True)
class PremiseParseBundle:
    """Verified result of the complete premise parsing workflow."""

    premises: list[str]
    draft_trees: list[FOLNode]
    schema: PremiseSchema
    trees: list[FOLNode]
    predicate_renames: list[dict[str, object]]
    verified: bool
    verification_issues: tuple[str, ...]

def _collect_predicates_in_order(node: FOLNode) -> list[tuple[str, int]]:
    if isinstance(node, AtomicNode):
        return [(node.predicate.name, len(node.arguments))]
    if isinstance(node, QuantifiedNode):
        predicates = _collect_predicates_in_order(node.body)
        if node.restrictor is not None:
            predicates.extend(_collect_predicates_in_order(node.restrictor))
        return predicates

    predicates = _collect_predicates_in_order(node.left)
    if node.right is not None:
        predicates.extend(_collect_predicates_in_order(node.right))
    return predicates

def _rename_in_node(node: FOLNode, remap: dict[tuple[str, int], str]) -> FOLNode:
    if isinstance(node, AtomicNode):
        key = (node.predicate.name, len(node.arguments))
        canonical_name = remap.get(key)
        if canonical_name is None:
            return node
        predicate = Predicate(
            name=canonical_name,
            arg_sorts=node.predicate.arg_sorts,
            description=node.predicate.description,
            aliases=[*node.predicate.aliases, node.predicate.name],
        )
        return AtomicNode(predicate=predicate, arguments=node.arguments)

    if isinstance(node, QuantifiedNode):
        return QuantifiedNode(
            quantifier=node.quantifier,
            variable=node.variable,
            body=_rename_in_node(node.body, remap),
            restrictor=(
                _rename_in_node(node.restrictor, remap)
                if node.restrictor is not None
                else None
            ),
        )

    return LogicalNode(
        operator=node.operator,
        left=_rename_in_node(node.left, remap),
        right=_rename_in_node(node.right, remap) if node.right is not None else None,
    )

def _best_schema_match(
    name: str,
    arity: int,
    predicates: list[PredicateSignature],
) -> PredicateSignature | None:
    exact = next(
        (
            predicate
            for predicate in predicates
            if predicate.arity == arity
            and (predicate.name == name or name in predicate.aliases)
        ),
        None,
    )
    if exact is not None:
        return exact

    candidates = [predicate for predicate in predicates if predicate.arity == arity]
    if not candidates:
        return None
    best = max(candidates, key=lambda predicate: _similarity(name, predicate.name))
    if _similarity(name, best.name) < _ALIGN_THRESHOLD:
        return None
    return best

def _similarity(left: str, right: str) -> float:
    left_words = _camel_words(left)
    right_words = _camel_words(right)
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 0.0

def _camel_words(name: str) -> frozenset[str]:
    """Split a predicate identifier into lowercase comparison tokens."""

    return frozenset(
        word.lower() for word in re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", name)
    )