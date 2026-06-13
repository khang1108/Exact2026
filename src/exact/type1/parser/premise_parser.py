"""High-level orchestration for parsing declarative Type 1 premises."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from exact.type1.ast.nodes import AtomicNode, ComparisonNode, FOLNode, LogicalNode, QuantifiedNode
from exact.type1.models.schemas import Predicate
from exact.type1.parser.frame_parser import ConstraintParser, PremiseFrameCompiler, PremiseFrameParser
from exact.type1.parser.parser import FOLParser
from exact.type1.parser.schemas import (
    PremiseFrameResult,
    PremiseParseBundle,
    PremiseSchema,
    _INTERROGATIVE_START,
    _OPTION_LINE,
    is_generic_class_constant,
    singularize_class_constant,
)

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient


class PremiseParser:
    """Parse and verify declarative premises using frame decomposition + atomic parsing."""

    def __init__(
        self,
        fol_parser: FOLParser,
        frame_parser: PremiseFrameParser | None = None,
        frame_compiler: PremiseFrameCompiler | None = None,
    ) -> None:
        if (frame_parser is None) != (frame_compiler is None):
            raise ValueError("frame_parser and frame_compiler must be configured together")
        if frame_parser is None and frame_compiler is None:
            client = getattr(fol_parser, "client", None)
            if client is not None:
                frame_parser = PremiseFrameParser(client)
                frame_compiler = PremiseFrameCompiler(fol_parser)

        self.fol_parser = fol_parser
        self.frame_parser = frame_parser
        self.frame_compiler = frame_compiler

    @classmethod
    def from_parser_client(cls, client: ParserClient) -> PremiseParser:
        """Construct all parser components from one shared client."""
        fol_parser = FOLParser(client)
        frame_parser = PremiseFrameParser(client)
        constraint_parser = ConstraintParser(client)
        frame_compiler = PremiseFrameCompiler(fol_parser, constraint_parser)
        return cls(fol_parser, frame_parser, frame_compiler)

    async def parse_premises(self, premises: list[str]) -> PremiseParseBundle:
        """Normalize, frame-decompose, compile, canonicalize, and verify premises."""

        normalized = [_normalize_premise(p) for p in premises]
        normalized = [p for p in normalized if p]
        if not normalized:
            raise ValueError("premises must contain at least one non-empty declarative statement")

        invalid = [p for p in normalized if not _is_declarative(p)]
        if invalid:
            raise ValueError(
                "PremiseParser accepts declarative premises only; rejected: "
                + "; ".join(repr(p) for p in invalid)
            )

        frames: list[PremiseFrameResult] | None = None
        if self.frame_parser is not None and self.frame_compiler is not None:
            frames = await self.frame_parser.parse_many(normalized)
            draft_trees = await self.frame_compiler.compile_many(normalized, frames)
        else:
            draft_trees = await self.fol_parser.parse_many(normalized)
        if frames is not None:
            draft_trees = [
                _repair_generic_class_constants(tree, frame)
                for tree, frame in zip(draft_trees, frames)  # type: ignore[arg-type]
            ]
        schema = PremiseSchema.from_trees(draft_trees)
        trees, renames = schema.canonicalize(draft_trees)
        issues = _verify_bundle(normalized, trees, schema, frames)

        return PremiseParseBundle(
            premises=normalized,
            draft_trees=draft_trees,
            schema=schema,
            trees=trees,
            predicate_renames=renames,
            verified=not issues,
            verification_issues=tuple(issues),
        )


def _is_declarative(premise: str) -> bool:
    """Reject obvious questions, commands, and multiple-choice option lines."""

    if premise.endswith("?") or _OPTION_LINE.match(premise):
        return False
    return _INTERROGATIVE_START.match(premise) is None


def _normalize_premise(premise: str) -> str:
    """Normalize whitespace without changing premise meaning."""

    return re.sub(r"\s+", " ", premise).strip()


_ONLY_IF_RE = re.compile(r"\bonly\s+(?:if|when)\b", re.IGNORECASE)
_NOT_NECESSARILY_RE = re.compile(r"\bnot\s+necessarily\b", re.IGNORECASE)
_NON_BLOCKING_SCHEMA_DIAGNOSTICS = (
    "SCHEMA_SIMILAR_PREDICATES:",
    "GENERIC_CLASS_USED_AS_CONSTANT:",  # auto-repaired before schema build; residual only
)

_RULE_LIKE_KINDS = frozenset({
    "universal_rule", "deontic_rule", "permission_rule",
    "prohibition_rule", "numeric_rule", "temporal_rule", "equivalence",
})

_NUMERIC_SIGNAL_RE = re.compile(
    r"\b(?:at\s+least|at\s+most|more\s+than|fewer\s+than|less\s+than|"
    r"greater\s+than|exactly|minimum|maximum|above|below|"
    r"\d+(?:\.\d+)?\s*(?:courses?|credits?|hours?|points?|percent|%))\b",
    re.IGNORECASE,
)
_TEMPORAL_SIGNAL_RE = re.compile(
    r"\b(?:before|after|by\s+\w+\s*\d|until|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s*\d{1,2})\b",
    re.IGNORECASE,
)


def _repair_generic_class_constants(
    tree: FOLNode,
    frame: PremiseFrameResult,
) -> FOLNode:
    """Promote a generic class-constant argument to a ForAll quantifier.

    Only applied when the frame identifies a rule-like premise AND the compiled
    tree is a bare AtomicNode (the fallback path produced something like
    ``Register(Students, BKDorm)`` instead of ``∀x[Student(x)].Register(x, BKDorm)``).
    The first matching class-constant argument becomes the domain restrictor.
    """
    if frame.kind not in _RULE_LIKE_KINDS:
        return tree
    if not isinstance(tree, AtomicNode):
        return tree

    for i, arg in enumerate(tree.arguments):
        if not is_generic_class_constant(arg):
            continue
        var = "x"
        restrictor_name = singularize_class_constant(arg)
        restrictor_pred = Predicate(
            name=restrictor_name,
            arg_sorts=["Entity"],
            aliases=[arg],
        )
        restrictor = AtomicNode(predicate=restrictor_pred, arguments=[var])
        new_args = list(tree.arguments)
        new_args[i] = var
        body = AtomicNode(predicate=tree.predicate, arguments=new_args)
        return QuantifiedNode(
            quantifier="FORALL",
            variable=var,
            body=body,
            restrictor=restrictor,
        )
    return tree


def _has_comparison_node(tree: FOLNode) -> bool:
    if isinstance(tree, ComparisonNode):
        return True
    if isinstance(tree, QuantifiedNode):
        return _has_comparison_node(tree.body) or (
            tree.restrictor is not None and _has_comparison_node(tree.restrictor)
        )
    if isinstance(tree, LogicalNode):
        return _has_comparison_node(tree.left) or (
            tree.right is not None and _has_comparison_node(tree.right)
        )
    return False


def _verify_bundle(
    premises: list[str],
    trees: list[FOLNode],
    schema: PremiseSchema,
    frames: list[PremiseFrameResult] | None = None,
) -> list[str]:
    issues = [
        diagnostic
        for diagnostic in schema.diagnostics
        if not diagnostic.startswith(_NON_BLOCKING_SCHEMA_DIAGNOSTICS)
    ]
    if len(premises) != len(trees):
        issues.append(
            f"expected one AST per premise, received {len(trees)} ASTs for {len(premises)} premises"
        )

    for index, (premise, tree) in enumerate(zip(premises, trees)):
        if _ONLY_IF_RE.search(premise):
            issues.append(
                f"ONLY_IF_DIRECTION_CHECK: premise {index + 1} contains 'only if/when' — "
                f"confirm FOL has the result as left_operand and the condition as right_operand"
            )
        frame_modality = frames[index].modality if frames and index < len(frames) else "none"
        if frame_modality == "not_necessarily" or _NOT_NECESSARILY_RE.search(premise):
            issues.append(
                "UNSUPPORTED_MODAL_NOT_NECESSARILY: "
                f"premise {index + 1} requires epistemic non-entailment reasoning"
            )
        has_cmp = _has_comparison_node(tree)
        if _NUMERIC_SIGNAL_RE.search(premise) and not has_cmp:
            issues.append(
                f"NUMERIC_CONSTRAINT_LOST: premise {index + 1} has numeric signals "
                f"but no ComparisonNode in the compiled AST"
            )
        if _TEMPORAL_SIGNAL_RE.search(premise) and not has_cmp:
            issues.append(
                f"TEMPORAL_CONSTRAINT_LOST: premise {index + 1} has temporal signals "
                f"but no ComparisonNode in the compiled AST"
            )

    return issues
