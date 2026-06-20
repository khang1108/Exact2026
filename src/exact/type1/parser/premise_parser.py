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
    repair_arity_drift,
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
            draft_trees, restrictor_repairs = _weaken_unwitnessed_rule_restrictors(
                draft_trees,
                frames,
            )
        else:
            restrictor_repairs = []
        draft_trees, arity_repairs = repair_arity_drift(draft_trees)
        schema = PremiseSchema.from_trees(draft_trees)
        trees, renames = schema.canonicalize(draft_trees)
        issues = (
            _verify_bundle(normalized, trees, schema, frames)
            + restrictor_repairs
            + arity_repairs
        )

        blocking = tuple(i for i in issues if _is_blocking_issue(i))
        warnings = tuple(i for i in issues if not _is_blocking_issue(i))

        epistemic_witness_indices = tuple(
            i for i, premise in enumerate(normalized) if _is_meta_epistemic(premise)
        )

        return PremiseParseBundle(
            premises=normalized,
            draft_trees=draft_trees,
            schema=schema,
            trees=trees,
            predicate_renames=renames,
            verified=not blocking,
            verification_issues=tuple(issues),
            blocking_issues=blocking,
            warnings=warnings,
            epistemic_witness_indices=epistemic_witness_indices,
        )


def _is_declarative(premise: str) -> bool:
    """Reject obvious questions, commands, and multiple-choice option lines."""

    if premise.endswith("?") or _OPTION_LINE.match(premise):
        return False
    return _INTERROGATIVE_START.match(premise) is None


# Meta-epistemic premises describe the *knowledge base* ("no premise states
# whether X", "it is unknown whether X"), not the world. They carry zero
# object-level content and must never compile to ¬X. Anchored to explicit
# epistemic lexemes so genuine quantified claims ("No one is qualified.",
# "No AI models use deep learning.") are NOT matched.
_META_EPISTEMIC_RE = re.compile(
    r"^\s*(?:"
    r"no\s+(?:premises?|information|statements?|mentions?|data|details?|records?|indications?)\b"
    r"|there\s+is\s+no\s+(?:premise|information|statement|mention|indication|record|data)\b"
    r"|it\s+is\s+not\s+(?:stated|specified|mentioned|indicated|known|clear|determined)\b"
    r"|it\s+is\s+(?:unknown|unclear|undetermined|unstated|uncertain)\b"
    r"|nothing\s+(?:is\s+)?(?:said|stated|states?|mentions?|mentioned|indicates?|indicated|specifies|specified|known)\b"
    r"|(?:the\s+)?premises?\s+(?:do(?:es)?\s+not|don'?t|doesn'?t)\s+(?:state|say|mention|indicate|specify)\b"
    r"|we\s+(?:do\s+not|don'?t|cannot|can'?t)\s+(?:know|tell|determine)\b"
    r")",
    re.IGNORECASE,
)


def _is_meta_epistemic(premise: str) -> bool:
    """True for premises that disclaim knowledge instead of asserting a fact."""

    return _META_EPISTEMIC_RE.match(premise) is not None


def _normalize_premise(premise: str) -> str:
    """Normalize whitespace without changing premise meaning."""

    return re.sub(r"\s+", " ", premise).strip()


_ONLY_IF_RE = re.compile(r"\bonly\s+(?:if|when)\b", re.IGNORECASE)
_NOT_NECESSARILY_RE = re.compile(r"\bnot\s+necessarily\b", re.IGNORECASE)
_NON_BLOCKING_SCHEMA_DIAGNOSTICS = (
    "SCHEMA_SIMILAR_PREDICATES:",
    "GENERIC_CLASS_USED_AS_CONSTANT:",  # auto-repaired before schema build; residual only
)

# Only structural failures that make the AST unusable block the solver. Soft
# diagnostics (numeric/temporal lost, direction checks, similar predicates,
# entity-encoded constants, arity drift) are warnings: the solver still runs and
# the warning is surfaced. See uncertainty_cause in the pipeline for routing.
_BLOCKING_ISSUE_PREFIXES = (
    "AST_INVALID",
    "NO_AST_FOR_PREMISE",
    "RAW_FOL_PARSE_FAILED",
    "expected one AST per premise",
    # "not necessarily" compiles to a plain implication that asserts the
    # opposite of the intended epistemic modality — the AST is actively wrong,
    # so it must block rather than mislead the solver.
    "UNSUPPORTED_MODAL_NOT_NECESSARILY",
)


def _is_blocking_issue(issue: str) -> bool:
    return issue.startswith(_BLOCKING_ISSUE_PREFIXES)

_RULE_LIKE_KINDS = frozenset({
    "universal_rule", "deontic_rule", "permission_rule",
    "prohibition_rule", "numeric_rule", "temporal_rule", "equivalence",
})

# "at least one/a/1" is existential (∃), not a counting constraint — it is
# correctly compiled to an Exists quantifier with no ComparisonNode, so it must
# not trip the numeric-signal check (negative lookahead excludes cardinality-one).
_NUMERIC_SIGNAL_RE = re.compile(
    r"\b(?:at\s+least(?!\s+(?:one|a|1)\b)|at\s+most|more\s+than|fewer\s+than|less\s+than|"
    r"greater\s+than|exactly|minimum|maximum|above|below|"
    r"\d+(?:\.\d+)?\s*(?:courses?|credits?|hours?|points?|percent|%)|"
    # Plain cardinal before a noun: "has 12 enrolled participants", "contains 3 elements"
    r"(?:has|have|contains?|includes?|holds?|with|of)\s+\d+(?:\.\d+)?\s+\w+)\b",
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


def _weaken_unwitnessed_rule_restrictors(
    trees: list[FOLNode],
    frames: list[PremiseFrameResult],
) -> tuple[list[FOLNode], list[str]]:
    """Remove synthetic class guards that can never participate in a proof.

    The frame compiler prepends ``restrictor_text`` to every rule antecedent.
    Some dataset rules use a class noun only as natural-language scoping
    ("students who complete X ...") without ever asserting ``Student(Sophia)``.
    Keeping that unwitnessed guard makes the rule vacuously true and blocks the
    intended chain. Only the compiler-added first conjunct is removed, and only
    when no ground fact with the same predicate signature exists anywhere.
    """

    grounded = _ground_atomic_signatures(trees)
    repaired: list[FOLNode] = []
    diagnostics: list[str] = []

    for index, (tree, frame) in enumerate(zip(trees, frames), start=1):
        updated, removed = _remove_unwitnessed_rule_restrictor(tree, frame, grounded)
        repaired.append(updated)
        if removed is not None:
            diagnostics.append(
                "UNWITNESSED_RESTRICTOR_WEAKENED: "
                f"premise {index} removed {removed[0]}/{removed[1]} from rule antecedent"
            )

    repaired.extend(trees[len(repaired):])
    return repaired, diagnostics


def _remove_unwitnessed_rule_restrictor(
    tree: FOLNode,
    frame: PremiseFrameResult,
    grounded: set[tuple[str, int]],
) -> tuple[FOLNode, tuple[str, int] | None]:
    if (
        frame.kind not in _RULE_LIKE_KINDS
        or frame.kind == "equivalence"
        or not frame.restrictor_text
        or not isinstance(tree, QuantifiedNode)
        or tree.quantifier != "FORALL"
        or not isinstance(tree.body, LogicalNode)
        or tree.body.operator != "IMPLIES"
        or tree.body.right is None
    ):
        return tree, None

    conjuncts = _flatten_and(tree.body.left)
    if len(conjuncts) < 2:
        return tree, None

    candidate = conjuncts[0]
    if (
        not isinstance(candidate, AtomicNode)
        or len(candidate.arguments) != 1
        or candidate.arguments[0] != tree.variable
    ):
        return tree, None

    signature = (candidate.predicate.name, len(candidate.arguments))
    if signature in grounded:
        return tree, None

    body = LogicalNode(
        operator="IMPLIES",
        left=_combine_and(conjuncts[1:]),
        right=tree.body.right,
    )
    return (
        QuantifiedNode(
            quantifier=tree.quantifier,
            variable=tree.variable,
            body=body,
            restrictor=tree.restrictor,
        ),
        signature,
    )


def _ground_atomic_signatures(trees: list[FOLNode]) -> set[tuple[str, int]]:
    signatures: set[tuple[str, int]] = set()
    for tree in trees:
        _collect_ground_atomic_signatures(tree, frozenset(), signatures)
    return signatures


def _collect_ground_atomic_signatures(
    node: FOLNode,
    bound_variables: frozenset[str],
    signatures: set[tuple[str, int]],
) -> None:
    if isinstance(node, AtomicNode):
        if all(argument not in bound_variables for argument in node.arguments):
            signatures.add((node.predicate.name, len(node.arguments)))
        return
    if isinstance(node, ComparisonNode):
        return
    if isinstance(node, QuantifiedNode):
        local_variables = bound_variables | {node.variable}
        _collect_ground_atomic_signatures(node.body, local_variables, signatures)
        if node.restrictor is not None:
            _collect_ground_atomic_signatures(node.restrictor, local_variables, signatures)
        return
    _collect_ground_atomic_signatures(node.left, bound_variables, signatures)
    if node.right is not None:
        _collect_ground_atomic_signatures(node.right, bound_variables, signatures)


def _flatten_and(node: FOLNode) -> list[FOLNode]:
    if isinstance(node, LogicalNode) and node.operator == "AND" and node.right is not None:
        return [*_flatten_and(node.left), *_flatten_and(node.right)]
    return [node]


def _combine_and(nodes: list[FOLNode]) -> FOLNode:
    result = nodes[0]
    for node in nodes[1:]:
        result = LogicalNode(operator="AND", left=result, right=node)
    return result


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
