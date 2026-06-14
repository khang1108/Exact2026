"""Frame-based premise parser and deterministic AST compiler.

Two-stage approach that replaces direct FOL generation for premises:
  1. PremiseFrameParser  — one LLM call per premise: identifies logical structure
     (kind, variable, restrictor, conditions, conclusions) without generating FOL.
  2. PremiseFrameCompiler — zero LLM calls: builds FOLNode from the frame by
     parsing each atomic fragment through FOLParser and composing deterministically.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from exact.type1.ast.nodes import (
    AtomicNode,
    ComparisonNode,
    DateTerm,
    FOLNode,
    FunctionTerm,
    LogicalNode,
    NumericTerm,
    QuantifiedNode,
)
from exact.type1.models.schemas import Predicate
from exact.type1.parser.schemas import (
    NumericConstraintResult,
    PremiseFrameResult,
    TemporalConstraintResult,
)
from exact.type1.prompts import (
    get_system_prompt_numeric_constraint,
    get_system_prompt_premise_frame,
    get_system_prompt_temporal_constraint,
)

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient
    from exact.type1.parser.parser import FOLParser

ChatMessage = dict[str, Any]


class PremiseFrameParser:
    """Parse each premise into a PremiseFrameResult via one LLM call per sentence."""

    def __init__(self, client: ParserClient) -> None:
        self.client = client

    async def parse_many(self, premises: list[str]) -> list[PremiseFrameResult]:
        """Dispatch all premises concurrently; vLLM batches them on the GPU."""
        messages_batch = [self._messages(p) for p in premises]
        return await self.client.parse_many_as(messages_batch, PremiseFrameResult)

    def _messages(self, premise: str) -> list[ChatMessage]:
        return [
            {"role": "system", "content": get_system_prompt_premise_frame().strip()},
            {"role": "user", "content": f'Input: "{premise}"'},
        ]


_TEMPORAL_OP_MAP: dict[str, str] = {
    "before": "<",
    "by": "<=",
    "until": "<=",
    "on": "=",
    "after": ">",
}


class ConstraintParser:
    """Parse numeric and temporal constraint sentences into ComparisonNode ASTs."""

    def __init__(self, client: ParserClient) -> None:
        self.client = client

    async def parse_numerics(self, texts: list[str]) -> list[ComparisonNode]:
        if not texts:
            return []
        messages_batch = [self._numeric_messages(t) for t in texts]
        results: list[NumericConstraintResult] = await self.client.parse_many_as(
            messages_batch, NumericConstraintResult
        )
        return [self._compile_numeric(r) for r in results]

    async def parse_temporals(self, texts: list[str]) -> list[ComparisonNode]:
        if not texts:
            return []
        messages_batch = [self._temporal_messages(t) for t in texts]
        results: list[TemporalConstraintResult] = await self.client.parse_many_as(
            messages_batch, TemporalConstraintResult
        )
        return [self._compile_temporal(r) for r in results]

    def _compile_numeric(self, r: NumericConstraintResult) -> ComparisonNode:
        return ComparisonNode(
            operator=r.operator,
            left=FunctionTerm(name=r.function_name, arguments=list(r.arguments)),
            right=NumericTerm(value=r.value),
        )

    def _compile_temporal(self, r: TemporalConstraintResult) -> ComparisonNode:
        operator = _TEMPORAL_OP_MAP.get(r.operator, "=")
        return ComparisonNode(
            operator=operator,  # type: ignore[arg-type]
            left=FunctionTerm(name=r.function_name, arguments=list(r.arguments)),
            right=DateTerm(value=r.date_value),
        )

    def _numeric_messages(self, text: str) -> list[ChatMessage]:
        return [
            {"role": "system", "content": get_system_prompt_numeric_constraint().strip()},
            {"role": "user", "content": f'Input: "{text}"'},
        ]

    def _temporal_messages(self, text: str) -> list[ChatMessage]:
        return [
            {"role": "system", "content": get_system_prompt_temporal_constraint().strip()},
            {"role": "user", "content": f'Input: "{text}"'},
        ]


class PremiseFrameCompiler:
    """Build FOLNode from a PremiseFrameResult without any further LLM calls.

    Each text fragment in the frame (restrictor_text, condition_texts, etc.) is
    force-parsed as an atomic predicate by the underlying FOLParser. The compiler
    then assembles the correct quantifier / operator skeleton deterministically.

    Falls back to a full recursive parse for ``meta_rule`` and ``unsupported``
    frames where the structure cannot be reliably decomposed.
    """

    def __init__(
        self,
        fol_parser: FOLParser,
        constraint_parser: ConstraintParser | None = None,
    ) -> None:
        self.fol_parser = fol_parser
        self.constraint_parser = constraint_parser

    async def compile_many(
        self,
        premises: list[str],
        frames: list[PremiseFrameResult],
    ) -> list[FOLNode]:
        """Compile all frames concurrently while preserving input order."""
        return list(await asyncio.gather(*(
            self._compile(premise, frame)
            for premise, frame in zip(premises, frames)
        )))

    async def _compile(self, premise: str, frame: PremiseFrameResult) -> FOLNode:
        kind = frame.kind
        var = frame.variable or "x"

        if kind in ("meta_rule", "unsupported"):
            return (await self.fol_parser.parse_many([premise]))[0]

        if kind in ("fact", "numeric_fact"):
            texts = frame.fact_texts or frame.conclusion_texts
            if not texts and not frame.numeric_constraints:
                return (await self.fol_parser.parse_many([premise]))[0]
            atomic_nodes = await self._parse_conclusions(texts, frame.modality) if texts else []
            numeric_nodes = await self._parse_numeric_constraints(frame.numeric_constraints)
            all_nodes: list[FOLNode] = list(atomic_nodes) + list(numeric_nodes)
            if not all_nodes:
                return (await self.fol_parser.parse_many([premise]))[0]
            return _and_nodes(all_nodes)

        if kind == "existential_fact":
            atomic_texts = (
                ([frame.restrictor_text] if frame.restrictor_text else [])
                + frame.condition_texts
            )
            numeric_nodes, temporal_nodes = await asyncio.gather(
                self._parse_numeric_constraints(frame.numeric_constraints),
                self._parse_temporal_constraints(frame.temporal_constraints),
            )
            atomic_nodes = await self._parse_atomics(atomic_texts) if atomic_texts else []
            all_nodes: list[FOLNode] = atomic_nodes + list(numeric_nodes) + list(temporal_nodes)
            if not all_nodes:
                return (await self.fol_parser.parse_many([premise]))[0]
            return QuantifiedNode(quantifier="EXISTS", variable=var, body=_and_nodes(all_nodes))

        if kind == "equivalence":
            lhs_texts = (
                ([frame.restrictor_text] if frame.restrictor_text else [])
                + frame.condition_texts
            )
            rhs_texts = frame.conclusion_texts
            if not lhs_texts or not rhs_texts:
                return (await self.fol_parser.parse_many([premise]))[0]
            lhs_nodes = await self._parse_atomics(lhs_texts)
            rhs_nodes = await self._parse_conclusions(rhs_texts, frame.modality)
            lhs = _and_nodes(lhs_nodes)
            rhs = _and_nodes(rhs_nodes)
            body = LogicalNode(operator="IFF", left=lhs, right=rhs)
            return QuantifiedNode(quantifier="FORALL", variable=var, body=body)

        # universal_rule, numeric_rule, deontic_rule, permission_rule,
        # prohibition_rule, temporal_rule — all map to ∀x.(antecedent → consequent)
        atomic_cond_texts = (
            ([frame.restrictor_text] if frame.restrictor_text else [])
            + frame.condition_texts
        )
        concl_texts = frame.conclusion_texts

        atomic_cond_nodes = await self._parse_atomics(atomic_cond_texts) if atomic_cond_texts else []
        numeric_nodes, temporal_nodes = await asyncio.gather(
            self._parse_numeric_constraints(frame.numeric_constraints),
            self._parse_temporal_constraints(frame.temporal_constraints),
        )
        all_cond_nodes: list[FOLNode] = list(atomic_cond_nodes) + list(numeric_nodes) + list(temporal_nodes)

        if not all_cond_nodes or not concl_texts:
            return (await self.fol_parser.parse_many([premise]))[0]

        antecedent = _and_nodes(all_cond_nodes)
        consequent = _and_nodes(
            await self._parse_conclusions(concl_texts, frame.modality)
        )
        body = LogicalNode(operator="IMPLIES", left=antecedent, right=consequent)
        return QuantifiedNode(quantifier="FORALL", variable=var, body=body)

    async def _parse_atomics(self, texts: list[str]) -> list[FOLNode]:
        """Force-parse each text as an atomic predicate, skipping the classifier."""
        return list(await asyncio.gather(*(
            self.fol_parser._parse_atomic(t) for t in texts  # noqa: SLF001
        )))

    async def _parse_numeric_constraints(self, texts: list[str]) -> list[ComparisonNode]:
        if not texts or self.constraint_parser is None:
            return []
        return await self.constraint_parser.parse_numerics(texts)

    async def _parse_temporal_constraints(self, texts: list[str]) -> list[ComparisonNode]:
        if not texts or self.constraint_parser is None:
            return []
        return await self.constraint_parser.parse_temporals(texts)

    async def _parse_conclusions(
        self,
        texts: list[str],
        modality: str,
    ) -> list[FOLNode]:
        """Parse conclusions and encode their deontic meaning in predicate names."""

        parse_texts = [_strip_not_necessarily(text) for text in texts]
        nodes = await self._parse_atomics(parse_texts)
        return [
            _apply_deontic_mapping(text, node, modality)
            for text, node in zip(texts, nodes)
        ]


def _and_nodes(nodes: list[FOLNode]) -> FOLNode:
    """Left-fold a list of FOL nodes into a nested AND tree."""
    result = nodes[0]
    for node in nodes[1:]:
        result = LogicalNode(operator="AND", left=result, right=node)
    return result


_NOT_NECESSARILY_RE = re.compile(r"\bnot\s+necessarily\b", re.IGNORECASE)
_PROHIBITED_RE = re.compile(
    r"\b(?:cannot|can\s+not|must\s+not|prohibited(?:\s+from)?|not\s+allowed\s+to)\b",
    re.IGNORECASE,
)
_REQUIRED_RE = re.compile(r"\brequired\s+to\b", re.IGNORECASE)
_MUST_RE = re.compile(r"\b(?:must|should)\b", re.IGNORECASE)
_ALLOWED_RE = re.compile(r"\b(?:may|allowed\s+to)\b", re.IGNORECASE)
_CAN_RE = re.compile(r"\bcan\b", re.IGNORECASE)
_ELIGIBLE_FOR_RE = re.compile(r"\beligible\s+for\b", re.IGNORECASE)


def _strip_not_necessarily(text: str) -> str:
    """Remove epistemic wording so it cannot become object-level negation."""

    return re.sub(r"\s+", " ", _NOT_NECESSARILY_RE.sub("", text)).strip()


def _apply_deontic_mapping(text: str, node: FOLNode, modality: str) -> FOLNode:
    """Rewrite one parsed conclusion into an explicit deontic predicate."""

    atom, parsed_negated = _unwrap_atomic_negation(node)
    if atom is None:
        return node

    effective_modality = _effective_modality(text, modality)
    if effective_modality == "not_necessarily":
        return atom

    if _ELIGIBLE_FOR_RE.search(text):
        rewritten = _rename_atomic(atom, "EligibleFor")
    elif effective_modality == "must":
        rewritten = _prefix_atomic(atom, "Required")
    elif effective_modality == "required":
        rewritten = _prefix_atomic(atom, "Requires")
    elif effective_modality in {"may", "allowed"}:
        rewritten = _prefix_atomic(atom, "Allowed")
    elif effective_modality == "can":
        rewritten = _prefix_atomic(atom, "Can")
    elif effective_modality == "prohibited":
        rewritten = _prefix_atomic(atom, "Can")
        return LogicalNode(operator="NOT", left=rewritten)
    else:
        rewritten = atom

    if parsed_negated:
        return LogicalNode(operator="NOT", left=rewritten)
    return rewritten


def _effective_modality(text: str, modality: str) -> str:
    if _NOT_NECESSARILY_RE.search(text) or modality == "not_necessarily":
        return "not_necessarily"
    if _PROHIBITED_RE.search(text) or modality == "prohibited":
        return "prohibited"
    if _REQUIRED_RE.search(text) or modality == "required":
        return "required"
    if _MUST_RE.search(text) or modality == "must":
        return "must"
    if _ALLOWED_RE.search(text) or modality in {"may", "allowed"}:
        return modality if modality in {"may", "allowed"} else "allowed"
    if _CAN_RE.search(text) or modality == "can":
        return "can"
    return "none"


def _unwrap_atomic_negation(node: FOLNode) -> tuple[AtomicNode | None, bool]:
    if isinstance(node, AtomicNode):
        return node, False
    if (
        isinstance(node, LogicalNode)
        and node.operator == "NOT"
        and isinstance(node.left, AtomicNode)
    ):
        return node.left, True
    return None, False


def _prefix_atomic(atom: AtomicNode, prefix: str) -> AtomicNode:
    name = atom.predicate.name
    if name.startswith(prefix):
        return atom

    for removable in (
        "NotAllowedTo",
        "ProhibitedFrom",
        "Prohibited",
        "Cannot",
        "Can",
        "Allowed",
        "Required",
        "Requires",
    ):
        if name.startswith(removable) and len(name) > len(removable):
            name = name[len(removable):]
            break
    return _rename_atomic(atom, f"{prefix}{name}")


def _rename_atomic(atom: AtomicNode, name: str) -> AtomicNode:
    if atom.predicate.name == name:
        return atom
    predicate = Predicate(
        name=name,
        arg_sorts=atom.predicate.arg_sorts,
        description=atom.predicate.description,
        aliases=[*atom.predicate.aliases, atom.predicate.name],
    )
    return AtomicNode(predicate=predicate, arguments=atom.arguments)
