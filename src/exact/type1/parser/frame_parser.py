"""Frame-based premise parser and deterministic AST compiler.

Two-stage approach that replaces direct FOL generation for premises:
  1. PremiseFrameParser  — one LLM call per premise: identifies logical structure
     (kind, variable, restrictor, conditions, conclusions) without generating FOL.
  2. PremiseFrameCompiler — zero LLM calls: builds FOLNode from the frame by
     parsing each atomic fragment through FOLParser and composing deterministically.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from exact.type1.ast.nodes import FOLNode, LogicalNode, QuantifiedNode
from exact.type1.parser.schemas import PremiseFrameResult
from exact.type1.prompts import get_system_prompt_premise_frame

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


class PremiseFrameCompiler:
    """Build FOLNode from a PremiseFrameResult without any further LLM calls.

    Each text fragment in the frame (restrictor_text, condition_texts, etc.) is
    force-parsed as an atomic predicate by the underlying FOLParser. The compiler
    then assembles the correct quantifier / operator skeleton deterministically.

    Falls back to a full recursive parse for ``meta_rule`` and ``unsupported``
    frames where the structure cannot be reliably decomposed.
    """

    def __init__(self, fol_parser: FOLParser) -> None:
        self.fol_parser = fol_parser

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
            if not texts:
                return (await self.fol_parser.parse_many([premise]))[0]
            nodes = await self._parse_atomics(texts)
            return _and_nodes(nodes)

        if kind == "existential_fact":
            texts = (
                ([frame.restrictor_text] if frame.restrictor_text else [])
                + frame.condition_texts
                + frame.numeric_constraints
            )
            if not texts:
                return (await self.fol_parser.parse_many([premise]))[0]
            nodes = await self._parse_atomics(texts)
            return QuantifiedNode(quantifier="EXISTS", variable=var, body=_and_nodes(nodes))

        if kind == "equivalence":
            lhs_texts = (
                ([frame.restrictor_text] if frame.restrictor_text else [])
                + frame.condition_texts
            )
            rhs_texts = frame.conclusion_texts
            if not lhs_texts or not rhs_texts:
                return (await self.fol_parser.parse_many([premise]))[0]
            all_nodes = await self._parse_atomics(lhs_texts + rhs_texts)
            lhs = _and_nodes(all_nodes[:len(lhs_texts)])
            rhs = _and_nodes(all_nodes[len(lhs_texts):])
            body = LogicalNode(operator="IFF", left=lhs, right=rhs)
            return QuantifiedNode(quantifier="FORALL", variable=var, body=body)

        # universal_rule, numeric_rule, deontic_rule, permission_rule,
        # prohibition_rule, temporal_rule — all map to ∀x.(antecedent → consequent)
        cond_texts = (
            ([frame.restrictor_text] if frame.restrictor_text else [])
            + frame.condition_texts
            + frame.numeric_constraints
            + frame.temporal_constraints
        )
        concl_texts = frame.conclusion_texts

        if not cond_texts or not concl_texts:
            return (await self.fol_parser.parse_many([premise]))[0]

        all_nodes = await self._parse_atomics(cond_texts + concl_texts)
        antecedent = _and_nodes(all_nodes[:len(cond_texts)])
        consequent = _and_nodes(all_nodes[len(cond_texts):])
        body = LogicalNode(operator="IMPLIES", left=antecedent, right=consequent)
        return QuantifiedNode(quantifier="FORALL", variable=var, body=body)

    async def _parse_atomics(self, texts: list[str]) -> list[FOLNode]:
        """Force-parse each text as an atomic predicate, skipping the classifier."""
        return list(await asyncio.gather(*(
            self.fol_parser._parse_atomic(t) for t in texts  # noqa: SLF001
        )))


def _and_nodes(nodes: list[FOLNode]) -> FOLNode:
    """Left-fold a list of FOL nodes into a nested AND tree."""
    result = nodes[0]
    for node in nodes[1:]:
        result = LogicalNode(operator="AND", left=result, right=node)
    return result
