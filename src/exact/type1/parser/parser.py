"""Recursive asynchronous natural-language to FOL AST parser.

``FOLParser`` owns recursive parsing semantics and deterministic repairs. It
does not load a model or make raw HTTP requests. Every LLM operation is routed
through the shared ``ParserClient``, which submits structured requests to the
self-hosted parser vLLM service.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from typing import Literal, TypeVar

from pydantic import BaseModel

from exact.type1.ast.classifier import PRONOUNS, _NOUN_STOP_WORDS, fast_classify
from exact.type1.ast.nodes import (
    AtomicNode,
    FOLNode,
    LogicalNode,
    QuantifiedNode,
    _extract_bound_var,
)
from exact.type1.models.schemas import Predicate
from exact.type1.parser.client import ParserClient
from exact.type1.parser.router import (
    ParserRequest,
    build_coreference_request,
    build_rephrase_request,
    build_sentence_request,
)
from exact.type1.parser.schemas import (
    AtomicResult,
    CoreferenceResult,
    LogicalResult,
    QuantifiedResult,
    RephraseResult,
)

LiteralQuantifier = Literal["ForAll", "ThereExists"]
ResultT = TypeVar("ResultT", bound=BaseModel)


class FOLParser:
    """Build recursive FOL ASTs using one shared asynchronous parser client.

    ``parse_many`` should be used for a problem's independent premises. It
    starts one task per premise, allowing vLLM to continuously batch their
    current recursive parsing steps on the GPU.
    """

    MAX_DEPTH = 8

    def __init__(self, client: ParserClient, *, max_depth: int = MAX_DEPTH) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must not be negative")
        self.client = client
        self.max_depth = max_depth

    async def parse_many(self, sentences: Iterable[str]) -> list[FOLNode]:
        """Parse independent sentences concurrently while preserving order."""

        return list(await asyncio.gather(*(self.parse(sentence) for sentence in sentences)))

    async def parse(
        self,
        sentence: str,
        *,
        depth: int = 0,
        parent: str = "",
        used_variables: frozenset[str] | None = None,
        force_quantifier: LiteralQuantifier | None = None,
    ) -> FOLNode:
        """Recursively parse one sentence into an FOL AST."""

        sentence = sentence.strip()
        if not sentence:
            raise ValueError("sentence must not be empty")

        used_variables = used_variables or frozenset()
        if depth > self.max_depth or self._is_recursive_loop(parent, sentence):
            return await self._parse_atomic(sentence)

        sentence_type = fast_classify(sentence)
        if sentence_type == "quantified":
            return await self._parse_quantified(
                sentence,
                depth=depth,
                used_variables=used_variables,
                force_quantifier=force_quantifier,
            )
        if sentence_type == "logical":
            return await self._parse_logical(
                sentence,
                depth=depth,
                used_variables=used_variables,
            )
        return await self._parse_atomic(sentence)

    async def rephrase(self, sentence: str) -> str:
        """Minimally rewrite one sentence using the dedicated rephrase prompt."""

        result = await self._execute(build_rephrase_request(sentence), RephraseResult)
        return result.rephrased

    async def _parse_atomic(self, sentence: str) -> FOLNode:
        result = await self._execute(build_sentence_request(sentence, kind="atomic"), AtomicResult)
        arguments = [self._normalize_constant(argument) for argument in result.arguments]
        predicate = Predicate(
            name=result.predicate,
            arg_sorts=["Entity"] * len(arguments),
            aliases=[],
        )
        node = AtomicNode(predicate=predicate, arguments=arguments)
        if result.negated:
            return LogicalNode(operator="NOT", left=node, right=None)
        return node

    async def _parse_quantified(
        self,
        sentence: str,
        *,
        depth: int,
        used_variables: frozenset[str],
        force_quantifier: LiteralQuantifier | None,
    ) -> QuantifiedNode:
        taken = ", ".join(sorted(used_variables)) if used_variables else "none"
        routed = build_sentence_request(
            f"[Already used variables: {taken}. Pick a different one.]\n{sentence}",
            kind="quantified",
        )
        result = await self._execute(routed, QuantifiedResult)

        quantifier = force_quantifier or self._override_quantifier(sentence, result.quantifier)
        scope_sentence = result.scope_sentence
        if result.variable not in scope_sentence:
            scope_sentence = re.sub(r"\b[a-z]\d*\b", result.variable, scope_sentence, count=1)

        body = await self.parse(
            scope_sentence,
            depth=depth + 1,
            parent=sentence,
            used_variables=used_variables | {result.variable},
        )
        return QuantifiedNode(
            quantifier="FORALL" if quantifier == "ForAll" else "EXISTS",
            variable=result.variable,
            body=body,
        )

    async def _parse_logical(
        self,
        sentence: str,
        *,
        depth: int,
        used_variables: frozenset[str],
    ) -> LogicalNode | QuantifiedNode:
        result = await self._execute(
            build_sentence_request(sentence, kind="logical"), LogicalResult
        )

        if result.operator == "NOT":
            left = await self.parse(
                result.left_operand,
                depth=depth + 1,
                parent=sentence,
                used_variables=used_variables,
            )
            return LogicalNode(operator="NOT", left=left)

        if result.right_operand is None:
            raise ValueError(f"{result.operator} parser result requires a right operand")

        right_sentence = result.right_operand
        force_universal = (
            result.operator == "IMPLIES"
            and re.search(r"^(a|an)\s+\w+", result.left_operand.lower().strip()) is not None
        )
        if result.operator == "IMPLIES" and PRONOUNS.search(right_sentence):
            right_sentence = await self._resolve_coreference(result.left_operand, right_sentence)

        # Parse left first because it may introduce a variable needed by right.
        left = await self.parse(
            result.left_operand,
            depth=depth + 1,
            parent=sentence,
            used_variables=used_variables,
            force_quantifier="ForAll" if force_universal else None,
        )
        bound_variable = _extract_bound_var(left)
        if bound_variable:
            right_sentence = self._inject_variable(
                right_sentence,
                bound_variable,
                result.left_operand,
            )

        right = await self.parse(
            right_sentence,
            depth=depth + 1,
            parent=sentence,
            used_variables=used_variables,
        )
        # Lift quantifier so it scopes over the whole expression, not just left.
        # ∀x.P(x) IMPLIES Q(x)  →  ∀x.(P(x) IMPLIES Q(x))
        if isinstance(left, QuantifiedNode):
            return QuantifiedNode(
                quantifier=left.quantifier,
                variable=left.variable,
                body=LogicalNode(operator=result.operator, left=left.body, right=right),
            )
        return LogicalNode(operator=result.operator, left=left, right=right)

    async def _resolve_coreference(self, left: str, right: str) -> str:
        result = await self._execute(build_coreference_request(left, right), CoreferenceResult)
        return result.resolved_right

    async def _execute(self, request: ParserRequest, schema: type[ResultT]) -> ResultT:
        result = await self.client.parse_request(request)
        if not isinstance(result, schema):
            raise TypeError(f"Expected {schema.__name__}, received {type(result).__name__}")
        return result

    @staticmethod
    def _override_quantifier(sentence: str, parsed: LiteralQuantifier) -> LiteralQuantifier:
        lowered = sentence.lower().strip()
        if re.search(r"^(a|an)\s+\w+", lowered):
            return "ThereExists"
        if re.search(r"\b(all|every|each|for\s+all|for\s+every)\b", lowered):
            return "ForAll"
        if re.search(r"\b(no\s+one|nobody|no\s+\w+)\b", lowered):
            return "ForAll"
        return parsed

    @staticmethod
    def _is_recursive_loop(parent: str, sentence: str) -> bool:
        if not parent:
            return False
        parent_tokens = set(parent.lower().split())
        sentence_tokens = set(sentence.lower().split())
        if not parent_tokens or not sentence_tokens:
            return False
        overlap = len(parent_tokens & sentence_tokens) / len(parent_tokens | sentence_tokens)
        return overlap > 0.85

    @staticmethod
    def _extract_head_noun(sentence: str) -> str | None:
        match = re.match(r"^(?:a|an|the)\s+(.+)", sentence.strip(), re.IGNORECASE)
        if not match:
            return None

        noun_tokens: list[str] = []
        for token in match.group(1).split():
            cleaned = re.sub(r"[^a-zA-Z0-9]", "", token).lower()
            if cleaned in _NOUN_STOP_WORDS:
                break
            if re.match(r".+(ing|ies|ied)$", cleaned) and noun_tokens:
                break
            noun_tokens.append(token)
            if len(noun_tokens) >= 3:
                break
        return "".join(word.capitalize() for word in noun_tokens) or None

    def _inject_variable(self, right: str, variable: str, left: str) -> str:
        entity = self._extract_head_noun(left)
        if not entity:
            return right

        spaced_entity = re.sub(r"([A-Z])", r" \1", entity).strip().lower()
        for pattern in (
            rf"\bthe\s+{re.escape(spaced_entity)}\b",
            rf"\b{re.escape(entity)}\b",
        ):
            injected = re.sub(pattern, variable, right, flags=re.IGNORECASE)
            if injected != right:
                return injected
        return right

    @staticmethod
    def _normalize_constant(name: str) -> str:
        name = name.strip()
        if re.match(r"^[xyz]\d*$", name):
            return name
        if re.match(r"^[A-Z][a-zA-Z0-9]*$", name):
            return name

        article_prefix = re.match(r"^(the|a|an)([A-Z][a-zA-Z0-9]*)$", name)
        if article_prefix:
            return article_prefix.group(2)

        name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.IGNORECASE).strip()
        parts = re.split(r"[\s_-]+", name)
        return "".join(
            part if part.isupper() and len(part) > 1 else part.capitalize()
            for part in parts
            if part
        )
