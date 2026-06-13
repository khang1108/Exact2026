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
    simplify,
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
        """Parse independent sentences concurrently while preserving order.

        Quantified sentences are first rephrased to IF-THEN form in a single
        batch; those that convert to logical avoid quantifier-nesting loops.
        Remaining quantified sentences fall back to the progress-check guard
        inside ``_parse_quantified``.
        """
        sentence_list = list(sentences)

        # Batch-rephrase all quantified sentences in one vLLM round-trip.
        rephrased = await asyncio.gather(*(
            self._rephrase_if_quantified(s) for s in sentence_list
        ))

        return list(await asyncio.gather(*(self._safe_parse(s) for s in rephrased)))

    async def _safe_parse(self, sentence: str) -> FOLNode:
        """Parse one sentence; on any failure return an opaque atomic placeholder.

        A premise/conclusion that cannot be parsed (e.g. the recursive splitter
        produced an empty operand) must not crash the whole request. The
        placeholder is an unconstrained 0-arity predicate, so Z3 simply treats it
        as unknown rather than entailing anything from it.
        """
        try:
            return await self.parse(sentence)
        except Exception:
            return self._fallback_node(sentence)

    @staticmethod
    def _fallback_node(sentence: str) -> FOLNode:
        words = re.findall(r"[A-Za-z0-9]+", sentence)
        name = "".join(w.capitalize() for w in words)[:60] or "Unparsed"
        return AtomicNode(predicate=Predicate(name=name, arg_sorts=[], aliases=[]), arguments=[])

    async def _rephrase_if_quantified(self, sentence: str) -> str:
        """Rephrase a sentence to IF-THEN if it is not already logical."""
        if fast_classify(sentence) == "logical":
            return sentence
        try:
            rephrased = await self.rephrase(sentence)
        except Exception:
            return sentence
        return rephrased if fast_classify(rephrased) == "logical" else sentence

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
        sentence_type = fast_classify(sentence)
        if depth > self.max_depth:
            return simplify(await self._parse_atomic(sentence))

        # A quantified sentence often becomes a near-identical logical sentence
        # after replacing its subject with a variable. That is structural
        # progress, even when token overlap is high.
        if (
            self._is_recursive_loop(parent, sentence)
            and fast_classify(parent) == sentence_type
        ):
            return simplify(await self._parse_atomic(sentence))

        if sentence_type == "quantified":
            return simplify(await self._parse_quantified(
                sentence,
                depth=depth,
                used_variables=used_variables,
                force_quantifier=force_quantifier,
            ))
        if sentence_type == "logical":
            return simplify(await self._parse_logical(
                sentence,
                depth=depth,
                used_variables=used_variables,
            ))
        return simplify(await self._parse_atomic(sentence))

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

        # Break quantifier-nesting loops only when the scope is still classified
        # as quantified. A same-length scope that became AND/IMPLIES is valid
        # structural progress and must be recursively parsed.
        scope_type = fast_classify(scope_sentence)
        no_progress = len(scope_sentence.split()) >= len(sentence.split()) * 0.85
        stuck_quantifier = scope_type == "quantified" and (
            self._is_recursive_loop(sentence, scope_sentence) or no_progress
        )
        if stuck_quantifier:
            body = await self._parse_atomic(scope_sentence)
        else:
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

        # Parse left first because it introduces the bound variable needed by right.
        left = await self.parse(
            result.left_operand,
            depth=depth + 1,
            parent=sentence,
            used_variables=used_variables,
            force_quantifier="ForAll" if force_universal else None,
        )
        bound_variable = _extract_bound_var(left)

        # Resolve pronouns in the right clause.
        # When a bound variable is already known, substitute directly — a 1.7 B model
        # tends to hallucinate the left's predicate when asked to resolve "it is not X"
        # against a left clause like "is not Y", producing "it is not Y" instead.
        if PRONOUNS.search(right_sentence):
            if bound_variable:
                right_sentence = self._substitute_pronouns(right_sentence, bound_variable)
            elif result.operator == "IMPLIES":
                right_sentence = await self._resolve_coreference(result.left_operand, right_sentence)

        if bound_variable:
            right_sentence = self._inject_variable(
                right_sentence,
                bound_variable,
                result.left_operand,
            )

        # Pass the bound variable as used so the right side doesn't re-quantify with
        # the same variable (e.g. ∀x.(P(x) IMPLIES ∀x.Q(x)) → use y instead).
        right = await self.parse(
            right_sentence,
            depth=depth + 1,
            parent=sentence,
            used_variables=used_variables | ({bound_variable} if bound_variable else frozenset()),
        )
        # Lift an antecedent quantifier only when its variable is genuinely used
        # by the consequent. Keep proposition-level implications such as
        # (∀x.P(x)) IMPLIES (∀y.Q(y)) as an implication between two formulas.
        if isinstance(left, QuantifiedNode) and _has_free_variable(right, left.variable):
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
    def _substitute_pronouns(text: str, variable: str) -> str:
        """Replace subject/object pronouns with a bound variable."""
        return re.sub(
            r"\b(it|its|they|them|their|he|his|him|she|her)\b",
            variable,
            text,
            flags=re.IGNORECASE,
        )

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

        # Dataset rules often switch between a class noun and a generic subject
        # in the consequent ("a Python code ... then the project ..."). Treat a
        # leading generic subject as the same bound entity, while leaving proper
        # names and object phrases untouched.
        injected = re.sub(
            r"^(?:the\s+)?(?:project|student|person|code|faculty\s+member|driver)\b",
            variable,
            right,
            count=1,
            flags=re.IGNORECASE,
        )
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


def _has_free_variable(
    node: FOLNode,
    variable: str,
    bound: frozenset[str] = frozenset(),
) -> bool:
    """Return whether ``variable`` occurs free in an AST."""

    if isinstance(node, AtomicNode):
        return variable not in bound and variable in node.arguments
    if isinstance(node, QuantifiedNode):
        return _has_free_variable(node.body, variable, bound | {node.variable})
    if _has_free_variable(node.left, variable, bound):
        return True
    return node.right is not None and _has_free_variable(node.right, variable, bound)
