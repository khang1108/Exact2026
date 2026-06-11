"""Route Type 1 parsing work to the correct prompt and response schema.

The router owns parser semantics: it selects a specialized system prompt,
formats the user message, and attaches the expected Pydantic response model.
The transport-only ``ParserClient`` executes the resulting requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel

from exact.type1.ast.classifier import fast_classify
from exact.type1.parser.client import ChatMessage
from exact.type1.parser.schemas import (
    AtomicResult,
    CoreferenceResult,
    LogicalResult,
    QuantifiedResult,
    RephraseResult,
)
from exact.type1.prompts import (
    get_system_prompt_atomic,
    get_system_prompt_coreference,
    get_system_prompt_logical,
    get_system_prompt_quantified,
    get_system_prompt_rephrase,
)

ParserKind = Literal["atomic", "logical", "quantified", "rephrase", "coreference"]
SentenceParserKind = Literal["atomic", "logical", "quantified"]

_SYSTEM_PROMPTS = {
    "atomic": get_system_prompt_atomic,
    "logical": get_system_prompt_logical,
    "quantified": get_system_prompt_quantified,
    "rephrase": get_system_prompt_rephrase,
    "coreference": get_system_prompt_coreference,
}

_RESULT_SCHEMAS: dict[ParserKind, type[BaseModel]] = {
    "atomic": AtomicResult,
    "logical": LogicalResult,
    "quantified": QuantifiedResult,
    "rephrase": RephraseResult,
    "coreference": CoreferenceResult,
}


@dataclass(frozen=True)
class ParserRequest:
    """One fully routed parser call ready for asynchronous execution."""

    kind: ParserKind
    messages: list[ChatMessage]
    schema: type[BaseModel]


def build_sentence_request(
    sentence: str,
    *,
    kind: SentenceParserKind | None = None,
) -> ParserRequest:
    """Build an atomic, logical, or quantified request for one sentence.

    When ``kind`` is omitted, the deterministic classifier chooses the parser.
    Explicit kinds are useful during recursive parsing when the caller already
    knows which grammar production must be extracted next.
    """

    sentence = sentence.strip()
    if not sentence:
        raise ValueError("sentence must not be empty")

    selected_kind = kind or cast(SentenceParserKind, fast_classify(sentence))
    return _build_request(selected_kind, f'Input: "{sentence}"')


def build_rephrase_request(sentence: str) -> ParserRequest:
    """Build a request that minimally rewrites a sentence before AST parsing."""

    sentence = sentence.strip()
    if not sentence:
        raise ValueError("sentence must not be empty")
    return _build_request("rephrase", f'Input: "{sentence}"')


def build_coreference_request(left_clause: str, right_clause: str) -> ParserRequest:
    """Build a request that resolves pronouns in one right-hand clause."""

    left_clause = left_clause.strip()
    right_clause = right_clause.strip()
    if not left_clause or not right_clause:
        raise ValueError("left_clause and right_clause must not be empty")

    user_message = f'Input left: "{left_clause}"\nInput right: "{right_clause}"'
    return _build_request("coreference", user_message)


def _build_request(kind: ParserKind, user_message: str) -> ParserRequest:
    """Combine one parser kind's prompt, user content, and response schema."""

    messages: list[ChatMessage] = [
        {"role": "system", "content": _SYSTEM_PROMPTS[kind]().strip()},
        {"role": "user", "content": user_message},
    ]
    return ParserRequest(kind=kind, messages=messages, schema=_RESULT_SCHEMAS[kind])
