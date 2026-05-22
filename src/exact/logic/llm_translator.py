"""LLM-to-IR translator for Type 1 logic.

The LLM is used as a semantic parser, not as the final judge. It converts
natural-language premises/questions into the small IR consumed by deterministic
symbolic solvers, following the Logic-LM/LINC architecture.
"""

from __future__ import annotations

from typing import Any, Protocol

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact.config import Settings, get_settings
from exact.logic.ir import Atom, Fact, ParsedPremise, Query, Rule
from exact.logic.parser import atom_from_text, parse_premise_to_ir, parse_question_to_query
from exact.llm_client import LLMClient


class JsonLLMClient(Protocol):
    def complete_json_sync(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]: ...


class AtomSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    negated: bool = False

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("atom text must not be empty")
        return value


class RuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conditions: list[AtomSpec] = Field(default_factory=list)
    conclusion: AtomSpec

    @field_validator("conditions")
    @classmethod
    def conditions_must_not_be_empty(cls, value: list[AtomSpec]) -> list[AtomSpec]:
        if not value:
            raise ValueError("rule conditions must not be empty")
        return value


class PremiseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_idx: int
    facts: list[AtomSpec] = Field(default_factory=list)
    rules: list[RuleSpec] = Field(default_factory=list)


class QuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: AtomSpec


class TranslationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premises: list[PremiseSpec]
    query: QuerySpec


def translate_with_llm(
    premises: list[str],
    question: str,
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> tuple[tuple[ParsedPremise, ...], Query]:
    """Translate a Type 1 instance into IR using a real LLM client."""

    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    raw = client.complete_json_sync(
        messages=_build_messages(premises, question),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    spec = TranslationSpec.model_validate(raw)
    return _spec_to_ir(spec, premises, question)


def translate_with_fallback(
    premises: list[str],
    question: str,
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> tuple[tuple[ParsedPremise, ...], Query, tuple[str, ...]]:
    """Try LLM translation, then fall back to the local parser with warnings."""

    settings = settings or get_settings()
    if llm_client is not None or settings.llm_provider == "local" or settings.llm_base_url:
        try:
            parsed, query = translate_with_llm(premises, question, llm_client, settings)
            return parsed, query, ()
        except Exception as exc:
            warnings = (f"LLM translation failed; heuristic parser used: {exc}",)
            return _heuristic_translation(premises, question, warnings)

    return _heuristic_translation(
        premises,
        question,
        ("No LLM client configured; heuristic parser used.",),
    )


def _heuristic_translation(
    premises: list[str],
    question: str,
    warnings: tuple[str, ...] = (),
) -> tuple[tuple[ParsedPremise, ...], Query, tuple[str, ...]]:
    parsed = tuple(parse_premise_to_ir(premise, idx) for idx, premise in enumerate(premises))
    query = parse_question_to_query(question)
    return parsed, query, warnings


def _build_messages(premises: list[str], question: str) -> list[ChatCompletionMessageParam]:
    premise_text = "\n".join(f"{idx}: {premise}" for idx, premise in enumerate(premises))
    return [
        {
            "role": "system",
            "content": (
                "You are a semantic parser for an educational logic QA system. "
                "Translate natural-language premises and the question into a compact Horn-style JSON IR. "
                "Do not answer the question. Use only the given text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return exactly this JSON shape:\n"
                "{\n"
                '  "premises": [\n'
                '    {"source_idx": 0, "facts": [{"text": "A", "negated": false}], '
                '"rules": [{"conditions": [{"text": "A"}], "conclusion": {"text": "B"}}]}\n'
                "  ],\n"
                '  "query": {"claim": {"text": "B", "negated": false}}\n'
                "}\n\n"
                "Rules:\n"
                "- source_idx must match the premise number shown below.\n"
                "- Use facts for directly stated atomic statements.\n"
                "- Use rules for if/then, who/that/when conditional rules, requirements, implications.\n"
                "- Split conjunctions into multiple condition atoms.\n"
                "- Preserve entities and predicates in simple English text.\n"
                "- Mark negated=true only for explicit negation.\n\n"
                f"Premises:\n{premise_text}\n\nQuestion:\n{question}"
            ),
        },
    ]


def _spec_to_ir(
    spec: TranslationSpec,
    raw_premises: list[str],
    raw_question: str,
) -> tuple[tuple[ParsedPremise, ...], Query]:
    parsed_by_idx: dict[int, ParsedPremise] = {}

    for premise_spec in spec.premises:
        source_idx = premise_spec.source_idx
        if source_idx < 0 or source_idx >= len(raw_premises):
            continue

        facts = tuple(
            Fact(
                atom=_atom_from_spec(atom_spec),
                source_idx=source_idx,
                text=raw_premises[source_idx],
            )
            for atom_spec in premise_spec.facts
        )
        rules = tuple(
            Rule(
                conditions=tuple(_atom_from_spec(atom_spec) for atom_spec in rule_spec.conditions),
                conclusion=_atom_from_spec(rule_spec.conclusion),
                source_idx=source_idx,
                text=raw_premises[source_idx],
            )
            for rule_spec in premise_spec.rules
        )
        parsed_by_idx[source_idx] = ParsedPremise(facts=facts, rules=rules)

    parsed = []
    for source_idx, premise in enumerate(raw_premises):
        parsed.append(parsed_by_idx.get(source_idx) or ParsedPremise(warnings=(f"No LLM IR for premise {source_idx}",)))

    return tuple(parsed), Query(claim=_atom_from_spec(spec.query.claim), raw_question=raw_question)


def _atom_from_spec(spec: AtomSpec) -> Atom:
    atom = atom_from_text(spec.text)
    return Atom(pred=atom.pred, args=atom.args, negated=spec.negated, text=atom.text)
