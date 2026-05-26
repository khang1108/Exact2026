"""LLM autoformalizer for the Type 1 logic pipeline.

The LLM is used as a semantic parser, not as the final judge. It converts
natural-language premises/questions into a compact Horn-style IR consumed by
deterministic symbolic solvers. The design follows the Logic-LM translator →
solver split, LINC's emphasis on predicate-consistent formalization, and the
SymbCoT/Logic-LM++ motivation for later verifier/repair stages.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact.config import Settings, get_settings
from exact.logic.ir import Atom, Fact, ParsedPremise, Query, Rule
from exact.logic.parser import atom_from_text, parse_premise_to_ir, parse_question_to_query
from exact.llm_client import LLMClient
from exact.logger import get_logger

logger = get_logger(__name__)

_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VARIABLE_RE = re.compile(r"^\?[a-z][a-z0-9_]*$")
_CONSTANT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_OPTION_LABELS = {"A", "B", "C", "D"}


class JsonLLMClient(Protocol):
    def complete_json_sync(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]: ...


class PredicateSpec(BaseModel):
    """Canonical predicate dictionary entry for one translated instance."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arity: int = Field(ge=0)
    gloss: str
    argument_roles: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def name_must_be_snake_case(cls, value: str) -> str:
        value = value.strip()
        if not _PREDICATE_RE.fullmatch(value):
            raise ValueError("predicate name must be snake_case")
        return value

    @field_validator("gloss")
    @classmethod
    def gloss_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("predicate gloss must not be empty")
        return value

    @field_validator("argument_roles")
    @classmethod
    def roles_must_not_be_blank(cls, value: list[str]) -> list[str]:
        return [role.strip() for role in value if role.strip()]


class AtomSpec(BaseModel):
    """LLM-produced atom using text plus optional canonical pred/args."""

    model_config = ConfigDict(extra="forbid")

    text: str
    negated: bool = False
    pred: str | None = None
    args: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("atom text must not be empty")
        return value

    @field_validator("pred")
    @classmethod
    def pred_must_be_snake_case(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not _PREDICATE_RE.fullmatch(value):
            raise ValueError("atom pred must be snake_case")
        return value

    @field_validator("args")
    @classmethod
    def args_must_be_variables_or_constants(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for arg in value:
            arg = arg.strip()
            if not arg:
                raise ValueError("atom args must not contain empty strings")
            if arg.startswith("?") and not _VARIABLE_RE.fullmatch(arg):
                raise ValueError("variables must use ?x-style snake_case names")
            if not arg.startswith("?") and not _CONSTANT_RE.fullmatch(arg):
                raise ValueError("constants must be lowercase snake_case")
            normalized.append(arg)
        return normalized


class RuleSpec(BaseModel):
    """Horn rule with conjunctive conditions and one conclusion atom."""

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
    """Formalization for one source premise index."""

    model_config = ConfigDict(extra="forbid")

    source_idx: int
    facts: list[AtomSpec] = Field(default_factory=list)
    rules: list[RuleSpec] = Field(default_factory=list)


class QuerySpec(BaseModel):
    """Formalized yes/no/unknown target claim."""

    model_config = ConfigDict(extra="forbid")

    claim: AtomSpec


class OptionSpec(BaseModel):
    """Formalized multiple-choice option goal."""

    model_config = ConfigDict(extra="forbid")

    label: str
    text: str
    goal: AtomSpec

    @field_validator("label")
    @classmethod
    def label_must_be_supported_option(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in _OPTION_LABELS:
            raise ValueError("option label must be one of A, B, C, D")
        return value

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("option text must not be empty")
        return value


class TranslationSpec(BaseModel):
    """Complete LLM autoformalization for premises, query, and options."""

    model_config = ConfigDict(extra="forbid")

    predicates: list[PredicateSpec] = Field(default_factory=list)
    premises: list[PremiseSpec]
    query: QuerySpec
    options: list[OptionSpec] = Field(default_factory=list)


def translate_with_llm(
    premises: list[str],
    question: str,
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> tuple[tuple[ParsedPremise, ...], Query]:
    """Translate Type 1 text into IR while leaving reasoning to solvers."""

    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    messages = _build_messages(premises, question)
    logger.info(
        "Starting LLM translation: premises=%s, question_chars=%s, max_tokens=%s",
        len(premises),
        len(question),
        settings.llm_max_tokens,
    )
    raw = client.complete_json_sync(
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    logger.info("Validating LLM translation schema")
    spec = TranslationSpec.model_validate(raw)
    logger.info("Converting LLM translation to IR")
    return _spec_to_ir(spec, premises, question)


def translate_with_fallback(
    premises: list[str],
    question: str,
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
    allow_heuristic_fallback: bool = True,
) -> tuple[tuple[ParsedPremise, ...], Query, tuple[str, ...]]:
    """Try LLM translation, then fall back to the local parser with warnings."""

    settings = settings or get_settings()
    if llm_client is not None or settings.llm_provider == "local" or settings.llm_base_url:
        try:
            parsed, query = translate_with_llm(premises, question, llm_client, settings)
            return parsed, query, ()
        except Exception as exc:
            if not allow_heuristic_fallback:
                raise RuntimeError(f"LLM translation failed and fallback is disabled: {exc}") from exc
            warnings = (f"LLM translation failed; heuristic parser used: {exc}",)
            return _heuristic_translation(premises, question, warnings)

    if not allow_heuristic_fallback:
        raise RuntimeError("No LLM client configured and fallback is disabled")

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
    """Build a compact autoformalization prompt for a local <=8B LLM."""

    premise_text = "\n".join(f"{idx}: {premise}" for idx, premise in enumerate(premises))
    schema_hint = (
        '{"predicates":[{"name":"predicate_name","arity":1,"gloss":"meaning",'
        '"argument_roles":["entity"]}],'
        '"premises":[{"source_idx":0,"facts":[{"text":"...","pred":"predicate_name",'
        '"args":["item"],"negated":false}],"rules":[{"conditions":[{"text":"...",'
        '"pred":"condition_name","args":["?x"],"negated":false}],'
        '"conclusion":{"text":"...","pred":"predicate_name","args":["?x"],'
        '"negated":false}}]}],'
        '"query":{"claim":{"text":"...","pred":"predicate_name","args":["sophia"],'
        '"negated":false}},'
        '"options":[]}'
    )
    examples = (
        "Example 1:\n"
        "Premise 0: Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.\n"
        "Premise 1: Sophia has completed the core curriculum.\n"
        "Question: Does Sophia qualify for advanced courses?\n"
        "JSON: {\"predicates\":["
        "{\"name\":\"completed_core_curriculum\",\"arity\":1,\"gloss\":\"entity completed core curriculum\",\"argument_roles\":[\"entity\"]},"
        "{\"name\":\"passed_science_assessment\",\"arity\":1,\"gloss\":\"entity passed science assessment\",\"argument_roles\":[\"entity\"]},"
        "{\"name\":\"qualified_for_advanced_courses\",\"arity\":1,\"gloss\":\"entity qualifies for advanced courses\",\"argument_roles\":[\"entity\"]}],"
        "\"premises\":[{\"source_idx\":0,\"facts\":[],\"rules\":[{\"conditions\":["
        "{\"text\":\"completed the core curriculum\",\"pred\":\"completed_core_curriculum\",\"args\":[\"?x\"],\"negated\":false},"
        "{\"text\":\"passed the science assessment\",\"pred\":\"passed_science_assessment\",\"args\":[\"?x\"],\"negated\":false}],"
        "\"conclusion\":{\"text\":\"qualified for advanced courses\",\"pred\":\"qualified_for_advanced_courses\",\"args\":[\"?x\"],\"negated\":false}}]},"
        "{\"source_idx\":1,\"facts\":[{\"text\":\"Sophia has completed the core curriculum\",\"pred\":\"completed_core_curriculum\",\"args\":[\"sophia\"],\"negated\":false}],\"rules\":[]}],"
        "\"query\":{\"claim\":{\"text\":\"Sophia qualifies for advanced courses\",\"pred\":\"qualified_for_advanced_courses\",\"args\":[\"sophia\"],\"negated\":false}},\"options\":[]}\n\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. Keep it compact and valid. Do not use markdown fences. Do not answer the question. "
                "Translate text into Horn-style predicates for a symbolic solver."
            ),
        },
        {
            "role": "user",
            "content": (
                "Task: build one predicate dictionary, then formalize premises, query, and MCQ options.\n"
                "Rules:\n"
                "- Output valid JSON only, matching this shape: "
                f"{schema_hint}\n"
                "- Reuse predicate names from predicates everywhere; never invent variants.\n"
                "- pred and constants must be lowercase snake_case; variables use ?x, ?y.\n"
                "- Generic rules use variables; named facts/goals use constants.\n"
                "- Standalone assertions go in facts, not rules.\n"
                "- Every rule must have at least one condition; never output conditions:[].\n"
                "- Split conjunctions into separate condition atoms.\n"
                "- Preserve source_idx exactly.\n"
                "- Do not translate A-D options; always set options to []. The pipeline evaluates MCQ options separately.\n"
                "- Mark negated=true only for explicit negation.\n\n"
                f"{examples}\n"
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
    """Convert validated LLM atom JSON to the solver IR."""

    if spec.pred:
        return Atom(
            pred=spec.pred.strip(),
            args=tuple(arg.strip() for arg in spec.args if arg.strip()),
            negated=spec.negated,
            text=spec.text,
        )

    atom = atom_from_text(spec.text)
    return Atom(pred=atom.pred, args=atom.args, negated=spec.negated, text=atom.text)
