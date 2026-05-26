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
from exact.logic.parser import atom_from_text
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
        for raw_arg in value:
            arg = raw_arg.strip()
            if not arg:
                raise ValueError("atom args must not contain empty strings")
            if arg.startswith("?"):
                if not _VARIABLE_RE.fullmatch(arg):
                    # Normalize malformed variable: keep only ?x prefix, e.g. "?x_student" → "?x"
                    var_match = re.match(r"\?[a-z][a-z0-9_]*", arg)
                    arg = var_match.group(0) if var_match else "?x"
            elif not _CONSTANT_RE.fullmatch(arg):
                # LLM sometimes embeds args as pred(?x) or mixed-case strings.
                # Extract the inner variable if present; otherwise slugify to snake_case.
                var_match = re.search(r"\?[a-z][a-z0-9_]*", arg)
                if var_match:
                    arg = var_match.group(0)
                else:
                    arg = re.sub(r"[^a-z0-9]+", "_", arg.lower()).strip("_") or "entity"
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


class PremisesOnlySpec(BaseModel):
    """LLM output for premise-only translation — no query field required."""

    model_config = ConfigDict(extra="ignore")

    predicates: list[PredicateSpec] = Field(default_factory=list)
    premises: list[PremiseSpec]


class QueryOnlySpec(BaseModel):
    """LLM output for query-only translation."""

    model_config = ConfigDict(extra="ignore")

    query: QuerySpec


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


def translate_premises_only_with_llm(
    premises: list[str],
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
    temperature: float | None = None,
) -> tuple[tuple[ParsedPremise, ...], tuple[str, ...], tuple[str, ...]]:
    """Translate premises to IR without a query for premise-level caching.

    Returns parsed premises and any warnings. Query translation is left to the
    caller so the same KB can serve multiple questions in the same premise group.
    """

    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    messages = _build_premises_only_messages(premises)
    logger.info("Starting premise-only LLM translation: premises=%s", len(premises))
    raw = client.complete_json_sync(
        messages=messages,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
    )
    spec = PremisesOnlySpec.model_validate(raw)
    parsed = _premise_specs_to_parsed(spec.premises, premises)
    predicate_names = tuple(predicate.name for predicate in spec.predicates)
    return parsed, (), predicate_names


def translate_query_only_with_llm(
    question: str,
    predicate_names: tuple[str, ...] = (),
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> Query:
    """Translate a Type 1 question into a solver query without retranslating premises."""

    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    messages = _build_query_only_messages(question, predicate_names=predicate_names)
    logger.info("Starting query-only LLM translation: question_chars=%s", len(question))
    raw = client.complete_json_sync(
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    spec = QueryOnlySpec.model_validate(raw)
    if predicate_names and spec.query.claim.pred not in set(predicate_names):
        # Soft mismatch: LLM used a predicate not in the KB dictionary (e.g. "implies").
        # Log and proceed — the solver will return Unknown rather than crashing.
        logger.warning(
            "Query predicate %r not in premise dictionary %r; solving will likely return Unknown",
            spec.query.claim.pred,
            list(predicate_names),
        )
    return Query(claim=_atom_from_spec(spec.query.claim), raw_question=question)


def _build_premises_only_messages(premises: list[str]) -> list[ChatCompletionMessageParam]:
    """Compact premise-only prompt for KnowledgeBase caching."""

    premise_text = "\n".join(f"{idx}: {premise}" for idx, premise in enumerate(premises))
    schema_hint = (
        '{"predicates":[{"name":"pred","arity":1,"gloss":"meaning","argument_roles":["entity"]}],'
        '"premises":[{"source_idx":0,"facts":[{"text":"...","pred":"pred","args":["item"],"negated":false}],'
        '"rules":[{"conditions":[{"text":"...","pred":"cond","args":["?x"],"negated":false}],'
        '"conclusion":{"text":"...","pred":"pred","args":["?x"],"negated":false}}]}]}'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. "
                "Translate premises into Horn-style predicates for a symbolic solver."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate these premises into Horn-style IR. Output valid JSON matching: {schema_hint}\n"
                "Rules:\n"
                "- pred and constants: lowercase snake_case; variables: ?x, ?y\n"
                "- Generic rules use variables (?x); named facts use constants (sofia)\n"
                "- Standalone assertions go in facts; implications go in rules\n"
                "- Every rule must have at least one condition; never output conditions:[]\n"
                "- Split conjunctions into separate condition atoms\n"
                "- Preserve source_idx exactly\n\n"
                f"Premises:\n{premise_text}"
            ),
        },
    ]


def _build_query_only_messages(
    question: str,
    predicate_names: tuple[str, ...] = (),
) -> list[ChatCompletionMessageParam]:
    """Compact query-only prompt for YNU/open-ended questions."""

    schema_hint = (
        '{"query":{"claim":{"text":"...","pred":"predicate_name",'
        '"args":["entity"],"negated":false}}}'
    )
    predicate_instruction = (
        "Allowed predicate names from premise translation: "
        f"{', '.join(predicate_names)}\n"
        "You must choose query.claim.pred from this allowed list.\n"
        if predicate_names
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. "
                "Translate the question target into one Horn-style query atom."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate this question target into valid JSON matching: {schema_hint}\n"
                "Rules:\n"
                "- Do not answer the question\n"
                f"{predicate_instruction}"
                "- pred and constants: lowercase snake_case; variables: ?x, ?y\n"
                "- Mark negated=true only for explicit negation\n\n"
                f"Question:\n{question}"
            ),
        },
    ]


def _premise_specs_to_parsed(
    premise_specs: list[PremiseSpec],
    raw_premises: list[str],
) -> tuple[ParsedPremise, ...]:
    """Convert validated PremiseSpec list to IR, filling gaps with warnings."""

    parsed_by_idx: dict[int, ParsedPremise] = {}
    for premise_spec in premise_specs:
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
                conditions=tuple(_atom_from_spec(c) for c in rule_spec.conditions),
                conclusion=_atom_from_spec(rule_spec.conclusion),
                source_idx=source_idx,
                text=raw_premises[source_idx],
            )
            for rule_spec in premise_spec.rules
        )
        parsed_by_idx[source_idx] = ParsedPremise(facts=facts, rules=rules)

    return tuple(
        parsed_by_idx.get(i) or ParsedPremise(warnings=(f"No LLM IR for premise {i}",))
        for i in range(len(raw_premises))
    )


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
    parsed = _premise_specs_to_parsed(spec.premises, raw_premises)
    return parsed, Query(claim=_atom_from_spec(spec.query.claim), raw_question=raw_question)


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
