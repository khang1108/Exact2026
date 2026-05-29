"""LLM autoformalizer for the Type 1 logic pipeline.

The LLM is used as a semantic parser, not as the final judge. It converts
natural-language premises/questions into a compact Horn-style IR consumed by
deterministic symbolic solvers. The design follows the Logic-LM translator →
solver split, LINC's emphasis on predicate-consistent formalization, and the
SymbCoT/Logic-LM++ motivation for later verifier/repair stages.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from openai.types.chat import ChatCompletionMessageParam
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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


def _normalize_pred_name(name: str) -> str:
    """Coerce any predicate/atom name string to lowercase snake_case."""
    # CamelCase → snake_case
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # Replace non-alphanumeric runs with underscore
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return name or "pred"


class JsonLLMClient(Protocol):
    def complete_json_sync(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]: ...


class PredicateSpec(BaseModel):
    """Canonical predicate dictionary entry for one translated instance."""

    model_config = ConfigDict(extra="ignore")

    name: str
    arity: int = Field(ge=0)
    gloss: str = ""
    argument_roles: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def name_must_be_snake_case(cls, value: str) -> str:
        value = _normalize_pred_name(value.strip())
        if not _PREDICATE_RE.fullmatch(value):
            raise ValueError(f"predicate name could not be normalized to snake_case: {value!r}")
        return value

    @field_validator("argument_roles")
    @classmethod
    def roles_must_not_be_blank(cls, value: list[str]) -> list[str]:
        return [role.strip() for role in value if role.strip()]


class AtomSpec(BaseModel):
    """LLM-produced atom with compact canonical pred/args fields."""

    model_config = ConfigDict(extra="ignore")

    text: str = ""
    negated: bool = False
    pred: str | None = None
    args: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def text_can_be_omitted(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def pred_or_text_must_exist(self) -> "AtomSpec":
        if not self.pred and not self.text:
            raise ValueError("atom must include pred or text")
        return self

    @field_validator("pred")
    @classmethod
    def pred_must_be_snake_case(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _normalize_pred_name(value.strip())
        if not _PREDICATE_RE.fullmatch(value):
            raise ValueError(f"atom pred could not be normalized to snake_case: {value!r}")
        return value

    @field_validator("args", mode="before")
    @classmethod
    def args_must_be_variables_or_constants(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        normalized: list[str] = []
        for raw_arg in value:
            arg = _coerce_atom_arg(raw_arg)
            if not arg:
                raise ValueError("atom args must not contain empty strings")
            if arg.startswith("?"):
                if len(arg) > 8:
                    # Long ?name strings (e.g. ?best_practices_project) are LLM-invented
                    # entity names with a spurious ? prefix — treat as ground constants.
                    arg = re.sub(r"[^a-z0-9]+", "_", arg[1:].lower()).strip("_") or "entity"
                elif not _VARIABLE_RE.fullmatch(arg):
                    # Short malformed variable: normalise to nearest valid form.
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

    model_config = ConfigDict(extra="ignore")

    conditions: list[AtomSpec] = Field(default_factory=list)
    conclusion: AtomSpec | None = None  # None means rule is malformed and will be dropped


class PremiseSpec(BaseModel):
    """Formalization for one source premise index."""

    model_config = ConfigDict(extra="ignore")

    source_idx: int
    facts: list[AtomSpec] = Field(default_factory=list)
    rules: list[RuleSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def drop_malformed_rules(self) -> "PremiseSpec":
        # Drop rules with empty conditions or missing conclusion (LLM sampling noise).
        # Also drop identity rules A → A which derive nothing new.
        def _is_identity(r: RuleSpec) -> bool:
            return (
                r.conclusion is not None
                and len(r.conditions) == 1
                and r.conditions[0].pred == r.conclusion.pred
                and r.conditions[0].args == r.conclusion.args
                and r.conditions[0].negated == r.conclusion.negated
            )

        self.rules = [
            r for r in self.rules
            if r.conditions and r.conclusion is not None and not _is_identity(r)
        ]
        return self


class QuerySpec(BaseModel):
    """Formalized yes/no/unknown target claim."""

    model_config = ConfigDict(extra="ignore")

    claim: AtomSpec


class OptionSpec(BaseModel):
    """Formalized multiple-choice option goal."""

    model_config = ConfigDict(extra="ignore")

    label: str
    text: str
    claim: AtomSpec = Field(validation_alias=AliasChoices("claim", "goal"))

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

    model_config = ConfigDict(extra="ignore")

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


class MCQOptionsSpec(BaseModel):
    """LLM output mapping each MCQ label to a KB-compatible atom."""

    model_config = ConfigDict(extra="ignore")

    options: list[OptionSpec]


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
        max_tokens=_translation_token_budget(settings.llm_max_tokens, len(premises)),
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
    base_budget = _premise_token_budget(settings.llm_max_tokens, len(premises))
    temp = settings.llm_temperature if temperature is None else temperature

    last_exc: Exception | None = None
    spec: PremisesOnlySpec | None = None
    for attempt, budget in enumerate([base_budget, min(8192, base_budget * 2)], start=1):
        try:
            raw = client.complete_json_sync(messages=messages, temperature=temp, max_tokens=budget)
        except ValueError as exc:
            msg = str(exc)
            if "invalid JSON" not in msg and "incomplete JSON" not in msg:
                raise
            last_exc = exc
            logger.warning(
                "Premise translation attempt %s/2 JSON error (budget=%s): %s", attempt, budget, msg[:200]
            )
            continue

        repaired = _repair_raw_output(raw)
        try:
            spec = PremisesOnlySpec.model_validate(repaired)
            break
        except ValidationError as exc:
            last_exc = exc
            logger.warning(
                "Premise translation attempt %s/2 schema error: %s", attempt, str(exc)[:300]
            )
            logger.debug("Raw LLM output (attempt %s): %s", attempt, json.dumps(raw)[:2000])
    else:
        raise last_exc  # type: ignore[misc]

    parsed = _premise_specs_to_parsed(spec.premises, premises)  # type: ignore[union-attr]
    # Start with the LLM's explicit predicate dictionary, then add every predicate that
    # actually appears in the IR (facts, rule conditions, rule conclusions).  The LLM
    # frequently omits conclusion predicates (e.g. "qualifies_for_scholarship") from its
    # explicit list, causing query translation to fail with "not in premise dictionary".
    explicit_names: set[str] = {p.name for p in spec.predicates}  # type: ignore[union-attr]
    ir_names: set[str] = set()
    for p in parsed:
        for fact in p.facts:
            ir_names.add(fact.atom.pred)
        for rule in p.rules:
            ir_names.add(rule.conclusion.pred)
            for cond in rule.conditions:
                ir_names.add(cond.pred)
    predicate_names = tuple(sorted(explicit_names | ir_names))
    return parsed, (), predicate_names


def translate_query_only_with_llm(
    question: str,
    predicate_names: tuple[str, ...] = (),
    entity_constants: tuple[str, ...] = (),
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> Query:
    """Translate a Type 1 question into a solver query without retranslating premises."""

    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    budget = _query_token_budget(settings.llm_max_tokens)
    logger.info("Starting query-only LLM translation: question_chars=%s", len(question))

    messages: list[ChatCompletionMessageParam] = _build_query_only_messages(
        question, predicate_names=predicate_names, entity_constants=entity_constants
    )
    predicate_set = set(predicate_names)
    spec: QueryOnlySpec | None = None

    for attempt in range(2):
        raw = client.complete_json_sync(
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=budget,
        )
        raw = _repair_raw_output(raw)
        spec = QueryOnlySpec.model_validate(raw)
        pred = spec.query.claim.pred

        if not predicate_names or pred in predicate_set:
            break

        if attempt == 0:
            logger.warning(
                "Query predicate %r not in premise dictionary; retrying with correction", pred
            )
            messages = [
                *messages,
                {"role": "assistant", "content": json.dumps(raw)},
                {
                    "role": "user",
                    "content": (
                        f"WRONG: pred='{pred}' is not in the allowed predicate list.\n"
                        f"You MUST use EXACTLY one of: {', '.join(predicate_names)}\n"
                        "Return corrected JSON with pred set to a name from the allowed list."
                    ),
                },
            ]
    else:
        pred = spec.query.claim.pred  # type: ignore[union-attr]
        closest = _closest_predicate(pred, predicate_names)
        if closest:
            logger.warning(
                "Query predicate %r not in premise dictionary; substituting closest match %r",
                pred,
                closest,
            )
            claim = spec.query.claim  # type: ignore[union-attr]
            spec = QueryOnlySpec(
                query=QuerySpec(
                    claim=AtomSpec(pred=closest, args=claim.args, negated=claim.negated)
                )
            )
        else:
            logger.warning(
                "Query predicate %r not in premise dictionary %r; solving will likely return Unknown",
                pred,
                list(predicate_names),
            )

    return Query(claim=_atom_from_spec(spec.query.claim), raw_question=question)  # type: ignore[union-attr]


def translate_mcq_options_with_llm(
    question: str,
    options: list[tuple[str, str]],
    predicate_names: tuple[str, ...],
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> dict[str, "Atom"]:
    """Translate MCQ option texts into KB-compatible atoms using the LLM.

    Returns a dict mapping label → Atom for successfully translated options.
    Missing labels mean translation failed for that option; caller should fall back.
    """

    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    messages = _build_mcq_options_messages(question, predicate_names)
    logger.info("Starting MCQ option translation: %s options", len(options))
    raw = client.complete_json_sync(
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=_mcq_options_token_budget(settings.llm_max_tokens),
    )
    spec = MCQOptionsSpec.model_validate(raw)
    valid_labels = {label for label, _ in options}
    result: dict[str, Atom] = {}
    for opt in spec.options:
        if opt.label not in valid_labels:
            continue
        try:
            result[opt.label] = _atom_from_spec(opt.claim)
        except Exception as exc:
            logger.debug("MCQ option %s atom conversion failed: %s", opt.label, exc)
    return result


def _build_premises_only_messages(premises: list[str]) -> list[ChatCompletionMessageParam]:
    """Compact premise-only prompt for KnowledgeBase caching."""

    premise_text = "\n".join(f"{idx}: {premise}" for idx, premise in enumerate(premises))
    schema_hint = (
        '{"predicates":[{"name":"pred","arity":1}],'
        '"premises":[{"source_idx":0,"facts":[{"pred":"pred","args":["item"],"negated":false}],'
        '"rules":[{"conditions":[{"pred":"cond","args":["?x"],"negated":false}],'
        '"conclusion":{"pred":"pred","args":["?x"],"negated":false}}]}]}'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. No extra text. "
                "Translate premises into compact Horn-style predicates for a symbolic solver. "
                "CRITICAL: never include a 'text' field in any atom object. "
                "CRITICAL: args must be an array of strings only, never nested objects."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate these premises into Horn-style IR. Output valid JSON matching: {schema_hint}\n"
                "Rules:\n"
                "- NEVER include 'text' fields in facts, conditions, or conclusions — omit them entirely\n"
                "- NEVER include 'gloss' or 'argument_roles' in predicates — only name and arity\n"
                "- pred and constants: lowercase snake_case; variables: ?x, ?y\n"
                "- args must be an array of strings only, e.g. [\"?x\"], never nested objects\n"
                "- Do not use pred values `and`, `or`, `either`, or `not`; split conjunctions and use negated=true for negation\n"
                "- Generic rules use variables (?x); named facts use constants (sofia)\n"
                "- Standalone assertions go in facts; implications go in rules\n"
                "- Every rule must have at least one condition; never output conditions:[]\n"
                "- CRITICAL: if a premise says 'If A and B then C', the rule must have TWO conditions: "
                "[{pred:A,args:[?x]},{pred:B,args:[?x]}] with conclusion {pred:C,args:[?x]}. "
                "NEVER write an identity rule where condition and conclusion share the same pred.\n"
                "- Split conjunctions into separate condition atoms\n"
                "- If a premise has alternatives with `or`, keep only Horn-compatible direct conditions and do not model disjunction\n"
                "- Preserve source_idx exactly\n\n"
                "Example multi-condition rule — Premise: 'If a student completes courses and passes exams, they graduate':\n"
                '{"source_idx":0,"facts":[],"rules":[{"conditions":[{"pred":"completes_courses","args":["?x"],"negated":false},{"pred":"passes_exams","args":["?x"],"negated":false}],"conclusion":{"pred":"graduates","args":["?x"],"negated":false}}]}\n\n'
                f"Premises:\n{premise_text}"
            ),
        },
    ]


def _build_query_only_messages(
    question: str,
    predicate_names: tuple[str, ...] = (),
    entity_constants: tuple[str, ...] = (),
) -> list[ChatCompletionMessageParam]:
    """Compact query-only prompt for YNU/open-ended questions."""

    schema_hint = (
        '{"query":{"claim":{"pred":"predicate_name","args":["entity"],"negated":false}}}'
    )
    predicate_instruction = (
        "CRITICAL — query.claim.pred MUST be EXACTLY one of these allowed names "
        "(no variants, no synonyms, no new inventions):\n"
        f"  {', '.join(predicate_names)}\n"
        "If the question's target concept is not literally in this list, pick the "
        "closest matching predicate from the list above.\n"
        if predicate_names
        else ""
    )
    entity_instruction = (
        "CRITICAL — entity names in args MUST be EXACTLY one of these known constants "
        "(copy verbatim, do NOT translate, rephrase, or alter spelling):\n"
        f"  {', '.join(entity_constants)}\n"
        if entity_constants
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. "
                "Translate the question target into one Horn-style query atom. "
                "Never put JSON objects inside args; args must be strings only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate this question target into valid JSON matching: {schema_hint}\n"
                "Rules:\n"
                "- Do not answer the question\n"
                f"{predicate_instruction}"
                f"{entity_instruction}"
                "- pred and constants: lowercase snake_case; variables: ?x, ?y\n"
                "- Prefer atom shape {\"pred\":\"name\",\"args\":[\"entity\"],\"negated\":false}; omit text fields\n"
                "- args must be an array of strings only, e.g. [\"student\"], never nested objects\n"
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
                conclusion=_atom_from_spec(rule_spec.conclusion),  # type: ignore[arg-type]
                source_idx=source_idx,
                text=raw_premises[source_idx],
            )
            for rule_spec in premise_spec.rules
            if rule_spec.conditions and rule_spec.conclusion is not None
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
        '{"predicates":[{"name":"predicate_name","arity":1}],'
        '"premises":[{"source_idx":0,"facts":[{"pred":"predicate_name","args":["item"],"negated":false}],'
        '"rules":[{"conditions":[{"pred":"condition_name","args":["?x"],"negated":false}],'
        '"conclusion":{"pred":"predicate_name","args":["?x"],"negated":false}}]}],'
        '"query":{"claim":{"pred":"predicate_name","args":["sophia"],"negated":false}},'
        '"options":[]}'
    )
    examples = (
        "Example 1:\n"
        "Premise 0: If a student completes assignments, the student passes.\n"
        "Premise 1: Sophia completes assignments.\n"
        "Question: Does Sophia pass?\n"
        "JSON: {\"predicates\":[{\"name\":\"completes_assignments\",\"arity\":1},{\"name\":\"passes\",\"arity\":1}],"
        "\"premises\":[{\"source_idx\":0,\"facts\":[],\"rules\":[{\"conditions\":[{\"pred\":\"completes_assignments\",\"args\":[\"?x\"],\"negated\":false}],"
        "\"conclusion\":{\"pred\":\"passes\",\"args\":[\"?x\"],\"negated\":false}}]},"
        "{\"source_idx\":1,\"facts\":[{\"pred\":\"completes_assignments\",\"args\":[\"sophia\"],\"negated\":false}],\"rules\":[]}],"
        "\"query\":{\"claim\":{\"pred\":\"passes\",\"args\":[\"sophia\"],\"negated\":false}},\"options\":[]}\n\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. Keep it compact and valid. Do not use markdown fences. Do not answer the question. "
                "Translate text into Horn-style predicates for a symbolic solver. "
                "Never put JSON objects inside args; args must be strings only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Task: build one predicate dictionary, then formalize premises, query, and MCQ options.\n"
                "Rules:\n"
                "- Output valid JSON only, matching this shape: "
                f"{schema_hint}\n"
                "- Keep output compact; omit atom text fields.\n"
                "- predicates may contain only name and arity; omit gloss and argument_roles unless needed.\n"
                "- Reuse predicate names from predicates everywhere; never invent variants.\n"
                "- pred and constants must be lowercase snake_case; variables use ?x, ?y.\n"
                "- args must be arrays of strings only, never nested JSON objects.\n"
                "- Do not use pred values `and`, `or`, `either`, or `not`; split conjunctions and use negated=true for negation.\n"
                "- Generic rules use variables; named facts/goals use constants.\n"
                "- Standalone assertions go in facts, not rules.\n"
                "- Every rule must have at least one condition; never output conditions:[].\n"
                "- Split conjunctions into separate condition atoms.\n"
                "- If a premise has alternatives with `or`, keep only Horn-compatible direct conditions and do not model disjunction.\n"
                "- Preserve source_idx exactly.\n"
                "- Do not translate A-D options; always set options to []. The pipeline evaluates MCQ options separately.\n"
                "- Mark negated=true only for explicit negation.\n\n"
                f"{examples}\n"
                f"Premises:\n{premise_text}\n\nQuestion:\n{question}"
            ),
        },
    ]


def _coerce_atom_arg(raw_arg: Any) -> str:
    if isinstance(raw_arg, str):
        return raw_arg.strip()
    if isinstance(raw_arg, dict):
        nested_args = raw_arg.get("args")
        if isinstance(nested_args, list) and nested_args:
            return _coerce_atom_arg(nested_args[0])
        pred = raw_arg.get("pred")
        if isinstance(pred, str) and pred.strip():
            return pred.strip()
        text = raw_arg.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return str(raw_arg).strip()


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
        text = spec.text or _format_atom_text(spec.pred, spec.args)
        return Atom(
            pred=spec.pred.strip(),
            args=tuple(arg.strip() for arg in spec.args if arg.strip()),
            negated=spec.negated,
            text=text,
        )

    atom = atom_from_text(spec.text)
    return Atom(pred=atom.pred, args=atom.args, negated=spec.negated, text=atom.text)


def _format_atom_text(pred: str, args: list[str]) -> str:
    args_text = ", ".join(args)
    return f"{pred}({args_text})" if args_text else pred


def _translation_token_budget(max_tokens: int, premise_count: int) -> int:
    # Each premise can produce nested rule JSON; 280 tokens/premise is more realistic.
    # Allow exceeding the global max_tokens cap when premises demand it.
    needed = max(1024, 512 + 280 * premise_count)
    return min(8192, max(needed, max_tokens))


def _premise_token_budget(max_tokens: int, premise_count: int) -> int:
    needed = max(1024, 512 + 280 * premise_count)
    return min(8192, max(needed, max_tokens))


def _query_token_budget(max_tokens: int) -> int:
    return min(max_tokens, 384)


def _mcq_options_token_budget(max_tokens: int) -> int:
    # 4 options × ~40 tokens each + JSON wrapper = ~256 tokens minimum
    return max(256, min(max_tokens, 512))


def _repair_atom_dict(atom: dict[str, Any], pred_map: dict[str, str]) -> None:
    """In-place: normalize pred name, remap via pred_map, drop text field.

    Also eliminates double negation: if the LLM embeds negation in the predicate
    name itself (e.g. ``not_follows_pep8``) *and* sets ``negated=True``, the two
    cancel each other out.  Strip the ``not_`` prefix and flip ``negated`` so the
    rest of the pipeline sees a clean positive or single-negation atom.
    """
    atom.pop("text", None)
    if "pred" in atom and isinstance(atom["pred"], str):
        old = atom["pred"]
        norm = pred_map.get(old, _normalize_pred_name(old))
        if norm.startswith("not_"):
            # LLM encoded negation in the pred name — strip it and flip negated flag.
            norm = norm[4:] or "pred"
            atom["negated"] = not bool(atom.get("negated", False))
        atom["pred"] = norm
    if "args" in atom and not isinstance(atom["args"], list):
        atom["args"] = [str(atom["args"])] if atom["args"] is not None else []


def _repair_atoms_list(atoms: list[Any], pred_map: dict[str, str]) -> None:
    for atom in atoms:
        if isinstance(atom, dict):
            _repair_atom_dict(atom, pred_map)


def _repair_raw_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize predicate names and fix common atom schema issues in raw LLM JSON.

    Performs a deep copy so the original is not mutated.
    """
    raw = json.loads(json.dumps(raw))  # deep copy
    pred_map: dict[str, str] = {}
    for pred in raw.get("predicates", []):
        if isinstance(pred, dict) and "name" in pred:
            old_name = str(pred["name"])
            new_name = _normalize_pred_name(old_name)
            pred["name"] = new_name
            pred_map[old_name] = new_name
    for premise in raw.get("premises", []):
        if not isinstance(premise, dict):
            continue
        _repair_atoms_list(premise.get("facts", []), pred_map)
        for rule in premise.get("rules", []):
            if isinstance(rule, dict):
                _repair_atoms_list(rule.get("conditions", []), pred_map)
                if isinstance(rule.get("conclusion"), dict):
                    _repair_atom_dict(rule["conclusion"], pred_map)
    query = raw.get("query")
    if isinstance(query, dict):
        claim = query.get("claim")
        if isinstance(claim, dict):
            _repair_atom_dict(claim, pred_map)
    return raw


def _closest_predicate(pred: str, predicate_names: tuple[str, ...]) -> str | None:
    """Return the predicate from predicate_names with the highest token-overlap with pred.

    Uses Jaccard similarity on underscore-split tokens. Returns None if no candidate
    scores at least 0.30 (prevents spurious substitutions for unrelated predicates).
    """
    if not predicate_names:
        return None
    pred_tokens = set(pred.split("_"))
    best_score, best_name = 0.0, None
    for name in predicate_names:
        name_tokens = set(name.split("_"))
        union_size = len(pred_tokens | name_tokens)
        overlap = len(pred_tokens & name_tokens) / union_size if union_size else 0.0
        if overlap > best_score:
            best_score, best_name = overlap, name
    return best_name if best_score >= 0.30 else None


def _build_mcq_options_messages(
    question: str,
    predicate_names: tuple[str, ...],
) -> list[ChatCompletionMessageParam]:
    """Prompt to translate each MCQ option text into a KB atom."""

    schema_hint = (
        '{"options":[{"label":"A","claim":{"pred":"name","args":["?x"],"negated":false}},'
        '{"label":"B","claim":{"pred":"name","args":["?x"],"negated":false}}]}'
    )
    predicate_list = ", ".join(predicate_names) if predicate_names else "(use descriptive snake_case)"
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. "
                "Map each MCQ option to ONE atom from the given predicate list."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate each MCQ option into a Horn atom. Output JSON matching: {schema_hint}\n"
                "Rules:\n"
                f"- Available predicates: {predicate_list}\n"
                "- Pick the predicate that best captures the option's core claim\n"
                "- negated=true if the option asserts the predicate does NOT hold\n"
                "- args: '?x' for generic entity, snake_case constant for a named entity\n"
                "- Include all options (A, B, C, D) that appear in the question\n\n"
                f"Question:\n{question}"
            ),
        },
    ]
