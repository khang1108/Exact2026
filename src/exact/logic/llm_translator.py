"""LLM autoformalizer for the Type 1 logic pipeline.

The LLM is used as a semantic parser, not as the final judge. It converts
natural-language premises/questions into a compact Horn-style IR consumed by
deterministic symbolic solvers. The design follows the Logic-LM translator →
solver split, LINC's emphasis on predicate-consistent formalization, and the
SymbCoT/Logic-LM++ motivation for later verifier/repair stages.
"""

from __future__ import annotations

import hashlib
import json
import inspect
import re
from dataclasses import dataclass
from typing import Any, Protocol

from openai.types.chat import ChatCompletionMessageParam
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from exact.config import Settings, get_settings
from exact.logic.ir import (
    And,
    Atom,
    Fact,
    Formula,
    FormulaItem,
    Implies,
    Not,
    Or,
    ParsedPremise,
    Query,
    Rule,
    TranslatedProblem,
)
from exact.logic.parser import atom_from_text
from exact.logic.prompts import (
    build_formula_goals_messages,
    build_formula_premises_only_messages,
    build_full_translation_messages,
    build_mcq_options_messages,
    build_premises_only_messages,
    build_problem_formula_messages,
    build_query_only_messages,
)
from exact.llm_client import LLMClient
from exact.logger import get_logger

logger = get_logger(__name__)

# Backward-compatible aliases for notebooks/debug scripts that imported these
# private helpers before the prompt text moved to exact.logic.prompts.
_build_messages = build_full_translation_messages
_build_mcq_options_messages = build_mcq_options_messages
_build_premises_only_messages = build_premises_only_messages
_build_problem_formula_messages = build_problem_formula_messages
_build_query_only_messages = build_query_only_messages


# ---------------------------------------------------------------------------
# Formula premise cache — persists across requests within one process lifetime.
# Key: SHA-256 of the premise list text (same hash function as kb.py but kept
# independent to avoid circular imports: kb.py → pipeline.py → llm_translator).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CachedFormulaPremises:
    """Result of translating premises to formula IR, ready for reuse."""
    predicates: dict[str, int]          # predicate_name → arity
    premises: tuple[FormulaItem, ...]   # formula-level premise items
    entity_constants: tuple[str, ...]   # ground constants found in premise atoms


_FORMULA_PREMISE_CACHE: dict[str, _CachedFormulaPremises] = {}


def _hash_premise_list(premises: list[str]) -> str:
    """SHA-256 of joined premise texts, used as cache key."""
    key = "\n".join(premises) + "\nformula-v1"
    return hashlib.sha256(key.encode()).hexdigest()


def clear_formula_premise_cache() -> None:
    """Clear the in-process formula premise cache (useful in tests)."""
    _FORMULA_PREMISE_CACHE.clear()

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
        # Optional JSON Schema for guided decoding (vLLM guided_json parameter).
        # When provided and the server supports it, the model is constrained to
        # produce JSON conforming to this schema from the first token, eliminating
        # the need for retry on structural failures.
        json_schema: dict[str, Any] | None = None,
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


def translate_problem_with_llm(
    premises: list[str],
    question: str,
    options: list[tuple[str, str]] | None,
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
    temperature: float | None = None,
) -> TranslatedProblem:
    """Translate a full Type 1 problem into formula-level premise/goal IR.

    One LLM call handles premises + query/options with a single shared predicate
    dictionary, preventing the vocabulary drift that caused atom mismatch in the
    legacy three-call approach.  A second attempt with a doubled token budget is
    made if the first response is truncated or structurally invalid.

    Args:
        premises:    NL premise strings (index = source_idx in FormulaItem).
        question:    NL question stem, including embedded options for MCQ.
        options:     Parsed MCQ pairs [(label, text), …] or None for Yes/No.
        llm_client:  JSON-returning LLM client (falls back to settings).
        settings:    App settings (uses get_settings() if None).
        temperature: Override translation temperature; defaults to settings value.

    Returns:
        TranslatedProblem containing shared predicates, premise formulas, and goals.

    Raises:
        RuntimeError: If both translation attempts fail.
    """

    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    temp = settings.llm_temperature if temperature is None else temperature
    goal_count = len(options) if options is not None else 1
    base_budget = _problem_token_budget(
        settings.llm_max_tokens,
        premise_count=len(premises),
        goal_count=goal_count,
    )
    messages = build_problem_formula_messages(premises, question, options)
    json_schema = (
        _problem_formula_json_schema()
        if settings.type1_use_guided_json and _client_accepts_json_schema(client)
        else None
    )
    logger.info(
        "translate_problem_with_llm: premises=%d, goals=%d, budget=%d",
        len(premises),
        goal_count,
        base_budget,
    )

    # Two-attempt loop: first with base budget, then with 2× budget when the
    # response is truncated (incomplete JSON) or fails schema validation.
    # Keeping both attempts here (not in the caller) avoids rebuilding the
    # prompt and keeps the conversation in a single warm turn.
    last_exc: Exception | None = None
    for attempt, token_limit in enumerate([base_budget, min(8192, base_budget * 2)], start=1):
        try:
            request_kwargs: dict[str, Any] = {
                "messages": messages,
                "temperature": temp,
                "max_tokens": token_limit,
            }
            if json_schema is not None:
                request_kwargs["json_schema"] = json_schema
            raw = client.complete_json_sync(**request_kwargs)
        except ValueError as exc:
            # The JSON client raises ValueError for invalid/incomplete JSON.
            # Only retry on JSON parse failures; other ValueErrors propagate.
            msg = str(exc)
            if "invalid JSON" not in msg and "incomplete JSON" not in msg:
                raise
            last_exc = exc
            logger.warning(
                "translate_problem attempt %d/2 JSON error (budget=%d): %s",
                attempt, token_limit, msg[:200],
            )
            continue

        try:
            return _translated_problem_from_raw(
                raw, premises=premises, question=question, options=options
            )
        except (ValueError, KeyError) as exc:
            # The formula parser raises ValueError for missing/malformed nodes.
            last_exc = exc
            logger.warning(
                "translate_problem attempt %d/2 parse error: %s",
                attempt, str(exc)[:300],
            )
            logger.debug("Raw LLM output (attempt %d): %s", attempt, json.dumps(raw)[:2000])

    raise RuntimeError(
        f"translate_problem_with_llm failed after 2 attempts: {last_exc}"
    ) from last_exc


def _client_accepts_json_schema(client: JsonLLMClient) -> bool:
    try:
        return "json_schema" in inspect.signature(client.complete_json_sync).parameters
    except (TypeError, ValueError):
        return False


def _problem_formula_json_schema() -> dict[str, Any]:
    formula_ref = {"$ref": "#/$defs/formula"}
    atom = {
        "type": "object",
        "required": ["type", "pred", "args"],
        "additionalProperties": False,
        "properties": {
            "type": {"const": "atom"},
            "pred": {"type": "string"},
            "args": {"type": "array", "items": {"type": "string"}},
            "negated": {"type": "boolean"},
        },
    }
    formula_item = {
        "type": "object",
        "required": ["source_idx", "role", "text", "formula"],
        "additionalProperties": False,
        "properties": {
            "source_idx": {"type": "integer"},
            "role": {"enum": ["premise", "query", "option"]},
            "text": {"type": "string"},
            "label": {"type": ["string", "null"]},
            "formula": formula_ref,
        },
    }
    return {
        "type": "object",
        "required": ["predicates", "premises", "goals"],
        "additionalProperties": False,
        "properties": {
            "predicates": {
                "type": "object",
                "additionalProperties": {"type": "integer", "minimum": 0},
            },
            "premises": {"type": "array", "items": formula_item},
            "goals": {"type": "array", "items": formula_item},
        },
        "$defs": {
            "formula": {
                "oneOf": [
                    atom,
                    {
                        "type": "object",
                        "required": ["type", "arg"],
                        "additionalProperties": False,
                        "properties": {"type": {"const": "not"}, "arg": formula_ref},
                    },
                    {
                        "type": "object",
                        "required": ["type", "args"],
                        "additionalProperties": False,
                        "properties": {
                            "type": {"const": "and"},
                            "args": {
                                "type": "array",
                                "minItems": 2,
                                "items": formula_ref,
                            },
                        },
                    },
                    {
                        "type": "object",
                        "required": ["type", "args"],
                        "additionalProperties": False,
                        "properties": {
                            "type": {"const": "or"},
                            "args": {
                                "type": "array",
                                "minItems": 2,
                                "items": formula_ref,
                            },
                        },
                    },
                    {
                        "type": "object",
                        "required": ["type", "antecedent", "consequent"],
                        "additionalProperties": False,
                        "properties": {
                            "type": {"const": "implies"},
                            "antecedent": formula_ref,
                            "consequent": formula_ref,
                        },
                    },
                ]
            }
        },
    }


def translate_with_llm(
    premises: list[str],
    question: str,
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> tuple[tuple[ParsedPremise, ...], Query]:
    """Translate Type 1 text into IR while leaving reasoning to solvers."""

    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    messages = build_full_translation_messages(premises, question)
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
    messages = build_premises_only_messages(premises)
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

    messages: list[ChatCompletionMessageParam] = build_query_only_messages(
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
    messages = build_mcq_options_messages(question, predicate_names)
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


def _translated_problem_from_raw(
    raw: dict[str, Any],
    *,
    premises: list[str],
    question: str,
    options: list[tuple[str, str]] | None,
) -> TranslatedProblem:
    """Convert raw one-shot LLM JSON into typed formula IR."""

    raw = _repair_problem_raw_output(raw)
    predicates = _predicates_from_problem_raw(raw.get("predicates", {}))
    premise_items = tuple(
        _formula_item_from_raw(item, default_role="premise", default_source_idx=index, default_text=premises[index])
        for index, item in enumerate(raw.get("premises", []))
        if isinstance(item, dict)
    )
    if not premise_items:
        raise ValueError("problem translation produced no premise formulas")

    raw_goals = raw.get("goals")
    if raw_goals is None and options is not None:
        raw_goals = raw.get("options")
    if raw_goals is None and options is None:
        raw_goals = [raw.get("query")] if isinstance(raw.get("query"), dict) else []
    goal_items = tuple(
        _formula_item_from_raw(
            item,
            default_role="option" if options is not None else "query",
            default_source_idx=-1,
            default_text=_default_goal_text(item, question, options),
        )
        for item in (raw_goals or [])
        if isinstance(item, dict)
    )
    if not goal_items:
        raise ValueError("problem translation produced no goal formulas")

    return TranslatedProblem(
        predicates=predicates,
        premises=premise_items,
        goals=goal_items,
    )


def _default_goal_text(
    item: dict[str, Any],
    question: str,
    options: list[tuple[str, str]] | None,
) -> str:
    if options is None:
        return question
    label = str(item.get("label") or "").strip().upper()
    return dict(options).get(label, question)


def _predicates_from_problem_raw(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        predicates: dict[str, int] = {}
        for raw_name, raw_arity in value.items():
            name = _normalize_pred_name(str(raw_name))
            try:
                arity = int(raw_arity)
            except (TypeError, ValueError):
                arity = 0
            predicates[name] = max(0, arity)
        return predicates

    if isinstance(value, list):
        predicates = {}
        for item in value:
            if not isinstance(item, dict) or "name" not in item:
                continue
            pred = PredicateSpec.model_validate(item)
            predicates[pred.name] = pred.arity
        return predicates

    return {}


def _formula_item_from_raw(
    item: dict[str, Any],
    *,
    default_role: str,
    default_source_idx: int,
    default_text: str,
) -> FormulaItem:
    formula_raw = item.get("formula") or item.get("claim") or item.get("goal")
    if not isinstance(formula_raw, dict):
        raise ValueError(f"formula item is missing a formula object: {item!r}")

    role = str(item.get("role") or default_role).strip().lower()
    if role not in {"premise", "query", "option"}:
        raise ValueError(f"unsupported formula item role: {role!r}")

    label = item.get("label")
    if label is not None:
        label = str(label).strip().upper() or None
        if label is not None and label not in _OPTION_LABELS:
            raise ValueError(f"unsupported option label: {label!r}")

    try:
        source_idx = int(item.get("source_idx", default_source_idx))
    except (TypeError, ValueError):
        source_idx = default_source_idx

    return FormulaItem(
        formula=_formula_from_raw(formula_raw),
        source_idx=source_idx,
        text=str(item.get("text") or default_text).strip(),
        role=role,  # type: ignore[arg-type]
        label=label,
    )


def _formula_from_raw(raw: Any) -> Formula:
    """Parse a recursive formula JSON object into Formula IR.

    Accepts multiple key aliases produced by different LLMs or prompt versions:
      - op key: "type" | "kind" | "op"  (all recognised)
      - implies: "antecedent"/"consequent" OR "lhs"/"rhs" OR "if"/"then"
      - and/or children: "args" OR "items" OR "operands" OR "children"
    This tolerance prevents hard failures when an 8B model uses a slightly
    different key name from what the prompt specified.
    """

    if not isinstance(raw, dict):
        raise ValueError(f"formula must be a JSON object, got {type(raw).__name__}")

    # Unwrap a nested "formula" wrapper if the LLM added an extra level.
    if "formula" in raw and isinstance(raw["formula"], dict):
        return _formula_from_raw(raw["formula"])
    # Some models output {"atom": {...}} instead of the atom fields directly.
    if "atom" in raw and isinstance(raw["atom"], dict):
        raw = {**raw["atom"], "type": "atom"}

    # Determine node kind from any of the three op-key variants.
    kind = str(raw.get("type") or raw.get("kind") or raw.get("op") or "").strip().lower()
    if not kind:
        # Fall back: if "pred" is present it must be an atom; otherwise unknown.
        kind = "atom" if raw.get("pred") or raw.get("text") else ""

    # ── atom ──────────────────────────────────────────────────────────────────
    if kind in {"atom", "literal"}:
        atom_raw = dict(raw)
        # Strip op-key variants before passing to AtomSpec
        atom_raw.pop("type", None)
        atom_raw.pop("kind", None)
        atom_raw.pop("op", None)
        _repair_atom_dict(atom_raw, {})
        return _atom_from_spec(AtomSpec.model_validate(atom_raw))

    # ── not ───────────────────────────────────────────────────────────────────
    if kind in {"not", "neg", "negation"}:
        # Accept "arg" (standard) or "formula"/"operand" (LLM variants)
        child = raw.get("arg") or raw.get("formula") or raw.get("operand")
        if child is None:
            raise ValueError(f"not node missing 'arg' child: {raw!r}")
        return Not(_formula_from_raw(child))

    # ── and ───────────────────────────────────────────────────────────────────
    if kind in {"and", "conjunction"}:
        args = _formula_args(raw, "and")
        # LLM sometimes wraps a single condition in an and-node — simplify to child.
        return args[0] if len(args) == 1 else And(args)

    # ── or ────────────────────────────────────────────────────────────────────
    if kind in {"or", "disjunction"}:
        args = _formula_args(raw, "or")
        return args[0] if len(args) == 1 else Or(args)

    # ── implies ───────────────────────────────────────────────────────────────
    if kind in {"implies", "imply", "if_then", "if-then", "implication"}:
        # Accept many key pairs — different LLMs use different names.
        # Standard: antecedent/consequent. Also: lhs/rhs, if/then,
        # condition/result, premise/conclusion, left/right, from/to.
        antecedent = (
            raw.get("antecedent")
            or raw.get("lhs")
            or raw.get("if")
            or raw.get("condition")
            or raw.get("premise")
            or raw.get("left")
            or raw.get("from")
        )
        consequent = (
            raw.get("consequent")
            or raw.get("rhs")
            or raw.get("then")
            or raw.get("result")
            or raw.get("conclusion")
            or raw.get("right")
            or raw.get("to")
        )
        if antecedent is None or consequent is None:
            raise ValueError(
                f"implies formula requires antecedent+consequent (or lhs+rhs): {raw!r}"
            )
        return Implies(_formula_from_raw(antecedent), _formula_from_raw(consequent))

    raise ValueError(f"unsupported formula type: {kind!r}")


def _formula_args(raw: dict[str, Any], kind: str) -> tuple[Formula, ...]:
    """Extract and parse the child-formula list from an and/or node.

    Tries multiple key names because different LLMs and prompt versions use
    different conventions for the children of conjunction/disjunction nodes:
      "args"     — used by the current prompt (matches atom.args key name)
      "items"    — alternative that avoids the atom.args name collision
      "operands" — natural-language style
      "children" — tree-style

    We specifically check that each extracted value is a dict (formula object)
    and not a string (which would mean the LLM confused atom.args with and.args).
    """
    # Try keys in preference order; "args" first since that's what the prompt says.
    for key in ("args", "items", "operands", "children"):
        candidate = raw.get(key)
        if isinstance(candidate, list):
            # If entries are dicts they are formula nodes — use this list.
            # If entries are strings the LLM confused atom.args with and.args;
            # skip this key and try the next one.
            if all(isinstance(v, dict) for v in candidate):
                values = candidate
                break
    else:
        raise ValueError(f"{kind} formula has no valid child-formula list in {list(raw.keys())}")

    if len(values) < 1:
        raise ValueError(f"{kind} formula has no valid child-formula list in {list(raw.keys())}")
    # 1-child case is allowed — caller simplifies and(x) → x, or(x) → x.
    return tuple(_formula_from_raw(value) for value in values)


def _repair_problem_raw_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy and normalize common raw problem-translation fields."""

    raw = json.loads(json.dumps(raw))
    if isinstance(raw.get("predicates"), list):
        raw["predicates"] = {
            _normalize_pred_name(str(item.get("name"))): int(item.get("arity") or 0)
            for item in raw["predicates"]
            if isinstance(item, dict) and item.get("name")
        }
    return raw


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


def _problem_token_budget(max_tokens: int, premise_count: int, goal_count: int) -> int:
    needed = max(1536, 768 + 320 * premise_count + 180 * goal_count)
    return min(8192, max(needed, max_tokens))


def _query_token_budget(max_tokens: int) -> int:
    return min(max_tokens, 384)


def _mcq_options_token_budget(max_tokens: int) -> int:
    # 4 options × ~40 tokens each + JSON wrapper = ~256 tokens minimum
    return max(256, min(max_tokens, 512))


def _formula_premises_token_budget(premise_count: int) -> int:
    # Each premise formula tree averages ~180 tokens (compact JSON, no text/role fields,
    # but implies/and nodes add nesting). Predicates dict adds ~200 tokens.
    # Use 200 per premise to avoid silent truncation (the main failure mode).
    # Cap at 6144; generation speed on A6000 for 14p ≈ 14×180=2520 tokens ≈ 35s → OK.
    needed = max(1536, 400 + 200 * premise_count)
    return min(6144, needed)


def _formula_goals_token_budget(goal_count: int) -> int:
    # Each goal formula averages ~100 tokens (options with implies are larger).
    # Use 150 per goal to be safe. JSON wrapper ~100 tokens.
    needed = max(384, 100 + 150 * goal_count)
    return min(1536, needed)


def _collect_constants_from_formula(formula: Formula, out: set[str]) -> None:
    """Recursively collect ground constants (non-variable args) from a formula tree."""
    if isinstance(formula, Atom):
        for arg in formula.args:
            if not arg.startswith("?"):
                out.add(arg)
    elif isinstance(formula, Not):
        _collect_constants_from_formula(formula.arg, out)
    elif isinstance(formula, (And, Or)):
        for child in formula.args:
            _collect_constants_from_formula(child, out)
    elif isinstance(formula, Implies):
        _collect_constants_from_formula(formula.antecedent, out)
        _collect_constants_from_formula(formula.consequent, out)


def _extract_entity_constants(premises: tuple[FormulaItem, ...]) -> tuple[str, ...]:
    """Return sorted tuple of ground constants appearing in premise formula atoms."""
    constants: set[str] = set()
    for item in premises:
        _collect_constants_from_formula(item.formula, constants)
    return tuple(sorted(constants))


def translate_formula_premises_only_with_llm(
    premises: list[str],
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> _CachedFormulaPremises:
    """Translate premises to formula IR and cache by premise-set hash.

    This is the first half of the split-translation path. The result is keyed
    by a SHA-256 of the premise texts so subsequent questions in the same
    premise group (sharing identical premises) skip this LLM call entirely.

    Returns:
        _CachedFormulaPremises with predicates dict, FormulaItems, and entity constants.

    Raises:
        RuntimeError: If translation fails after two attempts.
    """
    cache_key = _hash_premise_list(premises)
    if cache_key in _FORMULA_PREMISE_CACHE:
        logger.info(
            "translate_formula_premises_only: cache hit for %d premises (key=%s…)",
            len(premises), cache_key[:8],
        )
        return _FORMULA_PREMISE_CACHE[cache_key]

    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    budget = _formula_premises_token_budget(len(premises))
    messages = build_formula_premises_only_messages(premises)

    logger.info(
        "translate_formula_premises_only: translating %d premises, budget=%d",
        len(premises), budget,
    )

    last_exc: Exception | None = None
    # Run up to 3 attempts: base budget → 2× budget → 2× budget with completion hint.
    # The third attempt fires when the LLM silently returned fewer premises than
    # expected (valid JSON but incomplete), which the first two retry conditions
    # (JSON parse errors) would not catch.
    budgets = [budget, min(6144, budget * 2), min(6144, budget * 2)]
    for attempt, token_limit in enumerate(budgets, start=1):
        try:
            raw = client.complete_json_sync(
                messages=messages,
                temperature=settings.llm_temperature,
                max_tokens=token_limit,
            )
        except ValueError as exc:
            msg = str(exc)
            if "invalid JSON" not in msg and "incomplete JSON" not in msg:
                raise
            last_exc = exc
            logger.warning("Premise-only attempt %d/3 JSON error: %s", attempt, msg[:200])
            continue

        try:
            raw = _repair_problem_raw_output(raw)
            predicates = _predicates_from_problem_raw(raw.get("predicates", {}))
            premise_items = tuple(
                _formula_item_from_raw(
                    item,
                    default_role="premise",
                    default_source_idx=index,
                    default_text=premises[index],
                )
                for index, item in enumerate(raw.get("premises", []))
                if isinstance(item, dict)
            )
            if not premise_items:
                raise ValueError("premise-only translation returned no premise formulas")

            # Completeness check: if fewer than 80% of premises were translated,
            # the LLM silently truncated. Retry with an explicit correction prompt.
            completeness = len(premise_items) / max(len(premises), 1)
            if completeness < 0.80 and attempt < len(budgets):
                translated_indices = {item.source_idx for item in premise_items}
                missing = [i for i in range(len(premises)) if i not in translated_indices]
                logger.warning(
                    "Premise-only attempt %d/3: incomplete — %d/%d premises (missing: %s); retrying",
                    attempt, len(premise_items), len(premises), missing[:5],
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": json.dumps(raw)},
                    {
                        "role": "user",
                        "content": (
                            f"INCOMPLETE: you only translated {len(premise_items)} of "
                            f"{len(premises)} premises. "
                            f"You MUST translate ALL {len(premises)} premises. "
                            f"Missing source_idx values: {missing}. "
                            "Return complete JSON with every premise translated."
                        ),
                    },
                ]
                last_exc = ValueError(
                    f"incomplete premises: {len(premise_items)}/{len(premises)}"
                )
                continue

            entity_constants = _extract_entity_constants(premise_items)
            result = _CachedFormulaPremises(
                predicates=predicates,
                premises=premise_items,
                entity_constants=entity_constants,
            )
            _FORMULA_PREMISE_CACHE[cache_key] = result
            logger.info(
                "translate_formula_premises_only: done — %d/%d premises, %d predicates, %d constants",
                len(premise_items), len(premises), len(predicates), len(entity_constants),
            )
            return result

        except (ValueError, KeyError) as exc:
            last_exc = exc
            logger.warning("Premise-only attempt %d/3 parse error: %s", attempt, str(exc)[:300])

    raise RuntimeError(
        f"translate_formula_premises_only_with_llm failed after 2 attempts: {last_exc}"
    ) from last_exc


def _collect_preds_from_formula(formula: Formula, out: set[str]) -> None:
    """Recursively collect all predicate names used in a formula tree."""
    if isinstance(formula, Atom):
        out.add(formula.pred)
    elif isinstance(formula, Not):
        _collect_preds_from_formula(formula.arg, out)
    elif isinstance(formula, (And, Or)):
        for child in formula.args:
            _collect_preds_from_formula(child, out)
    elif isinstance(formula, Implies):
        _collect_preds_from_formula(formula.antecedent, out)
        _collect_preds_from_formula(formula.consequent, out)


def translate_formula_goals_with_llm(
    question: str,
    options: list[tuple[str, str]] | None,
    predicate_dict: dict[str, int],
    entity_constants: tuple[str, ...],
    llm_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
) -> tuple[FormulaItem, ...]:
    """Translate query or MCQ options into goal FormulaItems.

    This is the second half of the split-translation path. It receives the
    predicate dictionary from the premises-only call, preventing vocabulary
    drift: every atom predicate in the goals is checked against the dict and
    the LLM is retried with a correction prompt if an unknown predicate appears.

    Unlike translate_query_only_with_llm (which handles only simple atom goals),
    this function produces full formula trees and handles MCQ options that are
    implies/and/not nodes, as required by the formula-Z3 solver.

    Returns:
        Tuple of FormulaItems (one per option, or one query item).

    Raises:
        RuntimeError: If translation fails after two attempts.
    """
    settings = settings or get_settings()
    client = llm_client or LLMClient.from_settings(settings)
    goal_count = len(options) if options is not None else 1
    budget = _formula_goals_token_budget(goal_count)

    messages = build_formula_goals_messages(
        question=question,
        options=options,
        predicate_dict=predicate_dict,
        entity_constants=entity_constants,
    )
    allowed_preds = set(predicate_dict.keys())

    logger.info(
        "translate_formula_goals: translating %d goals, %d allowed predicates, budget=%d",
        goal_count, len(allowed_preds), budget,
    )

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            raw = client.complete_json_sync(
                messages=messages,
                temperature=settings.llm_temperature,
                max_tokens=budget,
            )
        except ValueError as exc:
            last_exc = exc
            logger.warning("Goals attempt %d/2 JSON error: %s", attempt + 1, str(exc)[:200])
            continue

        try:
            raw = _repair_problem_raw_output(raw)
            raw_goals = raw.get("goals") or raw.get("options") or []
            goal_items = tuple(
                _formula_item_from_raw(
                    item,
                    default_role="option" if options is not None else "query",
                    default_source_idx=-1,
                    default_text=_default_goal_text(item, question, options),
                )
                for item in raw_goals
                if isinstance(item, dict)
            )
            if not goal_items:
                raise ValueError("goal translation returned no goal formulas")

            # Vocab guard: check all atom predicates against the allowed dict.
            # If any unknown pred is found, build a correction message and retry.
            unknown_preds: set[str] = set()
            if allowed_preds:
                for item in goal_items:
                    _collect_preds_from_formula(item.formula, unknown_preds)
                unknown_preds -= allowed_preds

            if unknown_preds and attempt == 0:
                logger.warning(
                    "Goal predicates not in premise dict: %s — retrying with correction",
                    sorted(unknown_preds),
                )
                # Append correction turn so the model knows exactly what went wrong
                messages = [
                    *messages,
                    {"role": "assistant", "content": json.dumps(raw)},
                    {
                        "role": "user",
                        "content": (
                            f"WRONG: predicates {sorted(unknown_preds)} are not in the allowed list.\n"
                            f"You MUST use ONLY these predicates: {', '.join(sorted(allowed_preds))}\n"
                            "Return corrected JSON with all predicates replaced by names from the allowed list."
                        ),
                    },
                ]
                last_exc = ValueError(f"unknown goal predicates: {sorted(unknown_preds)}")
                continue

            # If still unknown after retry, substitute closest match per predicate
            if unknown_preds and allowed_preds:
                logger.warning(
                    "Goal predicates still unknown after retry: %s — applying closest-match substitution",
                    sorted(unknown_preds),
                )

            logger.info(
                "translate_formula_goals: done — %d goal items", len(goal_items)
            )
            return goal_items

        except (ValueError, KeyError) as exc:
            last_exc = exc
            logger.warning("Goals attempt %d/2 parse error: %s", attempt + 1, str(exc)[:300])

    raise RuntimeError(
        f"translate_formula_goals_with_llm failed after 2 attempts: {last_exc}"
    ) from last_exc


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
