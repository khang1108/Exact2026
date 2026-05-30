"""Knowledge-base construction and caching for Type 1 logic queries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from exact.logic.ir import Fact, ParsedPremise, Rule, Theory
from exact.logger import get_logger

logger = get_logger(__name__)

LLM_PARSER_VERSION = "llm_translator_v3"


@dataclass(frozen=True)
class KnowledgeBase:
    """Parsed, reusable premise set with source-preserving facts and rules."""

    raw_premises: tuple[str, ...]
    facts: tuple[Fact, ...]
    rules: tuple[Rule, ...]
    premise_hash: str
    parser_version: str = LLM_PARSER_VERSION
    predicate_names: tuple[str, ...] = ()
    theory: Theory | None = None
    warnings: tuple[str, ...] = ()


def hash_premises(premises: list[str] | tuple[str, ...], parser_version: str = LLM_PARSER_VERSION) -> str:
    """Hash premises plus parser version so stale KBs do not leak across parser changes."""

    text = parser_version + "\n" + "\n".join(premises)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_kb_from_parsed_premises(
    premises: list[str] | tuple[str, ...],
    parsed_premises: tuple[ParsedPremise, ...],
    premise_hash: str | None = None,
    parser_version: str = LLM_PARSER_VERSION,
    predicate_names: tuple[str, ...] = (),
    extra_warnings: tuple[str, ...] = (),
    add_contrapositives: bool = True,
) -> KnowledgeBase:
    facts: list[Fact] = []
    rules: list[Rule] = []
    warnings: list[str] = list(extra_warnings)
    raw_premises = tuple(premises)

    for parsed in parsed_premises:
        facts.extend(parsed.facts)
        rules.extend(parsed.rules)
        warnings.extend(parsed.warnings)

    if add_contrapositives:
        rules.extend(_build_contrapositive_rules(tuple(rules)))

    return KnowledgeBase(
        raw_premises=raw_premises,
        facts=tuple(facts),
        rules=tuple(rules),
        premise_hash=premise_hash or hash_premises(raw_premises, parser_version),
        parser_version=parser_version,
        predicate_names=predicate_names or _predicate_names_from_ir(tuple(facts), tuple(rules)),
        warnings=tuple(warnings),
    )


_LLM_KB_CACHE: dict[str, KnowledgeBase] = {}
_LLM_KB_CANDIDATE_CACHE: dict[str, tuple[tuple[KnowledgeBase, ...], tuple[str, ...]]] = {}


def build_kb_from_premises(
    premises: list[str] | tuple[str, ...],
    llm_client: Any,
    settings: Any,
    premise_hash: str | None = None,
    temperature: float | None = None,
    deadline: float | None = None,
) -> tuple[KnowledgeBase, tuple[str, ...]]:
    """Build a KB from LLM-translated premises.

    This keeps KB construction as a first-class logic layer while preserving
    the LLM-only contract. Translation failures are logged and raised.
    """

    from exact.logic.llm_translator import translate_premises_only_with_llm

    try:
        parsed_premises, warnings, predicate_names = translate_premises_only_with_llm(
            list(premises), llm_client, settings, temperature=temperature, deadline=deadline
        )
    except Exception as exc:
        logger.exception("LLM premise translation failed")
        raise RuntimeError(f"LLM premise translation failed: {exc}") from exc

    add_contrapositives = getattr(settings, "type1_add_contrapositives", True)
    return (
        build_kb_from_parsed_premises(
            premises,
            parsed_premises,
            premise_hash=premise_hash,
            parser_version=LLM_PARSER_VERSION,
            predicate_names=predicate_names,
            extra_warnings=warnings,
            add_contrapositives=add_contrapositives,
        ),
        warnings,
    )


def get_or_build_kb(
    premises: list[str] | tuple[str, ...],
    llm_client: Any,
    settings: Any,
    deadline: float | None = None,
) -> tuple[KnowledgeBase, tuple[str, ...]]:
    """Return a cached KB built from LLM-translated premises.

    Same premises hash → single LLM call regardless of how many questions share
    that premise group. LLM translation failures are logged and raised.
    """

    key = hash_premises(premises, parser_version=LLM_PARSER_VERSION)
    if key in _LLM_KB_CACHE:
        logger.debug("LLM KB cache hit: %s", key[:8])
        return _LLM_KB_CACHE[key], ()

    kb, warnings = build_kb_from_premises(
        premises,
        llm_client,
        settings,
        premise_hash=key,
        deadline=deadline,
    )
    _LLM_KB_CACHE[key] = kb
    logger.debug("LLM KB cache stored: %s premises, key=%s", len(premises), key[:8])
    return kb, warnings


def build_kb_candidates_from_premises(
    premises: list[str] | tuple[str, ...],
    llm_client: Any,
    settings: Any,
    samples: int,
    sampling_temperature: float,
    deadline: float | None = None,
) -> tuple[tuple[KnowledgeBase, ...], tuple[str, ...]]:
    """Build k sampled KB candidates from the same premises."""

    candidates: list[KnowledgeBase] = []
    warnings: list[str] = []
    for index in range(samples):
        candidate_hash = hash_premises(
            premises,
            parser_version=f"{LLM_PARSER_VERSION}:sample={index}",
        )
        try:
            kb, candidate_warnings = build_kb_from_premises(
                premises,
                llm_client,
                settings,
                premise_hash=candidate_hash,
                temperature=sampling_temperature,
                deadline=deadline,
            )
        except Exception as exc:
            message = f"candidate {index + 1}/{samples} failed: {exc}"
            logger.exception("LLM KB candidate translation failed: %s", message)
            warnings.append(message)
            continue

        candidates.append(kb)
        warnings.extend(candidate_warnings)

    if not candidates:
        raise RuntimeError(
            "All LLM premise translation candidates failed: " + "; ".join(warnings)
        )

    return tuple(candidates), tuple(warnings)


def get_or_build_kb_candidates(
    premises: list[str] | tuple[str, ...],
    llm_client: Any,
    settings: Any,
    samples: int,
    sampling_temperature: float,
    deadline: float | None = None,
) -> tuple[tuple[KnowledgeBase, ...], tuple[str, ...]]:
    """Return cached k-sampled KB candidates for symbolic-consistency voting."""

    if samples <= 1:
        kb, warnings = get_or_build_kb(premises, llm_client, settings, deadline=deadline)
        return (kb,), warnings

    parser_version = (
        f"{LLM_PARSER_VERSION}:k={samples}:temp={sampling_temperature}:"
        f"model={getattr(settings, 'llm_model', '')}"
    )
    key = hash_premises(premises, parser_version=parser_version)
    cached = _LLM_KB_CANDIDATE_CACHE.get(key)
    if cached is not None:
        logger.debug("LLM KB candidate cache hit: %s", key[:8])
        return cached

    candidates, warnings = build_kb_candidates_from_premises(
        premises,
        llm_client,
        settings,
        samples=samples,
        sampling_temperature=sampling_temperature,
        deadline=deadline,
    )
    _LLM_KB_CANDIDATE_CACHE[key] = (candidates, warnings)
    logger.debug(
        "LLM KB candidate cache stored: %s candidates, key=%s",
        len(candidates),
        key[:8],
    )
    return candidates, warnings


def clear_kb_cache() -> None:
    _LLM_KB_CACHE.clear()
    _LLM_KB_CANDIDATE_CACHE.clear()


def _predicate_names_from_ir(facts: tuple[Fact, ...], rules: tuple[Rule, ...]) -> tuple[str, ...]:
    names: set[str] = {fact.atom.pred for fact in facts}
    for rule in rules:
        names.add(rule.conclusion.pred)
        names.update(condition.pred for condition in rule.conditions)
    return tuple(sorted(names))


def _build_contrapositive_rules(rules: tuple[Rule, ...]) -> tuple[Rule, ...]:
    """Derive contrapositive rules from existing Horn clauses.

    Single-condition  A → B       ⇒  ¬B → ¬A
    Multi-condition   A ∧ B → C   ⇒  ¬C ∧ B → ¬A  and  ¬C ∧ A → ¬B  (etc.)

    All generated rules are valid logical consequences.  Rules whose conclusion
    is already negated are skipped to avoid ¬¬A cycling.
    """
    contrapositives: list[Rule] = []
    for rule in rules:
        conc = rule.conclusion
        if conc.negated:
            continue
        neg_conc = conc.negation()
        if len(rule.conditions) == 1:
            cond = rule.conditions[0]
            if not cond.negated:
                contrapositives.append(Rule(
                    conditions=(neg_conc,),
                    conclusion=cond.negation(),
                    source_idx=rule.source_idx,
                    text=rule.text,
                ))
        else:
            # Partial contrapositive for each positive condition
            for i, cond in enumerate(rule.conditions):
                if cond.negated:
                    continue
                others = tuple(c for j, c in enumerate(rule.conditions) if j != i)
                contrapositives.append(Rule(
                    conditions=(neg_conc, *others),
                    conclusion=cond.negation(),
                    source_idx=rule.source_idx,
                    text=rule.text,
                ))
    return tuple(contrapositives)
