"""Knowledge-base construction and caching for Type 1 logic queries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from exact.logic.ir import Fact, Rule, Theory
from exact.logic.ir import ParsedPremise
from exact.logic.parser import parse_premise_to_ir

PARSER_VERSION = "heuristic_horn_v1"


@dataclass(frozen=True)
class KnowledgeBase:
    """Parsed, reusable premise set with source-preserving facts and rules."""

    raw_premises: tuple[str, ...]
    facts: tuple[Fact, ...]
    rules: tuple[Rule, ...]
    premise_hash: str
    parser_version: str = PARSER_VERSION
    theory: Theory | None = None
    warnings: tuple[str, ...] = ()


_KB_CACHE: dict[str, KnowledgeBase] = {}


def hash_premises(premises: list[str] | tuple[str, ...], parser_version: str = PARSER_VERSION) -> str:
    """Hash premises plus parser version so stale KBs do not leak across parser changes."""

    text = parser_version + "\n" + "\n".join(premises)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_kb_from_premises(
    premises: list[str] | tuple[str, ...],
    premise_hash: str | None = None,
    parser_version: str = PARSER_VERSION,
) -> KnowledgeBase:
    """Build a KB once per premise set, following VERUS-LM's separation idea."""

    facts: list[Fact] = []
    rules: list[Rule] = []
    warnings: list[str] = []
    raw_premises = tuple(premises)

    for source_idx, premise in enumerate(raw_premises):
        parsed = parse_premise_to_ir(premise=premise, source_idx=source_idx)
        facts.extend(parsed.facts)
        rules.extend(parsed.rules)
        warnings.extend(parsed.warnings)

    return KnowledgeBase(
        raw_premises=raw_premises,
        facts=tuple(facts),
        rules=tuple(rules),
        premise_hash=premise_hash or hash_premises(raw_premises, parser_version),
        parser_version=parser_version,
        warnings=tuple(warnings),
    )


def build_kb_from_parsed_premises(
    premises: list[str] | tuple[str, ...],
    parsed_premises: tuple[ParsedPremise, ...],
    premise_hash: str | None = None,
    parser_version: str = PARSER_VERSION,
    extra_warnings: tuple[str, ...] = (),
) -> KnowledgeBase:
    facts: list[Fact] = []
    rules: list[Rule] = []
    warnings: list[str] = list(extra_warnings)
    raw_premises = tuple(premises)

    for parsed in parsed_premises:
        facts.extend(parsed.facts)
        rules.extend(parsed.rules)
        warnings.extend(parsed.warnings)

    return KnowledgeBase(
        raw_premises=raw_premises,
        facts=tuple(facts),
        rules=tuple(rules),
        premise_hash=premise_hash or hash_premises(raw_premises, parser_version),
        parser_version=parser_version,
        warnings=tuple(warnings),
    )


def get_or_build_kb(premises: list[str] | tuple[str, ...]) -> KnowledgeBase:
    key = hash_premises(premises)
    if key not in _KB_CACHE:
        _KB_CACHE[key] = build_kb_from_premises(premises, premise_hash=key)
    return _KB_CACHE[key]


def clear_kb_cache() -> None:
    _KB_CACHE.clear()
