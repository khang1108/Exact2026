"""Lightweight Type 1 parser.

This is intentionally conservative: it handles common regulation-style Horn
patterns now, while leaving a clean replacement boundary for LLM/CLOVER-style
decomposition later.
"""

from __future__ import annotations

import re
import unicodedata

from exact.logic.ir import Atom, Fact, ParsedPremise, Query, Rule

_IF_THEN_RE = re.compile(r"^\s*if\s+(.+?)\s*,?\s+then\s+(.+?)\.?\s*$", re.IGNORECASE)
_TRAILING_PUNCT_RE = re.compile(r"[\s.?!:;]+$")
_WORD_RE = re.compile(r"[a-z0-9]+")


def parse_premise_to_ir(premise: str, source_idx: int) -> ParsedPremise:
    """Parse one natural-language premise into facts/rules.

    Current MVP supports:
    - "If A then B." -> Rule(A -> B)
    - "If A and B, then C." -> Rule(A & B -> C)
    - "A." -> Fact(A)
    """

    text = premise.strip()
    if not text:
        return ParsedPremise(warnings=(f"premise {source_idx + 1} is empty",))

    match = _IF_THEN_RE.match(text)
    if match:
        antecedent, consequent = match.groups()
        conditions = tuple(_atom_from_clause(part) for part in _split_conjunction(antecedent))
        conclusion = _atom_from_clause(consequent)
        return ParsedPremise(
            rules=(
                Rule(
                    conditions=conditions,
                    conclusion=conclusion,
                    source_idx=source_idx,
                    text=text,
                ),
            )
        )

    return ParsedPremise(facts=(Fact(atom=_atom_from_clause(text), source_idx=source_idx, text=text),))


def parse_question_to_query(question: str) -> Query:
    """Parse a yes/no-style question into a target claim."""

    raw = question.strip()
    normalized = _strip_question_shell(raw)
    atom = _atom_from_clause(normalized)
    return Query(claim=atom, raw_question=raw, expects_negation=atom.negated)


def atom_from_text(text: str) -> Atom:
    """Public helper for tests and future LLM parser adapters."""

    return _atom_from_clause(text)


def _split_conjunction(text: str) -> list[str]:
    parts = re.split(r"\s+(?:and|&)\s+", text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def _atom_from_clause(text: str) -> Atom:
    clause = _clean_clause(text)
    negated = False

    for prefix in ("it is not true that ", "not ", "does not ", "do not ", "did not "):
        if clause.startswith(prefix):
            negated = True
            clause = clause[len(prefix) :].strip()
            break

    pred = _slugify(_canonicalize_clause(clause))
    if not pred:
        pred = "unknown"

    return Atom(pred=pred, negated=negated, text=clause)


def _strip_question_shell(question: str) -> str:
    text = _clean_clause(question)
    text = re.sub(
        r"^(?:based on the above premises,\s*)?(?:does|do|did|is|are|can|could|will|would|should)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+(?:hold|holds|follow|follows)$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _clean_clause(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().strip("\"'")
    text = _TRAILING_PUNCT_RE.sub("", text)
    return re.sub(r"\s+", " ", text).lower()


def _canonicalize_clause(text: str) -> str:
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    text = re.sub(r"\s+(?:is|are|was|were)\s+true$", "", text)
    return text.strip()


def _slugify(text: str) -> str:
    return "_".join(_WORD_RE.findall(text))
