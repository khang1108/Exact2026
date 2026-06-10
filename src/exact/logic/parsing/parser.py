"""Deterministic Horn parsing utilities for Type 1 logic tests and MCQ goals.

The production architecture follows the Logic-LM/LINC pattern: translate natural
language into symbolic form first, then let a deterministic solver reason over
that form. Runtime Type 1 premise translation is LLM-only; this module remains
as a small utility boundary for tests and for converting already-isolated MCQ
option text into solver atoms.
"""

from __future__ import annotations

import re
import unicodedata

from exact.logic.ir import Atom, Fact, ParsedPremise, Query, Rule

_IF_THEN_RE = re.compile(r"^\s*if\s+(.+?)\s*,?\s+then\s+(.+?)\.?\s*$", re.IGNORECASE)
_RELATIVE_RULE_RE = re.compile(
    r"^\s*(?:all\s+)?(?P<subject>[a-z][a-z\s-]*?)\s+who\s+"
    r"(?P<conditions>.+?)\s+(?P<conclusion>are|is|can|qualify|qualifies|receive|receives)\s+"
    r"(?P<tail>.+?)\.?\s*$",
    re.IGNORECASE,
)
_TRAILING_PUNCT_RE = re.compile(r"[\s.?!:;]+$")
_WORD_RE = re.compile(r"[a-z0-9]+")

_ENTITY_PREFIXES = (
    "professor",
    "dr",
    "student",
    "faculty member",
)
_SUBJECT_NOUNS = {
    "student",
    "students",
    "faculty member",
    "faculty members",
    "python code",
    "python project",
    "python projects",
    "python",
    "code",
    "project",
    "projects",
}
_SYNONYMS = {
    "research methodology course": "research methodology",
    "required community service hours": "community service",
    "university scholarship": "scholarship",
    "the university scholarship": "scholarship",
    "international program": "international program",
    "the international program": "international program",
    "science assessment": "science assessment",
    "the science assessment": "science assessment",
    "core curriculum": "core curriculum",
    "the core curriculum": "core curriculum",
    "capstone project": "capstone project",
    "a capstone project": "capstone project",
    "honors diploma": "honors diploma",
    "an honors diploma": "honors diploma",
    "faculty recommendation": "faculty recommendation",
    "a faculty recommendation": "faculty recommendation",
    "phd": "phd",
    "a phd": "phd",
    "pep 8 standards": "pep8",
    "pep 8 standard": "pep8",
    "clean and readable code": "clean_readable_code",
    "clean readable code": "clean_readable_code",
}
_LEADING_VERBS = {
    "are",
    "is",
    "was",
    "were",
    "be",
    "been",
    "can",
    "could",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "completed",
    "passed",
    "received",
    "awarded",
    "qualified",
    "qualify",
    "qualifies",
    "eligible",
    "holds",
    "hold",
    "maintains",
    "maintain",
    "follows",
    "follow",
}

_FILLER_WORDS = {
    "a",
    "an",
    "the",
    "all",
    "any",
    "required",
    "academic",
    "standards",
    "standard",
    "hours",
    "her",
    "his",
    "their",
}


def parse_premise_to_ir(premise: str, source_idx: int) -> ParsedPremise:
    """Parse one natural-language premise into facts or Horn rules."""

    text = premise.strip()
    if not text:
        return ParsedPremise(warnings=(f"premise {source_idx + 1} is empty",))

    rule = _parse_rule(text, source_idx)
    if rule is not None:
        return ParsedPremise(rules=(rule,))

    return ParsedPremise(facts=(Fact(atom=_atom_from_clause(text), source_idx=source_idx, text=text),))


def parse_question_to_query(question: str) -> Query:
    """Parse a yes/no-style question into a target claim atom."""

    raw = question.strip()
    normalized = _strip_question_shell(raw)
    atom = _atom_from_clause(normalized)
    return Query(claim=atom, raw_question=raw, expects_negation=atom.negated)


def atom_from_text(text: str) -> Atom:
    """Convert a short claim or MCQ option into the canonical Atom form."""

    return _atom_from_clause(text)


def _parse_rule(text: str, source_idx: int) -> Rule | None:
    """Return a Horn rule for supported conditional and relative-clause patterns."""

    if_then = _IF_THEN_RE.match(text)
    if if_then:
        antecedent, consequent = if_then.groups()
        conditions = tuple(_atom_from_clause(part, default_arg="?x") for part in _split_conditions(antecedent))
        conclusion = _atom_from_clause(consequent, default_arg="?x")
        return Rule(conditions=conditions, conclusion=conclusion, source_idx=source_idx, text=text)

    relative = _RELATIVE_RULE_RE.match(_clean_clause(text))
    if relative:
        conditions = tuple(
            _atom_from_clause(part, default_arg="?x")
            for part in _split_conditions(relative.group("conditions"))
        )
        conclusion_text = f"{relative.group('conclusion')} {relative.group('tail')}"
        conclusion = _atom_from_clause(conclusion_text, default_arg="?x")
        return Rule(conditions=conditions, conclusion=conclusion, source_idx=source_idx, text=text)

    return None


def _atom_from_clause(text: str, default_arg: str | None = None) -> Atom:
    """Build an Atom by extracting an optional entity and canonical predicate."""

    clause = _clean_clause(text)
    negated, clause = _strip_negation(clause)
    clause = _strip_subject_intro(clause)

    entity, predicate_clause = _extract_entity(clause)
    arg = entity or default_arg
    pred = _predicate_from_clause(predicate_clause)
    if not pred:
        pred = _slugify(_canonicalize_clause(clause)) or "unknown"

    args = (arg,) if arg else ()
    return Atom(pred=pred, args=args, negated=negated, text=clause)


def _split_conditions(text: str) -> list[str]:
    """Split a conjunction while preserving phrases such as clean and readable."""

    normalized = _clean_clause(text)
    normalized = re.sub(r"\bthey\b", "", normalized)
    normalized = re.sub(r"\bwho\b", "", normalized)
    parts = re.split(r"\s+(?:and|&)\s+(?=(?:has|have|had|passed|completed|received|holds?|maintains?|can|is|are|does|do|follows?|qualifies?|awarded)\b)", normalized)
    return [part.strip(" ,") for part in parts if part.strip(" ,")]


def _split_conditions_simple(text: str) -> list[str]:
    parts = re.split(r"\s+(?:and|&)\s+", text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def _strip_question_shell(question: str) -> str:
    """Remove question framing so the remaining text is a claim."""

    text = _clean_clause(question)
    text = re.sub(r"^based on the above premises,\s*", "", text)
    text = re.sub(r",?\s*according to the premises$", "", text)
    text = re.sub(r"^does\s+it\s+follow\s+that\s+", "", text)
    text = re.sub(
        r"^(?:does|do|did|is|are|can|could|will|would|should)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+(?:hold|holds|follow|follows)$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _strip_negation(clause: str) -> tuple[bool, str]:
    """Detect common explicit negation markers and remove them from the clause."""

    for prefix in ("it is not true that ", "not ", "does not ", "do not ", "did not ", "cannot "):
        if clause.startswith(prefix):
            return True, clause[len(prefix) :].strip()

    replacements = (
        (" does not ", " "),
        (" do not ", " "),
        (" did not ", " "),
        (" cannot ", " can "),
        (" is not ", " is "),
        (" are not ", " are "),
    )
    for marker, replacement in replacements:
        if marker in clause:
            return True, clause.replace(marker, replacement, 1).strip()

    return False, clause


def _strip_subject_intro(clause: str) -> str:
    """Remove generic subject introductions that should not become predicates."""

    return re.sub(
        r"^(?:all\s+)?(?:students?|faculty members?|python projects?|python code|a student|a faculty member)\s+",
        "",
        clause,
    ).strip()


def _extract_entity(clause: str) -> tuple[str | None, str]:
    """Extract a named entity constant from the start of a clause when present."""

    words = clause.split()
    if not words:
        return None, clause

    if len(words) >= 2 and " ".join(words[:2]) in {"professor john", "professor sophia"}:
        return _slugify(" ".join(words[:2])), " ".join(words[2:])

    first = words[0]
    if first not in _SUBJECT_NOUNS and first not in _LEADING_VERBS and first not in {"if", "then", "they", "it"}:
        if first in _ENTITY_PREFIXES and len(words) >= 2:
            return _slugify(" ".join(words[:2])), " ".join(words[2:])
        return _slugify(first), " ".join(words[1:])

    return None, clause


def _predicate_from_clause(clause: str) -> str:
    """Normalize educational verb phrases into stable predicate names."""

    clause = _canonicalize_clause(clause)
    patterns = (
        (r"^(?:has|have|had) completed (.+)$", "completed"),
        (r"^completed (.+)$", "completed"),
        (r"^(?:has|have|had) passed (.+)$", "passed"),
        (r"^passed (.+)$", "passed"),
        (r"^(?:has|have|had) received (.+)$", "received"),
        (r"^received (.+)$", "received"),
        (r"^(?:has|have|had) been awarded (.+)$", "awarded"),
        (r"^(?:is|are|was|were) awarded (.+)$", "awarded"),
        (r"^awarded (.+)$", "awarded"),
        (r"^(?:is|are|was|were) qualified for (.+)$", "qualified_for"),
        (r"^qualified for (.+)$", "qualified_for"),
        (r"^(?:is|are|was|were) eligible for (.+)$", "eligible_for"),
        (r"^eligible for (.+)$", "eligible_for"),
        (r"^qualif(?:y|ies) for (.+)$", "qualifies_for"),
        (r"^(?:can|could) teach (.+)$", "can_teach"),
        (r"^(?:can|could) supervise (.+)$", "can_supervise"),
        (r"^(?:can|could) serve on (.+)$", "can_serve_on"),
        (r"^(?:can|could) propose (.+)$", "can_propose"),
        (r"^holds? (.+)$", "holds"),
        (r"^maintains? (?:a )?gpa (?:above|of) (.+)$", "maintains_gpa"),
        (r"^follows? (.+)$", "follows"),
        (r"^(?:is|are|was|were) (.+)$", ""),
        (r"^(?:must be|needs? to be) (.+)$", ""),
        (r"^needs? (?:to pass|to complete|to receive)?\s*(.+)$", "needs"),
    )

    for regex, prefix in patterns:
        match = re.match(regex, clause)
        if not match:
            continue
        obj = _normalize_object(match.group(1))
        if not obj:
            return prefix
        return f"{prefix}_{obj}" if prefix else obj

    return _normalize_object(clause)


def _normalize_object(text: str) -> str:
    """Normalize noun/adjective objects into slug-safe predicate suffixes."""

    value = _canonicalize_clause(text)
    value = _SYNONYMS.get(value, value)
    if value in _SYNONYMS:
        value = _SYNONYMS[value]
    tokens = [token for token in _WORD_RE.findall(value) if token not in _FILLER_WORDS]
    return "_".join(tokens)


def _clean_clause(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().strip("\"'")
    text = _TRAILING_PUNCT_RE.sub("", text)
    text = text.replace("’", "'")
    return re.sub(r"\s+", " ", text).lower()


def _canonicalize_clause(text: str) -> str:
    text = re.sub(r"^(?:a|an|the)\s+", "", text.strip())
    text = re.sub(r"\s+(?:is|are|was|were)\s+true$", "", text)
    return text.strip()


def _slugify(text: str) -> str:
    return "_".join(_WORD_RE.findall(text))
