"""Whole-theory single-pass translation: NL problem -> consistent FOL.

One LLM call reads every premise + the question + any MCQ options together and
emits a single predicate dictionary plus one FOL string per premise / claim /
option, all over the shared vocabulary. Because the model sees the whole theory
at once and declares each predicate once, the same relation keeps the same name
across premises and the question — which the per-premise decomposition pipeline
could not guarantee. The emitted strings are parsed by ``parse_fol_string`` into
the same ``FOLNode`` AST the solver already consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from exact.type1.ast.nodes import FOLNode
from exact.type1.parser.fol_string_parser import FOLStringParseError, parse_fol_string
from exact.type1.parser.schemas import ParserResult
from exact.type1.prompts import get_system_prompt_theory_translate

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient


class PredicateDecl(ParserResult):
    """One predicate declared once and reused across the whole theory."""

    name: str
    arity: int
    gloss: str = ""


class OptionFOL(ParserResult):
    """One MCQ option translated to a FOL claim string."""

    label: str
    fol: str


class TranslatedTheory(ParserResult):
    """Raw LLM output: predicate dictionary + FOL strings for the whole theory."""

    predicates: list[PredicateDecl] = []
    premises: list[str] = []
    question_format: Literal["polar", "mcq", "open_wh"] = "polar"
    claim: str | None = None
    options: list[OptionFOL] = []


@dataclass
class TheoryTranslation:
    """Parsed translation: FOLNode ASTs plus diagnostics for unparsable strings."""

    premise_trees: list[FOLNode]
    premise_strings: list[str]
    # original 0-based premise index for each entry in premise_trees, so a tree
    # that fails to parse does not shift premises_used indices.
    premise_index_map: list[int]
    question_format: str
    claim_tree: FOLNode | None
    claim_string: str | None
    option_trees: dict[str, FOLNode]
    predicates: list[PredicateDecl]
    issues: list[str] = field(default_factory=list)


class TheoryTranslator:
    """Translate a full Type 1 problem to FOL in one LLM call."""

    def __init__(self, client: ParserClient, max_tokens: int = 2048) -> None:
        self.client = client
        self.max_tokens = max_tokens

    async def translate(
        self,
        premises: list[str],
        question: str,
        options: dict[str, str] | None = None,
        feedback: str | None = None,
        temperature: float = 0.0,
    ) -> TheoryTranslation:
        user = _render_problem(premises, question, options)
        if feedback:
            user += (
                "\n\nYOUR PREVIOUS TRANSLATION HAD PROBLEMS — fix them and re-emit the "
                "FULL JSON:\n" + feedback
            )
        raw = await self.client.parse_as(
            [
                {"role": "system", "content": get_system_prompt_theory_translate()},
                {"role": "user", "content": user},
            ],
            TranslatedTheory,
            max_tokens=self.max_tokens,
            temperature=temperature,
        )
        return _parse_translation(raw)


_LABEL_RE = re.compile(r"^\s*([A-Ea-e])\b")


def _normalize_label(label: str) -> str:
    """Reduce an option label to its bare letter (A-E) when present.

    The translator LLM sometimes echoes the whole option line as the label; the
    solver and scorer expect just the letter.
    """
    match = _LABEL_RE.match(label)
    return match.group(1).upper() if match else label.strip()


def _render_problem(
    premises: list[str], question: str, options: dict[str, str] | None
) -> str:
    lines = ["PREMISES:"]
    lines += [f"{i}. {p}" for i, p in enumerate(premises, start=1)]
    lines += ["", "QUESTION:", question]
    if options:
        lines += ["", "OPTIONS:"]
        lines += [f"{label}. {text}" for label, text in options.items()]
    return "\n".join(lines)


def _parse_translation(raw: TranslatedTheory) -> TheoryTranslation:
    issues: list[str] = []

    premise_trees: list[FOLNode] = []
    premise_strings: list[str] = []
    premise_index_map: list[int] = []
    for index, fol in enumerate(raw.premises):
        try:
            tree = parse_fol_string(fol)
        except FOLStringParseError as exc:
            issues.append(f"PREMISE_FOL_PARSE_FAILED: premise {index + 1}: {exc}")
            continue
        premise_trees.append(tree)
        premise_strings.append(fol)
        premise_index_map.append(index)

    claim_tree: FOLNode | None = None
    claim_string: str | None = raw.claim
    if raw.claim:
        try:
            claim_tree = parse_fol_string(raw.claim)
        except FOLStringParseError as exc:
            issues.append(f"CLAIM_FOL_PARSE_FAILED: {exc}")

    option_trees: dict[str, FOLNode] = {}
    for option in raw.options:
        if not option.fol:
            continue
        label = _normalize_label(option.label)
        try:
            option_trees[label] = parse_fol_string(option.fol)
        except FOLStringParseError as exc:
            issues.append(f"OPTION_FOL_PARSE_FAILED: {label}: {exc}")

    return TheoryTranslation(
        premise_trees=premise_trees,
        premise_strings=premise_strings,
        premise_index_map=premise_index_map,
        question_format=raw.question_format,
        claim_tree=claim_tree,
        claim_string=claim_string,
        option_trees=option_trees,
        predicates=raw.predicates,
        issues=issues,
    )
