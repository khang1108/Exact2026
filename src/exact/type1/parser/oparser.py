"""Role-aware MCQ option parser (deterministic-first).

``OParser`` classifies each option into one of nine roles using deterministic
regex rules in a fixed priority order, then realizes subjectless fragments into
declarative claims by recovering the subject from the question stem. An LLM
repair call is used only for fragments the deterministic templates cannot
resolve. No FOL is produced here — ``ClaimParser`` compiles selectable claims.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from exact.type1.parser.schemas import (
    FragmentRealizationResult,
    OptionClaim,
    OptionParseBundle,
    OptionRole,
)
from exact.type1.prompts import get_system_prompt_fragment_realization

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient

ChatMessage = dict[str, Any]

# --- deterministic role signals -------------------------------------------
_RAW_FOL_RE = re.compile(r"[∀∃→↔¬∧∨]|\bForAll\s*\(|\bExists\s*\(")
_PREMISE_REF_RE = re.compile(
    r"^premises?\s+\d+(?:\s*(?:,|and)\s*\d+)*\.?$", re.IGNORECASE
)
# A bare polarity token: "Yes." / "No," / "Uncertain" / "Cannot be determined".
# "No one is qualified." does NOT match (token is followed by a word, not [.,]).
_YNU_RE = re.compile(
    r"^(yes|no|uncertain|unknown)\s*[.,]|^(yes|no|uncertain|unknown)\s*$"
    r"|^cannot be determined\b",
    re.IGNORECASE,
)
_NONE_OF_ABOVE_RE = re.compile(r"\bnone of the (?:above|following)\b", re.IGNORECASE)
_MODAL_START_RE = re.compile(
    r"^(can|cannot|may|must|should|required|allowed|prohibited)\b", re.IGNORECASE
)
_PREDICATE_FRAGMENT_START_RE = re.compile(
    r"^(eligible|needs|need|only|requires|require|able|unable|ineligible|"
    r"prohibited from|allowed to)\b",
    re.IGNORECASE,
)
_CONJUNCTIVE_RE = re.compile(r"\bboth\b.*\band\b", re.IGNORECASE)
_LABEL_COMBO_RE = re.compile(r"\bboth\s+[A-E]\s+and\s+[A-E]\b", re.IGNORECASE)
_VAGUE_RE = re.compile(r"^(it depends|maybe|not sure|unclear)\b", re.IGNORECASE)

# --- subject recovery from the question stem ------------------------------
_SUBJECT_PATTERNS = [
    re.compile(r"\btrue\s+(?:about|for|of)\s+([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)?)", re.IGNORECASE),
    re.compile(r"\b(?:about|regarding|for|of)\s+([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)?)"),
    re.compile(r"\b(?:can|is|does|will|should|may|must)\s+([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)?)"),
]

_SELECTABLE_ROLES = frozenset({"FULL_CLAIM", "CONJUNCTIVE_CLAIM"})
_FRAGMENT_ROLES = frozenset({"SUBJECTLESS_MODAL_FRAGMENT", "PREDICATE_FRAGMENT"})


def classify(text: str) -> OptionRole:
    """Classify one normalized option text into a role (deterministic priority)."""

    t = text.strip()
    if not t:
        return "UNKNOWN"
    if _RAW_FOL_RE.search(t):
        return "RAW_FOL"
    if _PREMISE_REF_RE.match(t):
        return "PREMISE_REFERENCE"
    if _YNU_RE.match(t):
        return "YNU_ANSWER"
    if _NONE_OF_ABOVE_RE.search(t):
        return "NONE_OF_ABOVE"
    if _LABEL_COMBO_RE.search(t) or _VAGUE_RE.match(t):
        return "UNKNOWN"
    if _MODAL_START_RE.match(t):
        return "SUBJECTLESS_MODAL_FRAGMENT"
    if _PREDICATE_FRAGMENT_START_RE.match(t):
        return "PREDICATE_FRAGMENT"
    if _CONJUNCTIVE_RE.search(t):
        return "CONJUNCTIVE_CLAIM"
    return "FULL_CLAIM"


def _ynu_value(text: str) -> str:
    head = text.strip().lower()
    if head.startswith("yes"):
        return "yes"
    if head.startswith("no"):
        return "no"
    if head.startswith("uncertain") or head.startswith("unknown") or head.startswith("cannot"):
        return "uncertain"
    return "none"


def _premise_indices(text: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", text))


def recover_subject(stem: str) -> str | None:
    """Recover a proper-noun subject from the question stem, if present."""

    for pattern in _SUBJECT_PATTERNS:
        match = pattern.search(stem)
        if match:
            return match.group(1).strip()
    return None


def realize_fragment(role: str, fragment: str, subject: str) -> str:
    """Deterministically attach a recovered subject to a fragment claim."""

    frag = fragment.strip().rstrip(".")
    if role == "SUBJECTLESS_MODAL_FRAGMENT":
        # "Can be a research mentor" → "Professor Kim can be a research mentor"
        first, _, rest = frag.partition(" ")
        return f"{subject} {first.lower()}{(' ' + rest) if rest else ''}".strip() + "."
    # PREDICATE_FRAGMENT: "Eligible for X" → "Professor Kim is eligible for X"
    return f"{subject} is {frag[0].lower()}{frag[1:]}.".strip()


class OParser:
    """Interpret each MCQ option into a role-tagged OptionClaim."""

    def __init__(self, client: ParserClient) -> None:
        self.client = client

    async def parse_options(
        self,
        question_stem: str,
        options: dict[str, str],
        *,
        marker_style: str = "none",
        option_count: int | None = None,
        extraction_diagnostics: tuple[str, ...] = (),
    ) -> OptionParseBundle:
        """Classify and realize every option, returning an OptionParseBundle."""

        subject = recover_subject(question_stem)

        claims: list[OptionClaim] = []
        # Fragments that deterministic recovery cannot realize → batched LLM repair.
        pending_llm: list[tuple[int, str, str]] = []  # (index, role, normalized_text)

        for label, raw in options.items():
            normalized = raw.strip()
            role = classify(normalized)
            claim, needs_llm = self._build_claim(label, raw, normalized, role, subject)
            if needs_llm:
                pending_llm.append((len(claims), role, normalized))
            claims.append(claim)

        if pending_llm:
            results = await self._repair_fragments(question_stem, pending_llm)
            for (index, _role, _text), result in zip(pending_llm, results):
                claims[index] = self._apply_repair(claims[index], result)

        role_distribution: dict[str, int] = {}
        for claim in claims:
            role_distribution[claim.role] = role_distribution.get(claim.role, 0) + 1

        issues = self._verify(claims, option_count)

        return OptionParseBundle(
            options=tuple(claims),
            marker_style=marker_style,
            option_count=option_count if option_count is not None else len(claims),
            role_distribution=role_distribution,
            verified=not issues,
            issues=tuple(issues),
            extraction_diagnostics=extraction_diagnostics,
        )

    def _build_claim(
        self,
        label: str,
        raw: str,
        normalized: str,
        role: OptionRole,
        subject: str | None,
    ) -> tuple[OptionClaim, bool]:
        """Build an OptionClaim; return (claim, needs_llm_repair)."""

        diagnostics: list[str] = [f"ROLE_{role}"]

        if role in _SELECTABLE_ROLES:
            return self._claim(label, raw, normalized, role, claim_text=normalized,
                               is_selectable=True, diagnostics=diagnostics), False

        if role == "RAW_FOL":
            diagnostics.append("RAW_FOL_UNSUPPORTED")
            return self._claim(label, raw, normalized, role, raw_fol=normalized,
                               diagnostics=diagnostics), False

        if role == "PREMISE_REFERENCE":
            diagnostics.append("PREMISE_REFERENCE_UNSUPPORTED")
            return self._claim(label, raw, normalized, role,
                               premise_indices=_premise_indices(normalized),
                               diagnostics=diagnostics), False

        if role == "YNU_ANSWER":
            return self._claim(label, raw, normalized, role,
                               ynu_value=_ynu_value(normalized),
                               diagnostics=diagnostics), False

        if role == "NONE_OF_ABOVE":
            diagnostics.append("NONE_OF_ABOVE_SOLVER_POSTPROCESS_REQUIRED")
            return self._claim(label, raw, normalized, role, diagnostics=diagnostics), False

        if role in _FRAGMENT_ROLES:
            if subject is not None:
                realized = realize_fragment(role, normalized, subject)
                diagnostics.append("SUBJECT_RECOVERED")
                return self._claim(label, raw, normalized, role, claim_text=realized,
                                   is_selectable=True, diagnostics=diagnostics), False
            # No deterministic subject → try LLM repair.
            diagnostics.append("SUBJECT_MISSING")
            return self._claim(label, raw, normalized, role, diagnostics=diagnostics), True

        # UNKNOWN
        diagnostics.append("OPTION_UNSUPPORTED")
        return self._claim(label, raw, normalized, "UNKNOWN", diagnostics=diagnostics), False

    @staticmethod
    def _claim(
        label: str,
        raw: str,
        normalized: str,
        role: str,
        *,
        claim_text: str | None = None,
        raw_fol: str | None = None,
        premise_indices: tuple[int, ...] = (),
        ynu_value: str = "none",
        is_selectable: bool = False,
        diagnostics: list[str] | None = None,
    ) -> OptionClaim:
        return OptionClaim(
            label=label,
            original_text=raw,
            normalized_text=normalized,
            role=role,
            claim_text=claim_text,
            raw_fol=raw_fol,
            premise_indices=premise_indices,
            ynu_value=ynu_value,
            is_selectable=is_selectable,
            diagnostics=tuple(diagnostics or ()),
        )

    async def _repair_fragments(
        self,
        question_stem: str,
        pending: list[tuple[int, str, str]],
    ) -> list[FragmentRealizationResult]:
        messages_batch = [self._repair_messages(question_stem, text) for _, _, text in pending]
        return await self.client.parse_many_as(messages_batch, FragmentRealizationResult)

    def _apply_repair(
        self,
        claim: OptionClaim,
        result: FragmentRealizationResult,
    ) -> OptionClaim:
        diagnostics = list(claim.diagnostics)
        if result.subject_found and result.claim_text:
            diagnostics.append("SUBJECT_RECOVERED_LLM")
            return OptionClaim(
                label=claim.label,
                original_text=claim.original_text,
                normalized_text=claim.normalized_text,
                role=claim.role,
                claim_text=result.claim_text,
                raw_fol=None,
                premise_indices=(),
                ynu_value="none",
                is_selectable=True,
                diagnostics=tuple(diagnostics),
            )
        # Unresolved → demote to UNKNOWN, not sent to ClaimParser.
        diagnostics.append("FRAGMENT_REALIZATION_FAILED")
        return OptionClaim(
            label=claim.label,
            original_text=claim.original_text,
            normalized_text=claim.normalized_text,
            role="UNKNOWN",
            claim_text=None,
            raw_fol=None,
            premise_indices=(),
            ynu_value="none",
            is_selectable=False,
            diagnostics=tuple(diagnostics),
        )

    def _verify(self, claims: list[OptionClaim], option_count: int | None) -> list[str]:
        issues: list[str] = []
        count = option_count if option_count is not None else len(claims)
        if count not in (3, 4):
            issues.append(f"OPTION_COUNT_UNSUPPORTED: {count}")
        labels = [c.label for c in claims]
        if len(set(labels)) != len(labels):
            issues.append("OPTION_DUPLICATE_LABEL")
        for c in claims:
            if c.is_selectable and not c.claim_text:
                issues.append(f"OPTION_SELECTABLE_WITHOUT_CLAIM: {c.label}")
            if c.role in ("YNU_ANSWER", "NONE_OF_ABOVE") and c.claim_text is not None:
                issues.append(f"OPTION_SPECIAL_HAS_CLAIM: {c.label}")
        return issues

    def _repair_messages(self, question_stem: str, fragment: str) -> list[ChatMessage]:
        return [
            {"role": "system", "content": get_system_prompt_fragment_realization().strip()},
            {"role": "user", "content": f'Stem: "{question_stem}"\nFragment: "{fragment}"'},
        ]
