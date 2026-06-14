"""High-level orchestration for parsing one question into a verified QuerySpec.

Mirrors ``PremiseParser`` on the question side:

    QuestionSideParser
      ├── QParser        question                 → QuestionFrameResult
      ├── OParser        stem + options           → OptionParseBundle
      ├── ClaimParser    claim texts + schema     → FOLNode[]
      └── query_verifier QuerySpec                → supported / issues
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from exact.type1.parser import query_verifier
from exact.type1.parser.claim_parser import ClaimParser
from exact.type1.parser.oparser import OParser
from exact.type1.parser.options import MCQExtraction, extract_mcq
from exact.type1.parser.parser import FOLParser
from exact.type1.parser.qparser import QParser
from exact.type1.parser.schemas import (
    OptionClaim,
    OptionParseBundle,
    QuerySpec,
    QuestionParseBundle,
)

if TYPE_CHECKING:
    from exact.type1.ast.nodes import FOLNode
    from exact.type1.parser.client import ParserClient
    from exact.type1.parser.schemas import PremiseSchema


class QuestionSideParser:
    """Classify a question, interpret its options, and compile testable claims."""

    def __init__(
        self,
        qparser: QParser,
        oparser: OParser,
        claim_parser: ClaimParser,
    ) -> None:
        self.qparser = qparser
        self.oparser = oparser
        self.claim_parser = claim_parser

    @classmethod
    def from_parser_client(cls, client: ParserClient) -> QuestionSideParser:
        """Construct all question-side components from one shared client."""
        return cls(
            qparser=QParser(client),
            oparser=OParser(client),
            claim_parser=ClaimParser(FOLParser(client)),
        )

    async def parse_question(
        self,
        question: str,
        options: dict[str, str] | None,
        schema: PremiseSchema,
        *,
        extraction: MCQExtraction | None = None,
    ) -> QuestionParseBundle:
        """Classify the question and build a verified QuerySpec."""

        frame = await self.qparser.classify(question)

        main_claim_fol: FOLNode | None = None
        claim_renames: list[dict[str, object]] = []
        option_claims: tuple[OptionClaim, ...] = ()
        option_bundle: OptionParseBundle | None = None
        option_issues: list[str] = []

        # Options-presence is authoritative for MCQ — the classifier sometimes
        # mislabels an option-bearing question as polar.
        is_mcq = bool(options)
        question_format = "mcq" if is_mcq else frame.question_format
        solver_mode = frame.solver_mode

        if is_mcq:
            assert options is not None
            if extraction is None:
                extraction = extract_mcq(question)
            stem = extraction.stem if extraction.options else _strip_options(question)
            option_bundle = await self.oparser.parse_options(
                stem,
                options,
                marker_style=extraction.marker_style,
                option_count=len(options),
                extraction_diagnostics=extraction.diagnostics,
            )
            option_claims = await self._fill_option_fol(option_bundle.options, schema)
            option_issues = list(option_bundle.issues)
            # YNU-as-options: detect from option content even if the classifier
            # (which may not see separately-supplied options) missed it.
            if _looks_ynu(option_claims):
                solver_mode = "ynu_mapped"
            # ynu_mapped tests the stem proposition, not the options themselves.
            if solver_mode == "ynu_mapped" and frame.claim_text:
                fols, claim_renames = await self.claim_parser.parse_claims(
                    [frame.claim_text], schema
                )
                main_claim_fol = fols[0] if fols else None
        elif frame.question_format == "polar" and frame.claim_text:
            fols, claim_renames = await self.claim_parser.parse_claims(
                [frame.claim_text], schema
            )
            main_claim_fol = fols[0] if fols else None

        supported, issues = query_verifier.verify(
            question_format=question_format,
            solver_mode=solver_mode,
            main_claim_fol_present=main_claim_fol is not None,
            option_claims=option_claims,
        )

        spec = QuerySpec(
            question_format=question_format,
            solver_mode=solver_mode,
            can_interpretation=frame.can_interpretation,
            main_claim_text=frame.claim_text,
            negate_claim=frame.negate_claim,
            option_claims=option_claims,
            supported=supported,
            issues=tuple(issues + option_issues),
        )
        return QuestionParseBundle(
            question=question,
            spec=spec,
            main_claim_fol=main_claim_fol,
            claim_renames=claim_renames,
            option_bundle=option_bundle,
        )

    async def _fill_option_fol(
        self,
        claims: tuple[OptionClaim, ...],
        schema: PremiseSchema,
    ) -> tuple[OptionClaim, ...]:
        """Translate selectable option claim texts to canonicalized FOL."""

        solvable = [
            (i, c)
            for i, c in enumerate(claims)
            if c.is_selectable and c.claim_text
        ]
        if not solvable:
            return tuple(claims)

        texts = [c.claim_text for _, c in solvable if c.claim_text]
        fols, _ = await self.claim_parser.parse_claims(texts, schema)

        filled = list(claims)
        for (index, claim), fol in zip(solvable, fols):
            filled[index] = _with_fol(claim, fol)
        return tuple(filled)


def _with_fol(claim: OptionClaim, fol: FOLNode) -> OptionClaim:
    return OptionClaim(
        label=claim.label,
        original_text=claim.original_text,
        normalized_text=claim.normalized_text,
        role=claim.role,
        claim_text=claim.claim_text,
        raw_fol=claim.raw_fol,
        premise_indices=claim.premise_indices,
        ynu_value=claim.ynu_value,
        is_selectable=claim.is_selectable,
        fol=fol,
        diagnostics=claim.diagnostics,
    )


def _looks_ynu(option_claims: tuple[OptionClaim, ...]) -> bool:
    """True when the options are themselves Yes/No/Uncertain answers."""

    return sum(1 for c in option_claims if c.role == "YNU_ANSWER") >= 2


def _strip_options(question: str) -> str:
    """Return the question stem with any embedded A./A) option lines removed."""

    lines = question.split("\n")
    stem_lines: list[str] = []
    for line in lines:
        if re.match(r"^\s*[A-E][.)]\s+", line):
            break
        stem_lines.append(line)
    return "\n".join(stem_lines).strip() or question.strip()
