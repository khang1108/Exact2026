"""Claim → FOL bridge for the question side.

``ClaimParser`` pre-processes each claim text (possessive normalization,
IF_ALL_THEN_ALL meta-implication detection) before delegating to the
recursive ``FOLParser``, then canonicalizes predicates against the premise
schema so the solver shares one vocabulary with the premises.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from exact.type1.ast.nodes import LogicalNode

if TYPE_CHECKING:
    from exact.type1.ast.nodes import FOLNode
    from exact.type1.parser.parser import FOLParser
    from exact.type1.parser.schemas import PremiseSchema

# "if all/every X ..., then all/every Y ..."  →  meta-implication
_IF_ALL_THEN_ALL_RE = re.compile(
    r"^if\s+(?:all|every)\s+(.+?)\s*,\s*then\s+(?:all|every)\s+(.+)$",
    re.IGNORECASE,
)

# "John's GPA" / "Alice's score" → "GPA of John" / "score of Alice"
_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][a-zA-Z]*)(?:'s|'s)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)"
)


def _normalize_possessives(text: str) -> str:
    """Rewrite 'Owner's Attr' → 'Attr of Owner' so the atomic parser sees an attribute."""
    def _replace(m: re.Match) -> str:
        owner = m.group(1)
        attr = m.group(2)
        return f"{attr} of {owner}"
    return _POSSESSIVE_RE.sub(_replace, text)


class ClaimParser:
    """Translate claim texts to canonicalized FOL using the shared FOLParser."""

    def __init__(self, fol_parser: FOLParser) -> None:
        self.fol_parser = fol_parser

    async def parse_claims(
        self,
        claim_texts: list[str],
        schema: PremiseSchema,
    ) -> tuple[list[FOLNode], list[dict[str, object]]]:
        """Parse claim texts to FOL and rename predicates to schema canonicals."""

        if not claim_texts:
            return [], []

        # Build a flat parse batch, tracking which inputs need meta-implication assembly.
        # plan entry: (is_meta, flat_idx_antecedent, flat_idx_consequent_or_-1)
        plan: list[tuple[bool, int, int]] = []
        flat_texts: list[str] = []

        for text in claim_texts:
            normalized = _normalize_possessives(text.strip())
            m = _IF_ALL_THEN_ALL_RE.match(normalized)
            if m:
                i = len(flat_texts)
                flat_texts.append("all " + m.group(1).strip())
                flat_texts.append("all " + m.group(2).strip())
                plan.append((True, i, i + 1))
            else:
                i = len(flat_texts)
                flat_texts.append(normalized)
                plan.append((False, i, -1))

        raw = await self.fol_parser.parse_many(flat_texts)

        results: list[FOLNode] = []
        for is_meta, i1, i2 in plan:
            if is_meta:
                results.append(LogicalNode(operator="IMPLIES", left=raw[i1], right=raw[i2]))
            else:
                results.append(raw[i1])

        return schema.canonicalize(results)
