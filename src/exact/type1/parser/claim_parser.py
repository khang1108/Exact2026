"""Claim → FOL bridge for the question side.

``ClaimParser`` encapsulates the proven conclusion path that lived inline in the
pipeline: parse each claim sentence through the recursive ``FOLParser`` and then
canonicalize its predicate names against the premise schema so the solver shares
one vocabulary with the premises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exact.type1.ast.nodes import FOLNode
    from exact.type1.parser.parser import FOLParser
    from exact.type1.parser.schemas import PremiseSchema


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
        raw = await self.fol_parser.parse_many(claim_texts)
        return schema.canonicalize(raw)
