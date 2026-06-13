"""High-level orchestration for parsing declarative Type 1 premises."""

from __future__ import annotations

import re

from dataclasses import dataclass
from exact.type1.ast.nodes import FOLNode 
from exact.type1.parser.parser import FOLParser
from exact.type1.parser.schemas import (
    PremiseParseBundle,
    PremiseSchema,
    _OPTION_LINE,
    _INTERROGATIVE_START,
    _collect_predicates_in_order
)

class PremiseParser:
    """Parse and verify declarative premises using a low-level ``FOLParser``."""

    def __init__(self, fol_parser: FOLParser) -> None:
        self.fol_parser = fol_parser

    async def parse_premises(self, premises: list[str]) -> PremiseParseBundle:
        """Normalize, parse, canonicalize, and verify declarative premises."""

        normalized = [_normalize_premise(premise) for premise in premises]
        normalized = [premise for premise in normalized if premise]
        if not normalized:
            raise ValueError("premises must contain at least one non-empty declarative statement")

        invalid = [premise for premise in normalized if not _is_declarative(premise)]
        if invalid:
            raise ValueError(
                "PremiseParser accepts declarative premises only; rejected: "
                + "; ".join(repr(premise) for premise in invalid)
            )

        draft_trees = await self.fol_parser.parse_many(normalized)
        schema = PremiseSchema.from_trees(draft_trees)
        trees, renames = schema.canonicalize(draft_trees)
        issues = _verify_bundle(normalized, trees, schema)

        return PremiseParseBundle(
            premises=normalized,
            draft_trees=draft_trees,
            schema=schema,
            trees=trees,
            predicate_renames=renames,
            verified=not issues,
            verification_issues=tuple(issues),
        )

def _is_declarative(premise: str) -> bool:
    """Reject obvious questions, commands, and multiple-choice option lines."""

    if premise.endswith("?") or _OPTION_LINE.match(premise):
        return False
    return _INTERROGATIVE_START.match(premise) is None

def _normalize_premise(premise: str) -> str:
    """Normalize whitespace without changing premise meaning."""

    return re.sub(r"\s+", " ", premise).strip()

def _verify_bundle(
    premises: list[str],
    trees: list[FOLNode],
    schema: PremiseSchema,
) -> list[str]:
    issues: list[str] = []
    if len(premises) != len(trees):
        issues.append(
            f"expected one AST per premise, received {len(trees)} ASTs for {len(premises)} premises"
        )

    for index, tree in enumerate(trees):
        for name, arity in _collect_predicates_in_order(tree):
            if not schema.contains(name, arity):
                issues.append(
                    f"premise {index + 1} uses predicate {name}/{arity} outside the schema"
                )
    return issues
