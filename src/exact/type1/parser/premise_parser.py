"""High-level orchestration for parsing declarative Type 1 premises."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from exact.type1.ast.nodes import FOLNode
from exact.type1.parser.frame_parser import PremiseFrameCompiler, PremiseFrameParser
from exact.type1.parser.parser import FOLParser
from exact.type1.parser.schemas import (
    PremiseParseBundle,
    PremiseSchema,
    _INTERROGATIVE_START,
    _OPTION_LINE,
    _collect_predicates_in_order,
)

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient


class PremiseParser:
    """Parse and verify declarative premises using frame decomposition + atomic parsing."""

    def __init__(
        self,
        fol_parser: FOLParser,
        frame_parser: PremiseFrameParser,
        frame_compiler: PremiseFrameCompiler,
    ) -> None:
        self.fol_parser = fol_parser
        self.frame_parser = frame_parser
        self.frame_compiler = frame_compiler

    @classmethod
    def from_parser_client(cls, client: ParserClient) -> PremiseParser:
        """Construct all parser components from one shared client."""
        fol_parser = FOLParser(client)
        frame_parser = PremiseFrameParser(client)
        frame_compiler = PremiseFrameCompiler(fol_parser)
        return cls(fol_parser, frame_parser, frame_compiler)

    async def parse_premises(self, premises: list[str]) -> PremiseParseBundle:
        """Normalize, frame-decompose, compile, canonicalize, and verify premises."""

        normalized = [_normalize_premise(p) for p in premises]
        normalized = [p for p in normalized if p]
        if not normalized:
            raise ValueError("premises must contain at least one non-empty declarative statement")

        invalid = [p for p in normalized if not _is_declarative(p)]
        if invalid:
            raise ValueError(
                "PremiseParser accepts declarative premises only; rejected: "
                + "; ".join(repr(p) for p in invalid)
            )

        frames = await self.frame_parser.parse_many(normalized)
        draft_trees = await self.frame_compiler.compile_many(normalized, frames)
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
