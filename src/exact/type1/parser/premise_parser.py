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
)

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient


class PremiseParser:
    """Parse and verify declarative premises using frame decomposition + atomic parsing."""

    def __init__(
        self,
        fol_parser: FOLParser,
        frame_parser: PremiseFrameParser | None = None,
        frame_compiler: PremiseFrameCompiler | None = None,
    ) -> None:
        if (frame_parser is None) != (frame_compiler is None):
            raise ValueError("frame_parser and frame_compiler must be configured together")
        if frame_parser is None and frame_compiler is None:
            client = getattr(fol_parser, "client", None)
            if client is not None:
                frame_parser = PremiseFrameParser(client)
                frame_compiler = PremiseFrameCompiler(fol_parser)

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

        if self.frame_parser is not None and self.frame_compiler is not None:
            frames = await self.frame_parser.parse_many(normalized)
            draft_trees = await self.frame_compiler.compile_many(normalized, frames)
        else:
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


_ONLY_IF_RE = re.compile(r"\bonly\s+(?:if|when)\b", re.IGNORECASE)
_NON_BLOCKING_SCHEMA_DIAGNOSTICS = ("SCHEMA_SIMILAR_PREDICATES:",)


def _verify_bundle(
    premises: list[str],
    trees: list[FOLNode],
    schema: PremiseSchema,
) -> list[str]:
    issues = [
        diagnostic
        for diagnostic in schema.diagnostics
        if not diagnostic.startswith(_NON_BLOCKING_SCHEMA_DIAGNOSTICS)
    ]
    if len(premises) != len(trees):
        issues.append(
            f"expected one AST per premise, received {len(trees)} ASTs for {len(premises)} premises"
        )

    for index, premise in enumerate(premises):
        if _ONLY_IF_RE.search(premise):
            issues.append(
                f"ONLY_IF_DIRECTION_CHECK: premise {index + 1} contains 'only if/when' — "
                f"confirm FOL has the result as left_operand and the condition as right_operand"
            )

    return issues
