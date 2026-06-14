"""ClaimParser v2 — schema-aware claim → FOL bridge for the question side.

Stage 1 (``ClaimParser = FOLParser + schema.canonicalize``) sent every claim,
however irregular, straight into the recursive FOL parser and then renamed
predicates loosely. That produced pronoun leaks ("He can teach"), possessive
entities ("JohnsGPA"), and semantic drift (Requires → QualifiesFor).

v2 mirrors the premise side with five focused components:

    ClaimParser  (orchestrator)
      ├── ClaimNormalizer          pronoun / possessive / GPA / unicode
      ├── ClaimFrameRouter         detect high-risk forms before free FOL
      ├── SchemaGuidedFrameCompiler compile each frame (deterministic first)
      ├── SafeClaimCanonicalizer   strict, family/arity-guarded renaming
      └── ClaimVerifier            surface drift instead of hiding it

Only ``SIMPLE_ATOMIC`` / ``OBJECT_MODAL`` frames fall back to the free
``FOLParser``; structured frames (meta-implication, requirement/purpose, GPA)
are compiled deterministically so they cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from exact.type1.ast.nodes import (
    AtomicNode,
    ComparisonNode,
    FunctionTerm,
    LogicalNode,
    NumericTerm,
    QuantifiedNode,
)
from exact.type1.models.schemas import Predicate
from exact.type1.parser.schemas import _best_schema_match, _predicate_family

if TYPE_CHECKING:
    from exact.type1.ast.nodes import FOLNode
    from exact.type1.parser.parser import FOLParser
    from exact.type1.parser.schemas import PredicateSignature, PremiseSchema


# ---------------------------------------------------------------------------
# Claim frame
# ---------------------------------------------------------------------------

ClaimKind = str  # IF_ALL_THEN_ALL | REQUIREMENT | GPA_NUMERIC | OBJECT_MODAL | SIMPLE_ATOMIC


@dataclass
class ClaimFrame:
    """One claim routed to a compilation strategy with its extracted slots."""

    kind: ClaimKind
    text: str
    diagnostics: list[str] = field(default_factory=list)
    # IF_ALL_THEN_ALL
    antecedent: str | None = None
    consequent: str | None = None
    # REQUIREMENT ("X needs R to qualify for Y")
    subject: str | None = None
    requirement: str | None = None
    target: str | None = None
    # GPA_NUMERIC
    gpa_owner: str | None = None
    gpa_operator: str | None = None
    gpa_value: float | None = None


# ---------------------------------------------------------------------------
# 1. ClaimNormalizer
# ---------------------------------------------------------------------------

_UNICODE_APOSTROPHES = {"’": "'", "‘": "'", "ʼ": "'", "′": "'"}

_PRONOUN_RE = re.compile(r"\b(he|she|they|it|his|her|its|their)\b", re.IGNORECASE)
_POSSESSIVE_PRONOUNS = frozenset({"his", "her", "its", "their"})

# "John's GPA" / "Professor John's publications" → "GPA of John".
# The attribute is 1-2 words but a trailing verb/copula (is/are/has...) is
# excluded so "John's GPA is 3.8" → "GPA of John is 3.8" (not "GPA is of John").
_ATTR_VERB_STOP = r"(?:is|are|was|were|has|have|had|will|can|must|should|may|does|do|qualifies|meets)"
_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)?)'s\s+"
    rf"([A-Za-z]+(?:\s+(?!{_ATTR_VERB_STOP}\b)[A-Za-z]+)?)"
)


class ClaimNormalizer:
    """Deterministic surface cleanup applied before routing or FOL.

    Resolves pronouns to the question's target entity, rewrites possessive
    attributes into "Attr of Owner" form, and normalizes unicode apostrophes.
    """

    def normalize(self, text: str, target_entity: str | None) -> tuple[str, list[str]]:
        diagnostics: list[str] = []
        out = text.strip()
        for uni, ascii_ in _UNICODE_APOSTROPHES.items():
            if uni in out:
                out = out.replace(uni, ascii_)
                diagnostics.append("UNICODE_APOSTROPHE_NORMALIZED")

        if target_entity:
            resolved = self._resolve_pronouns(out, target_entity)
            if resolved != out:
                diagnostics.append("PRONOUN_RESOLVED")
                out = resolved

        possessive = self._normalize_possessives(out)
        if possessive != out:
            diagnostics.append("POSSESSIVE_NORMALIZED")
            out = possessive

        return out, diagnostics

    @staticmethod
    def _resolve_pronouns(text: str, entity: str) -> str:
        def _replace(m: re.Match) -> str:
            token = m.group(1)
            if token.lower() in _POSSESSIVE_PRONOUNS:
                return f"{entity}'s"
            return entity

        return _PRONOUN_RE.sub(_replace, text)

    @staticmethod
    def _normalize_possessives(text: str) -> str:
        def _replace(m: re.Match) -> str:
            owner = m.group(1)
            attr = m.group(2)
            return f"{attr} of {owner}"

        return _POSSESSIVE_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# 2. ClaimFrameRouter
# ---------------------------------------------------------------------------

_IF_ALL_THEN_ALL_RE = re.compile(
    r"^if\s+(?:all|every)\s+(.+?)\s*,\s*then\s+(?:all|every)\s+(.+)$",
    re.IGNORECASE,
)

# "X needs / requires / must have R to qualify for / be eligible for / be awarded Y"
_REQUIREMENT_RE = re.compile(
    r"^(.+?)\s+(?:needs?|requires?|must\s+have|must\s+complete|must\s+obtain)\s+(.+?)"
    r"\s+(?:to|in\s+order\s+to)\s+(?:qualify\s+for|be\s+eligible\s+for|be\s+awarded|"
    r"receive|get|obtain|earn)\s+(.+?)\.?$",
    re.IGNORECASE,
)

# "GPA of John is (at least) 3.8"  (after possessive normalization)
_GPA_RE = re.compile(
    r"\bGPA\s+of\s+([A-Z][A-Za-z ]*?)\s+is\s+"
    r"(at\s+least|at\s+most|above|below|greater\s+than|less\s+than|exactly)?\s*"
    r"([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

_OBJECT_MODAL_RE = re.compile(
    r"\b(can|cannot|may|must|should|is\s+allowed\s+to|is\s+permitted\s+to|"
    r"is\s+prohibited\s+from)\b",
    re.IGNORECASE,
)

_GPA_OP = {
    "at least": ">=",
    "at most": "<=",
    "above": ">",
    "greater than": ">",
    "below": "<",
    "less than": "<",
    "exactly": "=",
}


class ClaimFrameRouter:
    """Classify a normalized claim into a frame kind with extracted slots."""

    def route(self, text: str) -> ClaimFrame:
        stripped = text.strip()

        m = _IF_ALL_THEN_ALL_RE.match(stripped)
        if m:
            return ClaimFrame(
                kind="IF_ALL_THEN_ALL",
                text=stripped,
                antecedent="all " + m.group(1).strip(),
                consequent="all " + m.group(2).strip(),
            )

        m = _GPA_RE.search(stripped)
        if m:
            op = _GPA_OP.get((m.group(2) or "").strip().lower(), "=")
            return ClaimFrame(
                kind="GPA_NUMERIC",
                text=stripped,
                gpa_owner=m.group(1).strip(),
                gpa_operator=op,
                gpa_value=float(m.group(3)),
            )

        m = _REQUIREMENT_RE.match(stripped)
        if m:
            return ClaimFrame(
                kind="REQUIREMENT",
                text=stripped,
                subject=m.group(1).strip(),
                requirement=m.group(2).strip(),
                target=m.group(3).strip(),
            )

        if _OBJECT_MODAL_RE.search(stripped):
            return ClaimFrame(kind="OBJECT_MODAL", text=stripped)

        return ClaimFrame(kind="SIMPLE_ATOMIC", text=stripped)


# ---------------------------------------------------------------------------
# 3. SchemaGuidedFrameCompiler
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({"a", "an", "the", "of", "for", "to", "their", "his", "her"})


def _camel(phrase: str) -> str:
    """CamelCase a noun phrase into a single constant/predicate token."""
    words = [w for w in re.split(r"[\s\-]+", phrase.strip()) if w and w.lower() not in _STOPWORDS]
    return "".join(w[:1].upper() + w[1:] for w in words) if words else phrase.strip()


class SchemaGuidedFrameCompiler:
    """Compile frames to FOL — deterministic for structured kinds.

    Structured kinds (IF_ALL_THEN_ALL, REQUIREMENT, GPA_NUMERIC) are built
    directly so they cannot drift. SIMPLE_ATOMIC / OBJECT_MODAL claims are
    delegated to the free FOLParser in a single batch.
    """

    def __init__(self, fol_parser: FOLParser) -> None:
        self.fol_parser = fol_parser

    async def compile_frames(self, frames: list[ClaimFrame]) -> list[FOLNode]:
        # Collect every sub-text that needs the free parser into one batch.
        batch: list[str] = []
        plan: list[tuple[str, int, int]] = []  # (kind, idx_a, idx_b)

        for frame in frames:
            if frame.kind == "IF_ALL_THEN_ALL":
                a = len(batch)
                batch.append(frame.antecedent or "")
                batch.append(frame.consequent or "")
                plan.append(("IF_ALL_THEN_ALL", a, a + 1))
            elif frame.kind in ("SIMPLE_ATOMIC", "OBJECT_MODAL"):
                a = len(batch)
                batch.append(frame.text)
                plan.append((frame.kind, a, -1))
            else:
                plan.append((frame.kind, -1, -1))

        parsed = await self.fol_parser.parse_many(batch) if batch else []

        out: list[FOLNode] = []
        for frame, (kind, a, b) in zip(frames, plan):
            if kind == "IF_ALL_THEN_ALL":
                out.append(LogicalNode(operator="IMPLIES", left=parsed[a], right=parsed[b]))
            elif kind == "REQUIREMENT":
                out.append(self._compile_requirement(frame))
            elif kind == "GPA_NUMERIC":
                out.append(self._compile_gpa(frame))
            else:  # SIMPLE_ATOMIC / OBJECT_MODAL
                out.append(parsed[a])
        return out

    @staticmethod
    def _compile_requirement(frame: ClaimFrame) -> FOLNode:
        """Build Requires(Subject, Requirement, Target) — a requirement-family atom.

        Keeps the requirement and the goal distinct so the canonicalizer cannot
        collapse it to an achievement predicate (QualifiesFor) and lose meaning.
        """
        subject = _camel(frame.subject or "")
        requirement = _camel(frame.requirement or "")
        target = _camel(frame.target or "")
        predicate = Predicate(
            name="Requires",
            arg_sorts=["Entity", "Entity", "Entity"],
            aliases=[],
        )
        return AtomicNode(predicate=predicate, arguments=[subject, requirement, target])

    @staticmethod
    def _compile_gpa(frame: ClaimFrame) -> FOLNode:
        owner = _camel(frame.gpa_owner or "")
        return ComparisonNode(
            operator=frame.gpa_operator or "=",  # type: ignore[arg-type]
            left=FunctionTerm(name="GPA", arguments=[owner]),
            right=NumericTerm(value=frame.gpa_value or 0.0),
        )


# ---------------------------------------------------------------------------
# 4. SafeClaimCanonicalizer
# ---------------------------------------------------------------------------


class SafeClaimCanonicalizer:
    """Rename claim predicates to schema canonicals only when it is provably safe.

    A rename is applied only when the schema match shares arity, semantic key,
    and predicate family (the family guard already blocks requirement ↔
    achievement). Predicates with no safe match are left untouched so the
    verifier can report drift rather than the canonicalizer hiding it.
    """

    def canonicalize(
        self,
        trees: list[FOLNode],
        schema: PremiseSchema,
    ) -> tuple[list[FOLNode], list[dict[str, object]], list[str]]:
        predicates = list(schema.predicates)
        remap: dict[tuple[str, int], str] = {}
        diagnostics: list[str] = []

        for tree in trees:
            for name, arity in _collect_predicates(tree):
                if (name, arity) in remap:
                    continue
                match = _best_schema_match(name, arity, predicates)
                if match is None:
                    if not _schema_has(predicates, name, arity):
                        diagnostics.append(f"CLAIM_PREDICATE_UNMATCHED:{name}/{arity}")
                    continue
                if match.name != name:
                    remap[(name, arity)] = match.name

        renames = [
            {"from": name, "arity": arity, "to": canonical}
            for (name, arity), canonical in remap.items()
        ]
        canonicalized = [_rename(tree, remap) for tree in trees] if remap else trees
        canonicalized, constant_renames = schema.canonicalize_constants(canonicalized)
        return canonicalized, [*renames, *constant_renames], diagnostics


# ---------------------------------------------------------------------------
# 5. ClaimVerifier
# ---------------------------------------------------------------------------


class ClaimVerifier:
    """Flag claims the solver cannot use, so Uncertain is attributed honestly."""

    def verify(
        self,
        trees: list[FOLNode],
        schema: PremiseSchema,
    ) -> list[str]:
        predicates = list(schema.predicates)
        diagnostics: list[str] = []
        for tree in trees:
            for name, arity in _collect_predicates(tree):
                if _schema_has(predicates, name, arity):
                    continue
                # Predicate not in the premise vocabulary — unsolvable as-is.
                family = _predicate_family(name)
                if any(
                    _predicate_family(p.name) not in (None, family) and p.arity == arity
                    for p in predicates
                ):
                    diagnostics.append(f"CLAIM_SEMANTIC_DRIFT_BLOCKED:{name}/{arity}")
                else:
                    diagnostics.append(f"CLAIM_PREDICATE_UNMATCHED:{name}/{arity}")
        return diagnostics


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ClaimParser:
    """Translate claim texts to canonicalized FOL via the v2 component pipeline."""

    def __init__(self, fol_parser: FOLParser) -> None:
        self.fol_parser = fol_parser
        self.normalizer = ClaimNormalizer()
        self.router = ClaimFrameRouter()
        self.compiler = SchemaGuidedFrameCompiler(fol_parser)
        self.canonicalizer = SafeClaimCanonicalizer()
        self.verifier = ClaimVerifier()

    async def parse_claims(
        self,
        claim_texts: list[str],
        schema: PremiseSchema,
        *,
        target_entity: str | None = None,
    ) -> tuple[list[FOLNode], list[dict[str, object]]]:
        """Normalize → route → compile → safe-canonicalize. Returns (fols, renames)."""

        trees, renames, _diagnostics = await self.parse_claims_verbose(
            claim_texts, schema, target_entity=target_entity
        )
        return trees, renames

    async def parse_claims_verbose(
        self,
        claim_texts: list[str],
        schema: PremiseSchema,
        *,
        target_entity: str | None = None,
    ) -> tuple[list[FOLNode], list[dict[str, object]], list[str]]:
        """Full pipeline returning claim diagnostics (drift / unmatched / normalization).

        Diagnostics are returned (not stored) so a shared ClaimParser instance is
        safe under concurrent requests.
        """

        if not claim_texts:
            return [], [], []

        frames: list[ClaimFrame] = []
        for text in claim_texts:
            normalized, norm_diags = self.normalizer.normalize(text, target_entity)
            frame = self.router.route(normalized)
            frame.diagnostics.extend(norm_diags)
            frames.append(frame)

        trees = await self.compiler.compile_frames(frames)
        trees, renames, canon_diags = self.canonicalizer.canonicalize(trees, schema)
        verify_diags = self.verifier.verify(trees, schema)

        diagnostics: list[str] = []
        seen: set[str] = set()
        for diag in (
            [d for f in frames for d in f.diagnostics] + canon_diags + verify_diags
        ):
            if diag not in seen:
                seen.add(diag)
                diagnostics.append(diag)

        return trees, renames, diagnostics


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _collect_predicates(node: FOLNode) -> list[tuple[str, int]]:
    if isinstance(node, AtomicNode):
        return [(node.predicate.name, len(node.arguments))]
    if isinstance(node, ComparisonNode):
        return []
    if isinstance(node, LogicalNode):
        out = _collect_predicates(node.left)
        if node.right is not None:
            out.extend(_collect_predicates(node.right))
        return out
    if isinstance(node, QuantifiedNode):
        out = _collect_predicates(node.body)
        if node.restrictor is not None:
            out.extend(_collect_predicates(node.restrictor))
        return out
    return []


def _schema_has(predicates: list[PredicateSignature], name: str, arity: int) -> bool:
    return any(
        p.arity == arity and (p.name == name or name in p.aliases) for p in predicates
    )


def _rename(node: FOLNode, remap: dict[tuple[str, int], str]) -> FOLNode:
    if isinstance(node, ComparisonNode):
        return node
    if isinstance(node, AtomicNode):
        canonical = remap.get((node.predicate.name, len(node.arguments)))
        if canonical is None:
            return node
        predicate = Predicate(
            name=canonical,
            arg_sorts=node.predicate.arg_sorts,
            description=node.predicate.description,
            aliases=[*node.predicate.aliases, node.predicate.name],
        )
        return AtomicNode(predicate=predicate, arguments=node.arguments)
    if isinstance(node, LogicalNode):
        return LogicalNode(
            operator=node.operator,
            left=_rename(node.left, remap),
            right=_rename(node.right, remap) if node.right is not None else None,
        )
    if isinstance(node, QuantifiedNode):
        return QuantifiedNode(
            quantifier=node.quantifier,
            variable=node.variable,
            body=_rename(node.body, remap),
            restrictor=(
                _rename(node.restrictor, remap) if node.restrictor is not None else None
            ),
        )
    return node
