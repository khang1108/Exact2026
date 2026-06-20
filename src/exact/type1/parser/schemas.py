"""Structured response schemas produced by Type 1 parser operations.

Each parser operation has a narrow output contract. These Pydantic models are
sent to vLLM as JSON schemas and then validate the returned model content.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass

from exact.type1.ast import AtomicNode, FOLNode, LogicalNode, QuantifiedNode
from exact.type1.ast.nodes import ComparisonNode, FunctionTerm
from exact.type1.models.schemas import Predicate

_INTERROGATIVE_START = re.compile(
    r"^(?:can|could|did|do|does|how|is|may|should|was|were|what|when|where|"
    r"which|who|whom|whose|why|will|would)\b",
    re.IGNORECASE,
)
_OPTION_LINE = re.compile(r"^[A-E]\.\s+", re.IGNORECASE)
_NUMERIC_VALUE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[A-Za-z][A-Za-z0-9]*)?$"
)
_MONTH_VALUE = re.compile(
    r"^(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\d{1,2}(?:st|nd|rd|th)?(?:\d{2}|\d{4})?$",
    re.IGNORECASE,
)
_NUMBER_WORD_VALUE = re.compile(
    r"^(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million)\b",
    re.IGNORECASE,
)
_AUXILIARY_ACTIONS = frozenset(
    {
        "be",
        "can",
        "could",
        "do",
        "make",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
    }
)
_TOKEN_EQUIVALENTS = {
    "am": "be",
    "are": "be",
    "been": "be",
    "being": "be",
    "completed": "complete",
    "completing": "complete",
    "completion": "complete",
    "completions": "complete",
    "did": "do",
    "does": "do",
    "enrolled": "enroll",
    "enrolling": "enroll",
    "enrollment": "enroll",
    "enrollments": "enroll",
    "had": "have",
    "has": "have",
    "is": "be",
    "made": "make",
    "makes": "make",
    "making": "make",
    "predicted": "predict",
    "predicting": "predict",
    "prediction": "predict",
    "predictions": "predict",
    "predicts": "predict",
    "qualified": "qualify",
    "qualifies": "qualify",
    "qualifying": "qualify",
    "required": "require",
    "requires": "require",
    "requiring": "require",
    "was": "be",
    "were": "be",
}
_PLURAL_CLASS_NOUNS = frozenset(
    {
        "applicants", "applications", "assessments",
        "classes", "components", "conditions", "constraints",
        "controls", "courses", "criteria", "curricula",
        "departments", "documents",
        "employees", "entities",
        "faculty", "files",
        "items",
        "materials", "members", "methods", "models",
        "objects", "operations",
        "people", "policies", "processes", "programs",
        "records", "requests", "requirements", "resources", "rules",
        "services", "students", "subjects", "systems",
        "tasks", "teachers",
        "units", "users",
    }
)
_DEONTIC_PREDICATE_PREFIXES = frozenset({"allowed", "can", "required", "requires"})

# Predicate semantic families. Cross-family canonicalization is blocked so that
# e.g. Requires(Sophia, X) cannot be silently renamed to QualifiesFor(Sophia).
_PREDICATE_FAMILIES: dict[str, frozenset[str]] = {
    "requirement": frozenset({"require", "need", "must", "mandatory", "obligat", "prerequisit"}),
    "achievement": frozenset({"qualify", "eligible", "receive", "earn", "award", "achiev", "get"}),
    "action": frozenset({"pass", "complet", "submit", "take", "finish"}),
}

class ParserResult(BaseModel):
    """Base schema that rejects fields not requested by the parser operation."""

    model_config = ConfigDict(extra="forbid")


class RephraseResult(ParserResult):
    """Result of minimally rewriting a sentence for clearer FOL parsing."""

    rephrased: str


class QuantifiedResult(ParserResult):
    """Result of extracting one outer quantifier and its remaining sentence."""

    quantifier: Literal["ForAll", "ThereExists"]
    variable: str
    restrictor_sentence: str | None = None
    scope_sentence: str


class LogicalResult(ParserResult):
    """Result of splitting a sentence at its outermost logical operator."""

    operator: Literal["AND", "OR", "IMPLIES", "IFF", "NOT"]
    left_operand: str
    right_operand: str | None


class AtomicResult(ParserResult):
    """Result of extracting a predicate and arguments from an atomic sentence."""

    predicate: str
    arguments: list[str]
    negated: bool


class CoreferenceResult(ParserResult):
    """Result of resolving pronouns in a right-hand clause."""

    resolved_right: str


class PremiseFrameResult(ParserResult):
    """Structural decomposition of one premise before FOL generation.

    The frame identifies the logical role of each text fragment so the
    compiler can build the correct quantifier / operator skeleton deterministically.
    All text fragments must reference the entity through ``variable``.
    """

    kind: Literal[
        "fact",
        "universal_rule",
        "existential_fact",
        "equivalence",
        "numeric_fact",
        "numeric_rule",
        "deontic_rule",
        "permission_rule",
        "prohibition_rule",
        "temporal_rule",
        "meta_rule",
        "unsupported",
    ]

    variable: str | None = None
    restrictor_text: str | None = None

    condition_texts: list[str] = []
    conclusion_texts: list[str] = []
    fact_texts: list[str] = []
    numeric_constraints: list[str] = []
    temporal_constraints: list[str] = []

    modality: Literal[
        "none",
        "must",
        "can",
        "may",
        "allowed",
        "required",
        "prohibited",
        "not_necessarily",
    ] = "none"

    confidence: float = 1.0


class NumericConstraintResult(ParserResult):
    """Result of parsing one numeric constraint sentence (e.g. 'x has at least 5 courses')."""

    function_name: str
    arguments: list[str]
    operator: Literal["=", ">=", ">", "<=", "<", "!="]
    value: float


class TemporalConstraintResult(ParserResult):
    """Result of parsing one temporal constraint sentence (e.g. 'x enrolled before June 1')."""

    function_name: str
    arguments: list[str]
    operator: Literal["before", "after", "on", "by", "until"]
    date_value: str


class QuestionFrameResult(ParserResult):
    """Structural classification of one question before any FOL is produced.

    The frame identifies what kind of question this is, which solver mode should
    answer it, how to read a meta vs object-level "can", and—for polar
    questions—the single declarative claim to test. It never contains FOL.
    """

    question_format: Literal["polar", "mcq", "open_wh"]
    solver_mode: Literal[
        "entailment",
        "refutation",
        "strongest_conclusion",
        "fewest_premise",
        "premise_selection",
        "ynu_mapped",
        "unsupported",
    ]
    can_interpretation: Literal["none", "meta_inference", "object_modal"] = "none"
    claim_text: str | None = None
    negate_claim: bool = False
    # Resolution targets the ClaimNormalizer uses to fix pronouns/possessives.
    # target_entity: the named subject the claim is about ("Sophia", "Professor John").
    target_class: str | None = None
    target_property: str | None = None
    target_entity: str | None = None
    confidence: float = 1.0


OptionRole = Literal[
    "RAW_FOL",
    "PREMISE_REFERENCE",
    "YNU_ANSWER",
    "NONE_OF_ABOVE",
    "SUBJECTLESS_MODAL_FRAGMENT",
    "PREDICATE_FRAGMENT",
    "CONJUNCTIVE_CLAIM",
    "FULL_CLAIM",
    "UNKNOWN",
]


class FragmentRealizationResult(ParserResult):
    """LLM fallback for realizing one subjectless option fragment into a claim.

    Used only when deterministic subject recovery + templating cannot produce a
    complete declarative claim. Must preserve modality and negation.
    """

    claim_text: str | None = None
    subject_found: bool = False


# ------------------------------------------------------------------
# Schemas for Premise Parser
# ------------------------------------------------------------------
@dataclass(frozen=True)
class PredicateSignature:
    """One canonical predicate available to a Type 1 problem."""

    name: str
    arity: int
    aliases: tuple[str, ...] = ()
    argument_roles: tuple[str, ...] = ()
    value_type: Literal["bool", "number", "date", "entity"] | None = None
    introduced_by_premises: tuple[int, ...] = ()


@dataclass(frozen=True)
class ConstantSignature:
    """One constant observed in a Type 1 problem."""

    name: str
    aliases: tuple[str, ...] = ()
    sort: str | None = None
    introduced_by_premises: tuple[int, ...] = ()


@dataclass(frozen=True)
class PremiseSchema:
    """Canonical predicate vocabulary derived from premise ASTs."""

    predicates: tuple[PredicateSignature, ...]
    constants: tuple[ConstantSignature, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @classmethod
    def from_trees(cls, trees: list[FOLNode]) -> PremiseSchema:
        """Build a stable schema and independent diagnostics from raw ASTs."""

        occurrences = [
            (premise_index, atom)
            for premise_index, tree in enumerate(trees, start=1)
            for atom in _collect_atoms_in_order(tree)
        ]
        grouped: dict[tuple[int, tuple[str, ...]], list[tuple[int, AtomicNode]]] = {}
        for premise_index, atom in occurrences:
            key = (len(atom.arguments), _semantic_key(atom.predicate.name))
            grouped.setdefault(key, []).append((premise_index, atom))

        predicates: list[PredicateSignature] = []
        for group in grouped.values():
            names = _unique(item.predicate.name for _, item in group)
            canonical_name = _choose_canonical_name(names)
            aliases = tuple(name for name in names if name != canonical_name)
            predicates.append(
                PredicateSignature(
                    name=canonical_name,
                    arity=len(group[0][1].arguments),
                    aliases=aliases,
                    argument_roles=_infer_argument_roles([atom for _, atom in group]),
                    value_type=_infer_predicate_value_type([atom for _, atom in group]),
                    introduced_by_premises=tuple(
                        sorted({premise_index for premise_index, _ in group})
                    ),
                )
            )

        constants = _build_constant_signatures(trees)
        diagnostics = _build_schema_diagnostics(trees)
        return cls(
            predicates=tuple(predicates),
            constants=constants,
            diagnostics=diagnostics,
        )

    def canonicalize(self, trees: list[FOLNode]) -> tuple[list[FOLNode], list[dict[str, object]]]:
        """Rename predicates and entity constants to the shared schema vocabulary."""

        remap: dict[tuple[str, int], str] = {}
        for tree in trees:
            for name, arity in _collect_predicates_in_order(tree):
                match = _best_schema_match(name, arity, list(self.predicates))
                if match is not None and match.name != name:
                    remap[(name, arity)] = match.name

        renames = [
            {"from": name, "arity": arity, "to": canonical}
            for (name, arity), canonical in remap.items()
        ]
        canonicalized = (
            [_rename_in_node(tree, remap) for tree in trees] if remap else trees
        )
        canonicalized, constant_renames = self.canonicalize_constants(canonicalized)
        return canonicalized, [*renames, *constant_renames]

    def canonicalize_constants(
        self,
        trees: list[FOLNode],
    ) -> tuple[list[FOLNode], list[dict[str, object]]]:
        """Rename constants to exact aliases or one unambiguous semantic match."""

        remap: dict[str, str] = {}
        for tree in trees:
            for constant in _collect_constants_in_order(tree):
                match = _best_constant_match(constant, list(self.constants))
                if match is not None and match.name != constant:
                    remap[constant] = match.name

        renames = [
            {"kind": "constant", "from": name, "to": canonical}
            for name, canonical in remap.items()
        ]
        if not remap:
            return trees, renames
        return [_rename_constants_in_node(tree, remap) for tree in trees], renames

    def contains(self, name: str, arity: int) -> bool:
        """Return whether the exact canonical predicate is in the schema."""

        return any(
            predicate.name == name and predicate.arity == arity
            for predicate in self.predicates
        )


@dataclass(frozen=True)
class PremiseParseBundle:
    """Verified result of the complete premise parsing workflow."""

    premises: list[str]
    draft_trees: list[FOLNode]
    schema: PremiseSchema
    trees: list[FOLNode]
    predicate_renames: list[dict[str, object]]
    verified: bool
    verification_issues: tuple[str, ...]
    # Severity split: only blocking_issues set verified=False and skip the solver;
    # warnings are surfaced but the solver still runs (lower confidence).
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    # 0-based indices of meta-epistemic premises ("no premise states whether X")
    # that disclaim knowledge rather than assert a fact. Their FOL must be kept
    # out of the Z3 fact set, and they act as uncertainty witnesses.
    epistemic_witness_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class OptionClaim:
    """One MCQ option interpreted as a role-tagged, possibly testable claim."""

    label: str
    original_text: str
    normalized_text: str
    role: str
    claim_text: str | None
    raw_fol: str | None
    premise_indices: tuple[int, ...]
    ynu_value: str
    is_selectable: bool
    fol: FOLNode | None = None
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptionParseBundle:
    """Result of interpreting all options of one MCQ."""

    options: tuple[OptionClaim, ...]
    marker_style: str
    option_count: int
    role_distribution: dict[str, int]
    verified: bool
    issues: tuple[str, ...]
    extraction_diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class QuerySpec:
    """Text-level decision record for how to answer one question.

    Produced by the question-side parser before solving. Carries no FOL of its
    own; option FOL lives on each ``OptionClaim`` and the polar claim FOL lives
    on ``QuestionParseBundle``.
    """

    question_format: str
    solver_mode: str
    can_interpretation: str
    main_claim_text: str | None
    negate_claim: bool
    option_claims: tuple[OptionClaim, ...]
    supported: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class QuestionParseBundle:
    """Verified result of the complete question parsing workflow."""

    question: str
    spec: QuerySpec
    main_claim_fol: FOLNode | None
    claim_renames: list[dict[str, object]]
    option_bundle: OptionParseBundle | None = None

def _collect_predicates_in_order(node: FOLNode) -> list[tuple[str, int]]:
    return [
        (atom.predicate.name, len(atom.arguments))
        for atom in _collect_atoms_in_order(node)
    ]


def _collect_atoms_in_order(node: FOLNode) -> list[AtomicNode]:
    if isinstance(node, AtomicNode):
        return [node]
    if isinstance(node, ComparisonNode):
        return []
    if isinstance(node, QuantifiedNode):
        atoms = _collect_atoms_in_order(node.body)
        if node.restrictor is not None:
            atoms.extend(_collect_atoms_in_order(node.restrictor))
        return atoms
    # LogicalNode
    atoms = _collect_atoms_in_order(node.left)
    if node.right is not None:
        atoms.extend(_collect_atoms_in_order(node.right))
    return atoms

def _rename_in_node(node: FOLNode, remap: dict[tuple[str, int], str]) -> FOLNode:
    if isinstance(node, ComparisonNode):
        return node
    if isinstance(node, AtomicNode):
        key = (node.predicate.name, len(node.arguments))
        canonical_name = remap.get(key)
        if canonical_name is None:
            return node
        predicate = Predicate(
            name=canonical_name,
            arg_sorts=node.predicate.arg_sorts,
            description=node.predicate.description,
            aliases=[*node.predicate.aliases, node.predicate.name],
        )
        return AtomicNode(predicate=predicate, arguments=node.arguments)

    if isinstance(node, QuantifiedNode):
        return QuantifiedNode(
            quantifier=node.quantifier,
            variable=node.variable,
            body=_rename_in_node(node.body, remap),
            restrictor=(
                _rename_in_node(node.restrictor, remap)
                if node.restrictor is not None
                else None
            ),
        )

    return LogicalNode(
        operator=node.operator,
        left=_rename_in_node(node.left, remap),
        right=_rename_in_node(node.right, remap) if node.right is not None else None,
    )


def _rename_constants_in_node(node: FOLNode, remap: dict[str, str]) -> FOLNode:
    if isinstance(node, ComparisonNode):
        left = FunctionTerm(
            name=node.left.name,
            arguments=[remap.get(argument, argument) for argument in node.left.arguments],
        )
        right = node.right
        if isinstance(right, FunctionTerm):
            right = FunctionTerm(
                name=right.name,
                arguments=[remap.get(argument, argument) for argument in right.arguments],
            )
        return ComparisonNode(operator=node.operator, left=left, right=right)
    if isinstance(node, AtomicNode):
        return AtomicNode(
            predicate=node.predicate,
            arguments=[remap.get(argument, argument) for argument in node.arguments],
        )
    if isinstance(node, QuantifiedNode):
        return QuantifiedNode(
            quantifier=node.quantifier,
            variable=node.variable,
            body=_rename_constants_in_node(node.body, remap),
            restrictor=(
                _rename_constants_in_node(node.restrictor, remap)
                if node.restrictor is not None
                else None
            ),
        )
    return LogicalNode(
        operator=node.operator,
        left=_rename_constants_in_node(node.left, remap),
        right=(
            _rename_constants_in_node(node.right, remap)
            if node.right is not None
            else None
        ),
    )

def _map_atoms(node: FOLNode, fn: Callable[[AtomicNode], AtomicNode]) -> FOLNode:
    """Rebuild a tree, applying ``fn`` to every atom (comparisons left untouched)."""
    if isinstance(node, ComparisonNode):
        return node
    if isinstance(node, AtomicNode):
        return fn(node)
    if isinstance(node, QuantifiedNode):
        return QuantifiedNode(
            quantifier=node.quantifier,
            variable=node.variable,
            body=_map_atoms(node.body, fn),
            restrictor=(
                _map_atoms(node.restrictor, fn)
                if node.restrictor is not None
                else None
            ),
        )
    return LogicalNode(
        operator=node.operator,
        left=_map_atoms(node.left, fn),
        right=_map_atoms(node.right, fn) if node.right is not None else None,
    )


def repair_arity_drift(trees: list[FOLNode]) -> tuple[list[FOLNode], list[str]]:
    """Unify a predicate that drifts across arities down to its lowest arity.

    The atomic parser occasionally emits a higher-arity copy of a predicate whose
    surplus argument merely re-states the predicate itself (a self-nominalization,
    e.g. ``HasSupervisorApproval(Asha, SupervisorApproval)`` alongside a rule's
    unary ``HasSupervisorApproval(x)``). Z3 treats the two arities as distinct
    predicates, so entailment silently fails. When the surplus argument is such a
    self-nominalization, drop it so every occurrence shares one signature.
    """
    arities_by_name: dict[str, set[int]] = defaultdict(set)
    for tree in trees:
        for atom in _collect_atoms_in_order(tree):
            arities_by_name[atom.predicate.name].add(len(atom.arguments))
    targets = {
        name: min(arities)
        for name, arities in arities_by_name.items()
        if len(arities) > 1
    }
    if not targets:
        return trees, []

    repairs: list[str] = []

    def _self_nominalized(pred_name: str, argument: str) -> bool:
        pred_tokens = {
            _normalize_semantic_token(token) for token in _camel_word_list(pred_name)
        }
        arg_tokens = {
            _normalize_semantic_token(token) for token in _camel_word_list(argument)
        }
        return bool(arg_tokens) and arg_tokens <= pred_tokens

    def _repair_atom(atom: AtomicNode) -> AtomicNode:
        name = atom.predicate.name
        target = targets.get(name)
        if target is None or len(atom.arguments) <= target:
            return atom
        drop = {
            index
            for index, argument in enumerate(atom.arguments)
            if _self_nominalized(name, argument)
        }
        # Only repair when dropping the self-nominalized args reaches the target
        # arity — otherwise the surplus args are real and we leave the warning.
        if len(atom.arguments) - len(drop) != target:
            return atom
        kept = [index for index in range(len(atom.arguments)) if index not in drop]
        dropped = [atom.arguments[index] for index in sorted(drop)]
        sorts = atom.predicate.arg_sorts
        new_sorts = (
            [sorts[index] for index in kept] if len(sorts) == len(atom.arguments) else sorts
        )
        repairs.append(
            f"ARITY_DRIFT_REPAIRED: {name}/{len(atom.arguments)} → /{target} "
            f"(dropped self-nominalized {', '.join(dropped)})"
        )
        predicate = Predicate(
            name=name,
            arg_sorts=new_sorts,
            description=atom.predicate.description,
            aliases=list(atom.predicate.aliases),
        )
        return AtomicNode(predicate=predicate, arguments=[atom.arguments[i] for i in kept])

    repaired = [_map_atoms(tree, _repair_atom) for tree in trees]
    return repaired, list(dict.fromkeys(repairs))


def _predicate_family(name: str) -> str | None:
    """Return the semantic family of a predicate name, or None if unclassified."""
    tokens = {_normalize_semantic_token(t) for t in _camel_word_list(name)}
    for family, keywords in _PREDICATE_FAMILIES.items():
        if any(any(t.startswith(k) for k in keywords) for t in tokens):
            return family
    return None


def _best_schema_match(
    name: str,
    arity: int,
    predicates: list[PredicateSignature],
) -> PredicateSignature | None:
    exact = next(
        (
            predicate
            for predicate in predicates
            if predicate.arity == arity
            and (predicate.name == name or name in predicate.aliases)
        ),
        None,
    )
    if exact is not None:
        return exact

    claim_family = _predicate_family(name)
    semantic_key = _semantic_key(name)
    return next(
        (
            predicate
            for predicate in predicates
            if predicate.arity == arity
            and _semantic_key(predicate.name) == semantic_key
            # Block cross-family canonicalization (e.g. Requires → QualifiesFor).
            and not (
                claim_family is not None
                and _predicate_family(predicate.name) is not None
                and claim_family != _predicate_family(predicate.name)
            )
        ),
        None,
    )

def _camel_word_list(name: str) -> list[str]:
    return [
        word.lower()
        for word in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+", name)
    ]


def _semantic_key(name: str) -> tuple[str, ...]:
    raw_tokens = _camel_word_list(name)
    normalized = [_normalize_semantic_token(token) for token in raw_tokens]
    if raw_tokens and raw_tokens[0] in _DEONTIC_PREDICATE_PREFIXES:
        content = tuple(
            token for token in normalized[1:] if token not in _AUXILIARY_ACTIONS
        )
        return (f"modal:{raw_tokens[0]}", *content)
    content = tuple(token for token in normalized if token not in _AUXILIARY_ACTIONS)
    return content or tuple(normalized)


def _normalize_semantic_token(token: str) -> str:
    if token in _TOKEN_EQUIVALENTS:
        return _TOKEN_EQUIVALENTS[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def _choose_canonical_name(names: list[str]) -> str:
    return min(
        names,
        key=lambda name: (
            len(_camel_word_list(name)),
            len(name),
            names.index(name),
        ),
    )


def _infer_argument_roles(atoms: list[AtomicNode]) -> tuple[str, ...]:
    roles: list[str] = []
    for argument_index in range(len(atoms[0].arguments)):
        values = [atom.arguments[argument_index] for atom in atoms]
        if any(_is_date_value(value) or _is_numeric_value(value) for value in values):
            roles.append("value")
        elif argument_index == 0:
            roles.append("subject")
        elif argument_index == 1:
            roles.append("object")
        else:
            roles.append(f"argument_{argument_index + 1}")
    return tuple(roles)


def _infer_predicate_value_type(
    atoms: list[AtomicNode],
) -> Literal["bool", "number", "date", "entity"]:
    arguments = [argument for atom in atoms for argument in atom.arguments]
    if any(_is_date_value(argument) for argument in arguments):
        return "date"
    if any(_is_numeric_value(argument) for argument in arguments):
        return "number"
    return "bool"


def _build_constant_signatures(trees: list[FOLNode]) -> tuple[ConstantSignature, ...]:
    occurrences: dict[str, list[tuple[int, str, tuple[object, ...]]]] = {}
    for premise_index, tree in enumerate(trees, start=1):
        for constant, context in _collect_constant_occurrences(tree):
            occurrences.setdefault(constant.casefold(), []).append(
                (premise_index, constant, context)
            )

    keys = list(occurrences)
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    contexts = {
        key: {context for _, _, context in group}
        for key, group in occurrences.items()
    }
    token_sets = {
        key: frozenset(_normalize_semantic_token(token) for token in _camel_word_list(group[0][1]))
        for key, group in occurrences.items()
    }
    for key in keys:
        same_context_supersets = [
            candidate
            for candidate in keys
            if candidate != key
            and contexts[key] & contexts[candidate]
            and token_sets[key]
            and token_sets[key] < token_sets[candidate]
        ]
        if len(same_context_supersets) == 1:
            union(key, same_context_supersets[0])
        for candidate in keys:
            if (
                candidate != key
                and contexts[key] & contexts[candidate]
                and token_sets[key]
                and token_sets[key] == token_sets[candidate]
            ):
                union(key, candidate)

    grouped: dict[str, list[tuple[int, str, tuple[object, ...]]]] = {}
    for key, group in occurrences.items():
        grouped.setdefault(find(key), []).extend(group)

    signatures: list[ConstantSignature] = []
    for group in grouped.values():
        names = _unique(name for _, name, _ in group)
        canonical_name = _choose_canonical_name(names)
        signatures.append(
            ConstantSignature(
                name=canonical_name,
                aliases=tuple(name for name in names if name != canonical_name),
                sort=_infer_constant_sort(canonical_name),
                introduced_by_premises=tuple(
                    sorted({premise_index for premise_index, _, _ in group})
                ),
            )
        )
    return tuple(signatures)


def _best_constant_match(
    name: str,
    constants: list[ConstantSignature],
) -> ConstantSignature | None:
    folded = name.casefold()
    exact = next(
        (
            constant
            for constant in constants
            if constant.name.casefold() == folded
            or any(alias.casefold() == folded for alias in constant.aliases)
        ),
        None,
    )
    if exact is not None:
        return exact

    tokens = frozenset(
        _normalize_semantic_token(token) for token in _camel_word_list(name)
    )
    if not tokens:
        return None
    candidates = [
        constant
        for constant in constants
        if any(
            tokens <= candidate_tokens or candidate_tokens <= tokens
            for candidate_tokens in (
                frozenset(
                    _normalize_semantic_token(token)
                    for token in _camel_word_list(candidate)
                )
                for candidate in (constant.name, *constant.aliases)
            )
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


def _collect_constant_occurrences(
    node: FOLNode,
    bound_variables: frozenset[str] = frozenset(),
) -> list[tuple[str, tuple[object, ...]]]:
    if isinstance(node, AtomicNode):
        context = ("atomic", _semantic_key(node.predicate.name), len(node.arguments))
        return [
            (argument, (*context, index))
            for index, argument in enumerate(node.arguments)
            if argument not in bound_variables
        ]
    if isinstance(node, ComparisonNode):
        context = ("function", _semantic_key(node.left.name), len(node.left.arguments))
        occurrences = [
            (argument, (*context, index))
            for index, argument in enumerate(node.left.arguments)
            if argument not in bound_variables
        ]
        if isinstance(node.right, FunctionTerm):
            right_context = (
                "function",
                _semantic_key(node.right.name),
                len(node.right.arguments),
            )
            occurrences.extend(
                (argument, (*right_context, index))
                for index, argument in enumerate(node.right.arguments)
                if argument not in bound_variables
            )
        return occurrences
    if isinstance(node, QuantifiedNode):
        local_variables = bound_variables | {node.variable}
        occurrences = _collect_constant_occurrences(node.body, local_variables)
        if node.restrictor is not None:
            occurrences.extend(_collect_constant_occurrences(node.restrictor, local_variables))
        return occurrences

    occurrences = _collect_constant_occurrences(node.left, bound_variables)
    if node.right is not None:
        occurrences.extend(_collect_constant_occurrences(node.right, bound_variables))
    return occurrences


def _collect_constants_in_order(
    node: FOLNode,
    bound_variables: frozenset[str] = frozenset(),
) -> list[str]:
    if isinstance(node, AtomicNode):
        return [
            argument
            for argument in node.arguments
            if argument not in bound_variables
        ]
    if isinstance(node, ComparisonNode):
        return [a for a in node.left.arguments if a not in bound_variables]
    if isinstance(node, QuantifiedNode):
        local_variables = bound_variables | {node.variable}
        constants = _collect_constants_in_order(node.body, local_variables)
        if node.restrictor is not None:
            constants.extend(_collect_constants_in_order(node.restrictor, local_variables))
        return constants

    constants = _collect_constants_in_order(node.left, bound_variables)
    if node.right is not None:
        constants.extend(_collect_constants_in_order(node.right, bound_variables))
    return constants


def _build_schema_diagnostics(trees: list[FOLNode]) -> tuple[str, ...]:
    predicate_occurrences = [
        (premise_index, atom.predicate.name, len(atom.arguments))
        for premise_index, tree in enumerate(trees, start=1)
        for atom in _collect_atoms_in_order(tree)
    ]
    issues: list[str] = []

    arities_by_name: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for premise_index, name, arity in predicate_occurrences:
        arities_by_name[name][arity].add(premise_index)
    for name, arities in arities_by_name.items():
        if len(arities) > 1:
            issues.append(
                "ARITY_DRIFT: "
                f"{name} appears as "
                + ", ".join(f"/{a}" for a in sorted(arities))
            )

    signatures_by_semantics: dict[tuple[str, ...], list[tuple[str, int]]] = {}
    for _, name, arity in predicate_occurrences:
        signatures_by_semantics.setdefault(_semantic_key(name), []).append((name, arity))
    for signatures in signatures_by_semantics.values():
        unique_signatures = list(dict.fromkeys(signatures))
        if len({name for name, _ in unique_signatures}) > 1:
            issues.append(
                "SCHEMA_SIMILAR_PREDICATES: "
                "semantically similar predicates: "
                + ", ".join(f"{name}/{arity}" for name, arity in unique_signatures)
            )

    constant_premises: dict[str, set[int]] = defaultdict(set)
    constant_names: dict[str, str] = {}
    for premise_index, tree in enumerate(trees, start=1):
        for constant in _collect_constants_in_order(tree):
            key = constant.casefold()
            constant_names.setdefault(key, constant)
            constant_premises[key].add(premise_index)

    for key, premise_indices in constant_premises.items():
        constant = constant_names[key]
        introduced = ", ".join(map(str, sorted(premise_indices)))
        if _is_date_value(constant):
            issues.append(
                "SCHEMA_DATE_ENTITY_CONSTANT: "
                f"{constant} is a date encoded as an entity constant in premises {introduced}"
            )
        elif _is_numeric_value(constant):
            issues.append(
                "SCHEMA_NUMERIC_ENTITY_CONSTANT: "
                f"{constant} is a numeric value encoded as an entity constant "
                f"in premises {introduced}"
            )
        elif _is_plural_class_constant(constant):
            issues.append(
                "GENERIC_CLASS_USED_AS_CONSTANT: "
                f"{constant} is a generic class noun used as a constant in premises {introduced}"
            )

    return tuple(issues)


def _infer_constant_sort(value: str) -> str | None:
    if _is_date_value(value):
        return "date"
    if _is_numeric_value(value):
        return "number"
    if _is_plural_class_constant(value):
        return "class"
    return None


def _is_date_value(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    if _MONTH_VALUE.fullmatch(compact):
        return True
    if compact.isdigit() and len(compact) == 8:
        year, month, day = int(compact[:4]), int(compact[4:6]), int(compact[6:])
        return 1900 <= year <= 2200 and 1 <= month <= 12 and 1 <= day <= 31
    return False


def _is_numeric_value(value: str) -> bool:
    compact = re.sub(r"[\s,_-]+", "", value)
    return bool(_NUMERIC_VALUE.fullmatch(compact) or _NUMBER_WORD_VALUE.match(compact))


def _is_plural_class_constant(value: str) -> bool:
    words = _camel_word_list(value)
    if not words:
        return False
    final_word = words[-1]
    return final_word in _PLURAL_CLASS_NOUNS


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def is_generic_class_constant(name: str) -> bool:
    """Return True when a CamelCase constant name ends in a plural class noun.

    Examples: Students, AIModels, TransportationSystems, EligibleStudents.
    """
    return _is_plural_class_constant(name)


_IRREGULAR_SINGULARS = {
    "curricula": "curriculum",
    "criteria": "criterion",
    "people": "person",
    "faculty": "faculty",
}


def singularize_class_constant(name: str) -> str:
    """Return the singular CamelCase form of a plural class constant name.

    Examples:
        Students         → Student
        AIModels         → AIModel
        Applications     → Application
        EligibleStudents → EligibleStudent
        Curricula        → Curriculum
    """
    raw_words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+", name)
    if not raw_words:
        return name
    last = raw_words[-1].lower()
    singular = _IRREGULAR_SINGULARS.get(last)
    if singular is None:
        if last.endswith("ies") and len(last) > 3:
            singular = last[:-3] + "y"
        elif last.endswith("ses") and len(last) > 4:
            singular = last[:-2]
        elif last.endswith("s") and not last.endswith("ss") and len(last) > 2:
            singular = last[:-1]
        else:
            singular = last
    return "".join(raw_words[:-1]) + singular.capitalize()
