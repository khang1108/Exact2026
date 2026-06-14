"""Type 1 logic pipeline entry point.

Workflow per request:
  1. Parse and verify declarative premises through ``PremiseParser``.
  2. Classify the question + interpret options through ``QuestionSideParser``.
  3. Route on the resulting ``QuerySpec`` to the right Z3 check.
  4. Z3 entailment / refutation / YNU-mapping → answer label.
  5. Normalize the uncertain token and return answer with FOL debug info.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.config import get_settings
from exact.type1.ast import AtomicNode, FOLNode, QuantifiedNode
from exact.type1.ast.nodes import ComparisonNode
from exact.type1.parser import PremiseParser
from exact.type1.parser.options import extract_mcq
from exact.type1.proof_connectivity import (
    build_proof_connectivity_dashboard,
    interpret_z3_uncertainty,
)

if TYPE_CHECKING:
    from exact.type1.fallback import Type1FallbackReasoner
    from exact.type1.parser import QuestionSideParser
    from exact.type1.parser.schemas import OptionClaim, QuestionParseBundle
    from exact.type1.solvers import FOLSolver  # type: ignore[import-untyped]

# Internal uncertain literal emitted by the solver, normalized at the boundary.
_SOLVER_UNCERTAIN = "Uncertain"


async def run_type1_pipeline(
    payload: PredictionRequest,
    premise_parser: PremiseParser,
    question_parser: QuestionSideParser,
    solver: FOLSolver | None = None,
    fallback_reasoner: Type1FallbackReasoner | None = None,
) -> PredictionResponse:
    """Parse premises and the question through their workflows, then solve."""

    premises = [p.strip() for p in payload.premises or [] if p.strip()]
    if not premises:
        raise ValueError("Type 1 requests require at least one non-empty premise")

    premise_bundle = await premise_parser.parse_premises(premises)
    premise_fols = premise_bundle.trees

    # --- Question side: classify, interpret options, compile claims -----------
    options_dict = _normalize_options(payload.options)
    mcq_extraction = None
    if not options_dict:
        # Options may be embedded in the question body (A./A) lines).
        mcq_extraction = extract_mcq(payload.question)
        options_dict = mcq_extraction.options

    q_bundle = await question_parser.parse_question(
        payload.question,
        options_dict or None,
        premise_bundle.schema,
        extraction=mcq_extraction,
    )
    spec = q_bundle.spec
    is_mcq = spec.question_format == "mcq"
    question_type = QuestionType.MCQ if is_mcq else QuestionType.YNU
    proof_connectivity = build_proof_connectivity_dashboard(
        q_bundle,
        premise_bundle.schema,
    )

    # --- Z3 routing ----------------------------------------------------------
    solver_used = solver is not None and premise_bundle.verified and spec.supported
    sort_conflict = False
    premises_used: list[int] | None = None
    if solver_used:
        assert solver is not None
        try:
            raw_answer, premises_used = _solve(solver, premise_fols, spec, q_bundle)
        except ValueError as exc:
            if "SORT_CONFLICT" in str(exc):
                raw_answer = _SOLVER_UNCERTAIN
                sort_conflict = True
            else:
                raise
    else:
        raw_answer = _SOLVER_UNCERTAIN

    symbolic_answer = raw_answer
    symbolic_cause = _uncertainty_cause(
        solver, premise_bundle, spec, raw_answer, sort_conflict
    )
    uncertainty_interpretation = (
        interpret_z3_uncertainty(proof_connectivity)
        if symbolic_cause == "Z3_TRUE_UNCERTAIN"
        else None
    )
    fallback_used = False
    fallback_error: str | None = None
    fallback_explanation: str | None = None
    fallback_trigger = (
        symbolic_cause
        if raw_answer == _SOLVER_UNCERTAIN
        else (
            "RANKING_MODE_ADJUDICATION"
            if spec.solver_mode in {"fewest_premise", "strongest_conclusion", "premise_selection"}
            else None
        )
    )
    if fallback_trigger is not None and fallback_reasoner is not None:
        try:
            fallback = await fallback_reasoner.answer(
                premises=premises,
                question=payload.question,
                option_labels=(
                    [claim.label for claim in spec.option_claims]
                    if is_mcq
                    else ["Yes", "No"]
                ),
                options=options_dict or None,
            )
            if fallback.answer != _SOLVER_UNCERTAIN or raw_answer == _SOLVER_UNCERTAIN:
                raw_answer = fallback.answer
            fallback_used = True
            fallback_explanation = fallback.explanation
        except Exception as exc:  # Fallback failure must not hide symbolic diagnostics.
            fallback_error = f"{type(exc).__name__}: {exc}"

    cause = symbolic_cause if raw_answer == _SOLVER_UNCERTAIN else None
    answer = _normalize_answer(raw_answer)

    # --- Debug FOL text -------------------------------------------------------
    premise_items = [
        {
            "id": f"premise-{i}",
            "original_text": text,
            "fol": repr(tree),
            "ast": fol_node_to_dict(tree),
        }
        for i, (text, tree) in enumerate(
            zip(premise_bundle.premises, premise_fols),
            start=1,
        )
    ]
    question_items = _question_debug_items(q_bundle)

    fol_lines = [f"{item['id']}: {item['fol']}" for item in premise_items + question_items]
    fol_text = "\n".join(fol_lines)

    return PredictionResponse(
        id=payload.query_id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=question_type,
        answer=answer,
        explanation=(
            f"LLM fallback after {fallback_trigger}: {answer}. {fallback_explanation}"
            if fallback_used
            else f"Z3 {spec.solver_mode} result: {answer}"
            if cause is None
            else f"Answer: {answer} (cause: {cause})"
        ),
        fol=fol_text,
        cot=[
            (
                "Premise verification passed."
                if premise_bundle.verified
                else "Premise verification blocked; solver execution was skipped."
            ),
            f"Question classified as {spec.question_format}/{spec.solver_mode}"
            + (f" ({spec.can_interpretation})" if spec.can_interpretation != "none" else ""),
            f"Pipeline returned: {answer}"
            + (f" (cause: {cause})" if cause is not None else ""),
            *(
                [f"LLM fallback returned: {answer}"]
                if fallback_used
                else []
            ),
        ],
        premises=premise_bundle.premises,
        premises_used=premises_used if premises_used else None,
        confidence=None,
        routing_diagnostics={
            "stage": "z3_entailment",
            "solver_available": solver is not None,
            "solver_used": solver_used,
            "symbolic_answer": symbolic_answer,
            "uncertainty_cause": cause,
            "symbolic_uncertainty_cause": symbolic_cause,
            "z3_uncertainty_interpretation": uncertainty_interpretation,
            "fallback_used": fallback_used,
            "fallback_trigger": fallback_trigger,
            "fallback_error": fallback_error,
            "fallback_explanation": fallback_explanation,
            "proof_connectivity": proof_connectivity,
            "premise_bundle_verified": premise_bundle.verified,
            "premise_verification_issues": list(premise_bundle.verification_issues),
            "premise_blocking_issues": list(premise_bundle.blocking_issues),
            "premise_warnings": list(premise_bundle.warnings),
            "premise_predicate_renames": premise_bundle.predicate_renames,
            "query_spec": _query_spec_to_dict(q_bundle, proof_connectivity),
            "premise_schema": {
                "predicates": [
                    asdict(predicate) for predicate in premise_bundle.schema.predicates
                ],
                "constants": [
                    asdict(constant) for constant in premise_bundle.schema.constants
                ],
                "diagnostics": list(premise_bundle.schema.diagnostics),
            },
            "parsed_premises": premise_items,
            "parsed_question": question_items,
        },
    )


# ---------------------------------------------------------------------------
# Uncertainty attribution
# ---------------------------------------------------------------------------

# Maps a QuerySpec issue prefix to its uncertainty_cause bucket.
_ISSUE_CAUSE_MAP = (
    ("QUERY_MODE_DEFERRED", "QUERY_MODE_DEFERRED"),
    ("QUERY_NO_CLAIM", "NO_CLAIM_FOL"),
    ("QUERY_NO_YNU_OPTIONS", "NO_SOLVABLE_OPTIONS"),
    ("QUERY_NO_SOLVABLE_OPTIONS", "NO_SOLVABLE_OPTIONS"),
    ("QUERY_OPTIONS_UNSUPPORTED", "NO_SOLVABLE_OPTIONS"),
    ("QUERY_OPEN_WH_UNSUPPORTED", "QUERY_UNSUPPORTED"),
)


def _uncertainty_cause(
    solver: FOLSolver | None,
    premise_bundle: Any,
    spec: Any,
    raw_answer: str,
    sort_conflict: bool,
) -> str | None:
    """Attribute *why* the pipeline could not return a definite answer.

    Returns None when the solver ran and produced a definite Yes/No/label.
    Otherwise returns one bucket from the uncertainty taxonomy so eval can
    separate real logical uncertainty from parser/verifier/mode gaps.
    """

    if sort_conflict:
        return "Z3_SORT_CONFLICT"
    if solver is None:
        return "SOLVER_NOT_CONFIGURED"
    if not premise_bundle.verified:
        first = premise_bundle.blocking_issues[0] if premise_bundle.blocking_issues else "UNKNOWN"
        return f"PREMISE_VERIFICATION_BLOCKED:{first.split(':')[0]}"
    if not spec.supported:
        for issue in spec.issues:
            for prefix, bucket in _ISSUE_CAUSE_MAP:
                if issue.startswith(prefix):
                    if bucket == "QUERY_MODE_DEFERRED":
                        return f"QUERY_MODE_DEFERRED:{spec.solver_mode}"
                    return bucket
        return "QUERY_UNSUPPORTED"
    if raw_answer == _SOLVER_UNCERTAIN:
        return "Z3_TRUE_UNCERTAIN"
    return None


# ---------------------------------------------------------------------------
# Z3 routing
# ---------------------------------------------------------------------------

def _solve(
    solver: FOLSolver,
    premise_fols: list[FOLNode],
    spec: Any,
    q_bundle: QuestionParseBundle,
) -> tuple[str, list[int]]:
    """Dispatch to the right solver check based on the verified QuerySpec.

    Returns (answer, premises_used) where premises_used is a sorted list of
    0-based indices into premise_fols.  MCQ modes that do not use a single
    entailment check return [] for premises_used.
    """

    if spec.question_format != "mcq":
        # Polar entailment. negate_claim flips Yes/No when the question asks for falsity.
        assert q_bundle.main_claim_fol is not None
        answer, used = solver.check_ynu_with_used(premise_fols, q_bundle.main_claim_fol)
        return (_flip(answer) if spec.negate_claim else answer), used

    if spec.solver_mode == "ynu_mapped":
        assert q_bundle.main_claim_fol is not None
        ynu, used = solver.check_ynu_with_used(premise_fols, q_bundle.main_claim_fol)
        return _map_ynu_to_label(ynu, spec.option_claims), used

    option_fols = {c.label: c.fol for c in spec.option_claims if c.fol is not None}
    if spec.solver_mode == "refutation":
        return solver.check_mcq_refutation(premise_fols, option_fols), []
    if spec.solver_mode == "fewest_premise":
        return solver.check_mcq_fewest_premises(premise_fols, option_fols), []
    none_of_above_label = next(
        (c.label for c in spec.option_claims if c.role == "NONE_OF_ABOVE"), None
    )
    return solver.check_mcq(premise_fols, option_fols, none_of_above_label), []


def _flip(answer: str) -> str:
    if answer == "Yes":
        return "No"
    if answer == "No":
        return "Yes"
    return answer


def _map_ynu_to_label(ynu: str, option_claims: tuple[OptionClaim, ...]) -> str:
    """Map a Yes/No/Uncertain verdict to the label of the matching YNU option."""

    target = {"Yes": "yes", "No": "no", "Uncertain": "uncertain"}.get(ynu)
    for claim in option_claims:
        if claim.ynu_value == target:
            return claim.label
    return _SOLVER_UNCERTAIN


def _normalize_answer(answer: str) -> str:
    """Replace the internal uncertain literal with the configured output token."""

    if answer == _SOLVER_UNCERTAIN:
        return get_settings().type1_uncertain_token
    return answer


# ---------------------------------------------------------------------------
# Option normalisation
# ---------------------------------------------------------------------------

_YNU_TOKENS = frozenset({"yes", "no", "uncertain", "unknown"})


def _is_ynu_option_list(values: list[str]) -> bool:
    """True when every option is a bare Yes/No/Uncertain token (a polar set)."""
    return bool(values) and all(str(v).strip().lower() in _YNU_TOKENS for v in values)


def _normalize_options(options: Any) -> dict[str, str]:
    """Return {label: text} for MCQ options, or {} for YNU/polar.

    A list of bare Yes/No/Uncertain tokens is NOT a multiple-choice set — it is
    the answer space of a polar question, so it normalizes to {} (the pipeline
    then routes through the native YNU path instead of treating it as MCQ).
    """
    if options is None:
        return {}
    if isinstance(options, dict):
        if _is_ynu_option_list(list(options.values())):
            return {}
        return {str(k): str(v) for k, v in options.items()}
    if isinstance(options, list):
        if _is_ynu_option_list(options):
            return {}
        return {chr(ord("A") + i): str(v) for i, v in enumerate(options) if i < 5}
    return {}


# ---------------------------------------------------------------------------
# Debug serialisation
# ---------------------------------------------------------------------------

def _question_debug_items(q_bundle: QuestionParseBundle) -> list[dict[str, Any]]:
    spec = q_bundle.spec
    if spec.question_format == "mcq":
        return [
            {
                "id": claim.label,
                "original_text": claim.claim_text or claim.raw_fol or "",
                "fol": repr(claim.fol) if claim.fol is not None else None,
                "ast": fol_node_to_dict(claim.fol) if claim.fol is not None else None,
            }
            for claim in spec.option_claims
        ]
    return [
        {
            "id": "question",
            "original_text": spec.main_claim_text or q_bundle.question,
            "fol": repr(q_bundle.main_claim_fol) if q_bundle.main_claim_fol is not None else None,
            "ast": (
                fol_node_to_dict(q_bundle.main_claim_fol)
                if q_bundle.main_claim_fol is not None
                else None
            ),
        }
    ]


def _query_spec_to_dict(
    q_bundle: QuestionParseBundle,
    proof_connectivity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = q_bundle.spec
    connectivity_by_id = {
        claim["claim_id"]: claim
        for claim in (proof_connectivity or {}).get("claims", [])
    }
    result = {
        "question_format": spec.question_format,
        "solver_mode": spec.solver_mode,
        "can_interpretation": spec.can_interpretation,
        "main_claim_text": spec.main_claim_text,
        "main_claim_fol": (
            repr(q_bundle.main_claim_fol) if q_bundle.main_claim_fol is not None else None
        ),
        "negate_claim": spec.negate_claim,
        "supported": spec.supported,
        "issues": list(spec.issues),
        "marker_style": (
            q_bundle.option_bundle.marker_style if q_bundle.option_bundle else None
        ),
        "role_distribution": (
            q_bundle.option_bundle.role_distribution if q_bundle.option_bundle else None
        ),
        "extraction_diagnostics": (
            list(q_bundle.option_bundle.extraction_diagnostics)
            if q_bundle.option_bundle
            else []
        ),
        "option_claims": [
            _option_claim_to_dict(c, connectivity_by_id.get(c.label))
            for c in spec.option_claims
        ],
    }
    if proof_connectivity is not None:
        result["main_claim_proof_connectivity"] = connectivity_by_id.get("question")
    return result


def _option_claim_to_dict(
    c: OptionClaim,
    proof_connectivity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proof_diagnostics = (
        proof_connectivity.get("diagnostics", []) if proof_connectivity else []
    )
    result = {
        "label": c.label,
        "role": c.role,
        "normalized_text": c.normalized_text,
        "claim_text": c.claim_text,
        "ynu_value": c.ynu_value,
        "premise_indices": list(c.premise_indices),
        "raw_fol": c.raw_fol,
        "is_selectable": c.is_selectable,
        "fol": repr(c.fol) if c.fol is not None else None,
        "diagnostics": list(dict.fromkeys([*c.diagnostics, *proof_diagnostics])),
    }
    if proof_connectivity is not None:
        result["proof_connectivity"] = proof_connectivity
    return result


# ---------------------------------------------------------------------------
# AST → JSON serialisation
# ---------------------------------------------------------------------------

def fol_node_to_dict(node: FOLNode) -> dict[str, Any]:
    """Convert a recursive FOL dataclass tree into a JSON-safe dictionary."""

    if isinstance(node, AtomicNode):
        return {
            "type": "atomic",
            "predicate": node.predicate.model_dump(mode="json"),
            "arguments": list(node.arguments),
        }
    if isinstance(node, QuantifiedNode):
        return {
            "type": "quantified",
            "quantifier": node.quantifier,
            "variable": node.variable,
            "body": fol_node_to_dict(node.body),
            "restrictor": (
                fol_node_to_dict(node.restrictor)
                if node.restrictor is not None
                else None
            ),
        }
    if isinstance(node, ComparisonNode):
        return {
            "type": "comparison",
            "operator": node.operator,
            "left": {"name": node.left.name, "arguments": list(node.left.arguments)},
            "right": repr(node.right),
        }
    # LogicalNode
    return {
        "type": "logical",
        "operator": node.operator,  # type: ignore[union-attr]
        "left": fol_node_to_dict(node.left),  # type: ignore[union-attr]
        "right": (
            fol_node_to_dict(node.right)  # type: ignore[union-attr]
            if node.right is not None  # type: ignore[union-attr]
            else None
        ),
    }
