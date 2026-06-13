"""Type 1 logic pipeline entry point.

Workflow per request:
  1. Parse and verify declarative premises through ``PremiseParser``.
  2. Parse question options or the conclusion through the wrapped ``FOLParser``.
  3. Canonicalize conclusion predicate names against the premise schema.
  4. Z3 entailment check → "Yes" / "No" / "Uncertain" / option label.
  5. Return answer with FOL debug info.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.type1.ast import AtomicNode, FOLNode, QuantifiedNode
from exact.type1.ast.nodes import ComparisonNode
from exact.type1.parser import PremiseParser

if TYPE_CHECKING:
    from exact.type1.solvers import FOLSolver  # type: ignore[import-untyped]

async def run_type1_pipeline(
    payload: PredictionRequest,
    premise_parser: PremiseParser,
    solver: FOLSolver | None = None,
) -> PredictionResponse:
    """Parse premises and conclusions through their distinct workflows, then solve."""

    premises = [p.strip() for p in payload.premises or [] if p.strip()]
    if not premises:
        raise ValueError("Type 1 requests require at least one non-empty premise")

    premise_bundle = await premise_parser.parse_premises(premises)

    options_dict = _normalize_options(payload.options)
    is_mcq = bool(options_dict)
    question_type = QuestionType.MCQ if is_mcq else QuestionType.YNU

    if is_mcq:
        conclusion_sentences = list(options_dict.values())
    else:
        conclusion_sentences = [payload.question]

    conclusion_fols_raw = await premise_parser.fol_parser.parse_many(conclusion_sentences)
    conclusion_fols, conclusion_renames = premise_bundle.schema.canonicalize(
        conclusion_fols_raw
    )
    premise_fols = premise_bundle.trees

    # --- Z3 entailment -------------------------------------------------------
    solver_used = solver is not None and premise_bundle.verified
    if solver_used:
        if is_mcq:
            option_fols = dict(zip(options_dict.keys(), conclusion_fols))
            assert solver is not None
            answer = solver.check_mcq(premise_fols, option_fols)
        else:
            assert solver is not None
            answer = solver.check_ynu(premise_fols, conclusion_fols[0])
    else:
        answer = "Uncertain"

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
    conclusion_items = [
        {
            "id": label if is_mcq else "question",
            "original_text": text,
            "fol": repr(tree),
            "ast": fol_node_to_dict(tree),
        }
        for (label, text), tree in zip(
            (options_dict.items() if is_mcq else [("question", payload.question)]),
            conclusion_fols,
        )
    ]

    fol_lines = [f"{item['id']}: {item['fol']}" for item in premise_items + conclusion_items]
    fol_text = "\n".join(fol_lines)

    return PredictionResponse(
        id=payload.query_id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=question_type,
        answer=answer,
        explanation=(
            f"Z3 entailment result: {answer}"
            if solver_used
            else (
                "Z3 solver skipped because no solver was configured "
                "or premise verification failed."
            )
        ),
        fol=fol_text,
        cot=[
            (
                "Premise verification passed before parsing conclusions."
                if premise_bundle.verified
                else "Premise verification failed; solver execution was skipped."
            ),
            f"Pipeline returned: {answer}",
        ],
        premises=premise_bundle.premises,
        confidence=None,
        routing_diagnostics={
            "stage": "z3_entailment",
            "solver_available": solver is not None,
            "solver_used": solver_used,
            "premise_bundle_verified": premise_bundle.verified,
            "premise_verification_issues": list(premise_bundle.verification_issues),
            "premise_predicate_renames": premise_bundle.predicate_renames,
            "conclusion_predicate_renames": conclusion_renames,
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
            "parsed_conclusions": conclusion_items,
        },
    )


# ---------------------------------------------------------------------------
# Option normalisation
# ---------------------------------------------------------------------------

def _normalize_options(options: Any) -> dict[str, str]:
    """Return {label: text} for MCQ options, or {} for YNU."""
    if options is None:
        return {}
    if isinstance(options, dict):
        return {str(k): str(v) for k, v in options.items()}
    if isinstance(options, list):
        return {chr(ord("A") + i): str(v) for i, v in enumerate(options) if i < 5}
    return {}


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
