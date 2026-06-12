"""Type 1 logic pipeline entry point.

Workflow per request:
  1. Normalize premises and (MCQ options or question) into a flat sentence list.
  2. Parse all sentences in one concurrent batch → FOL ASTs.
  3. Z3 entailment check → "Yes" / "No" / "Uncertain" / option label.
  4. Return answer with FOL debug info.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.type1.ast import AtomicNode, FOLNode, QuantifiedNode
from exact.type1.parser import FOLParser

if TYPE_CHECKING:
    from exact.type1.solvers import FOLSolver  # type: ignore[import-untyped]


async def run_type1_pipeline(
    payload: PredictionRequest,
    parser: FOLParser,
    solver: FOLSolver | None = None,
) -> PredictionResponse:
    """Parse premises + conclusion in one batch, then solve via Z3."""

    premises = [p.strip() for p in payload.premises or [] if p.strip()]
    if not premises:
        raise ValueError("Type 1 requests require at least one non-empty premise")

    options_dict = _normalize_options(payload.options)
    is_mcq = bool(options_dict)
    question_type = QuestionType.MCQ if is_mcq else QuestionType.YNU

    # Build a flat list: premises first, then conclusion sentences.
    # MCQ: parse each option as a candidate conclusion.
    # YNU: parse the question itself as the conclusion.
    if is_mcq:
        conclusion_sentences = list(options_dict.values())
    else:
        conclusion_sentences = [payload.question]

    all_sentences = premises + conclusion_sentences
    all_trees = await parser.parse_many(all_sentences)

    premise_fols = all_trees[: len(premises)]
    conclusion_fols = all_trees[len(premises) :]

    # --- Z3 entailment -------------------------------------------------------
    if solver is not None:
        if is_mcq:
            option_fols = dict(zip(options_dict.keys(), conclusion_fols))
            answer = solver.check_mcq(premise_fols, option_fols)
        else:
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
        for i, (text, tree) in enumerate(zip(premises, premise_fols), start=1)
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
        explanation=f"Z3 entailment result: {answer}",
        fol=fol_text,
        cot=["Parsed all sentences concurrently.", f"Z3 solver returned: {answer}"],
        premises=premises,
        confidence=None,
        routing_diagnostics={
            "stage": "z3_entailment",
            "solver_available": solver is not None,
            "parsed_premises": premise_items,
            "parsed_conclusions": conclusion_items,
        },
    )


def _normalize_options(options: Any) -> dict[str, str]:
    """Return {label: text} for MCQ options, or {} for YNU."""
    if options is None:
        return {}
    if isinstance(options, dict):
        return {str(k): str(v) for k, v in options.items()}
    if isinstance(options, list):
        return {chr(ord("A") + i): str(v) for i, v in enumerate(options) if i < 5}
    return {}


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
        }
    return {
        "type": "logical",
        "operator": node.operator,
        "left": fol_node_to_dict(node.left),
        "right": fol_node_to_dict(node.right) if node.right is not None else None,
    }
