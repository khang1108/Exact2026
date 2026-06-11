"""Type 1 logic pipeline entry point.

The current stage translates all natural-language premises into recursive FOL
ASTs using the shared asynchronous parser service. A later reasoning stage will
consume these ASTs to select an answer; until then the pipeline reports an
explicit ``unknown`` answer instead of fabricating a solved result.
"""

from __future__ import annotations

from typing import Any

from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.type1.ast import AtomicNode, FOLNode, LogicalNode, QuantifiedNode
from exact.type1.parser import FOLParser


async def run_type1_pipeline(
    payload: PredictionRequest,
    parser: FOLParser,
) -> PredictionResponse:
    """Translate a Type 1 request's premises to FOL using concurrent inference."""

    # Step 1: Get all premises.
    premises = [premise.strip() for premise in payload.premises or [] if premise.strip()]
    if not premises:
        raise ValueError("Type 1 requests require at least one non-empty premise")

    # Step 2: Use FOLParser to parse all premises-NL to FOL.
    trees = await parser.parse_many(premises)
    parsed_premises = [
        {
            "id": f"premise-{index}",
            "original_text": premise,
            "fol": repr(tree),
            "ast": fol_node_to_dict(tree),
        }
        for index, (premise, tree) in enumerate(zip(premises, trees, strict=True), start=1)
    ]
    fol_text = "\n".join(f"{item['id']}: {item['fol']}" for item in parsed_premises)

    return PredictionResponse(
        id=payload.query_id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=QuestionType.MCQ if payload.options else QuestionType.YNU,
        answer="unknown",
        explanation=(
            "Premises were translated to FOL. The Type 1 reasoning and answer-selection "
            "stage is not connected yet."
        ),
        fol=fol_text,
        cot=["Parsed all Type 1 premises concurrently with the self-hosted parser model."],
        premises=premises,
        confidence=0.0,
        routing_diagnostics={
            "stage": "fol_parsing",
            "parsed_premises": parsed_premises,
        },
    )


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
    if isinstance(node, LogicalNode):
        return {
            "type": "logical",
            "operator": node.operator,
            "left": fol_node_to_dict(node.left),
            "right": fol_node_to_dict(node.right) if node.right is not None else None,
        }
    raise TypeError(f"Unsupported FOL node: {type(node).__name__}")
