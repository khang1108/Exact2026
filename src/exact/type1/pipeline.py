"""Type 1 logic pipeline entry point.

Workflow per request:
  1. Normalize premises and (MCQ options or question) into a flat sentence list.
  2. Parse all sentences in one concurrent batch → FOL ASTs.
  3. Align conclusion predicate names to premise predicate names (word similarity).
  4. Z3 entailment check → "Yes" / "No" / "Uncertain" / option label.
  5. Return answer with FOL debug info.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.type1.ast import AtomicNode, FOLNode, QuantifiedNode
from exact.type1.ast.nodes import LogicalNode
from exact.type1.models.schemas import Predicate
from exact.type1.parser import FOLParser

if TYPE_CHECKING:
    from exact.type1.llm_head import Type1LLMHead
    from exact.type1.solvers import FOLSolver  # type: ignore[import-untyped]

# Minimum Jaccard word-similarity to rename a conclusion predicate to a premise predicate.
_ALIGN_THRESHOLD = 0.6


async def run_type1_pipeline(
    payload: PredictionRequest,
    parser: FOLParser,
    solver: FOLSolver | None = None,
    llm_head: Type1LLMHead | None = None,
) -> PredictionResponse:
    """Parse → Z3 entailment → LLM head fallback when Z3 is Uncertain."""

    premises = [p.strip() for p in payload.premises or [] if p.strip()]
    if not premises:
        raise ValueError("Type 1 requests require at least one non-empty premise")

    options_dict = _normalize_options(payload.options)
    is_mcq = bool(options_dict)
    question_type = QuestionType.MCQ if is_mcq else QuestionType.YNU

    if is_mcq:
        conclusion_sentences = list(options_dict.values())
    else:
        conclusion_sentences = [payload.question]

    all_sentences = premises + conclusion_sentences
    all_trees = await parser.parse_many(all_sentences)

    premise_fols = all_trees[: len(premises)]
    conclusion_fols_raw = all_trees[len(premises) :]

    # Align: rename conclusion predicates that differ only by slight wording from
    # premise predicates (e.g. "QualifiesForUniversityScholarship" → "QualifiesForScholarship").
    conclusion_fols, renames = _align_predicates(conclusion_fols_raw, premise_fols)

    # --- Z3 entailment -------------------------------------------------------
    if solver is not None:
        if is_mcq:
            option_fols = dict(zip(options_dict.keys(), conclusion_fols))
            answer = solver.check_mcq(premise_fols, option_fols)
        else:
            answer = solver.check_ynu(premise_fols, conclusion_fols[0])
    else:
        answer = "Uncertain"

    # --- LLM head fallback (only when Z3 could not decide) -------------------
    answered_by = "z3"
    if answer == "Uncertain" and llm_head is not None:
        if is_mcq:
            answer = await llm_head.answer_mcq(premises, payload.question, options_dict)
        else:
            answer = await llm_head.answer_ynu(premises, payload.question)
        answered_by = "llm_head"

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
            "answered_by": answered_by,
            "solver_available": solver is not None,
            "llm_head_available": llm_head is not None,
            "predicate_renames": renames,
            "parsed_premises": premise_items,
            "parsed_conclusions": conclusion_items,
        },
    )


# ---------------------------------------------------------------------------
# Predicate alignment helpers
# ---------------------------------------------------------------------------

def _camel_words(name: str) -> frozenset[str]:
    """'QualifiesForScholarship' → frozenset({'qualifies', 'for', 'scholarship'})."""
    return frozenset(w.lower() for w in re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", name))


def _collect_predicates(node: FOLNode) -> set[tuple[str, int]]:
    """Return all (predicate_name, arity) pairs appearing in an AST."""
    if isinstance(node, AtomicNode):
        return {(node.predicate.name, len(node.arguments))}
    if isinstance(node, QuantifiedNode):
        return _collect_predicates(node.body)
    # LogicalNode
    result = _collect_predicates(node.left)
    if node.right is not None:
        result |= _collect_predicates(node.right)
    return result


def _rename_in_node(node: FOLNode, remap: dict[tuple[str, int], str]) -> FOLNode:
    """Return a copy of node with predicates renamed according to remap."""
    if isinstance(node, AtomicNode):
        key = (node.predicate.name, len(node.arguments))
        if key not in remap:
            return node
        new_pred = Predicate(
            name=remap[key],
            arg_sorts=node.predicate.arg_sorts,
            aliases=node.predicate.aliases,
        )
        return AtomicNode(predicate=new_pred, arguments=node.arguments)
    if isinstance(node, QuantifiedNode):
        return QuantifiedNode(
            node.quantifier, node.variable, _rename_in_node(node.body, remap)
        )
    # LogicalNode
    new_left = _rename_in_node(node.left, remap)
    new_right = _rename_in_node(node.right, remap) if node.right is not None else None
    return LogicalNode(node.operator, new_left, new_right)


def _align_predicates(
    conclusion_nodes: list[FOLNode],
    premise_nodes: list[FOLNode],
) -> tuple[list[FOLNode], list[dict]]:
    """
    For each conclusion predicate not in premises, find the best-matching premise
    predicate by CamelCase word-set Jaccard similarity and rename it if ≥ threshold.

    Returns (aligned_conclusions, rename_log).
    """
    # Build {(name, arity): word_set} from premises
    premise_vocab: dict[tuple[str, int], frozenset[str]] = {}
    for node in premise_nodes:
        for name, arity in _collect_predicates(node):
            premise_vocab[(name, arity)] = _camel_words(name)

    if not premise_vocab:
        return conclusion_nodes, []

    # Build rename map for conclusion predicates that are absent from premises
    remap: dict[tuple[str, int], str] = {}
    for node in conclusion_nodes:
        for name, arity in _collect_predicates(node):
            if (name, arity) in premise_vocab:
                continue  # exact match — nothing to do
            words = _camel_words(name)
            best_sim, best_name = 0.0, None
            for (pname, parity), pwords in premise_vocab.items():
                if parity != arity:
                    continue
                union = words | pwords
                sim = len(words & pwords) / len(union) if union else 0.0
                if sim > best_sim:
                    best_sim, best_name = sim, pname
            if best_name is not None and best_sim >= _ALIGN_THRESHOLD:
                remap[(name, arity)] = best_name

    rename_log = [
        {"from": k[0], "arity": k[1], "to": v} for k, v in remap.items()
    ]

    if not remap:
        return conclusion_nodes, rename_log

    return [_rename_in_node(n, remap) for n in conclusion_nodes], rename_log


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
        }
    return {
        "type": "logical",
        "operator": node.operator,
        "left": fol_node_to_dict(node.left),
        "right": fol_node_to_dict(node.right) if node.right is not None else None,
    }
