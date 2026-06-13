"""Type 1 logic pipeline entry point.

Workflow per request:
  1. Normalize premises and (MCQ options or question) into a flat sentence list.
  2. Parse all sentences in one concurrent batch → FOL ASTs.
  3. Align conclusion predicate names to premise predicate names (word similarity).
  4. Z3 entailment check → "Yes" / "No" / "Uncertain" / option label.
  5. If Uncertain and refiner available:
       a. 7B refiner inspects (NL, FOL) pairs → rephrased NL for bad translations.
       b. Re-parse corrected sentences with 1.7B parser.
       c. Re-align predicates, retry Z3 once.
  6. Return answer with FOL debug info.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.config import get_settings
from exact.type1.ast import AtomicNode, FOLNode, QuantifiedNode
from exact.type1.ast.nodes import LogicalNode
from exact.type1.models.schemas import Predicate
from exact.type1.parser import FOLParser

if TYPE_CHECKING:
    from exact.type1.refiner import Type1Refiner
    from exact.type1.solvers import FOLSolver  # type: ignore[import-untyped]

# Minimum Jaccard word-similarity to rename a conclusion predicate to a premise predicate.
_ALIGN_THRESHOLD = 0.6


async def run_type1_pipeline(
    payload: PredictionRequest,
    parser: FOLParser,
    solver: FOLSolver | None = None,
    refiner: Type1Refiner | None = None,
) -> PredictionResponse:
    """Run the Type 1 pipeline under a hard end-to-end deadline.

    Guarantees a response within ``type1_request_deadline_seconds``: if parse →
    solve → refine exceeds the budget the in-flight work is cancelled and a
    graceful "Uncertain" answer is returned instead of running unbounded.
    """
    deadline = get_settings().type1_request_deadline_seconds
    try:
        return await asyncio.wait_for(
            _run_type1_core(payload, parser, solver, refiner),
            timeout=deadline,
        )
    except asyncio.TimeoutError:
        return _timeout_response(payload, deadline)


async def _run_type1_core(
    payload: PredictionRequest,
    parser: FOLParser,
    solver: FOLSolver | None,
    refiner: Type1Refiner | None,
) -> PredictionResponse:
    """Parse → Z3 entailment → self-refinement loop if Uncertain."""

    premises = [p.strip() for p in payload.premises or [] if p.strip()]
    if not premises:
        raise ValueError("Type 1 requests require at least one non-empty premise")

    options_dict = _normalize_options(payload.options)
    is_mcq = bool(options_dict)
    question_type = QuestionType.MCQ if is_mcq else QuestionType.YNU
    option_labels = list(options_dict.keys()) if is_mcq else ["question"]

    conclusion_sentences = list(options_dict.values()) if is_mcq else [payload.question]

    all_trees = await parser.parse_many(premises + conclusion_sentences)
    premise_fols = all_trees[: len(premises)]
    conclusion_fols, renames = _align_predicates(all_trees[len(premises):], premise_fols)

    # --- Z3 entailment -------------------------------------------------------
    answer = _z3_check(solver, is_mcq, options_dict, premise_fols, conclusion_fols)

    # --- Self-refinement loop (only when Z3 is Uncertain) --------------------
    refinement_log: list[dict] = []
    if answer == "Uncertain" and refiner is not None:
        premise_fols, conclusion_fols, premises, conclusion_sentences, refinement_log = (
            await _refine_and_retry(
                premises, premise_fols,
                option_labels, conclusion_sentences, conclusion_fols,
                refiner, parser,
            )
        )
        renames_after, _ = _align_predicates(conclusion_fols, premise_fols)
        conclusion_fols = renames_after
        answer = _z3_check(solver, is_mcq, options_dict, premise_fols, conclusion_fols)

    answered_by = "z3"

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
            "id": label,
            "original_text": text,
            "fol": repr(tree),
            "ast": fol_node_to_dict(tree),
        }
        for label, text, tree in zip(option_labels, conclusion_sentences, conclusion_fols)
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
            "refiner_available": refiner is not None,
            "predicate_renames": renames,
            "refinement_log": refinement_log,
            "parsed_premises": premise_items,
            "parsed_conclusions": conclusion_items,
        },
    )


def _timeout_response(payload: PredictionRequest, deadline: float) -> PredictionResponse:
    """Graceful 'Uncertain' answer returned when the request exceeds its deadline."""
    is_mcq = bool(_normalize_options(payload.options))
    return PredictionResponse(
        id=payload.query_id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=QuestionType.MCQ if is_mcq else QuestionType.YNU,
        answer="Uncertain",
        explanation=f"Request exceeded the {deadline:.0f}s deadline; returning Uncertain.",
        fol="",
        cot=[f"Deadline of {deadline:.0f}s exceeded before the solver could decide."],
        premises=[p.strip() for p in payload.premises or [] if p.strip()],
        confidence=None,
        routing_diagnostics={"stage": "deadline_exceeded", "deadline_seconds": deadline},
    )


def _z3_check(
    solver: FOLSolver | None,
    is_mcq: bool,
    options_dict: dict[str, str],
    premise_fols: list[FOLNode],
    conclusion_fols: list[FOLNode],
) -> str:
    if solver is None:
        return "Uncertain"
    try:
        if is_mcq:
            return solver.check_mcq(premise_fols, dict(zip(options_dict.keys(), conclusion_fols)))
        return solver.check_ynu(premise_fols, conclusion_fols[0])
    except Exception:
        return "Uncertain"


async def _refine_and_retry(
    premises: list[str],
    premise_fols: list[FOLNode],
    option_labels: list[str],
    conclusion_sentences: list[str],
    conclusion_fols: list[FOLNode],
    refiner: Type1Refiner,
    parser: FOLParser,
) -> tuple[list[FOLNode], list[FOLNode], list[str], list[str], list[dict]]:
    """Call refiner, re-parse corrected sentences, return updated data + refinement log."""
    items = [
        {"id": f"premise-{i}", "nl": text, "fol": repr(fol)}
        for i, (text, fol) in enumerate(zip(premises, premise_fols), start=1)
    ] + [
        {"id": label, "nl": text, "fol": repr(fol)}
        for label, text, fol in zip(option_labels, conclusion_sentences, conclusion_fols)
    ]

    corrections = await refiner.refine(items)
    if not corrections:
        return premise_fols, conclusion_fols, premises, conclusion_sentences, []

    # Collect (target, idx, new_text) jobs in order so parse_many returns in sync.
    jobs: list[tuple[str, int, str]] = []
    for id_str, rephrased in corrections.items():
        m = re.match(r"^premise-(\d+)$", id_str)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(premises):
                jobs.append(("p", idx, rephrased))
            continue
        if id_str in option_labels:
            jobs.append(("c", option_labels.index(id_str), rephrased))

    if not jobs:
        return premise_fols, conclusion_fols, premises, conclusion_sentences, []

    new_trees = await parser.parse_many([text for _, _, text in jobs])

    new_premise_fols = list(premise_fols)
    new_conclusion_fols = list(conclusion_fols)
    new_premises = list(premises)
    new_conclusion_sentences = list(conclusion_sentences)
    refinement_log: list[dict] = []

    for (target, idx, text), tree in zip(jobs, new_trees):
        if target == "p":
            refinement_log.append({"id": f"premise-{idx+1}", "original": new_premises[idx], "rephrased": text})
            new_premises[idx] = text
            new_premise_fols[idx] = tree
        else:
            refinement_log.append({"id": option_labels[idx], "original": new_conclusion_sentences[idx], "rephrased": text})
            new_conclusion_sentences[idx] = text
            new_conclusion_fols[idx] = tree

    return new_premise_fols, new_conclusion_fols, new_premises, new_conclusion_sentences, refinement_log


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
