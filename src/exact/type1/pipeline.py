"""Type 1 logic pipeline entry point.

Workflow per request:
  1. Parse and verify declarative premises through ``PremiseParser``.
  2. Classify the question + interpret options through ``QuestionSideParser``.
  3. Route on the resulting ``QuerySpec`` to the right Z3 check.
  4. Z3 entailment / refutation / YNU-mapping → answer label.
  5. Normalize the uncertain token and return answer with FOL debug info.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
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
    from exact.type1.parser.theory_translator import TheoryTranslation, TheoryTranslator
    from exact.type1.solvers import FOLSolver  # type: ignore[import-untyped]

# Internal uncertain literal emitted by the solver, normalized at the boundary.
_SOLVER_UNCERTAIN = "Uncertain"


async def run_type1_pipeline(
    payload: PredictionRequest,
    premise_parser: PremiseParser,
    question_parser: QuestionSideParser,
    solver: FOLSolver | None = None,
    fallback_reasoner: Type1FallbackReasoner | None = None,
    theory_translator: TheoryTranslator | None = None,
    translator_mode: str | None = None,
) -> PredictionResponse:
    """Parse premises and the question through their workflows, then solve.

    ``translator_mode`` (per-request override, e.g. ``?translator=single_pass``)
    takes precedence over the EXACT_TYPE1_TRANSLATOR setting so A/B runs need no
    server restart.
    """

    premises = [p.strip() for p in payload.premises or [] if p.strip()]
    if not premises:
        raise ValueError("Type 1 requests require at least one non-empty premise")

    mode = translator_mode or get_settings().type1_translator
    use_single_pass = mode == "single_pass"
    if mode == "hybrid":
        # MCQ benefits from whole-theory consistent predicates; polar/YNU keeps
        # the decompose path (epistemic-witness Uncertain handling, etc.).
        use_single_pass = bool(_normalize_options(payload.options)) or bool(
            extract_mcq(payload.question).options
        )
    if use_single_pass and theory_translator is not None:
        return await run_type1_single_pass(
            payload, theory_translator, solver, fallback_reasoner
        )

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

    # --- Epistemic witnesses --------------------------------------------------
    # Meta-premises like "No premise states whether X" disclaim knowledge; their
    # FOL (often ¬X) must not reach Z3 as a fact. Keep them out of the solver
    # input (remapping indices), and if one is specifically about the queried
    # claim, the answer is decisively Uncertain with that premise as witness.
    witness_idx = set(premise_bundle.epistemic_witness_indices)
    solver_index_map = [i for i in range(len(premise_fols)) if i not in witness_idx]
    solver_fols = [premise_fols[i] for i in solver_index_map]
    claim_repr = repr(q_bundle.main_claim_fol) if q_bundle.main_claim_fol is not None else None
    matched_witness: int | None = None
    if claim_repr is not None:
        for i in sorted(witness_idx):
            if claim_repr in {repr(a) for a in _collect_atoms(premise_fols[i])}:
                matched_witness = i
                break

    # --- Z3 routing ----------------------------------------------------------
    solver_used = solver is not None and premise_bundle.verified and spec.supported
    sort_conflict = False
    premises_used: list[int] | None = None
    if solver_used:
        assert solver is not None
        try:
            raw_answer, used_local = _solve(solver, solver_fols, spec, q_bundle)
            premises_used = [solver_index_map[j] for j in used_local]
        except ValueError as exc:
            if "SORT_CONFLICT" in str(exc):
                raw_answer = _SOLVER_UNCERTAIN
                sort_conflict = True
            else:
                raise
    else:
        raw_answer = _SOLVER_UNCERTAIN

    # A witness about the queried claim is decisive: the premises explicitly say
    # they do not know, so the answer is Uncertain and that premise is cited.
    skip_fallback = matched_witness is not None
    if matched_witness is not None:
        raw_answer = _SOLVER_UNCERTAIN
        premises_used = [matched_witness]

    symbolic_answer = raw_answer
    symbolic_cause = (
        "EPISTEMIC_WITNESS_UNCERTAIN"
        if matched_witness is not None
        else _uncertainty_cause(solver, premise_bundle, spec, raw_answer, sort_conflict)
    )
    uncertainty_interpretation = (
        interpret_z3_uncertainty(proof_connectivity)
        if symbolic_cause == "Z3_TRUE_UNCERTAIN"
        else None
    )
    fallback_used = False
    fallback_error: str | None = None
    fallback_explanation: str | None = None
    # --- Fallback trigger -------------------------------------------------------
    # For ranking modes (fewest_premise, strongest_conclusion, premise_selection)
    # Z3 cannot answer meaningfully — these are minimisation/maximisation problems,
    # not entailment queries. Always adjudicate with the LLM for these modes,
    # regardless of whether Z3 happened to return a definite answer or Uncertain.
    # For entailment modes, only fall back when Z3 was uncertain.
    fallback_trigger: str | None
    if spec.solver_mode in {"fewest_premise", "strongest_conclusion", "premise_selection"}:
        fallback_trigger = "RANKING_MODE_ADJUDICATION"
    elif raw_answer == _SOLVER_UNCERTAIN:
        fallback_trigger = symbolic_cause
    elif is_mcq and any(
        "NUMERIC_CONSTRAINT_LOST" in issue
        for issue in premise_bundle.verification_issues
    ):
        # Numeric premises may have their values silently dropped by the FOL
        # parser. If Z3 answered an MCQ confidently but the bundle has numeric
        # loss warnings, the answer may be a false positive.
        fallback_trigger = "NUMERIC_CONSTRAINT_MCQ_VERIFY"
    else:
        fallback_trigger = None
    if skip_fallback:
        fallback_trigger = None
    if fallback_trigger is not None and fallback_reasoner is not None:
        try:
            # Open-ended: a wh-question, or a type-1 query with no options and no
            # testable polar claim. These get a free-form LLM answer (no label
            # set). Everything else carries Uncertain as a legal answer so the
            # fallback can decline to guess instead of over-committing.
            is_open_ended = spec.question_format == "open_wh" or (
                not is_mcq and q_bundle.main_claim_fol is None
            )
            fallback = await fallback_reasoner.answer(
                premises=premises,
                question=payload.question,
                option_labels=(
                    [c.label for c in spec.option_claims] + [_SOLVER_UNCERTAIN]
                    if is_mcq
                    else [] if is_open_ended else ["Yes", "No", _SOLVER_UNCERTAIN]
                ),
                options=options_dict or None,
            )
            if fallback.answer != _SOLVER_UNCERTAIN or raw_answer == _SOLVER_UNCERTAIN:
                symbolic_was_definite = raw_answer != _SOLVER_UNCERTAIN
                raw_answer = fallback.answer
                # Keep the symbolic proof's premises when Z3 already had a
                # definite answer (e.g. NUMERIC_CONSTRAINT_MCQ_VERIFY only
                # confirms it). Only adopt the LLM's premises when the fallback
                # actually decided an otherwise-Uncertain case.
                if fallback.premises_used and not symbolic_was_definite:
                    premises_used = fallback.premises_used
            fallback_used = True
            fallback_explanation = fallback.explanation
        except Exception as exc:  # Fallback failure must not hide symbolic diagnostics.
            fallback_error = f"{type(exc).__name__}: {exc}"

    # Provability questions ("Do the premises prove/establish X?") are decided
    # by entailment alone: if Z3 entailed the claim the answer is Yes, otherwise
    # the premises do NOT establish it → No (never Uncertain, and the LLM
    # fallback guess does not get to flip it to Yes).
    if not is_mcq and _is_provability_question(payload.question):
        raw_answer = "Yes" if symbolic_answer == "Yes" else "No"

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

    return PredictionResponse(
        id=payload.query_id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=question_type,
        answer=answer,
        explanation=_build_explanation(
            answer=answer,
            uncertain_token=get_settings().type1_uncertain_token,
            premises=premise_bundle.premises,
            premises_used=premises_used,
            is_mcq=is_mcq,
            spec=spec,
            fallback_explanation=fallback_explanation if fallback_used else None,
        ),
        fol=fol_lines,
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
            *(
                [f"LLM fallback error: {fallback_error}"]
                if fallback_error
                else []
            ),
        ],
        premises=premise_bundle.premises,
        premises_used=premises_used,  # None = solver not run; [] = uncertain; [0,1,...] = used
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

@dataclass
class _SinglePassAttempt:
    """One translate→refine→solve sample for self-consistency voting."""

    symbolic_answer: str
    premises_used: list[int] | None
    translation: "TheoryTranslation"
    refine_log: list[str]


async def _single_pass_attempt(
    payload: PredictionRequest,
    translator: TheoryTranslator,
    solver: FOLSolver | None,
    options_dict: dict[str, str],
    premises: list[str],
    max_refines: int,
    temperature: float,
    verify_refine: bool = False,
) -> _SinglePassAttempt:
    """Translate (with bounded refinement), optionally verify-refine, then solve.

    A translation failure (e.g. truncated/invalid JSON from the LLM) is caught
    and returned as an empty, unusable translation so the caller routes to the
    LLM fallback — never a hard request error.
    """
    refine_log: list[str] = []
    try:
        translation = await translator.translate(
            premises, payload.question, options_dict or None, temperature=temperature
        )
        for _ in range(max_refines):
            problems = _translation_problems(translation)
            if not problems:
                break
            refine_log.extend(problems)
            translation = await translator.translate(
                premises, payload.question, options_dict or None,
                feedback="\n".join(problems), temperature=temperature,
            )
        # Verify-and-refine: one checklist pass (coreference / deontic / epistemic
        # / predicate consistency) → corrected translation. Kept only if it still
        # parses to usable FOL; otherwise the original draft stands.
        if verify_refine:
            try:
                verified = await translator.verify(
                    premises, payload.question, options_dict or None, translation
                )
                if verified.premise_trees and not any(
                    "TRANSLATE_FAILED" in i for i in verified.issues
                ):
                    translation = verified
                    refine_log.append("VERIFY_REFINE_APPLIED")
            except Exception:
                pass  # verification is best-effort; keep the draft
    except Exception as exc:  # truncated JSON / model error -> route to fallback
        from exact.type1.parser.theory_translator import empty_translation
        return _SinglePassAttempt(
            _SOLVER_UNCERTAIN, None, empty_translation(f"{type(exc).__name__}: {exc}"), refine_log
        )

    premise_fols = translation.premise_trees
    is_mcq = translation.question_format == "mcq"
    raw_answer = _SOLVER_UNCERTAIN
    premises_used: list[int] | None = None
    if solver is not None and premise_fols:
        try:
            if is_mcq and translation.option_trees:
                raw_answer, used_local = solver.check_mcq_with_used(
                    premise_fols, translation.option_trees, None
                )
            elif not is_mcq and translation.claim_tree is not None:
                raw_answer, used_local = solver.check_ynu_with_used(
                    premise_fols, translation.claim_tree
                )
            else:
                used_local = []
            if used_local:
                premises_used = [translation.premise_index_map[j] for j in used_local]
            elif raw_answer != _SOLVER_UNCERTAIN:
                premises_used = []
        except ValueError as exc:
            if "SORT_CONFLICT" not in str(exc):
                raise
            raw_answer = _SOLVER_UNCERTAIN
    return _SinglePassAttempt(raw_answer, premises_used, translation, refine_log)


async def run_type1_single_pass(
    payload: PredictionRequest,
    translator: TheoryTranslator,
    solver: FOLSolver | None = None,
    fallback_reasoner: Type1FallbackReasoner | None = None,
) -> PredictionResponse:
    """Whole-theory single-pass translation path (EXACT_TYPE1_TRANSLATOR=single_pass).

    Translate premises + question + options in one LLM call to a shared FOL
    vocabulary, then reuse the existing Z3 checks and LLM fallback.
    """
    premises = [p.strip() for p in payload.premises or [] if p.strip()]
    options_dict = _normalize_options(payload.options)
    if not options_dict:
        options_dict = extract_mcq(payload.question).options

    settings = get_settings()
    k = max(1, settings.type1_self_consistency_samples)
    temperature = 0.0 if k == 1 else settings.type1_self_consistency_temperature

    # Adaptive K: a Z3-entailed answer is a proof, not a guess, so the first
    # sample that yields a DEFINITE symbolic answer is accepted as-is (K=1, the
    # common, fast case). Self-consistency voting only kicks in when sample 1 is
    # Uncertain — a signal the translation may have failed — where extra samples
    # plus a majority vote actually help. On a compute-bound GPU this turns most
    # MCQ requests from K calls into one.
    async def _attempt() -> _SinglePassAttempt:
        return await _single_pass_attempt(
            payload, translator, solver, options_dict, premises,
            settings.type1_single_pass_max_refines, temperature,
            settings.type1_verify_refine,
        )

    attempts: list[_SinglePassAttempt] = [await _attempt()]
    if attempts[0].symbolic_answer == _SOLVER_UNCERTAIN and k > 1:
        for _ in range(k - 1):
            attempts.append(await _attempt())
    votes: Counter[str] = Counter(a.symbolic_answer for a in attempts)

    # Majority vote over symbolic answers; a tie for first place abstains.
    ranked = votes.most_common()
    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        winner = ranked[0][0]
    else:
        winner = _SOLVER_UNCERTAIN
    representative = next(
        (a for a in attempts if a.symbolic_answer == winner), attempts[0]
    )

    translation = representative.translation
    refine_log = representative.refine_log
    vote_distribution = dict(votes)
    premise_fols = translation.premise_trees
    translate_failed = any("TRANSLATE_FAILED" in i for i in translation.issues)
    # MCQ-ness from the request (survives a failed/empty translation).
    is_mcq = bool(options_dict) or translation.question_format == "mcq"
    question_type = QuestionType.MCQ if is_mcq else QuestionType.YNU
    solver_used = solver is not None and bool(premise_fols)
    symbolic_answer = winner
    raw_answer = winner
    premises_used: list[int] | None = (
        representative.premises_used if winner != _SOLVER_UNCERTAIN else None
    )

    # Trust Z3 on a usable translation: a Z3 'Uncertain' is itself a valid
    # answer (the premises genuinely do not entail). Fall back to the LLM ONLY
    # when there is nothing to solve — a genuine open-ended (wh) question, or the
    # translation produced no usable FOL (failed/unparsable). This keeps Z3 the
    # decider and avoids the slow fallback path.
    fallback_used = False
    fallback_trigger: str | None = None
    fallback_explanation: str | None = None
    fallback_error: str | None = None
    is_open_ended = not is_mcq and not translate_failed and (
        translation.question_format == "open_wh" or translation.claim_tree is None
    )
    translation_usable = (not translate_failed) and bool(premise_fols) and (
        bool(translation.option_trees) if is_mcq else translation.claim_tree is not None
    )
    if (is_open_ended or not translation_usable) and fallback_reasoner is not None:
        fallback_trigger = "OPEN_ENDED" if is_open_ended else "TRANSLATION_UNUSABLE"
        fallback_labels = (
            [*(translation.option_trees or options_dict).keys(), _SOLVER_UNCERTAIN]
            if is_mcq
            else [] if is_open_ended else ["Yes", "No", _SOLVER_UNCERTAIN]
        )
        try:
            fb = await fallback_reasoner.answer(
                premises=premises,
                question=payload.question,
                option_labels=fallback_labels,
                options=options_dict or None,
            )
            raw_answer = fb.answer
            fallback_used = True
            fallback_explanation = fb.explanation
            if fb.premises_used:
                premises_used = fb.premises_used
        except Exception as exc:
            fallback_error = f"{type(exc).__name__}: {exc}"

    if not is_mcq and not is_open_ended and _is_provability_question(payload.question):
        raw_answer = "Yes" if symbolic_answer == "Yes" else "No"

    answer = _normalize_answer(raw_answer)

    premise_items = [
        {
            "id": f"premise-{translation.premise_index_map[i] + 1}",
            "original_text": (
                premises[translation.premise_index_map[i]]
                if translation.premise_index_map[i] < len(premises)
                else ""
            ),
            "fol": translation.premise_strings[i],
            "ast": fol_node_to_dict(tree),
        }
        for i, tree in enumerate(premise_fols)
    ]
    fol_lines = [f"{item['id']}: {item['fol']}" for item in premise_items]
    if translation.claim_string:
        fol_lines.append(f"claim: {translation.claim_string}")
    for label, tree in translation.option_trees.items():
        fol_lines.append(f"option-{label}: {tree!r}")

    explanation = _build_single_pass_explanation(
        answer=answer,
        uncertain_token=get_settings().type1_uncertain_token,
        premises=premises,
        premises_used=premises_used,
        fallback_explanation=fallback_explanation if fallback_used else None,
    )

    return PredictionResponse(
        id=payload.query_id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=question_type,
        answer=answer,
        explanation=explanation,
        fol=fol_lines,
        cot=[
            f"Translated whole theory in one pass "
            f"({len(premise_fols)} premises, {len(translation.predicates)} predicates).",
            f"Question format: {translation.question_format}.",
            f"Symbolic answer: {symbolic_answer}.",
            *(["LLM fallback adjudicated."] if fallback_used else []),
            *([f"LLM fallback error: {fallback_error}"] if fallback_error else []),
        ],
        premises=premises,
        premises_used=premises_used,
        confidence=None,
        routing_diagnostics={
            "stage": "single_pass_translation",
            "solver_available": solver is not None,
            "solver_used": solver_used,
            "symbolic_answer": symbolic_answer,
            "fallback_used": fallback_used,
            "fallback_trigger": fallback_trigger,
            "fallback_error": fallback_error,
            "question_format": translation.question_format,
            "predicates": [p.model_dump() for p in translation.predicates],
            "premise_fol": translation.premise_strings,
            "claim_fol": translation.claim_string,
            "option_fol": {
                label: repr(tree) for label, tree in translation.option_trees.items()
            },
            "translation_issues": translation.issues,
            "refine_log": refine_log,
            "vote_distribution": vote_distribution,
        },
    )


def _translation_problems(translation: TheoryTranslation) -> list[str]:
    """Refinement triggers: unparsable PREMISE FOL + a claim whose predicate no
    premise defines.

    Deliberately ignores OPTION orphans/parse-failures: an MCQ distractor that
    references a concept absent from the premises is *correct* (it is simply not
    entailed). Forcing it onto an existing predicate creates a false second
    entailment and makes Z3 return Uncertain.
    """
    problems = [i for i in translation.issues if i.startswith("PREMISE_FOL_PARSE_FAILED")]
    if translation.claim_tree is not None:
        premise_predicates = {
            atom.predicate.name
            for tree in translation.premise_trees
            for atom in _collect_atoms(tree)
        }
        for atom in _collect_atoms(translation.claim_tree):
            if atom.predicate.name not in premise_predicates:
                problems.append(
                    f"ORPHAN_PREDICATE: claim uses {atom.predicate.name} which no premise "
                    f"defines — reuse an existing premise predicate or recheck the translation"
                )
    return list(dict.fromkeys(problems))


def _build_single_pass_explanation(
    *,
    answer: str,
    uncertain_token: str,
    premises: list[str],
    premises_used: list[int] | None,
    fallback_explanation: str | None,
) -> str:
    """Human-readable explanation for the single-pass path (no QuerySpec)."""
    is_uncertain = answer == uncertain_token
    lines: list[str] = []
    if is_uncertain:
        lines.append(f"Answer: {answer} — the premises do not determine it.")
    else:
        lines.append(f"Answer: {answer}.")
    used = [i for i in (premises_used or []) if 0 <= i < len(premises)]
    if used and not is_uncertain:
        lines.append("Supported by:")
        lines.extend(f"  • Premise {i + 1}: {premises[i]}" for i in used)
    if fallback_explanation and fallback_explanation.strip():
        lines.append(f"Reasoning: {fallback_explanation.strip()}")
    return "\n".join(lines)


def _collect_atoms(node: FOLNode) -> list[AtomicNode]:
    """All atomic predicate applications under ``node`` (comparisons excluded)."""
    if isinstance(node, AtomicNode):
        return [node]
    if isinstance(node, ComparisonNode):
        return []
    if isinstance(node, QuantifiedNode):
        atoms = _collect_atoms(node.body)
        if node.restrictor is not None:
            atoms += _collect_atoms(node.restrictor)
        return atoms
    atoms = _collect_atoms(node.left)
    if node.right is not None:
        atoms += _collect_atoms(node.right)
    return atoms


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
    return solver.check_mcq_with_used(premise_fols, option_fols, none_of_above_label)


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


_PROVABILITY_RE = re.compile(
    r"\b(?:do|does|did)\s+the\s+premises\s+"
    r"(?:prove|establish|show|confirm|guarantee|imply|entail|demonstrate)\b"
    r"|\bguarantee[ds]?\b",
    re.IGNORECASE,
)


def _is_provability_question(question: str) -> bool:
    """True for 'do the premises prove/establish X?' / 'does X guarantee Y?'.

    These ask whether the claim is *entailed*; non-entailment answers No, not
    Uncertain.
    """
    return _PROVABILITY_RE.search(question) is not None


def _build_explanation(
    *,
    answer: str,
    uncertain_token: str,
    premises: list[str],
    premises_used: list[int] | None,
    is_mcq: bool,
    spec: Any,
    fallback_explanation: str | None,
) -> str:
    """Human-readable explanation grounded in the premises.

    Internal markers (Z3 mode, uncertainty cause, fallback trigger) stay out of
    this field — they live in ``cot`` / ``routing_diagnostics`` for debugging.
    """
    is_uncertain = answer == uncertain_token
    lines: list[str] = []

    # 1. Headline — the answer in plain words.
    if is_mcq:
        if is_uncertain:
            lines.append(f"Answer: {answer} — no option is supported by the premises.")
        else:
            opt = next((c for c in spec.option_claims if c.label == answer), None)
            text = (opt.claim_text or opt.normalized_text) if opt is not None else None
            lines.append(f"Answer: {answer}{f') {text}' if text else ''}")
    else:
        claim = spec.main_claim_text
        if is_uncertain:
            tail = f" whether {claim}" if claim else ""
            lines.append(
                f"Answer: {answer} — the premises do not give enough "
                f"information to determine{tail}."
            )
        else:
            lines.append(f"Answer: {answer}.")

    # 2. Grounding — quote the premises the conclusion rests on.
    used = [i for i in (premises_used or []) if 0 <= i < len(premises)]
    if used and not is_uncertain:
        lines.append("Supported by:")
        lines.extend(f"  • Premise {i + 1}: {premises[i]}" for i in used)

    # 3. Natural-language rationale from the LLM reasoner, when present.
    if fallback_explanation and fallback_explanation.strip():
        lines.append(f"Reasoning: {fallback_explanation.strip()}")

    return "\n".join(lines)


def _normalize_answer(answer: str) -> str:
    """Replace the internal uncertain literal with the configured output token."""

    if answer == _SOLVER_UNCERTAIN:
        return get_settings().type1_uncertain_token
    return answer


# ---------------------------------------------------------------------------
# Option normalisation
# ---------------------------------------------------------------------------

_YNU_TOKENS = frozenset({"yes", "no", "uncertain", "unknown"})
# Single uppercase letters A–E used as bare MCQ label placeholders.
_MCQ_LABEL_RE = re.compile(r"^[A-E]$")


def _is_ynu_option_list(values: list[str]) -> bool:
    """True when every option is a bare Yes/No/Uncertain token (a polar set)."""
    return bool(values) and all(str(v).strip().lower() in _YNU_TOKENS for v in values)


def _is_bare_mcq_label_list(values: list[str]) -> bool:
    """True when every item is a single uppercase letter (A-E) with no text content.

    Callers that send ``["A", "B", "C", "D"]`` as option labels instead of
    full option text trigger this check.  The pipeline then falls back to
    ``extract_mcq`` to parse the actual option sentences from the query body.
    """
    return bool(values) and all(_MCQ_LABEL_RE.match(str(v).strip()) for v in values)


def _normalize_options(options: Any) -> dict[str, str]:
    """Return {label: text} for MCQ options, or {} for YNU/polar/bare-labels.

    Returns {} in three cases:
    - ``None`` input
    - All values are Yes/No/Uncertain tokens  → route as polar YNU question
    - All values are single uppercase letters → bare label list; the pipeline
      will call ``extract_mcq`` to parse real option text from the query body
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
        if _is_bare_mcq_label_list(options):
            return {}  # force extract_mcq to parse text from query body
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
