"""Type 1 logic pipeline.

Type 1 uses an LLM semantic parser to build logic IR, then deterministic
symbolic reasoning to produce the answer. There is no local parser substitute:
missing or invalid LLM output is logged and raised.
"""

from __future__ import annotations

import inspect
import re
import time
from collections import Counter

from exact.config import Settings, get_settings
from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.llm_client import build_json_client_from_settings
from exact.logic.explain import explain_result, kb_to_fol_like_text
from exact.logic.kb import get_or_build_kb_candidates
from exact.logic.ir import (
    And,
    Atom,
    Compare,
    Exists,
    ForAll,
    Formula,
    Iff,
    Implies,
    InSet,
    Not,
    Or,
    SolveResult,
    TranslatedProblem,
    term_to_text,
)
from exact.logic.kb import KnowledgeBase
from exact.logic.translation import (
    JsonLLMClient,
    translate_formula_goals_with_llm,
    translate_formula_premises_only_with_llm,
    translate_mcq_options_with_llm,
    translate_problem_with_llm,
    translate_query_only_with_llm,
)
from exact.logic.parsing import atom_from_text
from exact.logger import get_logger, get_request_logger

logger = get_logger(__name__)
from exact.symbolic_solvers import ForwardChainSolver, Z3Solver
from exact.symbolic_solvers.z3_prop import FormulaZ3Result, Z3PropSolver


_OPTION_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:\(([A-D])\)|([A-D])[.):])[ \t]*(.*?)"
    r"(?=\n[ \t]*(?:\([A-D]\)|[A-D][.):])|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_OPTION_ORDER = ("A", "B", "C", "D")


def extract_options(question: str) -> list[tuple[str, str]]:
    options = [
        (
            (match.group(1) or match.group(2)).upper(),
            " ".join(match.group(3).split()),
        )
        for match in _OPTION_RE.finditer(question)
    ]
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for label, text in options:
        if label in seen or not text:
            continue
        seen.add(label)
        deduped.append((label, text))
    return deduped


def strip_options_from_question(question: str) -> str:
    first_option = _OPTION_RE.search(question)
    stem = question[: first_option.start()] if first_option else question
    return " ".join(stem.split())


def _solve_with_z3_fallback(kb: KnowledgeBase, claim: Atom, use_z3: bool = True) -> SolveResult:
    """Run ForwardChain; if Unknown and Z3 is enabled, try Z3 as tiebreaker."""
    result = ForwardChainSolver().solve(kb, claim)
    if result.label == "Unknown" and use_z3:
        z3_result = Z3Solver().solve(kb, claim)
        if z3_result.label != "Unknown":
            return z3_result
    return result


def evaluate_mcq_options(
    kb: KnowledgeBase,
    goals: list[tuple[str, Atom]],
    solver: ForwardChainSolver | None = None,
    use_z3_fallback: bool = False,
) -> dict[str, SolveResult]:
    if use_z3_fallback:
        return {label: _solve_with_z3_fallback(kb, goal, use_z3=True) for label, goal in goals}
    solver = solver or ForwardChainSolver()
    return {label: solver.solve(kb, goal) for label, goal in goals}


def decide_mcq_winner(results: dict[str, SolveResult], question_stem: str) -> str:
    if not results:
        return "A"

    ordered_labels = [label for label in _OPTION_ORDER if label in results]
    entailed = [label for label in ordered_labels if results[label].label == "Yes"]
    if len(entailed) == 1:
        return entailed[0]

    candidates = entailed or ordered_labels
    rank = {"Yes": 0, "Unknown": 1, "No": 2}

    def score(label: str) -> tuple[int, int, int]:
        result = results[label]
        proof_size = len(result.supporting_premises) if result.supporting_premises else 999
        return rank.get(result.label, 3), proof_size, _OPTION_ORDER.index(label)

    if entailed and "fewest premises" in question_stem.lower():
        return min(entailed, key=lambda label: (len(results[label].supporting_premises), _OPTION_ORDER.index(label)))

    return min(candidates, key=score)


_FALLBACK_MIN_BUDGET_S: float = 12.0  # skip optional LLM fallback if < this many seconds remain


def _remaining_budget_s(deadline: float | None) -> float:
    """Seconds left before the shared Type 1 soft deadline; inf when unbounded."""
    if deadline is None:
        return float("inf")
    return deadline - time.monotonic()


def _has_fallback_budget(deadline: float | None) -> bool:
    """True when enough wall-clock budget remains for optional LLM fallback."""
    return _remaining_budget_s(deadline) >= _FALLBACK_MIN_BUDGET_S


def _deadline_expired(deadline: float | None) -> bool:
    if deadline is None:
        return False
    return _remaining_budget_s(deadline) <= 0


def _deadline_exhausted_response(
    request: PredictionRequest,
    question_type: QuestionType,
    error: str,
) -> PredictionResponse:
    answer = "A" if question_type == QuestionType.MCQ else "Uncertain"
    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=question_type,
        answer=answer,
        explanation="The Type 1 reasoning budget was exhausted before a trusted proof was available.",
        fol=None,
        cot=["type1_deadline_exhausted: returned low-confidence fail-safe answer"],
        premises=[],
        confidence=0.0,
        error=error,
    )


def run_type1_pipeline(
    request: PredictionRequest,
    translator_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
    question_type: QuestionType | None = None,
) -> PredictionResponse:
    """Answer a Type 1 logic query with symbolic proof where possible.

    Primary path (when enabled): LLM formula translation + Z3 propositional
    entailment. On failure or when disabled, falls back to legacy translation,
    forward-chain/Z3, MCQ voting, and CoT paths under a shared soft deadline.

    Args:
        request: Type 1 prediction request (premises + question).
        translator_client: JSON LLM client for NL→logic translation; built from
            settings when omitted.
        settings: Pipeline configuration; loaded from environment when omitted.
        question_type: Routed question shape; defaults to YES_NO_UNCERTAIN when
            omitted.

    Returns:
        ``PredictionResponse`` with competition-formatted answer, explanation,
        optional FOL/COT fields, and confidence.
    """

    logger = get_request_logger(
        __name__,
        request_id=request.id,
        task_type=TaskType.TYPE1_LOGIC.value,
    )
    logger.info("Start Type 1 pipeline")

    settings = settings or get_settings()
    translator_client = translator_client or build_json_client_from_settings(settings)
    if translator_client is None:
        logger.error("Type 1 LLM-only pipeline has no configured JSON LLM client")
        raise RuntimeError(
            "Type 1 requires a JSON LLM client. Configure EXACT_LLM_BASE_URL for an "
            "OpenAI-compatible server or EXACT_LLM_PROVIDER=local for a Transformers model."
        )

    routed_question_type = question_type or QuestionType.YES_NO_UNCERTAIN
    premises = request.premises_nl or []
    # Shared budget for every Type 1 LLM call. Each branch passes the remaining
    # time into clients that support per-call timeout overrides.
    deadline = time.monotonic() + settings.type1_soft_deadline_s

    formula_z3_error: str | None = None
    if settings.type1_use_formula_z3:
        try:
            return _run_formula_z3_pipeline(
                request=request,
                routed_question_type=routed_question_type,
                premises=premises,
                translator_client=translator_client,
                settings=settings,
                logger=logger,
                deadline=deadline,
            )
        except Exception as exc:
            logger.exception("Type 1 formula-Z3 path failed")
            if not settings.type1_enable_legacy_fallback:
                raise RuntimeError(
                    f"Type 1 formula-Z3 path failed for request {request.id}: {exc}"
                ) from exc
            formula_z3_error = f"formula_z3_failed: {exc}"
            if _deadline_expired(deadline):
                return _deadline_exhausted_response(
                    request,
                    routed_question_type,
                    formula_z3_error,
                )

    samples = max(1, settings.type1_translation_samples)
    sampling_temperature = (
        settings.type1_sampling_temperature if samples > 1 else settings.llm_temperature
    )

    if routed_question_type == QuestionType.MCQ:
        try:
            kb_candidates, candidate_warnings = get_or_build_kb_candidates(
                premises,
                translator_client,
                settings,
                samples=samples,
                sampling_temperature=sampling_temperature,
                deadline=deadline,
            )
        except Exception as exc:
            logger.exception("Type 1 MCQ LLM premise translation failed")
            if _deadline_expired(deadline):
                return _deadline_exhausted_response(
                    request,
                    routed_question_type,
                    f"legacy_mcq_translation_failed: {exc}",
                )
            raise RuntimeError(
                f"Type 1 MCQ LLM premise translation failed for request {request.id}: {exc}"
            ) from exc
        return _run_mcq_vote_path(
            request,
            kb_candidates,
            routed_question_type,
            tuple(w for w in (formula_z3_error, *candidate_warnings) if w),
            translator_client=translator_client, settings=settings, deadline=deadline,
        )

    try:
        kb_candidates, candidate_warnings = get_or_build_kb_candidates(
            premises,
            translator_client,
            settings,
            samples=samples,
            sampling_temperature=sampling_temperature,
            deadline=deadline,
        )
    except Exception as exc:
        logger.exception("Type 1 LLM premise translation failed")
        if _deadline_expired(deadline):
            return _deadline_exhausted_response(
                request,
                routed_question_type,
                f"legacy_translation_failed: {exc}",
            )
        raise RuntimeError(
            f"Type 1 LLM premise translation failed for request {request.id}: {exc}"
        ) from exc

    candidate_results: list[tuple[KnowledgeBase, SolveResult]] = []
    query_errors: list[str] = []
    for index, kb in enumerate(kb_candidates, start=1):
        try:
            entity_constants = tuple(sorted({
                arg for fact in kb.facts for arg in fact.atom.args
                if not arg.startswith("?")
            }))
            query = translate_query_only_with_llm(
                question=request.question,
                predicate_names=kb.predicate_names,
                entity_constants=entity_constants,
                llm_client=translator_client,
                settings=settings,
                deadline=deadline,
            )
            candidate_results.append((
                kb,
                _solve_with_z3_fallback(kb, query.claim, use_z3=settings.type1_use_z3_fallback),
            ))
        except Exception as exc:
            message = f"query candidate {index}/{len(kb_candidates)} failed: {exc}"
            logger.exception("Type 1 LLM query translation failed: %s", message)
            query_errors.append(message)

    if not candidate_results:
        if _deadline_expired(deadline):
            return _deadline_exhausted_response(
                request,
                routed_question_type,
                "; ".join((*candidate_warnings, *query_errors)),
            )
        raise RuntimeError(
            f"Type 1 LLM query translation failed for request {request.id}: "
            + "; ".join((*candidate_warnings, *query_errors))
        )

    winner_label, vote_summary, vote_confidence = _vote_labels(
        [result.label for _, result in candidate_results],
        tie_order=("Unknown", "No", "Yes"),
    )
    kb, result = next(
        (candidate_kb, candidate_result)
        for candidate_kb, candidate_result in candidate_results
        if candidate_result.label == winner_label
    )
    explanation, cot, cited_premises = explain_result(result, kb)
    vote_line = f"symbolic_consistency_vote: {vote_summary}"
    warnings = tuple(w for w in (formula_z3_error, *candidate_warnings, *query_errors, *result.warnings) if w)

    response = PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=routed_question_type,
        # Convert internal "Unknown" → "Uncertain" for competition submission.
        # Internal logic (vote summary, CoT trigger below) still uses result.label.
        answer=_to_competition_label(result.label),
        explanation=f"{explanation} {vote_line}",
        fol=kb_to_fol_like_text(kb) or None,
        cot=[*cot, vote_line],
        premises=cited_premises,
        confidence=vote_confidence,
        error="; ".join(warnings) if warnings else None,
    )

    if result.label == "Unknown" and settings.type1_enable_cot_fallback:
        return _run_cot_unknown_fallback(
            request=request,
            response=response,
            llm_client=translator_client,
            settings=settings,
            logger=logger,
            deadline=deadline,
        )

    return response


def _run_formula_z3_pipeline(
    *,
    request: PredictionRequest,
    routed_question_type: QuestionType,
    premises: list[str],
    translator_client: JsonLLMClient,
    settings: Settings,
    logger,
    deadline: float | None = None,
) -> PredictionResponse:
    """Formula translation followed by Z3 propositional entailment.

    When settings.type1_formula_cache_premises is True (default), uses a
    split two-call approach:
      Call 1 (cached): premises only → predicate dict + FormulaItems
      Call 2 (fast):   query/options → goal FormulaItems using predicate dict

    This avoids re-translating premises for subsequent questions that share
    the same premise set (common in the dataset — avg 2, max 16 per group).
    Falls back to the one-shot path if the split path raises.
    """

    options = extract_options(request.question) if routed_question_type == QuestionType.MCQ else None
    if routed_question_type == QuestionType.MCQ and not options:
        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE1_LOGIC,
            question_type=routed_question_type,
            answer="A",
            explanation="No multiple-choice options were parsed, so the system returned default option A.",
            fol=None,
            cot=["formula_z3_skipped: no A-D options parsed"],
            premises=[],
            confidence=0.05,
            error="MCQ routed but no options were parsed.",
        )

    translated: TranslatedProblem | None = None
    split_error: str | None = None

    if settings.type1_formula_cache_premises:
        try:
            # Call 1: translate and cache premises (free on cache hit)
            premise_result = translate_formula_premises_only_with_llm(
                premises=premises,
                llm_client=translator_client,
                settings=settings,
                deadline=deadline,
            )
            # Call 2: translate goals using shared predicate vocabulary
            goal_items = translate_formula_goals_with_llm(
                question=request.question,
                options=options,
                predicate_dict=premise_result.predicates,
                entity_constants=premise_result.entity_constants,
                llm_client=translator_client,
                settings=settings,
                deadline=deadline,
            )
            translated = TranslatedProblem(
                predicates=premise_result.predicates,
                premises=premise_result.premises,
                goals=goal_items,
            )
            logger.info(
                "Formula-Z3 split translation: premises=%d goals=%d predicates=%d",
                len(translated.premises), len(translated.goals), len(translated.predicates),
            )
        except Exception as exc:
            split_error = f"split_translation_failed: {exc}"
            logger.warning("Split formula translation failed, falling back to one-shot: %s", exc)

    if translated is None:
        # One-shot fallback: all premises + goals in a single call
        translated = translate_problem_with_llm(
            premises=premises,
            question=request.question,
            options=options,
            llm_client=translator_client,
            settings=settings,
            deadline=deadline,
        )
        logger.info(
            "Formula-Z3 one-shot translation: premises=%d goals=%d predicates=%d",
            len(translated.premises), len(translated.goals), len(translated.predicates),
        )

    solver = Z3PropSolver()

    if routed_question_type == QuestionType.MCQ:
        response = _run_formula_z3_mcq_path(
            request=request,
            translated=translated,
            options=options or [],
            solver=solver,
            question_type=routed_question_type,
            llm_client=translator_client,
            settings=settings,
            deadline=deadline,
        )
    else:
        response = _run_formula_z3_query_path(
            request=request,
            translated=translated,
            solver=solver,
            question_type=routed_question_type,
            llm_client=translator_client,
            settings=settings,
            logger=logger,
            deadline=deadline,
        )

    # Attach split-path fallback warning if one-shot was used as fallback.
    if split_error:
        response = response.model_copy(
            update={"error": _join_errors(split_error, response.error)}
        )
    return response


def _run_formula_z3_query_path(
    *,
    request: PredictionRequest,
    translated: TranslatedProblem,
    solver: Z3PropSolver,
    question_type: QuestionType,
    llm_client: JsonLLMClient,
    settings: Settings,
    logger,
    deadline: float | None = None,
) -> PredictionResponse:
    result = solver.solve_query(translated)
    internal_answer = result.answer or "Unknown"
    answer = _to_competition_label(internal_answer)
    response = PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=question_type,
        answer=answer,
        explanation=_formula_query_explanation(result),
        fol=_translated_problem_to_fol_like_text(translated),
        cot=_formula_cot(translated, result, kind="query"),
        premises=_formula_premise_refs(translated, result.supporting_premises),
        confidence=_formula_query_confidence(result),
        error="; ".join(result.warnings) if result.warnings else None,
    )

    # Only run the optional CoT fallback when the symbolic solver returned Unknown
    # AND there is enough time budget remaining (>12s). Skip if low on budget to
    # guarantee we return the symbolic answer before the 60s hard cap.
    remaining = _remaining_budget_s(deadline)
    if (
        (internal_answer == "Unknown" or result.answer is None)
        and settings.type1_enable_cot_fallback
        and _has_fallback_budget(deadline)
    ):
        return _run_cot_unknown_fallback(
            request=request,
            response=response,
            llm_client=llm_client,
            settings=settings,
            logger=logger,
            deadline=deadline,
        )
    if not _has_fallback_budget(deadline) and internal_answer == "Unknown":
        logger.info(
            "Skipping CoT fallback: only %.1fs remaining (threshold=%.1fs)",
            remaining, _FALLBACK_MIN_BUDGET_S,
        )
    return response


def _run_formula_z3_mcq_path(
    *,
    request: PredictionRequest,
    translated: TranslatedProblem,
    options: list[tuple[str, str]],
    solver: Z3PropSolver,
    question_type: QuestionType,
    llm_client: JsonLLMClient,
    settings: Settings,
    deadline: float | None = None,
) -> PredictionResponse:
    stem = strip_options_from_question(request.question)
    result = solver.solve_mcq(translated, stem=stem)

    remaining = _remaining_budget_s(deadline)
    translation_warning = _mcq_translation_warning(translated)
    needs_fallback = result.answer is None or translation_warning is not None
    if needs_fallback and settings.type1_enable_cot_fallback and _has_fallback_budget(deadline):
        fallback = _run_mcq_llm_fallback(
            request=request,
            options=options,
            question_type=question_type,
            llm_client=llm_client,
            settings=settings,
            deadline=deadline,
        )
        if fallback is not None:
            return fallback.model_copy(
                update={
                    "fol": _translated_problem_to_fol_like_text(translated),
                    "cot": [
                        *_formula_cot(translated, result, kind="mcq"),
                        *(fallback.cot or []),
                    ],
                    "error": _join_errors(
                        "; ".join(result.warnings) if result.warnings else None,
                        translation_warning,
                        fallback.error,
                    ),
                }
            )
    if not _has_fallback_budget(deadline) and needs_fallback:
        logger.info(
            "Skipping MCQ LLM fallback: only %.1fs remaining (threshold=%.1fs)",
            remaining, _FALLBACK_MIN_BUDGET_S,
        )

    answer = result.answer or "Unknown"
    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=question_type,
        answer=answer,
        explanation=_formula_mcq_explanation(result),
        fol=_translated_problem_to_fol_like_text(translated),
        cot=_formula_cot(translated, result, kind="mcq"),
        premises=_formula_premise_refs(translated, result.supporting_premises),
        confidence=_formula_mcq_confidence(result),
        error=_join_errors(
            "; ".join(result.warnings) if result.warnings else None,
            translation_warning,
        ),
    )


_CONDITIONAL_OPTION_RE = re.compile(
    r"\b(if|then|when|whenever|unless|provided|implies?|only if)\b",
    re.IGNORECASE,
)


def _mcq_translation_warning(problem: TranslatedProblem) -> str | None:
    """Detect common small-model goal rewrites before trusting symbolic MCQ ranking."""

    suspicious = [
        item.label or "?"
        for item in problem.goals
        if (
            item.role == "option"
            and isinstance(item.formula, Implies)
            and (
                item.formula.antecedent == item.formula.consequent
                or not _CONDITIONAL_OPTION_RE.search(item.text)
            )
        )
    ]
    if not suspicious:
        return None
    return (
        "suspicious MCQ goal formalization for option(s) "
        f"{', '.join(suspicious)}; symbolic ranking requires verification"
    )


def _formula_query_explanation(result: FormulaZ3Result) -> str:
    answer = result.answer or "Unknown"
    if result.answer is None:
        return (
            "Formula-Z3 could not produce a trusted symbolic answer because the translated "
            "theory was inconsistent or incomplete."
        )
    return (
        f"Formula-Z3 judged the query as {answer}. It uses entailment checks where "
        "T entails phi iff T and not(phi) is UNSAT; the translated theory status was "
        f"{result.theory_status}."
    )


def _formula_mcq_explanation(result: FormulaZ3Result) -> str:
    if result.answer is None:
        return (
            "Formula-Z3 did not find a uniquely usable entailed option, so the case should "
            "be repaired or handled by the fail-safe fallback."
        )
    valid = ", ".join(result.valid_labels) if result.valid_labels else result.answer
    return (
        f"Option {result.answer} was selected by Formula-Z3 entailment. Entailed options: "
        f"{valid}. The translated theory status was {result.theory_status}."
    )


def _formula_query_confidence(result: FormulaZ3Result) -> float:
    if result.warnings or result.answer is None:
        return 0.20
    if result.answer in {"Yes", "No"}:
        return 0.78
    return 0.35


def _formula_mcq_confidence(result: FormulaZ3Result) -> float:
    if result.answer is None:
        return 0.10
    if len(result.valid_labels) == 1:
        return 0.80
    return 0.64


def _formula_cot(
    translated: TranslatedProblem,
    result: FormulaZ3Result,
    *,
    kind: str,
) -> list[str]:
    lines = [
        (
            "llm_formula_translation: "
            f"premises={len(translated.premises)}, goals={len(translated.goals)}, "
            f"predicates={len(translated.predicates)}"
        ),
        f"{result.mode}_theory_status: {result.theory_status}",
    ]
    if kind == "mcq":
        valid = ", ".join(result.valid_labels) if result.valid_labels else "none"
        lines.append(f"{result.mode}_mcq_valid_options: {valid}")
        if result.core_sizes:
            core_text = ", ".join(
                f"{label}={size}" for label, size in result.core_sizes
            )
            lines.append(f"{result.mode}_unsat_core_sizes: {core_text}")
    else:
        lines.append(f"{result.mode}_query_answer: {result.answer or 'Unknown'}")
    for warning in result.warnings:
        lines.append(f"{result.mode}_warning: {warning}")
    return lines


def _translated_problem_to_fol_like_text(problem: TranslatedProblem) -> str:
    predicate_text = ", ".join(
        f"{name}/{arity}" for name, arity in sorted(problem.predicates.items())
    )
    premise_lines = [
        f"P{item.source_idx + 1}: {_formula_to_text(item.formula)}"
        for item in problem.premises
    ]
    goal_lines = []
    for item in problem.goals:
        label = f"{item.label}: " if item.label else ""
        goal_lines.append(f"{item.role}:{label}{_formula_to_text(item.formula)}")
    return "\n".join(
        [
            f"Predicates: {predicate_text or '(implicit)'}",
            "Premises:",
            *premise_lines,
            "Goals:",
            *goal_lines,
        ]
    )


def _formula_to_text(formula: Formula) -> str:
    if isinstance(formula, Atom):
        args = ", ".join(term_to_text(arg) for arg in formula.args)
        atom = f"{formula.pred}({args})" if args else formula.pred
        return f"not {atom}" if formula.negated else atom
    if isinstance(formula, Not):
        return f"not ({_formula_to_text(formula.arg)})"
    if isinstance(formula, And):
        return "(" + " and ".join(_formula_to_text(arg) for arg in formula.args) + ")"
    if isinstance(formula, Or):
        return "(" + " or ".join(_formula_to_text(arg) for arg in formula.args) + ")"
    if isinstance(formula, Implies):
        return (
            f"({_formula_to_text(formula.antecedent)} -> "
            f"{_formula_to_text(formula.consequent)})"
        )
    if isinstance(formula, Iff):
        return f"({_formula_to_text(formula.left)} <-> {_formula_to_text(formula.right)})"
    if isinstance(formula, ForAll):
        return f"forall {', '.join(formula.variables)}. ({_formula_to_text(formula.body)})"
    if isinstance(formula, Exists):
        return f"exists {', '.join(formula.variables)}. ({_formula_to_text(formula.body)})"
    if isinstance(formula, Compare):
        return f"{term_to_text(formula.left)} {formula.op} {term_to_text(formula.right)}"
    if isinstance(formula, InSet):
        options = ", ".join(term_to_text(option) for option in formula.options)
        return f"{term_to_text(formula.member)} in {{{options}}}"
    return str(formula)


def _formula_premise_refs(
    problem: TranslatedProblem,
    supporting_premises: tuple[int, ...],
) -> list[str]:
    selected = set(supporting_premises)
    return [
        f"P{item.source_idx + 1}: {item.text}"
        for item in sorted(problem.premises, key=lambda item: item.source_idx)
        if item.source_idx in selected
    ]


def _run_cot_unknown_fallback(
    request: PredictionRequest,
    response: PredictionResponse,
    llm_client: JsonLLMClient,
    settings: Settings,
    logger,
    deadline: float | None = None,
) -> PredictionResponse:
    """Use LLM reasoning when symbolic consistency voting cannot prove a label."""

    messages = _build_cot_unknown_messages(request, response)
    try:
        raw = _complete_json_sync_with_deadline(
            llm_client,
            deadline=deadline,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        answer = _normalize_cot_answer(raw.get("answer"))
        explanation = str(raw.get("explanation") or "").strip()
        cot = _as_string_list(raw.get("cot"))
        confidence = _bounded_confidence(raw.get("confidence"), default=response.confidence or 0.35)
    except Exception as exc:
        logger.exception("Type 1 CoT fallback after symbolic Unknown failed")
        return response.model_copy(
            update={
                "error": _join_errors(response.error, f"CoT fallback failed: {exc}"),
            }
        )

    fallback_line = f"cot_fallback_after_symbolic_unknown: answer={answer}"
    return response.model_copy(
        update={
            "answer": answer,
            "explanation": (
                explanation
                or f"Symbolic reasoning returned Unknown, then the LLM reasoning fallback selected {answer}."
            ),
            "cot": [*(response.cot or []), *cot, fallback_line],
            "confidence": confidence,
            "error": response.error,
        }
    )


def _build_cot_unknown_messages(
    request: PredictionRequest,
    response: PredictionResponse,
) -> list[dict[str, str]]:
    premises = "\n".join(
        f"P{idx + 1}: {premise}" for idx, premise in enumerate(request.premises_nl or [])
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a careful educational logic reasoner. Return JSON only. "
                "Use the given premises only. The answer must be Yes, No, or Unknown."
            ),
        },
        {
            "role": "user",
            "content": (
                "The symbolic solver returned Unknown. Re-check the question with concise reasoning.\n"
                'Return JSON shaped as {"answer":"Yes|No|Unknown","explanation":"...",'
                '"cot":["step"],"confidence":0.0}.\n\n'
                f"Premises:\n{premises}\n\n"
                f"Question:\n{request.question}\n\n"
                f"Symbolic result:\nanswer={response.answer}\n"
                f"explanation={response.explanation}\n"
            ),
        },
    ]


def _normalize_cot_answer(value: object) -> str:
    """Map an LLM-produced answer token to the competition label.

    The internal solver uses "Unknown" for the undecidable case, but the
    competition rubric for Yes/No/Uncertain questions expects "Uncertain".
    We normalise here so both the symbolic path and the CoT fallback path
    always output the same external-facing label.
    """
    answer = str(value or "").strip().lower()
    if answer in {"yes", "true", "entailed"}:
        return "Yes"
    if answer in {"no", "false", "contradicted"}:
        return "No"
    if answer in {"unknown", "uncertain", "not enough information"}:
        # BTC confirmed the third label is "Uncertain", not "Unknown".
        return "Uncertain"
    raise ValueError(f"CoT fallback returned unsupported answer: {value!r}")


def _to_competition_label(internal_label: str) -> str:
    """Convert an internal solver label to the competition submission format.

    Internal solvers use "Unknown" for the undecidable case (matches Z3/
    forward-chain conventions). The competition expects "Uncertain" as the
    third valid answer for Yes/No/Uncertain questions.

    MCQ and other answer types are returned unchanged.
    """
    if internal_label == "Unknown":
        return "Uncertain"
    return internal_label


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _bounded_confidence(value: float, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def _join_errors(*errors: str | None) -> str | None:
    parts = [error for error in errors if error]
    return "; ".join(parts) if parts else None


def _complete_json_sync_with_deadline(
    client: JsonLLMClient,
    *,
    deadline: float | None,
    **kwargs,
):
    """Pass a per-call timeout when the client supports the remaining-budget contract."""

    if deadline is not None:
        remaining = _remaining_budget_s(deadline) - 0.5
        if remaining <= 0:
            raise TimeoutError("Type 1 fallback deadline exhausted before LLM call")
        try:
            parameters = inspect.signature(client.complete_json_sync).parameters
            accepts_timeout = "timeout_override" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            accepts_timeout = False
        if accepts_timeout:
            kwargs["timeout_override"] = remaining
    return client.complete_json_sync(**kwargs)


def _run_mcq_llm_fallback(
    request: PredictionRequest,
    options: list[tuple[str, str]],
    question_type: QuestionType,
    llm_client: JsonLLMClient,
    settings: Settings,
    deadline: float | None = None,
) -> PredictionResponse | None:
    """Direct LLM answer for MCQ when symbolic reasoning cannot prove any option."""
    premises_text = "\n".join(
        f"P{i + 1}: {p}" for i, p in enumerate(request.premises_nl or [])
    )
    options_text = "\n".join(f"{label}. {text}" for label, text in options)
    valid_labels = {label for label, _ in options}
    messages = [
        {
            "role": "system",
            "content": (
                "You are an educational logic reasoner. Return JSON only. "
                "Choose the best answer strictly from the given premises."
            ),
        },
        {
            "role": "user",
            "content": (
                "Symbolic reasoning could not prove any option. Select the best one using the premises.\n"
                'Return JSON: {"answer":"A","explanation":"...","confidence":0.0}\n\n'
                f"Premises:\n{premises_text}\n\n"
                f"Question:\n{request.question}\n\n"
                f"Options:\n{options_text}"
            ),
        },
    ]
    try:
        raw = _complete_json_sync_with_deadline(
            llm_client,
            deadline=deadline,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=min(settings.llm_max_tokens, 512),
        )
        answer = str(raw.get("answer", "")).strip().upper()
        if answer not in valid_labels:
            return None
        explanation = str(raw.get("explanation") or "").strip()
        confidence = _bounded_confidence(raw.get("confidence"), default=0.35)
        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE1_LOGIC,
            question_type=question_type,
            answer=answer,
            explanation=explanation or f"LLM selected option {answer} after no symbolic proof.",
            fol=None,
            cot=[f"mcq_llm_fallback: answer={answer}"],
            premises=[],
            confidence=confidence,
            error="No MCQ option symbolically entailed; LLM fallback used.",
        )
    except Exception as exc:
        logger.debug("MCQ LLM fallback failed: %s", exc)
        return None


def _run_mcq_path(
    request: PredictionRequest,
    kb: KnowledgeBase,
    question_type: QuestionType,
    translator_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
    deadline: float | None = None,
) -> PredictionResponse:
    options = extract_options(request.question)
    if not options:
        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE1_LOGIC,
            question_type=question_type,
            answer="A",
            explanation="No multiple-choice options were parsed, so the system returned the default option A.",
            fol=kb_to_fol_like_text(kb) or None,
            cot=["No option labels A-D were available for symbolic evaluation."],
            premises=[],
            confidence=0.05,
            error="MCQ routed but no options were parsed.",
        )

    stem = strip_options_from_question(request.question)

    # Translate options via LLM (using KB predicate vocabulary) then fall back per-option to text parser.
    translated: dict[str, Atom] = {}
    if translator_client is not None:
        try:
            translated = translate_mcq_options_with_llm(
                request.question, options, kb.predicate_names,
                translator_client, settings, deadline=deadline,
            )
        except Exception as exc:
            logger.debug("MCQ LLM option translation skipped: %s", exc)
    goals = [(label, translated.get(label) or atom_from_text(text)) for label, text in options]
    use_z3 = (settings or get_settings()).type1_use_z3_fallback
    results = evaluate_mcq_options(kb, goals, use_z3_fallback=use_z3)
    option_summary = _format_mcq_option_summary(results)
    no_entailed_option = all(result.label != "Yes" for result in results.values())

    if no_entailed_option:
        settings_eff = settings or get_settings()
        if translator_client is not None:
            fallback = _run_mcq_llm_fallback(
                request, options, question_type, translator_client, settings_eff,
                deadline=deadline,
            )
            if fallback is not None:
                return fallback.model_copy(
                    update={
                        "cot": [*(fallback.cot or []), option_summary],
                        "fol": kb_to_fol_like_text(kb) or None,
                    }
                )
        return PredictionResponse(
            id=request.id,
            task_type=TaskType.TYPE1_LOGIC,
            question_type=question_type,
            answer="Unknown",
            explanation=(
                "No MCQ option was symbolically entailed and LLM fallback did not produce a valid answer. "
                + option_summary
            ),
            fol=kb_to_fol_like_text(kb) or None,
            cot=[option_summary],
            premises=[],
            confidence=0.10,
            error="No MCQ option was symbolically entailed.",
        )

    winner = decide_mcq_winner(results, stem)
    winning_result = results[winner]
    explanation, proof_cot, cited_premises = explain_result(winning_result, kb)
    return PredictionResponse(
        id=request.id,
        task_type=TaskType.TYPE1_LOGIC,
        question_type=question_type,
        answer=winner,
        explanation=f"Option {winner} is selected. {explanation} {option_summary}",
        fol=kb_to_fol_like_text(kb) or None,
        cot=[*proof_cot, option_summary],
        premises=cited_premises,
        confidence=_mcq_confidence(results, winner),
    )


def _run_mcq_vote_path(
    request: PredictionRequest,
    kb_candidates: tuple[KnowledgeBase, ...],
    question_type: QuestionType,
    candidate_warnings: tuple[str, ...] = (),
    translator_client: JsonLLMClient | None = None,
    settings: Settings | None = None,
    deadline: float | None = None,
) -> PredictionResponse:
    responses: list[PredictionResponse] = [
        _run_mcq_path(request, kb, question_type, translator_client, settings, deadline)
        for kb in kb_candidates
    ]
    winner, vote_summary, confidence = _vote_labels(
        [response.answer for response in responses],
        tie_order=_OPTION_ORDER,
    )
    selected = next(response for response in responses if response.answer == winner)
    vote_line = f"symbolic_consistency_vote: {vote_summary}"
    warnings = [warning for warning in candidate_warnings if warning]
    if selected.error:
        warnings.append(selected.error)

    return selected.model_copy(
        update={
            "confidence": confidence,
            "explanation": f"{selected.explanation} {vote_line}",
            "cot": [*(selected.cot or []), vote_line],
            "error": "; ".join(warnings) if warnings else None,
        }
    )


def _format_mcq_option_summary(results: dict[str, SolveResult]) -> str:
    parts = [f"{label}: {results[label].label}" for label in _OPTION_ORDER if label in results]
    return "Option entailment results: " + ", ".join(parts) + "."


def _mcq_confidence(results: dict[str, SolveResult], winner: str) -> float:
    if results[winner].label == "Yes":
        entailed_count = sum(result.label == "Yes" for result in results.values())
        return 0.72 if entailed_count == 1 else 0.58
    if results[winner].label == "Unknown":
        return 0.28
    return 0.12


def _vote_labels(labels: list[str], tie_order: tuple[str, ...]) -> tuple[str, str, float]:
    if not labels:
        raise ValueError("Cannot vote over an empty label list")

    counts = Counter(labels)
    tie_rank = {label: index for index, label in enumerate(tie_order)}
    winner = min(
        counts,
        key=lambda label: (-counts[label], tie_rank.get(label, len(tie_rank)), label),
    )
    ordered_labels = sorted(
        counts,
        key=lambda label: (-counts[label], tie_rank.get(label, len(tie_rank)), label),
    )
    summary = ", ".join(f"{label}={counts[label]}" for label in ordered_labels)
    return winner, summary, counts[winner] / len(labels)
