from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Sequence

from exact.config import Settings, get_settings
from exact.datasets.type2_taxonomy import classify_type2_taxonomy
from exact.llm_client import has_json_llm_client_config
from exact.type2.geometry_context import build_geometry_prompt_context
from exact.type2.extraction.llm_structured import (
    PotCodeSpec,
    generate_final_explanation,
    generate_pot_code_candidates,
    repair_pot_code,
    build_llm_json_client,
    select_formula_ids,
)
from exact.type2.fallback.executor import ExecutionResult, execute_python
from exact.type2.formulas.knowledge import RetrievedFormulaContext, canonicalize_formula_ids
from exact.type2.schemas import Extraction, Type2SolveResult, Verification
from exact.type2.solving.physics_verifier import verify_against_physics_oracle
from exact.type2.solving.pot_verifier import OutputSanityResult, verify_output_sanity
from exact.type2.solving.solver import answer_conceptual, solve_extraction, solve_vector_template


POT_SOLVER_NOT_CONFIGURED = "type2_pot_solver_not_configured"
POT_SOLVER_FAILED = "type2_pot_solver_failed"


@dataclass(frozen=True)
class PotCandidateAttempt:
    index: int
    code_spec: PotCodeSpec
    execution: ExecutionResult
    repair_attempts: int
    verified: OutputSanityResult | None = None
    error: str | None = None
    agreement_count: int = 1


def _load_conceptual_kb() -> list[dict[str, Any]]:
    from exact.config import PACKAGE_DIR
    path = PACKAGE_DIR / "datasets" / "exact" / "type2_electricity_theory_kb.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _retrieve_theory_context(question: str) -> str:
    kb = _load_conceptual_kb()
    if not kb:
        return "No theoretical knowledge base available."

    query_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower()))
    scored_entries = []

    for entry in kb:
        text = " ".join([
            str(entry.get("subtopic_name") or ""),
            str(entry.get("description_subtopic") or ""),
            str(entry.get("topic_name") or ""),
            str(entry.get("description_topic") or ""),
            " ".join(entry.get("misconceptions") or [])
        ]).lower()
        entry_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text))
        overlap = len(query_tokens & entry_tokens)
        scored_entries.append((overlap, entry))

    scored_entries.sort(key=lambda x: x[0], reverse=True)
    top_entries = [item[1] for item in scored_entries[:1] if item[0] > 0]
    if not top_entries:
        top_entries = [entry for entry in kb[:1]]

    formatted = []
    for idx, entry in enumerate(top_entries, start=1):
        formatted.append(
            f"Theory Ref {idx} [{entry.get('subtopic_name')} - {entry.get('topic_name')}]:\n"
            f"- Concept: {_clip_context(str(entry.get('description_subtopic') or ''), 700)}\n"
            f"- Misconceptions: {_clip_context('; '.join(entry.get('misconceptions') or []), 250)}"
        )
    return "\n\n".join(formatted)


def _solve_conceptual(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
    settings: Settings,
) -> Type2SolveResult:
    curated = answer_conceptual(extraction) if settings.type2_use_concept_bank else None
    if curated is not None and curated.error is None:
        return curated

    client = build_llm_json_client(settings)
    if client is None:
        if curated is not None:
            return curated
        return _unconfigured_result(extraction)

    theory_context = _retrieve_theory_context(extraction.normalized_question)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert physics solver. Answer the conceptual physics question directly and concisely. "
                "Return exactly one JSON object and nothing else. The first character must be { and the last must be }. "
                "Use exactly these keys: answer, explanation, premises, cot. "
                "answer must be a short direct phrase. explanation must be one concise sentence. "
                "premises must contain at most 2 short strings. cot must be an empty list or at most 2 short public trace strings. "
                "Do not use markdown, code fences, LaTeX, or prose outside JSON."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{extraction.normalized_question}\n\n"
                f"Theoretical Reference Context:\n{theory_context}\n\n"
                f"Formula context:\n{_compact_formula_context(formula_context.context)}\n\n"
                "Return shape: "
                "{\"answer\":\"short answer\",\"explanation\":\"one sentence\","
                "\"premises\":[\"short premise\"],\"cot\":[]}"
            )
        }
    ]

    try:
        raw = client.complete_json_sync(
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=min(settings.type2_conceptual_max_tokens, 512),
        )
        answer = str(raw.get("answer") or "").strip()
        explanation = str(raw.get("explanation") or "").strip()
        cot = _as_string_list(raw.get("cot"))
        premises = _as_string_list(raw.get("premises"))

        return Type2SolveResult(
            answer=answer,
            unit=None,
            value=None,
            formula=None,
            extraction=extraction,
            verification=Verification(ok=True, message="Successfully solved conceptual question using RAG & LLM."),
            cot=[
                "Detected conceptual question.",
                "Retrieved theoretical knowledge from type2_electricity_theory_kb.json.",
                "Directly generated conceptual answer using LLM.",
                *cot
            ],
            premises=premises if premises else [explanation] if explanation else [answer],
            confidence=0.85,
            error=None,
        )
    except Exception as exc:
        return _failed_result(extraction, f"Conceptual LLM solver failed: {exc}")


def _clip_context(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _compact_formula_context(context: str, limit: int = 900) -> str:
    lines = [line for line in context.splitlines() if line.strip()]
    compact = "\n".join(lines[:4])
    return _clip_context(compact or "No formula context available.", limit)


def solve_with_pot(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
    settings: Settings | None = None,
    generate_explanation: bool = True,
) -> Type2SolveResult:
    settings = settings or get_settings()
    if extraction.kind.value == "conceptual":
        return _solve_conceptual(extraction, formula_context, settings)
    if not settings.type2_use_pot_solver:
        fallback = _try_executable_formula_fallback(
            extraction,
            formula_context,
            "PoT solver disabled by Type 2 config.",
            settings,
        )
        if fallback is not None:
            return fallback
        return _failed_result(extraction, "PoT solver disabled and no executable formula solved the extraction.")
    if settings.type2_deterministic_first and _should_use_deterministic_first(extraction, formula_context):
        fast_path = _try_executable_formula_fallback(
            extraction,
            formula_context,
            "Deterministic-first mode is enabled.",
            settings,
        )
        if fast_path is not None:
            fast_path.cot.insert(0, "Used deterministic executable formula before LLM code generation.")
            return fast_path
    elif settings.type2_deterministic_first:
        vector_fast_path = solve_vector_template(extraction)
        if vector_fast_path is not None:
            vector_fast_path.cot.insert(0, "Used deterministic vector template before LLM code generation.")
            return vector_fast_path
    if not has_json_llm_client_config(settings):
        fast_path = _try_executable_formula_fallback(extraction, formula_context, "LLM is disabled.", settings)
        if fast_path is not None:
            fast_path.cot.insert(0, "Used deterministic executable formula fast path because LLM is disabled.")
            return fast_path
    vector_fast_path = solve_vector_template(extraction)
    if vector_fast_path is not None:
        vector_fast_path.cot.insert(0, "Used deterministic vector template before LLM code generation.")
        return vector_fast_path

    # Let LLM select formulas from retrieved formulas
    selected_formula_ids = list(formula_context.formula_ids)
    solution_plan = list(formula_context.solution_plan)
    missing_variables = list(formula_context.missing_variables)
    choice = None
    try:
        quantities_summary = ", ".join(f"{k} = {v.value}" for k, v in extraction.quantities.items())
        extraction_summary = f"Target: {extraction.target}. Quantities: {quantities_summary}."
        choice = select_formula_ids(
            question=extraction.normalized_question,
            extraction_summary=extraction_summary,
            formula_summaries=formula_context.summaries,
            settings=settings,
        )
        if choice is not None and choice.formula_ids:
            selected_formula_ids = choice.formula_ids
            solution_plan = choice.solution_plan
            missing_variables = choice.missing_variables
    except Exception as exc:
        print(f"[!] Error selecting formulas with LLM: {exc}", flush=True)

    # Reconstruct formula_context to only contain selected formulas
    if choice is not None and choice.formula_ids:
        selected_set = set(selected_formula_ids)
        selected_summaries = [s for s in formula_context.summaries if str(s.get("id")) in selected_set]
        summary_map = {str(s.get("id")): s for s in selected_summaries}
        ordered_summaries = [summary_map[fid] for fid in selected_formula_ids if fid in summary_map]

        from exact.type2.formulas.knowledge import _format_context
        from types import SimpleNamespace
        selection_ns = SimpleNamespace(
            solution_plan=solution_plan,
            missing_variables=missing_variables,
            confidence=getattr(choice, "confidence", None),
            notes=getattr(choice, "notes", ()),
        )
        new_context_str = _format_context(ordered_summaries, selection=selection_ns)

        formula_context = RetrievedFormulaContext(
            formula_ids=tuple(selected_formula_ids),
            context=new_context_str,
            summaries=ordered_summaries,
            solution_plan=tuple(solution_plan),
            missing_variables=tuple(missing_variables),
            selector_confidence=getattr(choice, "confidence", None),
            selector_notes=tuple(getattr(choice, "notes", ())),
        )

    prompt_context = _build_solver_context(extraction, formula_context)
    geometry_prompt_context = build_geometry_prompt_context(
        extraction,
        unit_hint=_formula_context_unit_hint(formula_context),
    )
    if geometry_prompt_context:
        prompt_context = f"{prompt_context}\n\n{geometry_prompt_context}"
    try:
        candidate_count = max(1, settings.type2_pot_batch_size)
        candidates = generate_pot_code_candidates(
            extraction.normalized_question,
            "Use the retrieved formulas to solve the problem with Pint.",
            formula_context=prompt_context,
            candidate_count=candidate_count,
            temperature=_pot_generation_temperature(settings, candidate_count),
            settings=settings,
            debug_metadata={
                "attempt": 0,
                "batch_size": candidate_count,
                "max_repair_attempts": settings.type2_pot_max_retries,
                "trigger": "initial_generation",
                "geometry_context_included": bool(geometry_prompt_context),
            },
        )
    except Exception as exc:
        reason = f"LLM code generation returned invalid output: {exc}"
        fallback = _try_executable_formula_fallback(extraction, formula_context, reason, settings)
        if fallback is not None:
            return fallback
        return _failed_result(extraction, reason)
    if candidates is None:
        return _unconfigured_result(extraction)
    if not candidates:
        return _failed_result(extraction, "LLM code generation returned no usable PoT candidates.")

    selected, failed_attempts = _select_verified_pot_candidate(
        extraction,
        candidates,
        formula_context,
        settings,
        solver_context=prompt_context,
    )
    if selected is None:
        reason = _candidate_failure_reason(failed_attempts)
        fallback = _try_executable_formula_fallback(extraction, formula_context, reason, settings)
        if fallback is not None:
            return fallback
        verification = (
            failed_attempts[-1].verified.verification
            if failed_attempts and failed_attempts[-1].verified
            else Verification(False, reason)
        )
        return Type2SolveResult(
            answer="",
            unit=None,
            value=None,
            formula=None,
            extraction=extraction,
            verification=verification,
            cot=[
                f"Generated {len(candidates)} PoT candidate program(s) with the LLM.",
                "No PoT candidate passed execution and verification.",
            ],
            premises=[],
            confidence=0.0,
            error=POT_SOLVER_FAILED,
        )

    assert selected.verified is not None
    deterministic_conflict = _try_deterministic_conflict_fallback(
        selected.verified,
        extraction,
        formula_context,
        settings,
    )
    if deterministic_conflict is not None:
        return deterministic_conflict

    explanation = _final_explanation(
        extraction,
        selected.verified.answer,
        selected.verified.unit,
        formula_context,
        selected.code_spec,
        settings,
        generate_explanation=generate_explanation,
    )
    return Type2SolveResult(
        answer=selected.verified.answer,
        unit=selected.verified.unit,
        value=selected.verified.value,
        formula=None,
        extraction=extraction,
        verification=selected.verified.verification,
        cot=[
            "Retrieved formula context for the question.",
            _pot_generation_cot(len(candidates), selected),
            *(
                [f"Repaired the selected generated program {selected.repair_attempts} time(s) before execution succeeded."]
                if selected.repair_attempts
                else []
            ),
            f"Executed {len(candidates)} PoT candidate program(s) in the sandbox.",
            "Verified the numeric answer, unit, and formula IDs.",
            *explanation.cot,
        ],
        premises=explanation.premises,
        confidence=0.72,
        error=None,
    )


def _pot_generation_temperature(settings: Settings, candidate_count: int) -> float:
    if candidate_count <= 1:
        return settings.llm_temperature
    return settings.type2_pot_batch_temperature


def _select_verified_pot_candidate(
    extraction: Extraction,
    candidates: list[PotCodeSpec],
    formula_context: RetrievedFormulaContext,
    settings: Settings,
    solver_context: str = "",
) -> tuple[PotCandidateAttempt | None, list[PotCandidateAttempt]]:
    attempts: list[PotCandidateAttempt] = []
    verified_attempts: list[PotCandidateAttempt] = []

    for index, candidate in enumerate(candidates):
        code_spec, execution, repair_attempts, repair_error = _execute_with_repair_loop(
            extraction,
            candidate,
            formula_context,
            settings,
            repair_context=solver_context,
        )
        if repair_error is not None:
            attempts.append(
                PotCandidateAttempt(
                    index=index,
                    code_spec=code_spec,
                    execution=execution,
                    repair_attempts=repair_attempts,
                    error=repair_error,
                )
            )
            continue
        if not execution.ok:
            attempts.append(
                PotCandidateAttempt(
                    index=index,
                    code_spec=code_spec,
                    execution=execution,
                    repair_attempts=repair_attempts,
                    error=execution.error or f"execution failed after {repair_attempts} repair attempt(s)",
                )
            )
            continue

        unit = execution.ans_unit or code_spec.answer_unit
        verified = _verify_or_accept_execution(
            execution.ans,
            unit,
            code_spec.formula_ids_used,
            formula_context,
            extraction,
            settings,
        )
        attempt = PotCandidateAttempt(
            index=index,
            code_spec=code_spec,
            execution=execution,
            repair_attempts=repair_attempts,
            verified=verified,
            error=verified.error,
        )
        attempts.append(attempt)
        if verified.error is None:
            verified_attempts.append(attempt)

    if not verified_attempts:
        return None, attempts
    return _choose_agreeing_candidate(verified_attempts), attempts


def _choose_agreeing_candidate(attempts: list[PotCandidateAttempt]) -> PotCandidateAttempt:
    best = attempts[0]
    best_count = 1
    for candidate in attempts:
        count = sum(
            1
            for other in attempts
            if _verified_values_agree(candidate.verified, other.verified)
        )
        if (
            count > best_count
            or (
                count == best_count
                and (candidate.repair_attempts, candidate.index)
                < (best.repair_attempts, best.index)
            )
        ):
            best = candidate
            best_count = count
    return replace(best, agreement_count=best_count)


def _verified_values_agree(left: OutputSanityResult | None, right: OutputSanityResult | None) -> bool:
    if left is None or right is None:
        return False
    if left.value is None or right.value is None:
        return left.answer == right.answer and left.unit == right.unit
    try:
        converted = right.value.to(left.value.units)
        left_magnitude = float(left.value.magnitude)
        right_magnitude = float(converted.magnitude)
    except Exception:
        return left.answer == right.answer and left.unit == right.unit
    if not (math.isfinite(left_magnitude) and math.isfinite(right_magnitude)):
        return False
    scale = max(abs(left_magnitude), abs(right_magnitude), 1e-12)
    return abs(left_magnitude - right_magnitude) <= max(1e-9, 0.02 * scale)


def _candidate_failure_reason(attempts: list[PotCandidateAttempt]) -> str:
    errors = [attempt.error for attempt in attempts if attempt.error]
    if not errors:
        return "No PoT candidate was available."
    return errors[-1]


def _pot_generation_cot(candidate_count: int, selected: PotCandidateAttempt) -> str:
    if candidate_count <= 1:
        return "Generated a Pint-based Python program with the LLM."
    return (
        f"Generated {candidate_count} Pint-based Python candidate program(s) with the LLM; "
        f"selected candidate {selected.index + 1} with agreement count {selected.agreement_count}."
    )


def _execute_code_spec(spec: PotCodeSpec, timeout_seconds: float) -> ExecutionResult:
    code = _prepare_generated_code(spec.code, spec.formula_ids_used)
    return execute_python(code, timeout_seconds=timeout_seconds)


def _is_execution_successful(execution: ExecutionResult) -> bool:
    if not execution.ok:
        return False
    if execution.ans is None:
        return False
    if not str(execution.ans).strip():
        return False
    return True


def _execute_with_repair_loop(
    extraction: Extraction,
    code_spec: PotCodeSpec,
    formula_context: RetrievedFormulaContext,
    settings: Settings,
    repair_context: str = "",
) -> tuple[PotCodeSpec, ExecutionResult, int, str | None]:
    code_spec = _canonicalize_formula_ids(code_spec, formula_context)
    execution = _execute_code_spec(code_spec, settings.type2_pot_timeout)
    repair_attempts = 0
    max_retries = settings.type2_pot_max_retries

    while not _is_execution_successful(execution) and repair_attempts < max_retries:
        repair_attempts += 1
        if not execution.ok:
            err_msg = execution.error or "execution failed"
        else:
            err_msg = f"Program ran successfully but produced an empty or invalid value for 'ans'. Current ans value was: {execution.ans}."

        try:
            repaired = repair_pot_code(
                extraction.normalized_question,
                code_spec.code,
                err_msg,
                settings=settings,
                repair_context=repair_context,
                debug_metadata={
                    "attempt": repair_attempts,
                    "max_repair_attempts": max_retries,
                    "trigger": "repair_after_execution_failure",
                    "previous_error": err_msg,
                },
            )
        except Exception as exc:
            return code_spec, execution, repair_attempts, f"LLM code repair returned invalid output: {exc}"
        if repaired is None:
            return code_spec, execution, repair_attempts, err_msg

        code_spec = _canonicalize_formula_ids(repaired, formula_context)
        execution = _execute_code_spec(code_spec, settings.type2_pot_timeout)

    return code_spec, execution, repair_attempts, None


def _canonicalize_formula_ids(spec: PotCodeSpec, formula_context: RetrievedFormulaContext) -> PotCodeSpec:
    canonical_ids = canonicalize_formula_ids(spec.formula_ids_used, formula_context.summaries)
    if canonical_ids == spec.formula_ids_used:
        return spec
    return spec.model_copy(update={"formula_ids_used": canonical_ids})


def _verify_or_accept_execution(
    ans: object | None,
    unit: str | None,
    formula_ids_used: list[str],
    formula_context: RetrievedFormulaContext,
    extraction: Extraction,
    settings: Settings,
) -> OutputSanityResult:
    if ans is None or not str(ans).strip():
        return OutputSanityResult(
            verification=Verification(False, "PoT ans is empty."),
            answer="",
            unit=None,
            value=None,
            error="type2_output_sanity_failed",
        )
    ans_str = str(ans).strip()
    try:
        magnitude = float(ans_str)
        if not math.isfinite(magnitude):
            return OutputSanityResult(
                verification=Verification(False, "PoT ans is not finite."),
                answer="",
                unit=None,
                value=None,
                error="type2_output_sanity_failed",
            )
        if extraction.target in {"force", "electric_field"}:
            magnitude = abs(magnitude)
        formatted_ans = _format_number(magnitude)
    except (TypeError, ValueError):
        formatted_ans = ans_str

    return OutputSanityResult(
        verification=Verification(True, "PoT execution accepted; verification bypassed/disabled."),
        answer=formatted_ans,
        unit=(unit or "").strip() or None,
        value=None,
        error=None,
    )


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _try_executable_formula_fallback(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
    reason: str | None,
    settings: Settings,
) -> Type2SolveResult | None:
    if not settings.type2_use_executable_fallback:
        return None
    if not _should_use_deterministic_first(extraction, formula_context):
        return solve_vector_template(extraction)
    result = solve_extraction(extraction, preferred_formula_ids=formula_context.formula_ids)
    if result.error is not None:
        return None
    result.cot.insert(0, f"PoT solver failed; executable formula fallback was used. Reason: {reason or 'unknown'}")
    return result


def _should_use_deterministic_first(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
) -> bool:
    """Run scalar executable formulas first only for formula-like routes."""

    label = classify_type2_taxonomy(
        extraction.normalized_question,
        unit=_formula_context_unit_hint(formula_context),
    )
    return label.solve_method in {"direct_formula", "inverse_formula", "multi_step_formula"}


def _formula_context_unit_hint(formula_context: RetrievedFormulaContext) -> str:
    for summary in formula_context.summaries:
        output = str(summary.get("output") or summary.get("output_unit") or "").strip()
        if output:
            return output
    return ""


def _try_deterministic_conflict_fallback(
    pot_result: OutputSanityResult,
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
    settings: Settings,
) -> Type2SolveResult | None:
    if not settings.type2_use_executable_fallback:
        return None
    if not _should_use_deterministic_first(extraction, formula_context):
        return None
    deterministic = solve_extraction(extraction, preferred_formula_ids=formula_context.formula_ids)
    if deterministic.error is not None:
        return None
    if _results_numerically_agree(pot_result, deterministic):
        return None
    deterministic.cot.insert(
        0,
        (
            "PoT output disagreed with deterministic executable formula; "
            f"used deterministic result instead. PoT returned {pot_result.answer} {pot_result.unit or ''}."
        ).strip(),
    )
    return replace(deterministic, confidence=max(deterministic.confidence, 0.86))


def _results_numerically_agree(pot_result: OutputSanityResult, deterministic: Type2SolveResult) -> bool:
    pot_value = pot_result.value
    deterministic_value = deterministic.value
    if pot_value is None or deterministic_value is None:
        return True
    try:
        converted = pot_value.to(deterministic_value.units)
        pot_magnitude = float(converted.magnitude)
        deterministic_magnitude = float(deterministic_value.magnitude)
    except Exception:
        return True
    if not (math.isfinite(pot_magnitude) and math.isfinite(deterministic_magnitude)):
        return True
    scale = max(abs(deterministic_magnitude), abs(pot_magnitude), 1e-12)
    return abs(pot_magnitude - deterministic_magnitude) <= max(1e-9, 0.02 * scale)


def _strip_code_fence(code: str) -> str:
    text = code.strip()
    match = re.fullmatch(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _prepare_generated_code(
    code: str,
    formula_ids_used: Sequence[str] = (),
) -> str:
    stripped = _strip_code_fence(code)
    normalized = _normalize_pint_prefixes(stripped)
    normalized = _normalize_bare_pint_unit_aliases(normalized)
    normalized = _normalize_pint_constants(normalized)
    normalized = _rewrite_sqrt_calls_for_pint_quantities(normalized)
    normalized = _repair_omitted_formula_roots(normalized, formula_ids_used)
    return _remove_unused_optional_imports(normalized)


_ROOT_FORMULA_TARGETS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "current_from_power_resistance": (
        frozenset({"A", "ampere", "amperes"}),
        frozenset({"I", "i", "current", "ans"}),
    ),
    "voltage_from_power_resistance": (
        frozenset({"V", "volt", "volts"}),
        frozenset({"U", "u", "V", "v", "voltage", "ans"}),
    ),
    "voltage_from_capacitor_energy_capacitance": (
        frozenset({"V", "volt", "volts"}),
        frozenset({"U", "u", "V", "v", "voltage", "ans"}),
    ),
    "charge_from_capacitor_energy_capacitance": (
        frozenset({"C", "coulomb", "coulombs"}),
        frozenset({"Q", "q", "charge", "ans"}),
    ),
    "coulomb_equal_charges_from_force": (
        frozenset({"C", "coulomb", "coulombs"}),
        frozenset({"Q", "q", "charge", "ans"}),
    ),
}


def _repair_omitted_formula_roots(
    code: str,
    formula_ids_used: Sequence[str],
) -> str:
    """Restore a square root omitted from a cited inverse-square formula."""
    targets = [
        _ROOT_FORMULA_TARGETS[formula_id]
        for formula_id in formula_ids_used
        if formula_id in _ROOT_FORMULA_TARGETS
    ]
    if not targets:
        return code

    allowed_units = frozenset(unit for units, _ in targets for unit in units)
    allowed_names = frozenset(name for _, names in targets for name in names)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    transformer = _OmittedFormulaRootTransformer(allowed_units, allowed_names)
    transformed = transformer.visit(tree)
    if not transformer.changed:
        return code

    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


class _OmittedFormulaRootTransformer(ast.NodeTransformer):
    def __init__(
        self,
        allowed_units: frozenset[str],
        allowed_names: frozenset[str],
    ) -> None:
        self.allowed_units = allowed_units
        self.allowed_names = allowed_names
        self.changed = False
        self._repair_assignment = False

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        previous = self._repair_assignment
        self._repair_assignment = any(
            isinstance(target, ast.Name) and target.id in self.allowed_names
            for target in node.targets
        )
        node.value = self.visit(node.value)
        self._repair_assignment = previous
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        previous = self._repair_assignment
        self._repair_assignment = (
            isinstance(node.target, ast.Name) and node.target.id in self.allowed_names
        )
        if node.value is not None:
            node.value = self.visit(node.value)
        self._repair_assignment = previous
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if not self._repair_assignment or not _is_target_unit_conversion(
            node, self.allowed_units
        ):
            return node

        receiver = node.func.value
        if not isinstance(receiver, ast.BinOp) or _is_half_power(receiver):
            return node

        node.func.value = ast.BinOp(
            left=receiver,
            op=ast.Pow(),
            right=ast.Constant(value=0.5),
        )
        self.changed = True
        return node


def _is_target_unit_conversion(node: ast.Call, allowed_units: frozenset[str]) -> bool:
    if (
        not isinstance(node.func, ast.Attribute)
        or node.func.attr != "to"
        or len(node.args) != 1
        or node.keywords
    ):
        return False
    unit = node.args[0]
    return isinstance(unit, ast.Constant) and unit.value in allowed_units


def _is_half_power(node: ast.AST) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Pow):
        return False
    exponent = node.right
    if isinstance(exponent, ast.Constant):
        return exponent.value == 0.5
    return (
        isinstance(exponent, ast.BinOp)
        and isinstance(exponent.op, ast.Div)
        and isinstance(exponent.left, ast.Constant)
        and exponent.left.value == 1
        and isinstance(exponent.right, ast.Constant)
        and exponent.right.value == 2
    )


def _normalize_bare_pint_unit_aliases(code: str) -> str:
    """Repair common LLM unit aliases inside Pint expressions, ignoring string literals."""

    replacements = {
        "m": "ureg.meter",
        "s": "ureg.second",
        "kg": "ureg.kilogram",
        "C": "ureg.coulomb",
        "N": "ureg.newton",
        "J": "ureg.joule",
        "V": "ureg.volt",
        "A": "ureg.ampere",
        "ohm": "ureg.ohm",
    }
    assigned_names = _assigned_python_names(code)
    pattern = re.compile(
        r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|"
        r"(?P<op>[*/])\s*(?P<unit>" + "|".join(map(re.escape, replacements)) + r")\b"
        r"(?P<power>\s*\*\*\s*[-+]?\d+)?"
    )

    def replace(match: re.Match[str]) -> str:
        if match.group("op") is None:
            return match.group(0)
        unit = match.group("unit")
        if unit in assigned_names:
            return match.group(0)
        return f"{match.group('op')} {replacements[unit]}{match.group('power') or ''}"

    return pattern.sub(replace, code)


def _assigned_python_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _normalize_pint_constants(code: str) -> str:
    coulomb_constant = "8.9875517923e9 * ureg.newton * ureg.meter ** 2 / ureg.coulomb ** 2"
    code = code.replace("ureg.constants.Coulomb_constant", f"({coulomb_constant})")
    code = code.replace("ureg.constants.coulomb_constant", f"({coulomb_constant})")
    return code


def _rewrite_sqrt_calls_for_pint_quantities(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    rewritten = _SqrtToPower().visit(tree)
    ast.fix_missing_locations(rewritten)
    try:
        return ast.unparse(rewritten)
    except Exception:
        return code


class _SqrtToPower(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if len(node.args) != 1 or node.keywords:
            return node
        if _is_sqrt_call(node.func):
            return ast.BinOp(
                left=node.args[0],
                op=ast.Pow(),
                right=ast.Constant(value=0.5),
            )
        return node


def _is_sqrt_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Attribute):
        return func.attr == "sqrt" and isinstance(func.value, ast.Name) and func.value.id in {"sympy", "sp", "math"}
    if isinstance(func, ast.Name):
        return func.id == "sqrt"
    return False


def _remove_unused_optional_imports(code: str) -> str:
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped in {"import numpy as np", "import numpy"}:
            continue
        if stripped.startswith("from numpy import "):
            continue
        lines.append(line)
    return "\n".join(lines)


_PINT_PREFIX_SCALE = {
    "tera": "1e12",
    "giga": "1e9",
    "mega": "1e6",
    "kilo": "1e3",
    "centi": "1e-2",
    "milli": "1e-3",
    "micro": "1e-6",
    "nano": "1e-9",
    "pico": "1e-12",
}


def _normalize_pint_prefixes(code: str) -> str:
    """Convert LLM-written Pint prefix objects into numeric scale factors."""

    def replace_prefix(match: re.Match[str]) -> str:
        prefix = match.group(1)
        return f"{_PINT_PREFIX_SCALE[prefix]} * ureg."

    return re.sub(
        r"ureg\.(" + "|".join(_PINT_PREFIX_SCALE) + r")\s*\*\s*ureg\.",
        replace_prefix,
        code,
    )


def _build_solver_context(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
) -> str:
    quantities = "\n".join(
        f"- {key}: {quantity.value} ({quantity.evidence})"
        for key, quantity in extraction.quantities.items()
    )
    limited_formula_ids = _limited_formula_ids(formula_context)
    allowed_ids = "\n- ".join(limited_formula_ids) if limited_formula_ids else "None"
    solution_plan = "\n".join(f"- {step}" for step in formula_context.solution_plan)
    geometry_plan = "\n".join(f"- {step}" for step in _build_geometry_plan(extraction))
    return (
        f"Extraction:\n"
        f"- kind: {extraction.kind.value}\n"
        f"- target: {extraction.target}\n"
        f"- quantities:\n{quantities or '- none'}\n\n"
        f"Geometry/vector plan:\n{geometry_plan or '- no explicit vector geometry detected'}\n\n"
        f"Allowed formula IDs:\n- {allowed_ids}\n\n"
        f"Formula selector plan:\n{solution_plan or '- none'}\n\n"
        f"Retrieved formulas (top {len(limited_formula_ids)} only):\n"
        f"{_limited_formula_context(formula_context, limited_formula_ids)}"
    )


def _limited_formula_ids(formula_context: RetrievedFormulaContext) -> tuple[str, ...]:
    if formula_context.formula_ids:
        return tuple(formula_context.formula_ids)
    ids = [str(summary.get("id")) for summary in formula_context.summaries if summary.get("id")]
    return tuple(ids)


def _limited_formula_context(
    formula_context: RetrievedFormulaContext,
    formula_ids: tuple[str, ...],
) -> str:
    if not formula_ids:
        return "No formula context available."
    allowed = set(formula_ids)
    lines = [
        line
        for line in formula_context.context.splitlines()
        if any(formula_id in line for formula_id in allowed)
    ]
    if lines:
        return "\n".join(lines)
    summaries = [
        summary
        for summary in formula_context.summaries
        if str(summary.get("id")) in allowed
    ]
    return "\n".join(
        f"- {summary.get('id')}: {summary.get('expression') or summary.get('latex')}"
        for summary in summaries
        if summary.get("id")
    )


def _build_geometry_plan(extraction: Extraction) -> list[str]:
    lower = extraction.normalized_question.lower()
    plan: list[str] = []
    if extraction.target in {"force", "electric_field"}:
        plan.append("Treat force/field as vectors; do not add magnitudes unless directions are explicitly the same.")
    if "same direction" in lower:
        plan.append("Directions are collinear and the same, so resultant magnitudes add.")
    if "opposite direction" in lower:
        plan.append("Directions are collinear and opposite, so resultant magnitude is the absolute difference.")
    if any(marker in lower for marker in ("perpendicular", "90 degree", "90°", "right angle")):
        plan.append("Perpendicular components must be combined with Pythagoras or x/y components.")
    if "angle" in lower and "angle" in extraction.quantities:
        plan.append("Use the stated included angle in the cosine-rule resultant.")
    if "equilateral" in lower:
        plan.append("Equilateral triangle force/field resultants use a 60 degree angle between equal pairwise vectors.")
    if "perpendicular bisector" in lower:
        plan.append("For a perpendicular-bisector target point, resolve each source contribution into x/y components.")
    if "right-angled at a" in lower or "right angle at a" in lower:
        plan.append("The target at A has perpendicular AB and AC directions; compute missing side before combining vectors.")
    if any(key in extraction.quantities for key in ("charge_2", "charge_3", "length_2", "length_3")):
        plan.append("Preserve object identity: q1/q2/q3 and AB/AC/BC distances are not interchangeable.")
    return plan


def _final_explanation(
    extraction: Extraction,
    answer: str,
    unit: str | None,
    formula_context: RetrievedFormulaContext,
    code_spec: PotCodeSpec,
    settings: Settings,
    generate_explanation: bool,
):
    from exact.type2.extraction.llm_structured import FinalExplanationSpec

    unit_suffix = f" {unit}" if unit else ""
    explanation_context = _build_explanation_context(
        extraction,
        answer,
        unit,
        formula_context,
        code_spec,
    )
    if generate_explanation:
        try:
            spec = generate_final_explanation(
                extraction.normalized_question,
                answer,
                unit,
                formula_context.context,
                explanation_context,
                code_spec.formula_ids_used,
                settings=settings,
            )
        except Exception:
            spec = None
        if spec is not None:
            return spec

    fallback_explanation = _clean_code_explanation(
        code_spec.explanation,
        answer=answer,
        unit=unit,
    )
    return FinalExplanationSpec(
        explanation=fallback_explanation,
        premises=[] if not generate_explanation else _fallback_formula_premises(formula_context, code_spec),
        cot=[] if not generate_explanation else ["No LLM final explanation was available; used a concise verified fallback."],
    )


def _build_explanation_context(
    extraction: Extraction,
    answer: str,
    unit: str | None,
    formula_context: RetrievedFormulaContext,
    code_spec: PotCodeSpec,
) -> str:
    selected_formulas = [
        {
            "id": summary.get("id"),
            "expression": summary.get("expression") or summary.get("latex"),
            "conditions": list(summary.get("conditions") or ()),
        }
        for summary in formula_context.summaries
        if str(summary.get("id")) in set(code_spec.formula_ids_used)
    ]
    if not selected_formulas:
        selected_formulas = [
            {
                "id": summary.get("id"),
                "expression": summary.get("expression") or summary.get("latex"),
                "conditions": list(summary.get("conditions") or ()),
            }
            for summary in formula_context.summaries[:3]
        ]

    payload = {
        "question": extraction.normalized_question,
        "answer": answer,
        "unit": unit,
        "known_quantities": [
            {
                "name": quantity.name,
                "value": str(quantity.value),
                "evidence": quantity.evidence,
            }
            for quantity in extraction.quantities.values()
        ],
        "selected_formulas": selected_formulas,
        "solution_plan": list(formula_context.solution_plan),
        "numeric_work": _clean_code_explanation(
            code_spec.explanation,
            answer=answer,
            unit=unit,
        ),
        "style": (
            "Explain like a physics teacher in 2-4 natural sentences: intuition, formula, "
            "substitution, final answer. Do not mention code."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _clean_code_explanation(explanation: str | None, *, answer: str, unit: str | None) -> str:
    text = (explanation or "").strip()
    unit_suffix = f" {unit}" if unit else ""
    default = f"Using the selected physics relationship, the verified computation gives {answer}{unit_suffix}."
    if not text:
        return default
    looks_like_formula_dump = (
        len(text) > 700
        or text.count("\n- ") >= 3
        or "[executable;" in text
        or "[knowledge_json;" in text
    )
    mentions_implementation = any(term in text.lower() for term in ("python", "pint", "json", "code"))
    if looks_like_formula_dump or mentions_implementation:
        return default
    return text


def _fallback_formula_premises(
    formula_context: RetrievedFormulaContext,
    code_spec: PotCodeSpec,
) -> list[str]:
    used_ids = set(code_spec.formula_ids_used)
    summaries = [
        summary
        for summary in formula_context.summaries
        if not used_ids or str(summary.get("id")) in used_ids
    ]
    premises: list[str] = []
    for summary in summaries[:3]:
        formula_id = summary.get("id")
        expression = summary.get("expression") or summary.get("latex")
        if formula_id and expression:
            premises.append(f"{formula_id}: {expression}")
        elif expression:
            premises.append(str(expression))
    return premises


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _unconfigured_result(extraction: Extraction) -> Type2SolveResult:
    return Type2SolveResult(
        answer="",
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(False, "PoT-first solver requires a configured LLM client."),
        cot=["No LLM client is configured for the PoT-first Type 2 solver."],
        premises=[],
        confidence=0.0,
        error=POT_SOLVER_NOT_CONFIGURED,
    )


def _failed_result(extraction: Extraction, reason: str) -> Type2SolveResult:
    return Type2SolveResult(
        answer="",
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(False, reason),
        cot=["PoT-first solving attempted code generation/execution but failed."],
        premises=[],
        confidence=0.0,
        error=POT_SOLVER_FAILED,
    )
