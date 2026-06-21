from __future__ import annotations

from exact.common.schemas import PredictionRequest, PredictionResponse
from exact.config import Settings
from exact.llm_client import has_json_llm_client_config
from exact.type2.extraction.llm_structured import generate_explanation_from_trace
from exact.type2.trace import SolutionTrace, build_solution_trace


def finalize_type2_explanation(
    request: PredictionRequest,
    response: PredictionResponse,
    settings: Settings,
    *,
    verification_accepted: bool | None = None,
    verification_messages: tuple[str, ...] = (),
) -> PredictionResponse:
    """Attach SolutionTrace and optionally create one grounded LLM explanation."""

    trace = build_solution_trace(
        request,
        response,
        verification_accepted=verification_accepted,
        verification_messages=verification_messages,
    )
    diagnostics = dict(response.routing_diagnostics or {})
    update = {
        "reasoning": {"type": "solution_trace", "trace": trace.to_dict()},
        "routing_diagnostics": diagnostics,
    }
    if not _should_generate(settings, trace):
        diagnostics["type2_final_explanation"] = {
            "generated": False,
            "reason": _skip_reason(settings, trace),
        }
        return response.model_copy(update=update | _fallback_explanation_update(response, trace))

    try:
        generated = generate_explanation_from_trace(
            request.question,
            trace.to_dict(),
            settings=settings,
        )
    except Exception as exc:
        diagnostics["type2_final_explanation"] = {
            "generated": False,
            "reason": "generation_error",
            "error_type": type(exc).__name__,
        }
        generated = None

    if generated is None:
        diagnostics.setdefault(
            "type2_final_explanation",
            {"generated": False, "reason": "llm_unavailable"},
        )
        return response.model_copy(update=update | _fallback_explanation_update(response, trace))
    if not _explanation_matches_trace(generated.explanation, trace):
        diagnostics["type2_final_explanation"] = {
            "generated": False,
            "reason": "grounding_check_failed",
        }
        return response.model_copy(update=update | _fallback_explanation_update(response, trace))

    diagnostics["type2_final_explanation"] = {
        "generated": True,
        "reason": "solution_trace",
        "status": _explanation_status(trace),
    }
    update.update(
        {
            "explanation": generated.explanation.strip(),
            "premises": list(generated.premises) or list(trace.premises),
            "cot": list(generated.cot) or [step.description for step in trace.steps],
        }
    )
    return response.model_copy(update=update)


def _should_generate(settings: Settings, trace: SolutionTrace) -> bool:
    return (
        settings.type2_generate_explanation
        and has_json_llm_client_config(settings)
    )


def _skip_reason(settings: Settings, trace: SolutionTrace) -> str:
    if not settings.type2_generate_explanation:
        return "disabled"
    if not has_json_llm_client_config(settings):
        return "llm_not_configured"
    return "llm_generation_skipped"


def _explanation_matches_trace(explanation: str, trace: SolutionTrace) -> bool:
    text = explanation.strip()
    if not text:
        return False
    lowered = text.lower()
    if any(token in lowered for token in ("python", "json", "pipeline failure", "program-of-thought")):
        return False
    answer = trace.final_answer.strip()
    if answer and answer not in text:
        return False
    unit = trace.final_unit.strip()
    return not unit or unit.lower() in lowered


def _explanation_status(trace: SolutionTrace) -> str:
    if trace.verification_accepted and trace.final_answer.strip():
        return "verified"
    if trace.final_answer.strip():
        return "unverified"
    return "incomplete"


def _fallback_explanation_update(
    response: PredictionResponse,
    trace: SolutionTrace,
) -> dict[str, object]:
    composer = ExplanationComposer()
    answer = trace.final_answer.strip()
    unit = trace.final_unit.strip()
    explanation = composer.compose(trace, answer, unit)
    if not answer:
        explanation = _incomplete_explanation(trace)
    premises = list(trace.premises)
    if not premises and trace.formula_ids_used:
        premises = [f"Relevant formulas/rules: {', '.join(trace.formula_ids_used)}."]
    return {
        "explanation": explanation,
        "premises": premises or list(response.premises or ()),
        "cot": list(response.cot or ()) or [step.description for step in trace.steps],
    }


def _incomplete_explanation(trace: SolutionTrace) -> str:
    formulas = ", ".join(trace.formula_ids_used)
    premises = "; ".join(trace.premises[:2])
    warnings = "; ".join(trace.warnings[:2])
    parts = ["This problem can be approached by identifying the relevant physical principles from the available solution trace."]
    if formulas:
        parts.append(f"Relevant formulas or rules include: {formulas}.")
    elif premises:
        parts.append(f"Useful known relationships from the trace are: {premises}.")
    if warnings:
        parts.append(f"A final answer was not confirmed because: {warnings}.")
    else:
        parts.append("A final numeric answer is still missing, so the next step is to map the known quantities to the appropriate formula and complete the substitution carefully.")
    return " ".join(parts)


class ExplanationComposer:
    """Deterministic fallback composer retained for non-LLM deployments."""

    def __init__(self, settings=None) -> None:
        self.settings = settings

    def compose(self, trace: SolutionTrace, answer: str, unit: str) -> str:
        sentences = [step.description for step in trace.steps if step.kind != "verification"]
        if trace.formula_ids_used:
            sentences.insert(0, f"We start from the relevant relation(s): {', '.join(trace.formula_ids_used)}.")
        if answer:
            if trace.verification_accepted:
                sentences.append(f"Therefore, the verified final answer is {answer}{(' ' + unit) if unit else ''}.")
            else:
                sentences.append(f"This gives the best available final answer: {answer}{(' ' + unit) if unit else ''}.")
        return " ".join(part.strip() for part in sentences if part.strip())
