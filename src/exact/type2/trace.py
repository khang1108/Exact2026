from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from exact.common.schemas import PredictionRequest, PredictionResponse


@dataclass(frozen=True)
class SolutionStep:
    step_id: str
    kind: str
    description: str
    expression: str | None = None
    result_value: str | None = None
    result_unit: str | None = None


@dataclass(frozen=True)
class SolutionTrace:
    query_id: str | None
    question_text: str
    solver_used: str
    steps: tuple[SolutionStep, ...]
    final_answer: str
    final_unit: str
    verification_accepted: bool
    verification_messages: tuple[str, ...] = ()
    premises: tuple[str, ...] = ()
    formula_ids_used: tuple[str, ...] = ()
    retrieved_formula_ids: tuple[str, ...] = ()
    generated_code: str = ""
    code_semantic_summary: str = ""
    computed_values: dict[str, Any] = field(default_factory=dict)
    solver_context: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_solution_trace(
    request: PredictionRequest,
    response: PredictionResponse,
    *,
    verification_accepted: bool | None = None,
    verification_messages: tuple[str, ...] = (),
    computed_values: dict[str, Any] | None = None,
) -> SolutionTrace:
    """Build the canonical explanation context from a completed Type 2 response."""

    existing = _existing_trace(response)
    if existing is not None:
        return existing

    diagnostics = dict(response.routing_diagnostics or {})
    steps = tuple(
        SolutionStep(
            step_id=f"s{index}",
            kind="solution_step",
            description=str(description),
        )
        for index, description in enumerate(response.cot or (), start=1)
        if str(description).strip()
    )
    accepted = (
        verification_accepted
        if verification_accepted is not None
        else response.error is None and bool(str(response.answer or "").strip())
    )
    solver_context = _solver_context(diagnostics)
    formula_ids = _string_tuple(
        diagnostics.get("formula_ids_used")
        or diagnostics.get("formula_ids")
        or diagnostics.get("retrieved_formula_ids")
        or solver_context.get("selected_rule")
    )
    retrieved_formula_ids = _string_tuple(diagnostics.get("retrieved_formula_ids"))
    return SolutionTrace(
        query_id=response.query_id or request.id,
        question_text=request.question,
        solver_used=_solver_name(diagnostics),
        steps=steps,
        final_answer=str(response.answer or ""),
        final_unit=str(response.unit or ""),
        verification_accepted=accepted,
        verification_messages=verification_messages,
        premises=tuple(
            str(item)
            for item in response.premises or ()
            if str(item).strip() and not str(item).startswith("diagnostics=")
        ),
        formula_ids_used=formula_ids,
        retrieved_formula_ids=retrieved_formula_ids,
        generated_code=str(diagnostics.get("generated_code") or ""),
        code_semantic_summary=str(diagnostics.get("code_semantic_summary") or ""),
        computed_values=dict(computed_values or diagnostics.get("computed_values") or {}),
        solver_context=solver_context,
        assumptions=_string_tuple(diagnostics.get("assumptions")),
        warnings=_string_tuple(diagnostics.get("verification_warnings")),
    )


def _existing_trace(response: PredictionResponse) -> SolutionTrace | None:
    reasoning = response.reasoning or {}
    if reasoning.get("type") != "solution_trace" or not isinstance(reasoning.get("trace"), dict):
        return None
    raw = dict(reasoning["trace"])
    try:
        raw["steps"] = tuple(SolutionStep(**step) for step in raw.get("steps", ()))
        raw.setdefault("solver_context", {})
        for key in (
            "verification_messages",
            "premises",
            "formula_ids_used",
            "retrieved_formula_ids",
            "assumptions",
            "warnings",
        ):
            raw[key] = tuple(raw.get(key, ()))
        trace = SolutionTrace(**raw)
        if trace.final_answer != str(response.answer or ""):
            return None
        if trace.final_unit != str(response.unit or ""):
            return None
        return trace
    except (TypeError, ValueError):
        return None


def _solver_name(diagnostics: dict[str, Any]) -> str:
    selected_route = diagnostics.get("selected_route")
    if isinstance(selected_route, dict) and selected_route.get("selected_solver"):
        return str(selected_route["selected_solver"])
    return str(
        diagnostics.get("solver")
        or diagnostics.get("selected_solver")
        or diagnostics.get("predicted_method")
        or diagnostics.get("domain")
        or "type2_pipeline"
    )


def _solver_context(diagnostics: dict[str, Any]) -> dict[str, Any]:
    selected_route = diagnostics.get("selected_route")
    if not isinstance(selected_route, dict):
        return {}
    solver_result = selected_route.get("solver_result")
    if not isinstance(solver_result, dict):
        solver_result = {}
    solver_diagnostics = solver_result.get("diagnostics")
    if not isinstance(solver_diagnostics, dict):
        solver_diagnostics = {}
    context = {
        "domain": selected_route.get("domain"),
        "system_type": selected_route.get("system_type"),
        "selected_rule": solver_result.get("selected_rule")
        or solver_diagnostics.get("selected_rule"),
        "result": solver_result.get("result") or solver_diagnostics.get("result"),
        "normalized_inputs": solver_diagnostics.get("normalized_inputs") or {},
        "intermediate_values": solver_diagnostics.get("intermediate_values") or {},
    }
    return {key: value for key, value in context.items() if value not in (None, {}, [])}


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()
