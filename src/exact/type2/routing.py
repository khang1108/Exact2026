from __future__ import annotations

from typing import Any

from exact.datasets.type2_taxonomy import classify_type2_taxonomy
from exact.type2.formulas.knowledge import RetrievedFormulaContext
from exact.type2.schemas import Extraction


def build_routing_diagnostics(
    question: str,
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
    *,
    request_id: str | None = None,
    gold_or_dataset_method: str | None = None,
    gold_solver_family: str | None = None,
    gold_formula_family: str | None = None,
) -> dict[str, Any]:
    """Predict deterministic Type 2 routing without changing solver behavior."""

    label = classify_type2_taxonomy(
        question=question,
        unit=_target_unit_hint(formula_context),
    )
    solver_checks = [
        _check_text_solver(label.solver_family, label.solve_method, extraction),
        _check_formula_executor(label.solver_family, label.solve_method, extraction, formula_context),
        _check_vector_superposition(label.solve_method, extraction),
        _check_electrostatic_graph(label.solve_method, extraction),
        _check_circuit_network_graph(label.solve_method, extraction),
        _check_capacitor_network_graph(label.solve_method, extraction),
        _check_equilibrium_solver(label.solve_method, extraction),
    ]
    eligible = [item["solver"] for item in solver_checks if item["eligible"]]
    ineligible = [
        {"solver": item["solver"], "reason": item["reason"]}
        for item in solver_checks
        if not item["eligible"]
    ]

    return {
        "id": request_id,
        "gold_solver_family": gold_solver_family,
        "gold_or_dataset_method": gold_or_dataset_method,
        "gold_formula_family": gold_formula_family,
        "predicted_solver_family": label.solver_family,
        "predicted_method": label.solve_method,
        "formula_family": label.question_type,
        "eligible_solvers": eligible,
        "ineligible_solvers": ineligible,
        "extraction_target": extraction.target,
        "extraction_kind": extraction.kind.value,
        "extracted_quantities": sorted(extraction.quantities),
        "retrieved_formula_ids": list(formula_context.formula_ids),
        "actual_solver": None,
        "fallback_solver": None,
        "fallback_used": None,
        "verification_passed": None,
        "correct": None,
        "diagnostics_only": True,
    }


def mark_current_solver_used(diagnostics: dict[str, Any], *, error: str | None) -> dict[str, Any]:
    updated = dict(diagnostics)
    updated["actual_solver"] = "llm_pot"
    updated["fallback_solver"] = "llm_pot"
    updated["fallback_used"] = False
    updated["verification_passed"] = error is None
    return updated


def _check_text_solver(solver_family: str, solve_method: str, extraction: Extraction) -> dict[str, Any]:
    if solver_family == "text_solver" or solve_method == "conceptual_reasoning":
        return _eligible("text_solver")
    if extraction.kind.value == "conceptual":
        return _eligible("text_solver")
    return _ineligible("text_solver", "question is not classified as conceptual")


def _check_formula_executor(
    solver_family: str,
    solve_method: str,
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
) -> dict[str, Any]:
    if solver_family != "formula_executor":
        if solve_method in {"geometry_vector_graph", "vector_superposition"}:
            return _ineligible("formula_executor", "target appears vector-valued, not scalar direct formula")
        return _ineligible("formula_executor", f"predicted solver family is {solver_family}")
    if not formula_context.formula_ids:
        return _ineligible("formula_executor", "no formulas were retrieved")
    if extraction.target is None:
        return _ineligible("formula_executor", "extraction has no target quantity")
    return _eligible("formula_executor")


def _check_vector_superposition(solve_method: str, extraction: Extraction) -> dict[str, Any]:
    if solve_method != "vector_superposition":
        return _ineligible("vector_superposition", f"predicted method is {solve_method}")
    if extraction.target not in {"force", "electric_field"}:
        return _ineligible("vector_superposition", "target is not force or electric_field")
    if not _has_any_quantity(extraction, "force", "force_2", "electric_field", "electric_field_2"):
        return _ineligible("vector_superposition", "missing vector magnitudes/components")
    return _eligible("vector_superposition")


def _check_electrostatic_graph(solve_method: str, extraction: Extraction) -> dict[str, Any]:
    if solve_method not in {"geometry_vector_graph", "electrostatic_force_graph"}:
        return _ineligible("electrostatic_graph", f"predicted method is {solve_method}")
    if extraction.target not in {"force", "electric_field"}:
        return _ineligible("electrostatic_graph", "target is not force or electric_field")
    if not _has_any_quantity(extraction, "charge", "charge_2", "charge_3"):
        return _ineligible("electrostatic_graph", "missing source/target charge quantities")
    if not _has_any_quantity(extraction, "length", "length_2", "length_3"):
        return _ineligible("electrostatic_graph", "missing distance quantities")
    return _eligible("electrostatic_graph")


def _check_circuit_network_graph(solve_method: str, extraction: Extraction) -> dict[str, Any]:
    if solve_method != "circuit_network_graph":
        return _ineligible("circuit_network_graph", f"predicted method is {solve_method}")
    if not _has_any_quantity(extraction, "resistance", "resistance_2", "current", "voltage"):
        return _ineligible("circuit_network_graph", "missing circuit quantities")
    return _eligible("circuit_network_graph")


def _check_capacitor_network_graph(solve_method: str, extraction: Extraction) -> dict[str, Any]:
    if solve_method != "capacitor_network_graph":
        return _ineligible("capacitor_network_graph", f"predicted method is {solve_method}")
    if not _has_any_quantity(extraction, "capacitance", "capacitance_2", "charge", "voltage"):
        return _ineligible("capacitor_network_graph", "missing capacitor quantities")
    return _eligible("capacitor_network_graph")


def _check_equilibrium_solver(solve_method: str, extraction: Extraction) -> dict[str, Any]:
    if solve_method != "equilibrium_solve":
        return _ineligible("equilibrium_solver", f"predicted method is {solve_method}")
    if extraction.target not in {"length", "force", "electric_field", "charge"}:
        return _ineligible("equilibrium_solver", "target is not a supported equilibrium unknown")
    return _eligible("equilibrium_solver")


def _target_unit_hint(formula_context: RetrievedFormulaContext) -> str:
    for summary in formula_context.summaries:
        output = str(summary.get("output") or summary.get("output_unit") or "").strip()
        if output:
            return output
    return ""


def _has_any_quantity(extraction: Extraction, *names: str) -> bool:
    return any(name in extraction.quantities for name in names)


def _eligible(solver: str) -> dict[str, Any]:
    return {"solver": solver, "eligible": True, "reason": ""}


def _ineligible(solver: str, reason: str) -> dict[str, Any]:
    return {"solver": solver, "eligible": False, "reason": reason}
