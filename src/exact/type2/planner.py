from __future__ import annotations

from collections import deque

from exact.type2.schemas import FormulaPlan, FormulaRetrievalResult, FormulaStep, PhysicsIR


def build_formula_plan(ir: PhysicsIR, formula_context: FormulaRetrievalResult, settings=None) -> FormulaPlan:
    target_dimension = ir.target.dimension or "unknown"
    known_names = set(ir.knowns)
    target_name = ir.target.symbol or ir.target.dimension or "answer"
    missing_variables: list[str] = []
    direct = _build_direct_plan(formula_context, known_names, target_dimension)
    if direct is not None:
        return direct

    multistep = _build_multistep_plan(formula_context, known_names, target_dimension, target_name)
    if multistep is not None:
        return multistep

    for formula in formula_context.retrieved:
        if formula.output_dimension != target_dimension:
            continue
        unmet = [name for name in formula.required if name not in known_names]
        missing_variables.extend(unmet)
    return FormulaPlan(
        target=target_name,
        target_dimension=target_dimension,
        steps=[],
        missing_variables=sorted(set(missing_variables)),
        confidence=0.0,
        notes=["NO_SAFE_FORMULA_PLAN"],
    )


def _build_direct_plan(formula_context: FormulaRetrievalResult, known_names: set[str], target_dimension: str) -> FormulaPlan | None:
    for formula in formula_context.retrieved:
        required = list(formula.required)
        if formula.output_dimension == target_dimension and set(required).issubset(known_names):
            return FormulaPlan(
                target=formula.output,
                target_dimension=target_dimension,
                steps=[
                    FormulaStep(
                        step_id="plan_1",
                        formula_id=formula.id,
                        input_variables=required,
                        output_variable=formula.output,
                        output_dimension=formula.output_dimension,
                    )
                ],
                confidence=0.9,
                notes=["SAFE_DIRECT_PLAN"],
            )
    return None


def _build_multistep_plan(
    formula_context: FormulaRetrievalResult,
    known_names: set[str],
    target_dimension: str,
    target_name: str,
) -> FormulaPlan | None:
    formulas = [formula for formula in formula_context.retrieved if formula.output_dimension]
    queue = deque([(known_names, [])])
    visited = {frozenset(known_names)}
    best_missing: list[str] = []

    while queue:
        available, plan = queue.popleft()
        if len(plan) >= 3:
            continue
        for formula in formulas:
            if formula.output in available:
                continue
            required = set(formula.required)
            unmet = sorted(required - available)
            if unmet:
                if formula.output_dimension == target_dimension:
                    best_missing.extend(unmet)
                continue
            step_id = f"plan_{len(plan) + 1}"
            depends_on = [step.step_id for step in plan if step.output_variable in required]
            step = FormulaStep(
                step_id=step_id,
                formula_id=formula.id,
                input_variables=list(formula.required),
                output_variable=formula.output,
                output_dimension=formula.output_dimension,
                depends_on=depends_on,
            )
            next_available = set(available)
            next_available.add(formula.output)
            next_plan = [*plan, step]
            if formula.output_dimension == target_dimension:
                confidence = max(0.5, 0.9 - 0.1 * (len(next_plan) - 1))
                return FormulaPlan(
                    target=target_name,
                    target_dimension=target_dimension,
                    steps=next_plan,
                    confidence=confidence,
                    notes=["SAFE_MULTISTEP_PLAN"],
                )
            frozen = frozenset(next_available)
            if frozen in visited:
                continue
            visited.add(frozen)
            queue.append((next_available, next_plan))
    return None
