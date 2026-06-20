from __future__ import annotations

from math import atan2, degrees, hypot, isfinite

from exact.type2.formulas.bank import FORMULAS
from exact.type2.schemas import SolveContext, SolveResult, SolutionStep, SolverEligibility
from exact.type2.solving.units import ureg


class ConceptualSolver:
    name = "ConceptualSolver"

    def can_solve(self, ctx: SolveContext) -> SolverEligibility:
        eligible = ctx.ir.question_kind == "conceptual"
        return SolverEligibility(eligible=eligible, confidence=0.9 if eligible else 0.0, reason=None if eligible else "not conceptual")

    def solve(self, ctx: SolveContext) -> SolveResult:
        target = ctx.ir.target.raw_text or "the requested concept"
        text = f"The problem is conceptual, so the answer is explained using the identified physics principle for {target}."
        return SolveResult(
            success=True,
            answer="",
            unit="",
            solver_name=self.name,
            solution_steps=[SolutionStep(step_id="s1", kind="conceptual", description=text)],
            confidence=0.6,
        )


class FormulaGraphSolver:
    name = "FormulaGraphSolver"

    def can_solve(self, ctx: SolveContext) -> SolverEligibility:
        eligible = bool(ctx.formula_plan.steps) and not ctx.ir.flags.get("requires_vector") and not ctx.ir.flags.get("requires_network")
        return SolverEligibility(eligible=eligible, confidence=ctx.formula_plan.confidence if eligible else 0.0, reason=None if eligible else "no safe scalar plan")

    def solve(self, ctx: SolveContext) -> SolveResult:
        values = _build_known_values(ctx)
        solution_steps: list[SolutionStep] = []
        formula_ids_used: list[str] = []
        intermediate_values: dict[str, float | str] = {}
        final_answer = None
        final_unit = None

        for index, plan_step in enumerate(ctx.formula_plan.steps, start=1):
            formula = next((item for item in FORMULAS if item.id == plan_step.formula_id), None)
            if formula is None:
                return SolveResult(success=False, solver_name=self.name, error="FORMULA_NOT_FOUND")
            try:
                if formula.callable is None:
                    return SolveResult(success=False, solver_name=self.name, error="FORMULA_NOT_EXECUTABLE")
                result = formula.callable(values)
            except Exception as exc:
                return SolveResult(success=False, solver_name=self.name, error=str(exc))
            magnitude = getattr(result, "magnitude", None)
            if magnitude is None or not isfinite(float(magnitude)):
                return SolveResult(success=False, solver_name=self.name, error="INVALID_NUMERIC_RESULT")
            canonical_result = _canonicalize_quantity(result, plan_step.output_dimension)
            values[plan_step.output_variable] = canonical_result
            answer = f"{float(magnitude):.6g}"
            answer = f"{float(canonical_result.magnitude):.6g}"
            unit = _normalize_unit(str(canonical_result.units))
            final_answer = answer
            final_unit = unit
            formula_ids_used.append(formula.id)
            intermediate_values[plan_step.output_variable] = answer
            formula_step_id = f"s{len(solution_steps) + 1}"
            solution_steps.append(
                SolutionStep(
                    step_id=formula_step_id,
                    kind="formula",
                    description=formula.explanation_template,
                    formula_id=formula.id,
                    expression=formula.expression,
                    result_variable=plan_step.output_variable,
                    depends_on=list(plan_step.depends_on),
                )
            )
            solution_steps.append(
                SolutionStep(
                    step_id=f"s{len(solution_steps) + 1}",
                    kind="intermediate_result" if index < len(ctx.formula_plan.steps) else "final_result",
                    description=f"Substitute the known values and compute {plan_step.output_variable}.",
                    result_variable=plan_step.output_variable,
                    result_value=answer,
                    result_unit=unit,
                    depends_on=[formula_step_id],
                )
            )

        return SolveResult(
            success=True,
            answer=final_answer,
            unit=final_unit,
            solver_name=self.name,
            formula_ids_used=formula_ids_used,
            intermediate_values=intermediate_values,
            solution_steps=solution_steps,
            confidence=ctx.formula_plan.confidence,
        )


class VectorGeometrySolver:
    name = "VectorGeometrySolver"

    def can_solve(self, ctx: SolveContext) -> SolverEligibility:
        eligible = bool(ctx.ir.flags.get("requires_vector"))
        return SolverEligibility(eligible=eligible, confidence=0.4 if eligible else 0.0, reason=None if eligible else "vector reasoning not required")

    def solve(self, ctx: SolveContext) -> SolveResult:
        vector_group = ctx.ir.vector_contribution_groups[0] if ctx.ir.vector_contribution_groups else None
        if vector_group is not None and vector_group.relation == "same_direction":
            return _solve_same_direction_forces(ctx)
        if vector_group is not None and vector_group.count == 2 and vector_group.magnitude_symbol:
            return _solve_equal_vector_group(ctx, vector_group)
        values = _build_known_values(ctx)
        text = ctx.ir.question_text.lower()
        charges = sorted(
            [name for name, quantity in ctx.ir.knowns.items() if quantity.dimension == "charge"],
        )
        distances = sorted(
            [name for name, quantity in ctx.ir.knowns.items() if quantity.dimension in {"distance", "length"}],
        )
        if ctx.ir.target.dimension not in {"force", "electric_field"}:
            return SolveResult(success=False, solver_name=self.name, error="UNSUPPORTED_VECTOR_TARGET")
        if len(charges) < 2 or not distances:
            return SolveResult(success=False, solver_name=self.name, error="INSUFFICIENT_VECTOR_VALUES")

        component_vectors: list[tuple[float, float, str]] = []
        solution_steps: list[SolutionStep] = []
        formula_ids_used: list[str] = []
        target_quantity_name = "charge_3" if "on q3" in text or len(charges) >= 3 else ("charge_2" if ctx.ir.target.dimension == "force" and len(charges) >= 2 else None)

        source_charges = [name for name in charges if name != target_quantity_name]
        if not source_charges:
            return SolveResult(success=False, solver_name=self.name, error="NO_VECTOR_SOURCES")

        angle_deg = next(iter(ctx.ir.geometry.angles_degrees.values()), 60.0 if "equilateral" in text else 90.0 if "right angle" in text or "perpendicular" in text else None)
        if len(source_charges) == 1:
            return SolveResult(success=False, solver_name=self.name, error="SINGLE_VECTOR_SOURCE")

        magnitudes: list[tuple[float, str]] = []
        for index, source_name in enumerate(source_charges[:2], start=1):
            known = {
                source_name: values[source_name],
                distances[0]: values[distances[0]],
            }
            if ctx.ir.target.dimension == "force":
                if target_quantity_name is None or target_quantity_name not in values:
                    return SolveResult(success=False, solver_name=self.name, error="MISSING_TARGET_CHARGE")
                known[target_quantity_name] = values[target_quantity_name]
                formula = next(item for item in FORMULAS if item.id == "coulomb_force")
            else:
                formula = next(item for item in FORMULAS if item.id == "electric_field_point_charge")
            try:
                if formula.callable is None:
                    return SolveResult(success=False, solver_name=self.name, error="FORMULA_NOT_EXECUTABLE")
                result = formula.callable(known)
            except Exception as exc:
                return SolveResult(success=False, solver_name=self.name, error=str(exc))
            canonical_result = _canonicalize_quantity(result, ctx.ir.target.dimension)
            magnitude = float(canonical_result.to_base_units().magnitude)
            magnitudes.append((magnitude, _normalize_unit(str(canonical_result.units))))
            formula_ids_used.append(formula.id)
            solution_steps.append(
                SolutionStep(
                    step_id=f"s{len(solution_steps) + 1}",
                    kind="formula",
                    description=f"Compute the magnitude of contribution {index} from {source_name}.",
                    formula_id=formula.id,
                    expression=formula.expression,
                    result_variable=f"contribution_{index}",
                )
            )

        theta = float(angle_deg if angle_deg is not None else 90.0)
        for index, (magnitude, unit) in enumerate(magnitudes, start=1):
            angle = 0.0 if index == 1 else theta
            x_component = magnitude if angle == 0.0 else magnitude * __import__("math").cos(__import__("math").radians(angle))
            y_component = 0.0 if angle == 0.0 else magnitude * __import__("math").sin(__import__("math").radians(angle))
            component_vectors.append((x_component, y_component, unit))
            solution_steps.append(
                SolutionStep(
                    step_id=f"s{len(solution_steps) + 1}",
                    kind="vector_decomposition",
                    description=f"Resolve contribution {index} into x and y components using angle {angle:.6g} deg.",
                    result_variable=f"contribution_{index}",
                    result_value=f"({x_component:.6g}, {y_component:.6g})",
                    result_unit=unit,
                )
            )

        total_x = sum(item[0] for item in component_vectors)
        total_y = sum(item[1] for item in component_vectors)
        final_magnitude = hypot(total_x, total_y)
        unit = component_vectors[0][2]
        direction_deg = degrees(atan2(total_y, total_x)) if ctx.ir.target.wants_direction or ctx.ir.geometry.direction_required else None
        answer = f"{final_magnitude:.6g}"
        solution_steps.append(
            SolutionStep(
                step_id=f"s{len(solution_steps) + 1}",
                kind="vector_sum",
                description="Sum the vector components to obtain the resultant vector.",
                result_variable="resultant",
                result_value=f"({total_x:.6g}, {total_y:.6g})",
                result_unit=unit,
            )
        )
        final_description = "Compute the magnitude of the resultant vector."
        if direction_deg is not None:
            final_description = f"Compute the magnitude and direction of the resultant vector. Direction = {direction_deg:.6g} deg from +x."
        solution_steps.append(
            SolutionStep(
                step_id=f"s{len(solution_steps) + 1}",
                kind="final_result",
                description=final_description,
                result_variable=ctx.ir.target.dimension or "resultant",
                result_value=answer,
                result_unit=unit,
            )
        )
        warnings = []
        if angle_deg is None:
            warnings.append("Used default right-angle vector composition because no explicit geometry angle was extracted.")
        return SolveResult(
            success=True,
            answer=answer,
            unit=unit,
            solver_name=self.name,
            formula_ids_used=formula_ids_used,
            intermediate_values={"resultant_x": f"{total_x:.6g}", "resultant_y": f"{total_y:.6g}"},
            solution_steps=solution_steps,
            confidence=0.65 if not warnings else 0.45,
            warnings=warnings,
        )


class NetworkCircuitSolver:
    name = "NetworkCircuitSolver"

    def can_solve(self, ctx: SolveContext) -> SolverEligibility:
        eligible = bool(ctx.ir.flags.get("requires_network"))
        return SolverEligibility(eligible=eligible, confidence=0.8 if eligible else 0.0, reason=None if eligible else "network reasoning not required")

    def solve(self, ctx: SolveContext) -> SolveResult:
        text = ctx.ir.question_text.lower()
        relation_types = {relation.type for relation in ctx.ir.relations}
        is_parallel = "parallel" in text or "connected_in_parallel" in relation_types
        is_series = "series" in text or "connected_in_series" in relation_types
        if not is_parallel and not is_series:
            return SolveResult(success=False, solver_name=self.name, error="UNSUPPORTED_NETWORK_TOPOLOGY")
        resistances = [q for q in ctx.extraction.quantities if q.dimension == "resistance" and q.value is not None]
        voltage = next((q for q in ctx.extraction.quantities if q.dimension == "voltage" and q.value is not None), None)
        if len(resistances) < 2 or voltage is None:
            return SolveResult(success=False, solver_name=self.name, error="INSUFFICIENT_NETWORK_VALUES")
        if is_parallel:
            reciprocal = sum(1.0 / q.value for q in resistances if q.value)
            if reciprocal == 0:
                return SolveResult(success=False, solver_name=self.name, error="INVALID_NETWORK_VALUES")
            equivalent = 1.0 / reciprocal
            reduction_expression = "1/Req = sum(1/Ri)"
            reduction_description = "Reduce the parallel resistors to an equivalent resistance."
        else:
            equivalent = sum(q.value for q in resistances if q.value is not None)
            reduction_expression = "Req = sum(Ri)"
            reduction_description = "Reduce the series resistors to an equivalent resistance."
        current = voltage.value / equivalent
        return SolveResult(
            success=True,
            answer=f"{current:.6g}",
            unit="A",
            solver_name=self.name,
            formula_ids_used=["parallel_resistance_two" if is_parallel else "series_resistance", "ohm_current"],
            intermediate_values={"Req": f"{equivalent:.6g}"},
            solution_steps=[
                SolutionStep(step_id="s1", kind="network_reduction", description=reduction_description, expression=reduction_expression, result_variable="Req", result_value=f"{equivalent:.6g}", result_unit="ohm"),
                SolutionStep(step_id="s2", kind="final_result", description="Use Ohm's law on the equivalent resistance to get the total current.", expression="I = V/Req", result_variable="I", result_value=f"{current:.6g}", result_unit="A", depends_on=["s1"]),
            ],
            confidence=0.95,
        )


class PoTSolver:
    name = "PoTSolver"

    def can_solve(self, ctx: SolveContext) -> SolverEligibility:
        return SolverEligibility(eligible=True, confidence=0.1, reason="fallback solver")

    def solve(self, ctx: SolveContext) -> SolveResult:
        return SolveResult(success=False, solver_name=self.name, error="NEED_MORE_FORMULA_CONTEXT")


def _build_known_values(ctx: SolveContext) -> dict[str, object]:
    values: dict[str, object] = {}
    dimension_counts: dict[str, int] = {}
    for quantity in ctx.extraction.quantities:
        if quantity.value is None or not quantity.unit:
            continue
        canonical_name = quantity.name
        if canonical_name in values:
            dimension = quantity.dimension or canonical_name
            dimension_counts[dimension] = dimension_counts.get(dimension, 1) + 1
            canonical_name = f"{canonical_name}_{dimension_counts[dimension]}"
        values[canonical_name] = quantity.value * ureg(quantity.unit)
    return values


def _normalize_unit(unit: str) -> str:
    normalized = unit.replace("ohm", "ohm").replace("ampere", "A").replace("volt", "V").replace("newton", "N")
    normalized = normalized.replace("coulomb", "C").replace("farad", "F").replace("joule", "J").replace("watt", "W")
    normalized = normalized.replace("N / C", "N/C").replace("V / m", "V/m")
    return normalized


def _canonicalize_quantity(quantity, output_dimension: str | None):
    if output_dimension == "power":
        return quantity.to("W")
    if output_dimension == "energy":
        return quantity.to("J")
    if output_dimension == "current":
        return quantity.to("A")
    if output_dimension == "voltage":
        return quantity.to("V")
    if output_dimension == "resistance":
        return quantity.to("ohm")
    if output_dimension == "capacitance":
        return quantity.to("F")
    if output_dimension == "charge":
        return quantity.to("C")
    if output_dimension in {"distance", "length"}:
        return quantity.to("m")
    if output_dimension == "speed":
        return quantity.to("m/s")
    if output_dimension == "magnetic_field":
        return quantity.to("T")
    if output_dimension == "force":
        return quantity.to("N")
    if output_dimension == "electric_field":
        try:
            return quantity.to("N/C")
        except Exception:
            return quantity.to("V/m")
    return quantity


def _solve_equal_vector_group(ctx: SolveContext, group) -> SolveResult:
    quantity = next(
        (
            item
            for item in ctx.extraction.quantities
            if item.symbol == group.magnitude_symbol or item.name == group.magnitude_symbol.lower()
        ),
        None,
    )
    if quantity is None or quantity.value is None or not quantity.unit:
        return SolveResult(success=False, solver_name=VectorGeometrySolver.name, error="INSUFFICIENT_VECTOR_VALUES")
    angle_deg = group.angle_between_deg
    if angle_deg is None:
        return SolveResult(success=False, solver_name=VectorGeometrySolver.name, error="MISSING_VECTOR_ANGLE")
    magnitude = quantity.value
    unit = quantity.unit
    theta = float(angle_deg)
    x1, y1 = magnitude, 0.0
    x2 = magnitude * __import__("math").cos(__import__("math").radians(theta))
    y2 = magnitude * __import__("math").sin(__import__("math").radians(theta))
    total_x = x1 + x2
    total_y = y1 + y2
    final_magnitude = hypot(total_x, total_y)
    direction_deg = degrees(atan2(total_y, total_x)) if ctx.ir.target.wants_direction or ctx.ir.geometry.direction_required else None
    description = f"Combine two equal vector contributions of magnitude {magnitude:.6g} {unit} separated by {theta:.6g} deg."
    final_description = "Compute the magnitude of the resultant vector."
    if direction_deg is not None:
        final_description = f"Compute the magnitude and direction of the resultant vector. Direction = {direction_deg:.6g} deg from +x."
    return SolveResult(
        success=True,
        answer=f"{final_magnitude:.6g}",
        unit=unit,
        solver_name=VectorGeometrySolver.name,
        formula_ids_used=[],
        intermediate_values={"resultant_x": f"{total_x:.6g}", "resultant_y": f"{total_y:.6g}"},
        solution_steps=[
            SolutionStep(step_id="s1", kind="vector_decomposition", description=description, result_variable="contribution_group", result_value=f"({x1:.6g}, 0), ({x2:.6g}, {y2:.6g})", result_unit=unit),
            SolutionStep(step_id="s2", kind="vector_sum", description="Sum the vector components to obtain the resultant vector.", result_variable="resultant", result_value=f"({total_x:.6g}, {total_y:.6g})", result_unit=unit, depends_on=["s1"]),
            SolutionStep(step_id="s3", kind="final_result", description=final_description, result_variable=ctx.ir.target.dimension or "resultant", result_value=f"{final_magnitude:.6g}", result_unit=unit, depends_on=["s2"]),
        ],
        confidence=0.8,
    )


def _solve_same_direction_forces(ctx: SolveContext) -> SolveResult:
    forces = [item for item in ctx.extraction.quantities if item.dimension == "force" and item.value is not None and item.unit]
    if len(forces) < 2:
        return SolveResult(success=False, solver_name=VectorGeometrySolver.name, error="INSUFFICIENT_VECTOR_VALUES")
    total = sum(float(item.value) for item in forces[:2])
    unit = _normalize_unit(forces[0].unit or "N")
    return SolveResult(
        success=True,
        answer=f"{total:.6g}",
        unit=unit,
        solver_name=VectorGeometrySolver.name,
        solution_steps=[
            SolutionStep(
                step_id="s1",
                kind="vector_sum",
                description="For forces acting in the same direction, add their magnitudes.",
                result_variable=ctx.ir.target.dimension or "resultant_force",
                result_value=f"{total:.6g}",
                result_unit=unit,
            )
        ],
        confidence=0.85,
    )
