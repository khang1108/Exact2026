from __future__ import annotations

from exact.type2.schemas import SolveContext, SolveResult, VerificationResult


_DIMENSION_TO_UNITS = {
    "electric_field": {"N/C", "V/m"},
    "force": {"N"},
    "magnetic_field": {"T"},
    "potential": {"V"},
    "energy": {"J"},
    "current": {"A"},
    "resistance": {"ohm"},
    "power": {"W"},
    "voltage": {"V"},
    "capacitance": {"F"},
}


def verify_result(result: SolveResult, ctx: SolveContext) -> VerificationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not result.success:
        errors.append(result.error or "solve failed")
    if result.answer in (None, "") and ctx.ir.question_kind != "conceptual":
        errors.append("missing answer")
    expected_units = _DIMENSION_TO_UNITS.get(ctx.ir.target.dimension or "", set())
    if expected_units and result.unit and result.unit not in expected_units:
        errors.append(f"unit {result.unit} incompatible with target {ctx.ir.target.dimension}")
    if expected_units and not result.unit and ctx.ir.question_kind != "conceptual":
        errors.append("missing unit")
    if ctx.ir.target.wants_direction and not any(step.kind in {"vector_sum", "final_result"} and "direction" in step.description.lower() for step in result.solution_steps):
        warnings.append("direction requested but not explicitly derived")
    if not result.solution_steps:
        errors.append("missing solution steps")
    accepted = not errors
    confidence = result.confidence if accepted else 0.0
    return VerificationResult(accepted=accepted, confidence=confidence, errors=errors, warnings=warnings)
