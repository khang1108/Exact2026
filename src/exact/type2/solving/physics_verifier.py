from __future__ import annotations

import math
from dataclasses import dataclass

import pint

from exact.type2.schemas import Extraction, Verification
from exact.type2.solving.electrostatic_graph import (
    build_electrostatic_graph,
    solve_electrostatic_force_graph,
)
from exact.type2.solving.vector_solver import solve_geometry_vector_problem
from exact.type2.solving.pot_verifier import OutputSanityResult
from exact.type2.solving.units import parse_quantity


STRUCTURED_STATE_ERROR = "type2_structured_state_verification_failed"
PHYSICS_ORACLE_ERROR = "type2_physics_oracle_verification_failed"


@dataclass(frozen=True)
class StructuredStateResult:
    verification: Verification
    problem_type: str | None
    error: str | None = None


@dataclass(frozen=True)
class PhysicsOracleResult:
    verification: Verification
    reference_answer: str | None
    reference_unit: str | None
    error: str | None = None


def verify_structured_state(extraction: Extraction) -> StructuredStateResult:
    if extraction.target not in {"force", "electric_field"}:
        return StructuredStateResult(
            verification=Verification(True, "No structured physics state required for this target."),
            problem_type=None,
            error=None,
        )

    lower = extraction.normalized_question.lower()
    if not any(
        marker in lower
        for marker in (
            "charge",
            "charges",
            "coulomb",
            "q1",
            "q2",
            "q3",
            "electric force",
            "electric field",
            "field strength",
        )
    ):
        return StructuredStateResult(
            verification=Verification(True, "Force problem is not recognized as Coulomb/vector state."),
            problem_type=None,
            error=None,
        )

    if extraction.target == "electric_field":
        vector_result = solve_geometry_vector_problem(extraction)
        if vector_result is None:
            return StructuredStateResult(
                verification=Verification(
                    False,
                    "Electric-field vector problem state could not be fully resolved into source charges, field point, and geometry.",
                ),
                problem_type="coulomb_vector",
                error=STRUCTURED_STATE_ERROR,
            )
        return StructuredStateResult(
            verification=Verification(
                True,
                "Resolved electric-field vector state with deterministic geometry.",
            ),
            problem_type="coulomb_vector",
            error=None,
        )

    graph = build_electrostatic_graph(extraction)
    if graph is None:
        return StructuredStateResult(
            verification=Verification(
                False,
                "Coulomb/vector problem state could not be fully resolved into charge nodes, target, and geometry.",
            ),
            problem_type="coulomb_vector",
            error=STRUCTURED_STATE_ERROR,
        )

    source_count = len(graph.charges) - 1
    if source_count < 1:
        return StructuredStateResult(
            verification=Verification(False, "Coulomb/vector state has no source charges."),
            problem_type="coulomb_vector",
            error=STRUCTURED_STATE_ERROR,
        )

    return StructuredStateResult(
        verification=Verification(
            True,
            f"Resolved Coulomb/vector state with layout `{graph.layout}` and {source_count} source charge(s).",
        ),
        problem_type="coulomb_vector",
        error=None,
    )


def verify_against_physics_oracle(
    extraction: Extraction,
    candidate: OutputSanityResult,
    *,
    relative_tolerance: float = 0.02,
    absolute_tolerance: float = 1e-9,
) -> PhysicsOracleResult:
    reference = solve_electrostatic_force_graph(extraction)
    if reference is None:
        reference = _solve_resultant_force_oracle(extraction)

    if reference is None:
        return PhysicsOracleResult(
            verification=Verification(True, "No deterministic physics oracle applies to this problem."),
            reference_answer=None,
            reference_unit=None,
            error=None,
        )

    if reference.value is None or reference.unit is None:
        return PhysicsOracleResult(
            verification=Verification(False, "Deterministic physics oracle failed to compute reference answer."),
            reference_answer=None,
            reference_unit=None,
            error=PHYSICS_ORACLE_ERROR,
        )

    if candidate.value is not None:
        try:
            candidate_value = candidate.value.to(reference.unit)
            candidate_number = float(candidate_value.magnitude)
        except Exception:
            candidate_number = _candidate_number_from_answer(candidate, reference.unit)
    else:
        candidate_number = _candidate_number_from_answer(candidate, reference.unit)

    reference_number = float(reference.value.to(reference.unit).magnitude)
    absolute_error = abs(candidate_number - reference_number)
    denominator = max(abs(reference_number), absolute_tolerance)
    relative_error = absolute_error / denominator
    if absolute_error <= absolute_tolerance or relative_error <= relative_tolerance:
        return PhysicsOracleResult(
            verification=Verification(
                True,
                f"Candidate agrees with deterministic Coulomb/vector oracle: {reference.answer} {reference.unit}.",
            ),
            reference_answer=reference.answer,
            reference_unit=reference.unit,
            error=None,
        )

    return PhysicsOracleResult(
        verification=Verification(
            False,
            (
                "Candidate disagrees with deterministic Coulomb/vector oracle: "
                f"candidate={candidate.answer} {candidate.unit}, "
                f"reference={reference.answer} {reference.unit}, "
                f"relative_error={relative_error:.6g}, absolute_error={absolute_error:.6g}."
            ),
        ),
        reference_answer=reference.answer,
        reference_unit=reference.unit,
        error=PHYSICS_ORACLE_ERROR,
    )


def _candidate_number_from_answer(candidate: OutputSanityResult, reference_unit: str) -> float:
    value = parse_quantity(float(candidate.answer), candidate.unit or reference_unit).to(reference_unit)
    number = float(value.magnitude)
    if not math.isfinite(number):
        raise ValueError("candidate answer is not finite")
    return number


def _solve_resultant_force_oracle(extraction: Extraction):
    lower = extraction.normalized_question.lower()
    q = extraction.quantities
    if extraction.target != "force" or "force" not in q or "force_2" not in q:
        return None

    f1 = q["force"].value.to("N")
    f2 = q["force_2"].value.to("N")
    if "same direction" in lower:
        value = f1 + f2
        message = "same_direction_resultant"
    elif "opposite direction" in lower or "opposite directions" in lower:
        value = abs(f1 - f2)
        message = "opposite_direction_resultant"
    elif any(marker in lower for marker in ("perpendicular", "90 degree", "90°", "right angle")):
        value = (f1**2 + f2**2) ** 0.5
        message = "perpendicular_resultant"
    elif "angle" in lower and "angle" in q:
        theta = q["angle"].value.to("radian").magnitude
        value = (f1**2 + f2**2 + 2 * f1 * f2 * math.cos(theta)) ** 0.5
        message = "included_angle_resultant"
    else:
        return None

    return _OracleReference(
        answer=_format_number(float(value.to("N").magnitude)),
        unit="N",
        value=value.to("N"),
        message=message,
    )


@dataclass(frozen=True)
class _OracleReference:
    answer: str
    unit: str
    value: pint.Quantity
    message: str


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
