from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from exact.type2.schemas import Type2QuestionKind


class PotMode(str, Enum):
    NUMERIC_SINGLE = "numeric_single"
    NUMERIC_MULTI_JSON = "numeric_multi_json"
    SYMBOLIC_EXPR_JSON = "symbolic_expr_json"
    LOCATION_OR_ANGLE_NUMERIC = "location_or_angle_numeric"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ExecutionPolicy:
    pot_mode: PotMode
    solver_family: str | None = None
    solve_method: str | None = None
    use_direction_classifier: bool = False
    use_conceptual_classifier: bool = False


def build_execution_policy(
    answer_mode: Type2QuestionKind | str,
    solver_family: str | None = None,
    solve_method: str | None = None,
) -> ExecutionPolicy:
    kind = _kind_value(answer_mode)

    if kind in {
        Type2QuestionKind.NUMERICAL.value,
        "scalar_numeric",
        Type2QuestionKind.MIXED.value,
    }:
        return ExecutionPolicy(
            pot_mode=PotMode.NUMERIC_SINGLE,
            solver_family=solver_family,
            solve_method=solve_method,
        )

    if kind == "multi_value_numeric":
        return ExecutionPolicy(pot_mode=PotMode.NUMERIC_MULTI_JSON, solver_family=solver_family, solve_method=solve_method)

    if kind == "symbolic_formula":
        return ExecutionPolicy(pot_mode=PotMode.SYMBOLIC_EXPR_JSON, solver_family=solver_family, solve_method=solve_method)

    if kind == "location_or_angle_numeric":
        return ExecutionPolicy(pot_mode=PotMode.LOCATION_OR_ANGLE_NUMERIC, solver_family=solver_family, solve_method=solve_method)

    if kind == "directional_output":
        return ExecutionPolicy(pot_mode=PotMode.DISABLED, use_direction_classifier=True, solver_family=solver_family, solve_method=solve_method)

    if kind in {Type2QuestionKind.CONCEPTUAL.value, "qualitative_conceptual"}:
        return ExecutionPolicy(pot_mode=PotMode.DISABLED, use_conceptual_classifier=True, solver_family=solver_family, solve_method=solve_method)

    return ExecutionPolicy(pot_mode=PotMode.DISABLED, solver_family=solver_family, solve_method=solve_method)


def validate_fallback_output(kind: Type2QuestionKind | str, output: Any) -> bool:
    if not output:
        return False

    kind_value = _kind_value(kind)

    if kind_value in {
        Type2QuestionKind.NUMERICAL.value,
        "scalar_numeric",
        Type2QuestionKind.MIXED.value,
    }:
        return is_single_numeric_answer(output)

    if kind_value == "multi_value_numeric":
        return is_parts_json(output)

    if kind_value == "symbolic_formula":
        return is_parseable_sympy_expression(output)

    if kind_value == "location_or_angle_numeric":
        return is_numeric_with_expected_target(output)

    if kind_value == "directional_output":
        return is_canonical_direction(output)

    if kind_value in {Type2QuestionKind.CONCEPTUAL.value, "qualitative_conceptual"}:
        return is_short_text_answer(output)

    return False


def _kind_value(kind: Type2QuestionKind | str) -> str:
    return kind.value if isinstance(kind, Type2QuestionKind) else str(kind)


def is_single_numeric_answer(output: Any) -> bool:
    if not isinstance(output, str):
        output = str(output)
    return bool(re.search(r"\d", output))


def is_parts_json(output: Any) -> bool:
    if not isinstance(output, str):
        return False
    try:
        data = json.loads(output)
        return "parts" in data and isinstance(data["parts"], list)
    except Exception:
        return False


def is_parseable_sympy_expression(output: Any) -> bool:
    if not isinstance(output, str):
        return False
    try:
        data = json.loads(output)
        return "expression" in data and "symbols" in data
    except Exception:
        return False


def is_numeric_with_expected_target(output: Any) -> bool:
    return is_single_numeric_answer(output)


def is_canonical_direction(output: Any) -> bool:
    if not isinstance(output, str):
        return False
    valid = {
        "toward_q1",
        "toward_q2",
        "toward_positive_charge",
        "toward_negative_charge",
        "left",
        "right",
        "up",
        "down",
        "clockwise",
        "counterclockwise",
        "same_direction_as_larger_force",
        "opposite_direction_to_larger_force",
        "zero_no_direction",
        "unknown",
    }
    return output.strip() in valid


def is_short_text_answer(output: Any) -> bool:
    if not isinstance(output, str):
        return False
    try:
        data = json.loads(output)
        return "answer" in data and "reason" in data
    except Exception:
        return False
