from __future__ import annotations

from typing import Any

from exact.type2.measurement.schemas import MeasuredQuantity
from exact.type2.measurement.uncertainty_policy import absolute_uncertainty, relative_uncertainty


def evaluate_ast(ast: Any, values: dict[str, MeasuredQuantity]) -> float:
    if isinstance(ast, str):
        return values[ast].value
    if isinstance(ast, (int, float)):
        return float(ast)
    op = ast["op"]
    args = ast.get("args", ())
    if op == "const":
        return float(ast["value"])
    if op == "add":
        return sum(evaluate_ast(arg, values) for arg in args)
    if op == "sub":
        first, second = args
        return evaluate_ast(first, values) - evaluate_ast(second, values)
    if op == "mul":
        result = 1.0
        for arg in args:
            result *= evaluate_ast(arg, values)
        return result
    if op == "div":
        first, second = args
        return evaluate_ast(first, values) / evaluate_ast(second, values)
    if op == "pow":
        base, exponent = args
        return evaluate_ast(base, values) ** float(exponent)
    raise ValueError(f"unsupported AST op {op}")


def propagated_relative_error(ast: Any, values: dict[str, MeasuredQuantity]) -> tuple[float, dict[str, float]]:
    contributions: dict[str, float] = {}
    total = _relative_contribution(ast, values, 1.0, contributions)
    return total, contributions


def propagated_absolute_error_for_sum(ast: Any, values: dict[str, MeasuredQuantity]) -> float:
    if isinstance(ast, str):
        return absolute_uncertainty(values[ast])
    if isinstance(ast, (int, float)):
        return 0.0
    op = ast["op"]
    args = ast.get("args", ())
    if op in {"add", "sub"}:
        return sum(propagated_absolute_error_for_sum(arg, values) for arg in args)
    raise ValueError("absolute addition/subtraction propagation only supports add/sub AST")


def is_additive_ast(ast: Any) -> bool:
    if isinstance(ast, (str, int, float)):
        return True
    return ast.get("op") in {"add", "sub", "const"} and all(is_additive_ast(arg) for arg in ast.get("args", ()))


def _relative_contribution(ast: Any, values: dict[str, MeasuredQuantity], exponent_factor: float, contributions: dict[str, float]) -> float:
    if isinstance(ast, str):
        contribution = abs(exponent_factor) * relative_uncertainty(values[ast])
        contributions[ast] = contributions.get(ast, 0.0) + contribution
        return contribution
    if isinstance(ast, (int, float)):
        return 0.0
    op = ast["op"]
    args = ast.get("args", ())
    if op == "const":
        return 0.0
    if op in {"mul", "div"}:
        return sum(_relative_contribution(arg, values, exponent_factor, contributions) for arg in args)
    if op == "pow":
        base, exponent = args
        return _relative_contribution(base, values, exponent_factor * float(exponent), contributions)
    raise ValueError(f"relative propagation does not support `{op}`")

