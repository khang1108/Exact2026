from __future__ import annotations

from typing import Any


def solved(
    solver: str,
    selected_rule: str,
    result: dict[str, Any],
    *,
    normalized_inputs: dict[str, Any] | None = None,
    intermediate_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "solver": solver,
        "status": "solved",
        "selected_rule": selected_rule,
        "normalized_inputs": normalized_inputs or {},
        "intermediate_values": intermediate_values or {},
        "result": result,
    }


def partial(
    solver: str,
    reason: str,
    result: dict[str, Any],
    *,
    missing: list[str] | None = None,
    intermediate_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "solver": solver,
        "status": "partial",
        "reason": reason,
        "missing": missing or [],
        "intermediate_values": intermediate_values or {},
        "result": result,
        "fallback_recommended": True,
    }


def unsolved(
    solver: str,
    reason: str,
    *,
    missing: list[str] | None = None,
    fallback_recommended: bool = True,
) -> dict[str, Any]:
    return {
        "solver": solver,
        "status": "unsolved",
        "reason": reason,
        "missing": missing or [],
        "fallback_recommended": fallback_recommended,
    }
