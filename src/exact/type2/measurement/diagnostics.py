from __future__ import annotations

from typing import Any


def solved(solver: str, selected_rule: str, result: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"solver": solver, "status": "solved", "selected_rule": selected_rule, **extra, "result": result}


def unsolved(solver: str, reason: str, *, missing: list[str] | None = None, fallback_recommended: bool = True) -> dict[str, Any]:
    return {
        "solver": solver,
        "status": "unsolved",
        "reason": reason,
        "missing": missing or [],
        "fallback_recommended": fallback_recommended,
    }

