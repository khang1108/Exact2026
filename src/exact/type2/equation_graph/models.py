from __future__ import annotations

from dataclasses import dataclass

import pint


@dataclass(frozen=True)
class EquationStep:
    formula_id: str
    output: str
    expression: str
    value: pint.Quantity


@dataclass(frozen=True)
class EquationGraphResult:
    solved: bool
    answer: str = ""
    unit: str | None = None
    value: pint.Quantity | None = None
    formula_ids_used: tuple[str, ...] = ()
    steps: tuple[EquationStep, ...] = ()
    error: str | None = None

