from __future__ import annotations

from exact.type2.electromagnetism.diagnostics import unsolved
from exact.type2.electromagnetism.schemas import ElectromagnetismContract


def solve(contract: ElectromagnetismContract) -> dict:
    return unsolved("direct_formula_solver", f"no direct shortcut registered for `{contract.target.quantity}`")
