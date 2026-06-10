from __future__ import annotations

from exact.type2.electromagnetism.diagnostics import unsolved
from exact.type2.electromagnetism.schemas import ElectromagnetismContract


def solve(contract: ElectromagnetismContract) -> dict:
    return unsolved("conceptual_state_solver", f"no conceptual state rule registered for `{contract.target.quantity}`")
