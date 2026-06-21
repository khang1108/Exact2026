from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    passed: bool
    message: str
    severity: Literal["error", "warning"] = "error"
    confidence_delta: float = 0.0


@dataclass(frozen=True)
class PhysicsSanityReport:
    accepted: bool
    confidence_delta: float
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    rule_results: tuple[RuleResult, ...] = field(default_factory=tuple)

    @property
    def message(self) -> str:
        if self.accepted:
            return "Physics sanity checks passed."
        return "; ".join(self.errors) or "Physics sanity checks failed."

