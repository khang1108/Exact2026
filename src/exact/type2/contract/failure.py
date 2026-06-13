from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractFailure:
    status: str
    reason: str
    missing: tuple[str, ...] = ()
    fallback_recommended: bool = True

