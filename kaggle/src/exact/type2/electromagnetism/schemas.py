from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


OutputKind = Literal["numeric", "symbolic", "boolean", "magnitude_direction", "conceptual"]


@dataclass(frozen=True)
class EMQuantityValue:
    value: float
    unit: str
    original: str = ""


@dataclass(frozen=True)
class EMTarget:
    quantity: str
    unit: str | None = None
    output: OutputKind = "numeric"


@dataclass(frozen=True)
class EMComponent:
    id: str
    kind: str
    properties: dict[str, EMQuantityValue] = field(default_factory=dict)
    role: str | None = None


@dataclass(frozen=True)
class EMEvidence:
    text: str
    mapped_to: dict[str, Any]


@dataclass(frozen=True)
class ElectromagnetismContract:
    target: EMTarget
    system_type: str
    components: tuple[EMComponent, ...] = ()
    knowns: dict[str, EMQuantityValue] = field(default_factory=dict)
    source: dict[str, EMQuantityValue | str | bool] = field(default_factory=dict)
    geometry: dict[str, EMQuantityValue | str | bool] = field(default_factory=dict)
    state: dict[str, EMQuantityValue | str | bool] = field(default_factory=dict)
    condition: dict[str, EMQuantityValue | str | bool] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)
    convention: dict[str, Any] = field(default_factory=dict)
    primary: dict[str, EMQuantityValue | str | bool] = field(default_factory=dict)
    secondary: dict[str, EMQuantityValue | str | bool] = field(default_factory=dict)
    coil: dict[str, EMQuantityValue | str | bool] = field(default_factory=dict)
    flux_change: dict[str, EMQuantityValue | str | bool] = field(default_factory=dict)
    domain: str = "electromagnetism"
    parse_confidence: float = 0.0
    evidence: tuple[EMEvidence, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class EMValidationIssue:
    reason: str
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatedEMContract:
    contract: ElectromagnetismContract
