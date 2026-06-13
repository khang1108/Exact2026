from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MeasurementQuantity:
    value: float
    unit: str
    original: str = ""


@dataclass(frozen=True)
class Uncertainty:
    value: float
    unit: str
    source: str = "given"


@dataclass(frozen=True)
class MeasuredQuantity:
    symbol: str
    value: float
    unit: str
    original: str = ""
    absolute_uncertainty: Uncertainty | None = None
    relative_uncertainty: float | None = None
    percentage_uncertainty: float | None = None


@dataclass(frozen=True)
class MeasurementTarget:
    quantities: tuple[str, ...]
    of: str
    unit: str | None = None


@dataclass(frozen=True)
class MeasurementEvidence:
    text: str
    mapped_to: dict[str, Any]


@dataclass(frozen=True)
class MeasurementContract:
    system_type: str
    target: MeasurementTarget
    measured_quantities: dict[str, MeasuredQuantity] = field(default_factory=dict)
    true_value: MeasurementQuantity | None = None
    measured_value: MeasurementQuantity | None = None
    measurements: tuple[MeasurementQuantity, ...] = ()
    instrument: dict[str, MeasurementQuantity | str] = field(default_factory=dict)
    derived_quantity: dict[str, Any] = field(default_factory=dict)
    error_model: dict[str, Any] = field(default_factory=dict)
    error_policy: dict[str, Any] = field(default_factory=dict)
    rounding_policy: dict[str, Any] = field(default_factory=dict)
    domain: str = "measurement_error"
    parse_confidence: float = 0.0
    evidence: tuple[MeasurementEvidence, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class MeasurementValidationIssue:
    reason: str
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatedMeasurementContract:
    contract: MeasurementContract

