from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DdtQuantity:
    name: str
    value: float
    unit: str


@dataclass
class DdtContract:
    family: str
    target: str
    quantities: dict[str, DdtQuantity] = field(default_factory=dict)
    relation: str | None = None
    notes: list[str] = field(default_factory=list)
    source: str = "unknown"


@dataclass
class DdtAnswer:
    answer: str
    unit: str | None
    explanation: str
    cot: list[str]
    confidence: float = 0.9
