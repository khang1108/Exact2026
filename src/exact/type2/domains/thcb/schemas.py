from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ThcbQuantity:
    name: str
    value: float
    unit: str


@dataclass
class ThcbContract:
    family: str
    target: str
    quantities: dict[str, ThcbQuantity] = field(default_factory=dict)
    readings: list[ThcbQuantity] = field(default_factory=list)
    relation: str | None = None
    requested_outputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ThcbAnswer:
    answer: str
    unit: str | None
    explanation: str
    cot: list[str]
    confidence: float = 1.0
