from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ScalarQuantity:
    name: str
    value: float
    unit: str

@dataclass
class FunctionQuantity:
    symbol: str          # "I" or "q"
    expression: str      # normalized Python-like expression, e.g. "2*sin(100*pi*t)"
    output_unit: str     # "A" or "C"
    variable: str = "t"

@dataclass
class EnergyExtraction:
    family: str
    target: str

    C: ScalarQuantity | None = None
    V: ScalarQuantity | None = None
    Q: ScalarQuantity | None = None
    L: ScalarQuantity | None = None
    I: ScalarQuantity | None = None
    E: ScalarQuantity | None = None
    E_total: ScalarQuantity | None = None
    E_electric: ScalarQuantity | None = None
    E_magnetic: ScalarQuantity | None = None
    
    energy_ratio: float | None = None
    energy_ratio_target: str | None = None
    relative_permittivity: float | None = None
    
    t: ScalarQuantity | None = None

    I_t: FunctionQuantity | None = None
    q_t: FunctionQuantity | None = None
