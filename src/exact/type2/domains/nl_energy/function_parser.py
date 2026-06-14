from __future__ import annotations

import math
import re
from exact.type2.domains.nl_energy.schemas import FunctionQuantity

def _parse_amp(amp_str: str) -> float:
    if amp_str == "" or amp_str == "+":
        return 1.0
    if amp_str == "-":
        return -1.0
    if "√" in amp_str:
        parts = amp_str.split("√")
        first = float(parts[0]) if parts[0] not in ("", "+", "-") else (1.0 if parts[0] != "-" else -1.0)
        second = math.sqrt(float(parts[1]))
        return first * second
    return float(amp_str)

def _parse_sinusoidal(func: FunctionQuantity) -> tuple[float, str, str]:
    expr = func.expression.lower().replace(" ", "")
    expr = expr.replace("×10^", "e").replace("x10^", "e").replace("*10^", "e")
    expr = expr.replace("×10⁻", "e-").replace("x10⁻", "e-").replace("*10⁻", "e-")
    expr = expr.replace("⁻", "-")
    
    superscripts = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
    for sup, digit in superscripts.items():
        expr = expr.replace(sup, digit)
        
    expr = expr.replace("π", "pi").replace("\\pi", "pi")
    
    match = re.match(r"^(.*?)(sin|cos)\((.*?)\)$", expr)
    if not match:
        raise ValueError(f"Could not parse sinusoidal expression: {func.expression}")
        
    amp_str, trig_func, inner = match.groups()
    amp = _parse_amp(amp_str)
    
    return amp, trig_func, inner

def evaluate_sinusoidal(func: FunctionQuantity, time_seconds: float) -> float:
    amp, trig_func, inner = _parse_sinusoidal(func)
    
    inner = inner.replace("t", "").replace("*", "")
    if "pi" in inner:
        inner = inner.replace("pi", "")
        if inner == "" or inner == "-":
            inner += "1"
        omega = float(inner) * math.pi
    else:
        omega = float(inner) if inner else 1.0
        
    angle = omega * time_seconds
    
    if trig_func == "sin":
        return amp * math.sin(angle)
    else:
        return amp * math.cos(angle)

def get_amplitude(func: FunctionQuantity) -> float:
    amp, _, _ = _parse_sinusoidal(func)
    return abs(amp)
