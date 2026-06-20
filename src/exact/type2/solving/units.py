from __future__ import annotations

import pint


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def parse_quantity(value: float, unit: str) -> pint.Quantity:
    unit = unit.replace("Ω", "ohm").replace("Ω", "ohm").replace("μ", "u").replace("µ", "u")
    if unit.lower() in {"turns/m", "turn/m", "turns per meter", "turn per meter"}:
        unit = "1/m"
    q = Q_(value, unit)
    low = unit.lower()
    
    # Specific known mappings to base SI units used in our domains
    mapping = {
        "mj": "J", "uj": "J", "nj": "J", "kw": "W", "mw": "W", 
        "uf": "F", "nf": "F", "pf": "F", "mf": "F",
        "uc": "C", "nc": "C", "pc": "C", "mc": "C",
        "mh": "H", "uh": "H",
        "cm": "m", "mm": "m", "km": "m",
        "ma": "A", "ka": "A",
        "kv": "V", "mv": "V",
        "kohm": "ohm",
        "kpa": "Pa",
        "mn": "N", "mt": "T",
        "uwb": "Wb", "mwb": "Wb"
    }
    
    if low in mapping:
        return q.to(mapping[low])
        
    if low.startswith("μ") or low.startswith("µ"):
        base = unit[1:]
        if base.lower() in ("f", "c", "h", "a", "v", "j", "w", "s", "m", "t", "wb", "n"):
            try:
                return q.to(base)
            except pint.errors.DimensionalityError:
                pass
                
    if low.startswith("m") and len(low) > 1:
        base = unit[1:]
        if base.lower() in ("f", "c", "h", "a", "v", "j", "w", "s", "t", "wb", "n"):
            try:
                return q.to(base)
            except pint.errors.DimensionalityError:
                pass

    return q
