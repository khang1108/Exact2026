from __future__ import annotations

from typing import Any

from exact.type2.solver_contract.models import ParsedQuantity


UNIT_ALIASES = {
    "uC": "microcoulomb",
    "μC": "microcoulomb",
    "mC": "millicoulomb",
    "nC": "nanocoulomb",
    "cm": "centimeter",
    "mm": "millimeter",
    "degree": "degree",
    "degrees": "degree",
    "deg": "degree",
    "V/m": "volt / meter",
    "N/C": "newton / coulomb",
}


def normalize_unit_text(text: str) -> str:
    s = text.strip()
    s = s.replace("µ", "μ")
    s = s.replace("×10^", "e")
    s = s.replace(" x 10^", "e")
    s = s.replace("10^-", "1e-")
    s = s.replace("10⁻", "1e-")
    s = s.replace("u C", "uC")
    
    # Replace known aliases
    # We do a simple split and replace for the unit part if it's separated by space
    # or just rely on pint handling the string if we clean it up enough.
    # We can also do a naive word boundary replacement.
    for alias, formal in UNIT_ALIASES.items():
        if s.endswith(alias) and (len(s) == len(alias) or s[-len(alias)-1] == " "):
            s = s[: -len(alias)].strip() + " " + formal
            break
            
    return s


def safe_parse_quantity(raw: str, ureg: Any) -> ParsedQuantity:
    try:
        cleaned = normalize_unit_text(raw)
        q = ureg.Quantity(cleaned)
        return ParsedQuantity(raw=raw, quantity=q, ok=True)
    except Exception as e:
        return ParsedQuantity(raw=raw, quantity=None, ok=False, error=type(e).__name__)
