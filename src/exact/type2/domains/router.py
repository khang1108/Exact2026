from __future__ import annotations

import re

def route_domain(question_id: str | None, question_text: str) -> str:
    """Route a Type 2 question to a specific domain pipeline based on ID or content."""
    
    # Prefix routing
    if question_id:
        if "LD" in question_id or "DT" in question_id:
            return "LD"
        if "TD" in question_id:
            return "TD"
        if "NL" in question_id:
            return "NL_ENERGY"

    # Conservative keyword fallback for NL_ENERGY if no ID is present
    text = question_text.lower()
    
    nl_keywords = [
        "capacitor energy",
        "electric field energy",
        "magnetic field energy",
        "inductor energy",
        "lc circuit",
        "oscillation",
        "maximum current",
        "maximum charge",
        "si unit of energy",
        "energy versus current",
        "energy versus capacitance",
    ]
    
    for kw in nl_keywords:
        if kw in text:
            return "NL_ENERGY"
            
    # Regex fallback for I(t), q(t), U(t) etc with sin/cos
    if re.search(r"\b[iqvu]\(t\).*?(sin|cos)", text):
        return "NL_ENERGY"

    return "GENERIC"
