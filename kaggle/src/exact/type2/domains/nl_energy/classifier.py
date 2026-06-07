from __future__ import annotations

import re

def classify_nl_energy_family(question: str) -> str:
    """Classify an NL energy question into a specific mathematical or conceptual family."""
    text = question.lower()

    if "si unit" in text or "unit of" in text:
        return "CONCEPTUAL_UNIT"

    if "shape of the graph" in text or "shape of graph" in text or "versus current" in text or "versus capacitance" in text or "versus voltage" in text:
        return "GRAPH_SHAPE"

    if "lc circuit" in text or "oscillation circuit" in text or "lc oscillation" in text or "ideal lc" in text:
        return "LC_CONSERVATION"

    # Check for time-dependent patterns like I(t), q(t), u(t)
    has_time_func = bool(re.search(r"\b[iqvu]\(t\)", text) or re.search(r"(sin|cos)\s*\(", text))
    
    has_cap = "capacitor" in text or "capacitance" in text
    has_ind = "inductor" in text or "inductance" in text
    
    if has_time_func:
        if has_cap and not has_ind:
            return "TIME_DEPENDENT_CAPACITOR_ENERGY"
        if has_ind and not has_cap:
            return "TIME_DEPENDENT_INDUCTOR_ENERGY"
            
        # Fallbacks for missing explicit capacitor/inductor components
        if any(kw in text for kw in ("u(t)", "v(t)", "q(t)", "voltage", "charge", "electric")):
            return "TIME_DEPENDENT_CAPACITOR_ENERGY"
        if any(kw in text for kw in ("i(t)", "current", "coil", "solenoid", "magnetic")):
            return "TIME_DEPENDENT_INDUCTOR_ENERGY"

    if has_cap and not has_ind:
        return "CAPACITOR_ENERGY"
        
    if has_ind and not has_cap:
        return "INDUCTOR_ENERGY"
        
    if "electric field energy" in text and "magnetic field energy" in text:
        return "LC_CONSERVATION"

    return "UNKNOWN"
