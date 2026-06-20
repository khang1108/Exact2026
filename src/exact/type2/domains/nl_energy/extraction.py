from __future__ import annotations

import re
from exact.type2.domains.nl_energy.schemas import EnergyExtraction, ScalarQuantity, FunctionQuantity
from exact.config import Settings
from exact.type2.extraction.llm_structured import parse_with_llm

def extract_nl_energy_quantities(question: str) -> EnergyExtraction:
    """Extract scalars and functions from NL energy questions using regex."""
    text = question.lower()
    text = text.replace("µ", "μ") # unify micro sign to greek mu
    
    # Normalize superscripts and scientific notation globally
    superscripts = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}
    for sup, digit in superscripts.items():
        text = text.replace(sup, digit)
    text = re.sub(r"\s*[\*x×]\s*10\^?", "e", text)
    
    # Initialize extraction
    ext = EnergyExtraction(family="UNKNOWN", target="energy") # family/target will be set by pipeline
    
    # 1. Extract FunctionQuantities (e.g. U(t) = 120sin(2000t))
    # match pattern like: u(t) = 120sin(2000t) or i(t) = 2sin(100pit)
    # look for u, i, q
    
    func_pattern = r"\b([uiqv]|voltage|current|charge)\s*(?:\(t\))?\s*(?:=|is given by|is)\s*(.*?(?:sin|cos)\(.*?\))"
    for match in re.finditer(func_pattern, text):
        symbol = match.group(1).upper()
        expr = match.group(2).replace(" ", "")
        
        # map u, v, voltage -> V, i, current -> I, q, charge -> Q
        if symbol in ("U", "V", "VOLTAGE"):
            symbol = "V"
        elif symbol in ("I", "CURRENT"):
            symbol = "I"
        elif symbol in ("Q", "CHARGE"):
            symbol = "Q"
        
        unit = "V" if symbol == "V" else ("A" if symbol == "I" else "C")
        
        fq = FunctionQuantity(symbol=symbol, expression=expr, output_unit=unit)
        if symbol == "I":
            ext.I_t = fq
        elif symbol == "Q":
            ext.q_t = fq
        elif symbol == "V":
            # For schema simplicity, we might just map V(t) to an extended field, 
            # or treat q_t as the generic function field.
            # Let's map it to q_t for now or add V_t conceptually.
            # We don't have V_t in schema, but we can store it in q_t and check symbol
            ext.q_t = fq
            ext.q_t.symbol = "V"

    # 2. Extract Scalar Quantities
    # e.g. 10 \mu F, 200 V, 0.5 H, 1.8 A, 1 ms
    
    # Capacitance
    c_match = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(μf|uf|nf|pf|mf|f)\b", text)
    if c_match:
        ext.C = ScalarQuantity(name="C", value=float(c_match.group(1)), unit=c_match.group(2))
        
    # Inductance
    l_match = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(μh|uh|mh|h)\b", text)
    if l_match:
        ext.L = ScalarQuantity(name="L", value=float(l_match.group(1)), unit=l_match.group(2))
        
    # Voltage
    v_match = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(kv|v)\b", text)
    if v_match:
        ext.V = ScalarQuantity(name="V", value=float(v_match.group(1)), unit=v_match.group(2))
        
    # Current
    i_match = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(ma|a)\b", text)
    if i_match:
        ext.I = ScalarQuantity(name="I", value=float(i_match.group(1)), unit=i_match.group(2))
        
    # Charge
    q_match = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(μc|uc|mc|c)\b", text)
    if q_match:
        ext.Q = ScalarQuantity(name="Q", value=float(q_match.group(1)), unit=q_match.group(2))
        
    # Energy
    for m in re.finditer(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(μj|uj|mj|j)\b", text):
        val = float(m.group(1))
        unit = m.group(2)
        sq = ScalarQuantity(name="E", value=val, unit=unit)
        
        window = text[max(0, m.start()-35):m.start()]
        if "total" in window or "circuit" in window or "maximum" in window:
            if not ext.E_total: ext.E_total = sq
        elif "electric" in window or "capacitor" in window:
            if not ext.E_electric: ext.E_electric = sq
        elif "magnetic" in window or "inductor" in window:
            if not ext.E_magnetic: ext.E_magnetic = sq
        else:
            if not ext.E: ext.E = sq
        
    # Time
    t_match = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(ms|s)\b", text)
    if t_match:
        ext.t = ScalarQuantity(name="t", value=float(t_match.group(1)), unit=t_match.group(2))

    eps_match = re.search(
        r"(?:relative\s+permittivity|dielectric\s+constant)"
        r"(?:\s*\([^)]*\))?"
        r"(?:\s+of)?"
        r"(?:\s*(?:ε|epsilon|eps)(?:_?r)?)?"
        r"\s*(?:=|is|of)?\s*"
        r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        text,
    )
    if eps_match:
        ext.relative_permittivity = float(eps_match.group(1))
        
    # Energy Ratio
    if any(kw in text for kw in ("electric energy equals magnetic energy", "equal to magnetic energy", "equal to the magnetic energy", "energies are equal", "electric and magnetic energy are equal", "magnetic energy equals electric energy")):
        ext.energy_ratio = 0.5
        ext.energy_ratio_target = "electric"
        
    ratio_match = re.search(r"(magnetic|electric)\s+energy.*?(\d+(?:\.\d+)?|\d+/\d+)\s*(%|times)?\s*(?:of)?\s*(?:the)?\s*total\s*(?:energy)?", text)
    if ratio_match and ext.energy_ratio is None:
        target = ratio_match.group(1)
        val_str = ratio_match.group(2)
        if "/" in val_str:
            num, den = val_str.split("/")
            val = float(num) / float(den)
        else:
            val = float(val_str)
            
        if ratio_match.group(3) == "%" or val > 1.0: # if it's "75", it's 75%. If "0.75", it's 0.75.
            val = val / 100.0 if val > 1.0 else val
        ext.energy_ratio = val
        ext.energy_ratio_target = target
        
    return ext


def run_llm_extraction_repair(question: str, ext: EnergyExtraction, settings: Settings | None = None) -> EnergyExtraction:
    """Uses the LLM to extract any missing scalar parameters if deterministic solver failed."""
    spec = parse_with_llm(question, settings=settings)
    if not spec:
        return ext
        
    # Map LLM spec quantities to ext
    for q in spec.quantities:
        name = q.name.lower()
        val = q.value
        unit = q.unit.lower()
        
        if "capac" in name and not ext.C:
            ext.C = ScalarQuantity(name="C", value=val, unit=unit)
        elif ("volt" in name or "potential" in name) and not ext.V:
            ext.V = ScalarQuantity(name="V", value=val, unit=unit)
        elif "induct" in name and not ext.L:
            ext.L = ScalarQuantity(name="L", value=val, unit=unit)
        elif "current" in name and not ext.I:
            ext.I = ScalarQuantity(name="I", value=val, unit=unit)
        elif "charge" in name and not ext.Q:
            ext.Q = ScalarQuantity(name="Q", value=val, unit=unit)
        elif "energy" in name:
            sq = ScalarQuantity(name="E", value=val, unit=unit)
            if "total" in name and not ext.E_total:
                ext.E_total = sq
            elif "electric" in name and not ext.E_electric:
                ext.E_electric = sq
            elif "magnetic" in name and not ext.E_magnetic:
                ext.E_magnetic = sq
            elif not ext.E:
                ext.E = sq
        elif "time" in name and not ext.t:
            ext.t = ScalarQuantity(name="t", value=val, unit=unit)
        elif ("permittivity" in name or "dielectric" in name) and ext.relative_permittivity is None:
            ext.relative_permittivity = val
            
    return ext
