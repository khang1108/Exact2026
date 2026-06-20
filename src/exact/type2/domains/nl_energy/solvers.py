from __future__ import annotations

import math
from exact.type2.domains.nl_energy.schemas import EnergyExtraction
from exact.type2.domains.nl_energy.function_parser import get_amplitude, evaluate_sinusoidal

def to_si(val: float, unit: str) -> float:
    unit = unit.lower()
    if unit.startswith("m") and unit not in ("m", "m^2", "m^3"): return val * 1e-3
    if unit.startswith("μ") or unit.startswith("u"): return val * 1e-6
    if unit.startswith("n"): return val * 1e-9
    if unit.startswith("p"): return val * 1e-12
    if unit.startswith("k"): return val * 1e3
    return val

def _solve_nl_energy_internal(ext: EnergyExtraction, text: str) -> tuple[float | str, str] | None:
    
    # 1. CONCEPTUAL_UNIT
    if ext.family == "CONCEPTUAL_UNIT":
        return ("J", "") # We return J as the string answer

    # 2. GRAPH_SHAPE (optional, simplified)
    if ext.family == "GRAPH_SHAPE":
        if "versus current" in text or "versus voltage" in text:
            return ("upward parabola", "")
        if "versus capacitance" in text:
            return ("upward straight line", "")

    # Pre-compute SI values
    c = to_si(ext.C.value, ext.C.unit) if ext.C else None
    v = to_si(ext.V.value, ext.V.unit) if ext.V else None
    q = to_si(ext.Q.value, ext.Q.unit) if ext.Q else None
    l = to_si(ext.L.value, ext.L.unit) if ext.L else None
    i = to_si(ext.I.value, ext.I.unit) if ext.I else None
    e = to_si(ext.E.value, ext.E.unit) if ext.E else None
    t = to_si(ext.t.value, ext.t.unit) if ext.t else None
    eps_r = ext.relative_permittivity

    # 3. TIME_DEPENDENT_CAPACITOR_ENERGY
    if ext.family == "TIME_DEPENDENT_CAPACITOR_ENERGY":
        if ext.q_t and c and t is not None:
            val = evaluate_sinusoidal(ext.q_t, t)
            if ext.q_t.symbol == "V":
                ans = 0.5 * c * (val ** 2)
            else:
                ans = (val ** 2) / (2 * c)
            return (ans, "J")
        # Maximum energy
        if ext.q_t and c and any(kw in text for kw in ("maximum", "peak", "greatest", "max")):
            val_max = get_amplitude(ext.q_t)
            if ext.q_t.symbol == "V":
                ans = 0.5 * c * (val_max ** 2)
            else:
                ans = (val_max ** 2) / (2 * c)
            return (ans, "J")

    # 4. TIME_DEPENDENT_INDUCTOR_ENERGY
    if ext.family == "TIME_DEPENDENT_INDUCTOR_ENERGY":
        if ext.I_t and l and t is not None:
            inst_i = evaluate_sinusoidal(ext.I_t, t)
            ans = 0.5 * l * (inst_i ** 2)
            return (ans, "J")
        # Maximum energy
        if ext.I_t and l and any(kw in text for kw in ("maximum", "peak", "greatest", "max")):
            i_max = get_amplitude(ext.I_t)
            ans = 0.5 * l * (i_max ** 2)
            return (ans, "J")

    # 5. LC_CONSERVATION
    if ext.family == "LC_CONSERVATION":
        if ext.E_total and ext.E_electric and "magnetic" in text:
            w_tot = to_si(ext.E_total.value, ext.E_total.unit)
            w_c = to_si(ext.E_electric.value, ext.E_electric.unit)
            return (w_tot - w_c, "J")
        if ext.E_total and ext.E_magnetic and "electric" in text:
            w_tot = to_si(ext.E_total.value, ext.E_total.unit)
            w_l = to_si(ext.E_magnetic.value, ext.E_magnetic.unit)
            return (w_tot - w_l, "J")
        if ext.E_electric and ext.E_magnetic and ("total" in text or "maximum" in text):
            w_c = to_si(ext.E_electric.value, ext.E_electric.unit)
            w_l = to_si(ext.E_magnetic.value, ext.E_magnetic.unit)
            return (w_c + w_l, "J")
            
        if "magnetic" in text and "electric" in text and "maximum" in text:
            if "what is the value" in text or "reaches its maximum" in text:
                return (0.0, "J")

        # Energy ratio questions
        if ext.energy_ratio is not None and ext.energy_ratio_target:
            ratio = ext.energy_ratio
            if ext.energy_ratio_target == "magnetic":
                mag_ratio = ratio
                elec_ratio = 1.0 - ratio
            else:
                elec_ratio = ratio
                mag_ratio = 1.0 - ratio
                
            if "current" in text or "i" in text.split():
                ans = math.sqrt(mag_ratio) * 100.0
                return (ans, "%")
            if "charge" in text or "q" in text.split() or "voltage" in text or "u" in text.split():
                ans = math.sqrt(elec_ratio) * 100.0
                return (ans, "%")

    # 6. CAPACITOR_ENERGY
    if ext.family == "CAPACITOR_ENERGY":
        if c and v and eps_r and "dielectric" in text:
            base_energy = 0.5 * c * (v ** 2)
            if "disconnected" in text:
                return (base_energy / eps_r, "J")
            if "remains connected" in text or "still connected" in text or "connected to the voltage source" in text:
                return (base_energy * eps_r, "J")
        if c and v:
            return (0.5 * c * (v ** 2), "J")
        if c and q:
            return ((q ** 2) / (2 * c), "J")
        if q and v:
            return (0.5 * q * v, "J")
        if c and e and (any(kw in text for kw in ("voltage", "potential difference", "potential across", "u (v)", "u(v)", "across its plates", "across it", "unit: v")) or "u" in text.split()):
            return (math.sqrt(2 * e / c), "V")
        if c and e and ("charge" in text or "q" in text.split()):
            return (math.sqrt(2 * c * e), "C")
        if e and v and ("capacitance" in text or "capacitor" in text or "c" in text.split()):
            return (2 * e / (v ** 2), "F")
        if e and q and ("capacitance" in text or "capacitor" in text or "c" in text.split()):
            return ((q ** 2) / (2 * e), "F")
            
        # disconnected plates separation logic
        if "disconnected" in text and "doubled" in text and e:
            return (2.0 * e, "J")
        if "disconnected" in text and "tripled" in text and e:
            return (3.0 * e, "J")
        if "disconnected" in text and "quadrupled" in text and e:
            return (4.0 * e, "J")
        if "disconnected" in text and "halved" in text and e:
            return (0.5 * e, "J")
            
        if "disconnected" in text and "doubled" in text and "times the initial" in text:
            return (2.0, "")
        if "disconnected" in text and "tripled" in text and "times the initial" in text:
            return (3.0, "")
        if "disconnected" in text and "quadrupled" in text and "times the initial" in text:
            return (4.0, "")
        if "disconnected" in text and "halved" in text and "times the initial" in text:
            return (0.5, "")

    # 7. INDUCTOR_ENERGY
    if ext.family == "INDUCTOR_ENERGY":
        if l and i:
            return (0.5 * l * (i ** 2), "J")
        if e and i:
            return ((2 * e) / (i ** 2), "H")
        if e and l:
            return (math.sqrt((2 * e) / l), "A")
            
        # Proportional changes based on current (E = 0.5 * L * I^2)
        if "current" in text and "halved" in text and e:
            return (0.25 * e, "J")
        if "current" in text and "doubled" in text and e:
            return (4.0 * e, "J")
        if "current" in text and "tripled" in text and e:
            return (9.0 * e, "J")

    return None


def format_scalar_answer(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".")

def solve_nl_energy(ext: EnergyExtraction, question: str) -> tuple[float | str, str] | None:
    text = question.lower()
    ans_tuple = _solve_nl_energy_internal(ext, text)
    if ans_tuple is None:
        return None
        
    ans, unit = ans_tuple
    if not isinstance(ans, float):
        return ans_tuple
        
    # Check for target unit hints in the question
    if unit == "J":
        if "in mj" in text: return (ans * 1e3, "mJ")
        if "in uj" in text or "in μj" in text: return (ans * 1e6, "μJ")
        if "in nj" in text: return (ans * 1e9, "nJ")
    elif unit == "H":
        if "in mh" in text: return (ans * 1e3, "mH")
        if "in uh" in text or "in μh" in text: return (ans * 1e6, "μH")
    elif unit == "F":
        if "in mf" in text: return (ans * 1e3, "mF")
        if "in uf" in text or "in μf" in text: return (ans * 1e6, "μF")
        if "in nf" in text: return (ans * 1e9, "nF")
        if "in pf" in text: return (ans * 1e12, "pF")
    elif unit == "A":
        if "in ma" in text: return (ans * 1e3, "mA")
        if "in ua" in text or "in μa" in text: return (ans * 1e6, "μA")
    elif unit == "V":
        if "in mv" in text: return (ans * 1e3, "mV")
        if "in kv" in text: return (ans * 1e-3, "kV")
    elif unit == "C":
        if "in mc" in text: return (ans * 1e3, "mC")
        if "in uc" in text or "in μc" in text: return (ans * 1e6, "μC")
        if "in nc" in text: return (ans * 1e9, "nC")
        if "in pc" in text: return (ans * 1e12, "pC")
        
    return ans_tuple
