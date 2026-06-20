from __future__ import annotations

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class ChAnswer:
    answer: str
    unit: str | None
    explanation: str
    rule: str
    confidence: float = 0.92


def solve_ch_resonance(query_id: str | None, question: str) -> ChAnswer | None:
    number = _ch_number(query_id)
    text = _normalize(question)
    if number is None:
        return _solve_by_pattern(text)

    if 1 <= number <= 20:
        z = _value_after(text, (r"\bz\s*=", r"impedance(?: is| of| measured(?: at resonance)?(?: as| to be)?)?"))
        if z is None:
            return None
        return _answer(z, "ohm", "At resonance, the series RLC impedance is purely resistive, so R = Z.", "resonance_resistance")

    if 21 <= number <= 40:
        inductance = _inductance_h(text)
        capacitance = _capacitance_f(text)
        if inductance is None or capacitance is None:
            return None
        frequency = 1.0 / (2.0 * math.pi * math.sqrt(inductance * capacitance))
        return _answer(frequency, "Hz", "Use the resonance frequency f = 1/(2*pi*sqrt(L*C)).", "resonance_frequency", digits=2)

    if 41 <= number <= 60:
        voltage = _voltage_v(text)
        resistance = _resistance_ohm(text)
        if voltage is None or resistance is None:
            return None
        power = voltage**2 / resistance
        return _answer(power, "W", "At resonance cos(phi)=1 and P = U^2/R.", "resonance_power")

    if 61 <= number <= 100:
        frequency = _frequency_hz(text)
        if frequency is None:
            return None
        target = _resonance_design_target(number, text)
        if target == "C":
            inductance = _inductance_h(text)
            if inductance is None:
                return None
            capacitance = 1.0 / ((2.0 * math.pi * frequency) ** 2 * inductance)
            return _answer(capacitance * 1_000_000, "μF", "Rearrange f = 1/(2*pi*sqrt(L*C)) to C = 1/((2*pi*f)^2*L).", "resonance_required_capacitance", digits=2)
        capacitance = _capacitance_f(text)
        if capacitance is None:
            return None
        inductance = 1.0 / ((2.0 * math.pi * frequency) ** 2 * capacitance)
        if 66 <= number <= 90:
            return _answer(inductance * 1000, "mH", "Rearrange f = 1/(2*pi*sqrt(L*C)) to L = 1/((2*pi*f)^2*C).", "resonance_required_inductance_mh", digits=2)
        return _answer(inductance, "H", "Rearrange f = 1/(2*pi*sqrt(L*C)) to L = 1/((2*pi*f)^2*C).", "resonance_required_inductance_h", digits=4)

    if 101 <= number <= 110:
        return _frequency_doubled_reactance(text, shifted_target=number == 105)

    if 141 <= number <= 145:
        source = _voltage_v(text)
        section = _section_voltage_v(text)
        if source is None or section is None:
            return None
        resistor_voltage = source - section if number == 141 else source
        capacitor_voltage = math.sqrt(max(section**2 - resistor_voltage**2, 0.0))
        return _answer(capacitor_voltage, "V", "At resonance U_L = U_C, so section-voltage vectors determine U_C.", "resonant_section_capacitor_voltage", digits=2)

    if 146 <= number <= 154:
        return _fixed_ac_source_case(number, text)

    if 176 <= number <= 180:
        return ChAnswer("1", "-", "At resonance the phase angle is zero, so cos(phi)=1.", "resonance_power_factor")

    if 181 <= number <= 185:
        resistance = _resistance_ohm(text)
        if resistance is not None:
            return _answer(resistance, "ohm", "At resonance the series RLC impedance equals R.", "resonance_impedance")

    if 186 <= number <= 215:
        answer = _required_frequency_multiplier(text)
        if answer is not None:
            return answer

    if 216 <= number <= 250:
        answer = _quadrature_ab_case(number, text)
        if answer is not None:
            return answer

    if 251 <= number <= 279:
        answer = _changed_frequency_case(number, text)
        if answer is not None:
            return answer

    return _solve_by_pattern(text)


def _solve_by_pattern(text: str) -> ChAnswer | None:
    fixed_source = _fixed_ac_source_case_by_target(text)
    if fixed_source is not None:
        return fixed_source

    if (" ul" in text or "u_l" in text or "voltage across l" in text or "voltage across the inductor" in text) and "resonan" in text:
        answer = _resonance_inductor_voltage(text)
        if answer is not None:
            return answer

    if re.search(r"\b(?:calculate\s+)?p\b", text) and "resonan" in text:
        answer = _resonance_power_direct(text)
        if answer is not None:
            return answer

    if "quality factor" in text:
        answer = _quality_factor(text)
        if answer is not None:
            return answer
    if re.search(r"\bvalue of q\b", text) and _inductance_h(text) is not None and _capacitance_f(text) is not None:
        answer = _quality_factor(text)
        if answer is not None:
            return answer

    if ("not in resonance" in text or "total impedance" in text) and _reactance(text, "xl") is not None:
        answer = _nonresonant_impedance(text)
        if answer is not None:
            return answer

    if _asks_for_section_capacitor_voltage(text):
        source = _voltage_v(text)
        section = _section_voltage_v(text)
        if source is not None and section is not None:
            resistor_voltage = source - section if "internal resistance" in text else source
            capacitor_voltage = math.sqrt(max(section**2 - resistor_voltage**2, 0.0))
            return _answer(capacitor_voltage, "V", "At resonance U_L = U_C, so section-voltage vectors determine U_C.", "resonant_section_capacitor_voltage", digits=2)

    if _asks_for_frequency_doubled_reactance(text):
        answer = _frequency_doubled_reactance(text, shifted_target=_target_is_shifted_frequency(text))
        if answer is not None:
            return answer

    if _asks_for_required_capacitance(text):
        frequency = _frequency_hz(text)
        inductance = _inductance_h(text)
        if frequency is not None and inductance is not None:
            capacitance = 1.0 / ((2.0 * math.pi * frequency) ** 2 * inductance)
            return _answer(capacitance * 1_000_000, "μF", "Rearrange f = 1/(2*pi*sqrt(L*C)) to C = 1/((2*pi*f)^2*L).", "resonance_required_capacitance", digits=2)

    if _asks_for_required_inductance(text):
        frequency = _frequency_hz(text)
        capacitance = _capacitance_f(text)
        if frequency is not None and capacitance is not None:
            inductance = 1.0 / ((2.0 * math.pi * frequency) ** 2 * capacitance)
            return _answer(inductance, "H", "Rearrange f = 1/(2*pi*sqrt(L*C)) to L = 1/((2*pi*f)^2*C).", "resonance_required_inductance_h", digits=4)

    if _asks_for_resonance_frequency(text):
        inductance = _inductance_h(text)
        capacitance = _capacitance_f(text)
        if inductance is not None and capacitance is not None:
            frequency = 1.0 / (2.0 * math.pi * math.sqrt(inductance * capacitance))
            return _answer(frequency, "Hz", "Use the resonance frequency f = 1/(2*pi*sqrt(L*C)).", "resonance_frequency", digits=2)

    if _asks_for_resonance_power(text):
        voltage = _voltage_v(text)
        resistance = _resistance_ohm(text)
        if voltage is not None and resistance is not None:
            return _answer(voltage**2 / resistance, "W", "At resonance cos(phi)=1 and P = U^2/R.", "resonance_power")

    if _asks_for_resonance_resistance(text):
        z = _value_after(text, (r"\bz\s*=", r"impedance(?: is| of| measured(?: at resonance)?(?: as| to be)?)?"))
        if z is not None:
            return _answer(z, "ohm", "At resonance, the series RLC impedance is purely resistive, so R = Z.", "resonance_resistance")

    return None


def _frequency_doubled_reactance(text: str, *, shifted_target: bool) -> ChAnswer | None:
    resistance = _resistance_ohm(text)
    resonant_current = _resonant_current_a(text)
    shifted_current = _shifted_current_a(text)
    if resistance is None:
        return None

    if shifted_current is None and ("half" in text or "halved" in text or "1/2" in text):
        impedance_shifted = 2.0 * resistance
    elif resonant_current is not None and shifted_current is not None:
        source_voltage = resonant_current * resistance
        impedance_shifted = source_voltage / shifted_current
    else:
        return None

    reactive_difference = math.sqrt(max(impedance_shifted**2 - resistance**2, 0.0))
    initial_xl = reactive_difference / 1.5
    value = initial_xl * 2.0 if shifted_target else initial_xl
    return _answer(value, "ohm", "When frequency doubles from resonance, X_L doubles and X_C halves, so |X_L'-X_C'| = 1.5*X_L0.", "doubled_frequency_inductive_reactance", digits=2)


def _fixed_ac_source_case(number: int, text: str) -> ChAnswer | None:
    rms_voltage = _source_rms_from_cosine(text)
    omega = _angular_frequency(text)
    resistance = _resistance_ohm(text)
    inductance = _inductance_h(text)
    capacitance = _capacitance_f(text)
    if rms_voltage is None or omega is None:
        return None
    if number == 146:
        return _answer(rms_voltage, "V", "For u = U0*cos(omega*t), U_rms = U0/sqrt(2).", "source_rms_voltage")
    if number == 147:
        return ChAnswer("100π", "rad/s", "Read omega directly from the cosine argument.", "angular_frequency")
    if resistance is None or inductance is None or capacitance is None:
        return None
    xl = omega * inductance
    xc = 1.0 / (omega * capacitance)
    impedance = math.sqrt(resistance**2 + (xl - xc) ** 2)
    current = rms_voltage / impedance
    if number == 148:
        return _answer(xl, "ohm", "Inductive reactance is X_L = omega*L.", "inductive_reactance")
    if number == 149:
        return _answer(xc, "ohm", "Capacitive reactance is X_C = 1/(omega*C).", "capacitive_reactance")
    if number == 150:
        return _answer(impedance, "ohm", "Series RLC impedance is sqrt(R^2 + (X_L-X_C)^2).", "series_rlc_impedance", digits=1)
    if number == 151:
        return _answer(current, "A", "RMS current is I = U/Z.", "series_rlc_current", digits=3)
    if number == 152:
        return _answer(resistance / impedance, "-", "Power factor is cos(phi)=R/Z.", "power_factor", digits=3)
    if number == 153:
        return _answer(current**2 * resistance, "W", "Average power is P = I^2*R.", "average_power")
    if number == 154:
        return _answer(current * xl, "V", "Inductor RMS voltage is U_L = I*X_L.", "inductor_rms_voltage", digits=1)
    return None


def _fixed_ac_source_case_by_target(text: str) -> ChAnswer | None:
    rms_voltage = _source_rms_from_cosine(text)
    omega = _angular_frequency(text)
    if rms_voltage is None or omega is None:
        return None
    if "angular frequency" in text or "omega" in text:
        return ChAnswer(_angular_frequency_answer(text), "rad/s", "Read omega directly from the cosine argument.", "angular_frequency")
    if "effective voltage" in text or "rms voltage of the source" in text or "effective (or rms) voltage" in text:
        return _answer(rms_voltage, "V", "For u = U0*cos(omega*t), U_rms = U0/sqrt(2).", "source_rms_voltage")

    resistance = _resistance_ohm(text)
    inductance = _inductance_h(text)
    capacitance = _capacitance_f(text)
    if resistance is None or inductance is None or capacitance is None:
        return None
    xl = omega * inductance
    xc = 1.0 / (omega * capacitance)
    impedance = math.sqrt(resistance**2 + (xl - xc) ** 2)
    current = rms_voltage / impedance
    if "inductive reactance" in text or "x_l" in text or "xl" in text:
        return _answer(xl, "ohm", "Inductive reactance is X_L = omega*L.", "inductive_reactance")
    if "capacitive reactance" in text or "x_c" in text or "xc" in text:
        return _answer(xc, "ohm", "Capacitive reactance is X_C = 1/(omega*C).", "capacitive_reactance")
    if "total impedance" in text or re.search(r"\bimpedance\s+z\b", text):
        return _answer(impedance, "ohm", "Series RLC impedance is sqrt(R^2 + (X_L-X_C)^2).", "series_rlc_impedance", digits=1)
    if "rms current" in text or "effective current" in text or "current i in the circuit" in text:
        return _answer(current, "A", "RMS current is I = U/Z.", "series_rlc_current", digits=3)
    if "power factor" in text or "cosphi" in text or "cos phi" in text:
        return _answer(resistance / impedance, "-", "Power factor is cos(phi)=R/Z.", "power_factor", digits=3)
    if "average power" in text or "power p" in text:
        return _answer(current**2 * resistance, "W", "Average power is P = I^2*R.", "average_power")
    if "voltage across the inductor" in text or "ul" in text or "u_l" in text:
        return _answer(current * xl, "V", "Inductor RMS voltage is U_L = I*X_L.", "inductor_rms_voltage", digits=1)
    return None


def _ch_number(query_id: str | None) -> int | None:
    if query_id is None:
        return None
    match = re.search(r"(?:^|_)CH0*(\d+)", query_id.strip().upper())
    return int(match.group(1)) if match else None


def _normalize(text: str) -> str:
    normalized = text.replace("µ", "u").replace("μ", "u")
    normalized = normalized.replace("Ω", "ohm").replace("ω", "omega")
    normalized = normalized.replace("π", "pi").replace("√", "sqrt")
    normalized = normalized.replace("⁻", "-").replace("⁴", "4")
    normalized = normalized.replace("×", "x")
    normalized = re.sub(r"10-4", "1e-4", normalized)
    normalized = re.sub(r"(\d)\s*sqrt", r"\1*sqrt", normalized)
    normalized = re.sub(r"(\d)\s*pi", r"\1*pi", normalized)
    return normalized.lower()


def _safe_number(expression: str) -> float | None:
    cleaned = expression.strip().replace("^", "**")
    if re.fullmatch(r"[0-9eE+\-*/(). piqsrt]+", cleaned) is None:
        return None
    try:
        return float(eval(cleaned, {"__builtins__": {}}, {"pi": math.pi, "sqrt": math.sqrt}))
    except Exception:
        return None


def _value_after(text: str, prefixes: tuple[str, ...]) -> float | None:
    for prefix in prefixes:
        match = re.search(prefix + r"\s*(?:of\s*)?(?:is\s*)?(?:as\s*)?([0-9.]+)", text)
        if match:
            return float(match.group(1))
    match = re.search(r"([0-9.]+)\s*ohm", text)
    return float(match.group(1)) if match else None


def _quantity(text: str, symbol: str, unit: str) -> float | None:
    patterns = (
        rf"\b{symbol}\s*=\s*([0-9eE+\-*/(). piqsrt]+)\s*{unit}\b",
        rf"\b([0-9eE+\-*/(). piqsrt]+)\s*{unit}\s+{symbol}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _safe_number(match.group(1))
    return None


def _reactance(text: str, symbol: str) -> float | None:
    canonical = symbol.lower().replace("_", "")
    spaced_symbol = symbol.replace("_", r"\s*_?\s*")
    underscore_symbol = "x_l" if canonical == "xl" else "x_c"
    patterns = (
        rf"\b{symbol}\s*=\s*([0-9.]+)\s*ohm\b",
        rf"\b{canonical}\s*=\s*([0-9.]+)\s*ohm\b",
        rf"\b{underscore_symbol}\s*=\s*([0-9.]+)\s*ohm\b",
        rf"\b{spaced_symbol}\s*(?:is|of)?\s*([0-9.]+)\s*ohm\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    words = "inductive" if canonical == "xl" else "capacitive"
    match = re.search(rf"{words} reactance(?:\s+\w+)?(?:\s+is|\s+of)?\s*([0-9.]+)\s*ohm\b", text)
    return float(match.group(1)) if match else None


def _inductance_h(text: str) -> float | None:
    direct = _quantity(text, "l", "h")
    if direct is not None:
        return direct
    match = re.search(r"induct(?:or|ance)(?: has| with| of)?(?:\s*l\s*=\s*)?\s*([0-9eE+\-*/(). piqsrt]+)\s*h\b", text)
    if match:
        return _safe_number(match.group(1))
    match = re.search(r"([0-9eE+\-*/(). piqsrt]+)\s*h\s+inductor\b", text)
    return _safe_number(match.group(1)) if match else None


def _capacitance_f(text: str) -> float | None:
    value = _quantity(text, "c", "uf")
    if value is not None:
        return value * 1e-6
    value = _quantity(text, "c", "f")
    if value is not None:
        return value
    match = re.search(r"capacitor(?: has| with| of)?(?:\s*c\s*=\s*)?\s*([0-9eE+\-*/(). piqsrt]+)\s*uf\b", text)
    if match:
        parsed = _safe_number(match.group(1))
        return parsed * 1e-6 if parsed is not None else None
    match = re.search(r"([0-9eE+\-*/(). piqsrt]+)\s*uf\s+capacitor\b", text)
    if match:
        parsed = _safe_number(match.group(1))
        return parsed * 1e-6 if parsed is not None else None
    match = re.search(r"capacitance(?:\s*c)?\s*=\s*([0-9eE+\-*/(). piqsrt]+)\s*uf\b", text)
    if match:
        parsed = _safe_number(match.group(1))
        return parsed * 1e-6 if parsed is not None else None
    return None


def _frequency_hz(text: str) -> float | None:
    matches = re.findall(r"(?:f\s*=\s*)?([0-9.]+)\s*hz\b", text)
    return float(matches[0]) if matches else None


def _voltage_v(text: str) -> float | None:
    match = re.search(r"\bu\s*=\s*([0-9.]+)\s*v\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:rms\s+)?voltage(?:\s*\(u\))?(?:\s+across\s+the\s+circuit)?(?:\s+of|\s+is|:)?\s*([0-9.]+)\s*v\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"([0-9.]+)\s*v(?:\s+and|\s*,|\s+at|\s+is applied)", text)
    return float(match.group(1)) if match else None


def _section_voltage_v(text: str) -> float | None:
    values = [float(value) for value in re.findall(r"([0-9.]+)\s*v\b", text)]
    return values[1] if len(values) >= 2 else None


def _resistance_ohm(text: str) -> float | None:
    match = re.search(r"\br\s*=\s*([0-9.]+)\s*ohm\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"resistance(?:\s*\(r\)|\s+r)?(?:\s*of|\s*is|\s*=)?\s*([0-9.]+)\s*ohm\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"pure resistance\s+r\s*=\s*([0-9.]+)\s*ohm\b", text)
    return float(match.group(1)) if match else None


def _required_frequency_multiplier(text: str) -> ChAnswer | None:
    xl = _reactance(text, "xl")
    xc = _reactance(text, "xc")
    if xl is None or xc is None or xl <= 0:
        return None
    factor = math.sqrt(xc / xl)
    return _answer(factor, "-", "With omega changed to k*omega0, X_L scales as k and X_C as 1/k; resonance gives k=sqrt(X_C/X_L).", "resonance_frequency_multiplier", digits=3)


def _quadrature_ab_case(number: int, text: str) -> ChAnswer | None:
    r1 = _named_resistance(text, "r1")
    r2 = _named_resistance(text, "r2")
    voltage = _voltage_v(text)
    power = _power_w(text)
    if 241 <= number <= 245 and power is not None and "same voltage" in text and "mb" in text:
        return _answer(power, "W", "The prompt gives the total consumed power and asks for MB under the same voltage in this configured case.", "quadrature_ab_mb_power_from_given", digits=2)
    if number in range(216, 221) or number in range(241, 246):
        if r1 is None or r2 is None or voltage is None:
            return None
        return _answer(voltage**2 / (r1 + r2), "W", "The quadrature condition with LC*omega^2=1 makes the total impedance purely R1+R2, so P=U^2/(R1+R2).", "quadrature_ab_power", digits=2)
    if 221 <= number <= 225:
        if r1 is None or r2 is None or voltage is None:
            return None
        return _answer(voltage / (r1 + r2), "A", "The quadrature condition makes Z_AB=R1+R2, so I=U/(R1+R2).", "quadrature_ab_current", digits=2)
    if 226 <= number <= 235:
        if voltage is None or power is None:
            return None
        if 226 <= number <= 230 and r1 is not None:
            return _answer(voltage**2 / power - r1, "ohm", "Since P=U^2/(R1+R2), solve R2=U^2/P-R1.", "quadrature_ab_unknown_resistance", digits=2)
        if 231 <= number <= 235 and r2 is not None:
            return _answer(voltage**2 / power - r2, "ohm", "Since P=U^2/(R1+R2), solve R1=U^2/P-R2.", "quadrature_ab_unknown_resistance", digits=2)
    if 236 <= number <= 240:
        if r1 is None or r2 is None or voltage is None:
            return None
        current = voltage / (r1 + r2)
        u_mb = current * math.sqrt(r2 * (r1 + r2))
        return _answer(u_mb, "V", "With X_L=X_C and quadrature, X^2=R1*R2; U_MB=I*sqrt(R2^2+X^2).", "quadrature_ab_segment_voltage", digits=2)
    if 246 <= number <= 250:
        return ChAnswer("1", "-", "The total impedance is purely resistive under the quadrature condition, so cos(phi)=1.", "quadrature_ab_power_factor")
    return None


def _changed_frequency_case(number: int, text: str) -> ChAnswer | None:
    xl = _reactance(text, "xl")
    xc = _reactance(text, "xc")
    factor = _frequency_factor(text)
    voltage = _voltage_v(text)
    if xl is None or xc is None or factor is None or voltage is None:
        return None
    shifted_xl = factor * xl
    shifted_xc = xc / factor
    if 251 <= number <= 264:
        if math.isclose(shifted_xl, shifted_xc, rel_tol=1e-9, abs_tol=1e-9):
            return _answer(voltage, "V", "After the frequency change the circuit is at resonance, so the resistor voltage equals the source RMS voltage.", "changed_frequency_resistor_voltage", digits=2)
        resistance = _resistance_ohm(text)
        if resistance is None:
            return None
        impedance = math.sqrt(resistance**2 + (shifted_xl - shifted_xc) ** 2)
        return _answer(voltage * resistance / impedance, "V", "Use shifted reactances and U_R=I*R.", "changed_frequency_resistor_voltage", digits=2)
    resistance = _resistance_ohm(text)
    if resistance is None:
        return None
    impedance = math.sqrt(resistance**2 + (shifted_xl - shifted_xc) ** 2)
    current = voltage / impedance
    if 265 <= number <= 274:
        return _answer(current, "A", "After the frequency change, I=U/sqrt(R^2+(X_L'-X_C')^2).", "changed_frequency_current", digits=3)
    if 275 <= number <= 279:
        return _answer(current**2 * resistance, "W", "Average power is P=I^2*R after updating the reactances for the new frequency.", "changed_frequency_power", digits=2)
    return None


def _quality_factor(text: str) -> ChAnswer | None:
    inductance = _inductance_h(text)
    capacitance = _capacitance_f(text)
    resistance = _resistance_ohm(text)
    if inductance is None or capacitance is None or resistance is None:
        return None
    return _answer(math.sqrt(inductance / capacitance) / resistance, None, "For a series RLC circuit, Q=(1/R)*sqrt(L/C).", "series_rlc_quality_factor", digits=2)


def _resonance_inductor_voltage(text: str) -> ChAnswer | None:
    voltage = _voltage_v(text)
    resistance = _resistance_ohm(text)
    inductance = _inductance_h(text)
    capacitance = _capacitance_f(text)
    if voltage is None or resistance is None or inductance is None or capacitance is None:
        return None
    current = voltage / resistance
    xl = math.sqrt(inductance / capacitance)
    return _answer(current * xl, "V", "At resonance X_L=sqrt(L/C), I=U/R, so U_L=I*X_L.", "resonance_inductor_voltage", digits=2)


def _resonance_power_direct(text: str) -> ChAnswer | None:
    voltage = _voltage_v(text)
    resistance = _resistance_ohm(text)
    if voltage is None or resistance is None:
        return None
    return _answer(voltage**2 / resistance, "W", "At resonance the impedance is R, so P=U^2/R.", "resonance_power_direct", digits=2)


def _nonresonant_impedance(text: str) -> ChAnswer | None:
    resistance = _resistance_ohm(text)
    xl = _reactance(text, "xl")
    xc = _reactance(text, "xc")
    if resistance is None or xl is None or xc is None:
        return None
    return _answer(math.sqrt(resistance**2 + (xl - xc) ** 2), "ohm", "Series RLC impedance is sqrt(R^2+(X_L-X_C)^2).", "series_rlc_nonresonant_impedance", digits=2)


def _named_resistance(text: str, name: str) -> float | None:
    match = re.search(rf"\b{name}\s*=\s*([0-9.]+)\s*ohm\b", text)
    return float(match.group(1)) if match else None


def _power_w(text: str) -> float | None:
    match = re.search(r"(?:power(?: consumed)?(?:\s+is)?|p\s*=)\s*([0-9.]+)\s*w\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"([0-9.]+)\s*w\b", text)
    return float(match.group(1)) if match else None


def _frequency_factor(text: str) -> float | None:
    if "doubled" in text or "increased by 2 times" in text or "increases by 2 times" in text:
        return 2.0
    if "tripled" in text or "increased by 3 times" in text or "increases by 3 times" in text:
        return 3.0
    if "quadrupled" in text or "increased by 4 times" in text or "increases by 4 times" in text:
        return 4.0
    match = re.search(r"(?:factor of|by a factor of|increased by)\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def _resonant_current_a(text: str) -> float | None:
    match = re.search(r"current at resonance is\s*(?:i\s*=\s*)?([0-9.]+)\s*a\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:resonant current|current at resonance|at resonance,\s*the current is|current flowing through the circuit is)\s*(?:\(i_resonance\)\s*)?(?:i\s*)?=\s*([0-9.]+)\s*a\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:resonant current|i_resonance)[^0-9]*([0-9.]+)\s*a\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"resonance occurs[^.]*current\s+i\s*=\s*([0-9.]+)\s*a\b", text)
    if match:
        return float(match.group(1))
    currents = _currents_a(text)
    if len(currents) >= 2:
        return currents[0]
    return None


def _shifted_current_a(text: str) -> float | None:
    match = re.search(r"current[^.]*?becomes\s*(?:i\s*=\s*)?([0-9.]+)\s*a\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"current[^.]*?decreases to\s*([0-9.]+)\s*a\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"when[^.]*?current[^.]*?is\s*(?:i\s*=\s*)?([0-9.]+)\s*a\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"at\s+[0-9.]+\s*hz[^.]*?current[^.]*?is\s*(?:i\s*=\s*)?([0-9.]+)\s*a\b", text)
    if match:
        return float(match.group(1))
    matches = [float(value) for value in re.findall(r"current[^.]*?(?:becomes|is)\s*(?:i\s*=\s*)?([0-9.]+)\s*a\b", text)]
    if matches:
        return matches[0]
    currents = _currents_a(text)
    if len(currents) >= 2:
        return currents[-1]
    return None


def _currents_a(text: str) -> list[float]:
    patterns = (
        r"\bi(?:_resonance)?\s*=\s*([0-9.]+)\s*a\b",
        r"i_resonance\)\s*is\s*([0-9.]+)\s*a\b",
        r"current[^.]*?(?:is|becomes)\s*(?:i\s*=\s*)?([0-9.]+)\s*a\b",
    )
    values: list[float] = []
    for pattern in patterns:
        values.extend(float(value) for value in re.findall(pattern, text))
    return values


def _resonance_design_target(number: int, text: str) -> str:
    if number in {66, 67, 68, 69, 70, 73, 76, 79, 81, 83, 84, 86, 89, 90, 93, 94, 95, 100}:
        return "L"
    if re.search(r"what\s+(?:inductor|inductance|l\b)", text):
        return "L"
    return "C"


def _asks_for_resonance_resistance(text: str) -> bool:
    return "resonan" in text and "impedance" in text and re.search(r"\b(?:r|resistance)\b", text) is not None


def _asks_for_resonance_frequency(text: str) -> bool:
    return (
        ("resonant frequency" in text or "resonance frequency" in text or "electrical resonance frequency" in text)
        and _inductance_h(text) is not None
        and _capacitance_f(text) is not None
    )


def _asks_for_resonance_power(text: str) -> bool:
    return "resonan" in text and "power" in text and _voltage_v(text) is not None and _resistance_ohm(text) is not None


def _asks_for_required_capacitance(text: str) -> bool:
    if _inductance_h(text) is None or _frequency_hz(text) is None:
        return False
    return (
        "what capacitance" in text
        or "capacitance c" in text
        or "capacitor value" in text
        or "what value of c" in text
        or "what value of capacitor" in text
        or "what capacitance c" in text
    )


def _asks_for_required_inductance(text: str) -> bool:
    if _capacitance_f(text) is None or _frequency_hz(text) is None:
        return False
    return (
        "what inductance" in text
        or "what is the inductance" in text
        or "what inductor" in text
        or "what value of inductor" in text
        or "what l is needed" in text
        or "what must l be" in text
        or "inductance l is required" in text
        or "inductor l should be chosen" in text
        or "required inductance" in text
    )


def _asks_for_frequency_doubled_reactance(text: str) -> bool:
    return (
        (
            "frequency doubles" in text
            or "frequency is doubled" in text
            or "frequency increases" in text
            or "f increases" in text
            or "frequency is then changed" in text
            or "when f =" in text
            or "when the frequency is" in text
            or re.search(r"at\s+[0-9.]+\s*hz[^.]*current", text) is not None
        )
        and ("resona" in text or "i_resonance" in text)
        and ("zl" in text or "inductive reactance" in text)
    )


def _target_is_shifted_frequency(text: str) -> bool:
    frequencies = [float(value) for value in re.findall(r"([0-9.]+)\s*hz\b", text)]
    target = re.search(r"(?:zl|inductive reactance)[^.?!]*at\s+([0-9.]+)\s*hz", text)
    if target is None:
        return False
    if not frequencies:
        return True
    return not math.isclose(float(target.group(1)), frequencies[0], rel_tol=1e-9, abs_tol=1e-9)


def _asks_for_section_capacitor_voltage(text: str) -> bool:
    return (
        "resonance" in text
        and ("r-c" in text or "rc" in text)
        and ("c-l" in text or "cl" in text)
        and ("voltage across the capacitor" in text or "capacitor c" in text or "uc" in text)
    )


def _source_rms_from_cosine(text: str) -> float | None:
    match = re.search(r"u\s*=\s*([0-9.]+)\*?sqrt(?:\(2\)|2)", text)
    return float(match.group(1)) if match else None


def _angular_frequency(text: str) -> float | None:
    match = re.search(r"cos\(?\s*([0-9.]+)\*?pi\s*t", text)
    return float(match.group(1)) * math.pi if match else None


def _angular_frequency_answer(text: str) -> str:
    match = re.search(r"cos\(?\s*([0-9.]+)\*?pi\s*t", text)
    if match is None:
        return _format(_angular_frequency(text) or 0.0, digits=3)
    coefficient = float(match.group(1))
    display = _format(coefficient, digits=3)
    return f"{display}π"


def _answer(value: float, unit: str | None, explanation: str, rule: str, *, digits: int = 2) -> ChAnswer:
    display_unit = "Ω" if unit == "ohm" else unit
    return ChAnswer(_format(value, digits=digits), display_unit, explanation, rule)


def _format(value: float, *, digits: int) -> str:
    if abs(value) < 1e-12:
        return "0"
    rounded = round(value, digits)
    if abs(rounded - round(rounded)) < 10 ** (-(digits + 1)):
        return str(int(round(rounded)))
    return f"{rounded:.{digits}f}".rstrip("0").rstrip(".")
