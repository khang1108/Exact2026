from __future__ import annotations

import math

from exact.type2.domains.thcb.schemas import ThcbAnswer, ThcbContract, ThcbQuantity


def solve_thcb_contract(contract: ThcbContract) -> ThcbAnswer | None:
    if contract.family in {"MEASUREMENT_ERROR", "ERROR_PROPAGATION"}:
        return _solve_measurement(contract)
    if contract.family in {"PARALLEL_CIRCUIT", "SIMPLE_CIRCUIT"}:
        return _solve_circuit(contract)
    if contract.family == "CONCEPTUAL_CIRCUIT" or contract.target == "conceptual":
        return _solve_conceptual(contract)
    return None


def _solve_measurement(contract: ThcbContract) -> ThcbAnswer | None:
    q = contract.quantities
    measured = _first(q, "measured", "voltage", "current", "resistance")
    actual = q.get("actual")
    abs_err = q.get("absolute_error") or q.get("least_count")

    if contract.family == "ERROR_PROPAGATION":
        propagated = _solve_error_propagation(contract)
        if propagated is not None:
            return propagated

    if contract.readings and ("mean" in contract.requested_outputs or "mean_absolute_error" in contract.requested_outputs):
        mean = sum(item.value for item in contract.readings) / len(contract.readings)
        mad = sum(abs(item.value - mean) for item in contract.readings) / len(contract.readings)
        unit = contract.readings[0].unit if contract.readings else None
        if contract.target == "random_error":
            return _answer(mad, unit, "Computed random error as mean absolute deviation.")
        return _multi([mean, mad], f"{unit}; {unit}" if unit else None, "Computed mean value and mean absolute error.")

    if actual is not None and measured is not None:
        err = abs(measured.value - actual.value)
        if contract.target == "absolute_and_relative_error":
            return _multi([err, 100.0 * err / abs(actual.value)], f"{actual.unit}; %", "Computed absolute and percentage relative error.")
        if contract.target == "relative_error":
            return _answer(100.0 * err / abs(actual.value), "%", "Computed percentage relative error from true and measured values.")
        return _answer(err, actual.unit, "Computed absolute error from true and measured values.")

    if contract.target == "maximum_possible" and measured is not None and abs_err is not None:
        return _answer(measured.value + abs_err.value, measured.unit, "Added uncertainty to the measured value.")

    if contract.target == "absolute_error" and abs_err is not None:
        return _answer(abs_err.value, abs_err.unit, "Used instrument least count/uncertainty as absolute error.")

    if contract.target == "relative_error" and measured is not None and abs_err is not None:
        return _answer(100.0 * abs_err.value / abs(measured.value), "%", "Computed percentage relative error.")

    return None


def _solve_error_propagation(contract: ThcbContract) -> ThcbAnswer | None:
    q = contract.quantities
    lower_notes = " ".join(contract.notes).lower()
    voltage = q.get("voltage")
    current = q.get("current")
    resistance = q.get("resistance")
    abs_err = q.get("absolute_error")

    # Explicit direct targets
    if contract.target == "voltage_absolute_error" and voltage is not None:
        dv = _error_for(q, "voltage")
        if dv is not None:
            return _answer(dv, "V", "Returned extracted voltage absolute error.")

    if contract.target == "current_absolute_error" and current is not None:
        di = _error_for(q, "current")
        if di is not None:
            return _answer(di, "A", "Returned extracted current absolute error.")

    if voltage is not None and current is not None:
        dv = _error_for(q, "voltage")
        di = _error_for(q, "current")
        if dv is not None and di is not None:
            rel = dv / abs(voltage.value) + di / abs(current.value)
            
            if contract.target in {"power_absolute_error", "power_relative_error"} or "power" in contract.target:
                power = voltage.value * current.value
                if "relative" in contract.target:
                    return _answer(100.0 * rel, "%", "For P=UI, relative errors add.")
                return _answer(power * rel, "W", "For P=UI, absolute error is P times summed relative error.")

            # Check original condition: if resistance is present, we skip default resistance calculation
            if contract.target not in {"resistance_absolute_error", "resistance_relative_error"} and resistance is not None:
                return None

            if contract.target in {"resistance_absolute_error", "resistance_relative_error"} or "absolute" in contract.target or "relative" in contract.target:
                resistance_value = voltage.value / current.value
                if "relative" in contract.target:
                    return _answer(100.0 * rel, "%", "For R=U/I, relative errors add.")
                return _answer(resistance_value * rel, "Ω", "For R=U/I, absolute error is R times summed relative error.")

    if abs_err is not None and lower_notes:
        return _answer(abs_err.value, abs_err.unit, "Used extracted propagated error.")
    return None


def _solve_circuit(contract: ThcbContract) -> ThcbAnswer | None:
    conceptual = _solve_conceptual(contract)
    if conceptual is not None:
        return conceptual

    q = contract.quantities
    voltage = q.get("voltage")
    total_current = q.get("total_current")
    resistances = _numbered(q, "resistance")
    currents = _numbered(q, "current")
    powers = _numbered(q, "power")

    if contract.target == "branch_current" and total_current is not None and currents:
        known = sum(item.value for item in currents)
        return _answer(abs(total_current.value - known), total_current.unit, "Used total current as the sum of parallel branch currents.")

    if contract.target == "total_current" and currents:
        return _answer(sum(item.value for item in currents), currents[0].unit, "Summed branch currents in parallel.")

    if contract.target == "branch_currents" and voltage is not None and resistances:
        values = [voltage.value / r.value for r in resistances]
        if "total_current" in contract.requested_outputs:
            values.append(sum(values))
        return _multi(values, "A", "Applied Ohm's law to each parallel branch.")

    if contract.target == "total_current" and voltage is not None and resistances:
        values = [voltage.value / r.value for r in resistances]
        return _answer(sum(values), "A", "Summed Ohm-law branch currents in parallel.")

    if contract.target == "equivalent_resistance" and len(resistances) >= 2:
        reciprocal = sum(1.0 / r.value for r in resistances)
        return _answer(1.0 / reciprocal, "Ω", "Computed equivalent resistance for parallel branches.")

    if contract.target == "power":
        if voltage is not None and total_current is not None:
            return _answer(voltage.value * total_current.value, "W", "Used P=UI.")
        if powers:
            return _answer(sum(item.value for item in powers), powers[0].unit, "Summed component powers.")

    if contract.target == "branch_power" and powers:
        total = q.get("total_power")
        if total is not None:
            divisor = 2 if len(powers) == 0 else max(2, int(round(total.value / powers[0].value)))
            return _answer(total.value / divisor, total.unit, "Split total power equally across identical lamps.")
        if len(powers) == 1:
            return _answer(powers[0].value, powers[0].unit, "Used extracted branch power.")
    if contract.target == "branch_power":
        total = q.get("total_power")
        if total is not None:
            return _answer(total.value / 2.0, total.unit, "Split total power equally across two identical lamps.")

    if contract.target == "current":
        power = q.get("power")
        if power is not None and voltage is not None:
            return _answer(power.value / voltage.value, "A", "Used I=P/U.")
    return None


def _solve_conceptual(contract: ThcbContract) -> ThcbAnswer | None:
    if contract.relation == "resistance_down_current_up":
        return ThcbAnswer(
            answer="current increases",
            unit=None,
            explanation="For fixed voltage, Ohm's law gives I=U/R.",
            cot=["Matched THCB circuit concept."],
        )
    if contract.relation == "current_up_brightness_up":
        return ThcbAnswer(
            answer="the lamp shines brighter because the current increases",
            unit=None,
            explanation="Larger current increases lamp power and brightness.",
            cot=["Matched THCB circuit concept."],
        )
    if contract.relation == "lower_resistance_brighter":
        return ThcbAnswer(
            answer="brighter because the current is higher",
            unit=None,
            explanation="Parallel bulbs share voltage; lower resistance draws more current.",
            cot=["Matched THCB circuit concept."],
        )
    return None


def _error_for(q: dict[str, ThcbQuantity], base: str) -> float | None:
    if f"{base}_error" in q:
        return q[f"{base}_error"].value
    if "absolute_error" in q:
        return q["absolute_error"].value
    return None


def _first(q: dict[str, ThcbQuantity], *names: str) -> ThcbQuantity | None:
    for name in names:
        if name in q:
            return q[name]
    return None


def _numbered(q: dict[str, ThcbQuantity], base: str) -> list[ThcbQuantity]:
    items = []
    if base in q:
        items.append(q[base])
    for index in range(1, 6):
        key = f"{base}_{index}"
        if key in q:
            items.append(q[key])
    return items


def _answer(value: float, unit: str | None, explanation: str) -> ThcbAnswer:
    return ThcbAnswer(_format(value), unit, explanation, ["Solved THCB deterministic contract."])


def _multi(values: list[float], unit: str | None, explanation: str) -> ThcbAnswer:
    return ThcbAnswer("; ".join(_format(value) for value in values), unit, explanation, ["Solved THCB deterministic contract."])


def _format(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")
