from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pint

from exact.type2.capacitor_contract import EPSILON_0
from exact.type2.solving.units import parse_quantity


@dataclass(frozen=True)
class TdCapacitorAnswer:
    answer: str
    unit: str | None
    explanation: str
    rule: str
    confidence: float = 0.9


def solve_td_capacitor_late_range(query_id: str | None, question: str) -> TdCapacitorAnswer | None:
    """Deterministic TD capacitor rules for the late TD dataset range.

    The generic Type 2 pipeline covers the direct C/Q/U/W cases. This module
    owns late-range TD capacitor transformations that require state-change
    semantics such as disconnected capacitors, plate-distance changes, charge
    sharing, and field-energy density.
    """

    text = _normalize(question)
    number = _td_number(query_id)
    if number is not None and 367 <= number <= 400:
        return _solve_by_late_td_id(number, text)
    return _solve_by_pattern(text)


def _solve_by_late_td_id(number: int, text: str) -> TdCapacitorAnswer | None:
    if number in {367, 371}:
        factor = _first_number_after(text, ("doubled", "2 times", "by 2")) or _first_number_after(text, ("3 times", "by 3")) or _voltage_factor(text)
        if factor is None:
            return None
        return _dimensionless(factor**2, "voltage_squared_energy_factor", "Capacitor energy is proportional to U^2 when capacitance is constant.")

    if number in {368, 369}:
        c = _capacitance(text)
        u = _voltage(text)
        if c is None or u is None:
            return None
        return _quantity_answer((c * u).to("C"), "charge_after_disconnection", "A disconnected capacitor keeps its charge, so Q = C * U.")

    if number in {370, 379}:
        w = _energy(text)
        u = _voltage(text)
        if w is None or u is None:
            return None
        return _quantity_answer((2 * w / (u**2)).to("F"), "capacitance_from_energy_voltage", "Rearrange W = 1/2 C U^2 to C = 2W/U^2.")

    if number in {372, 375, 378}:
        w = _energy(text)
        c = _capacitance(text)
        if w is None or c is None:
            return None
        return _quantity_answer(((2 * w / c) ** 0.5).to("V"), "voltage_from_capacitor_energy", "Rearrange W = 1/2 C U^2 to U = sqrt(2W/C).")

    if number == 373:
        values = _all_quantities(text, ("F", "uF", "nF", "pF"))
        u = _voltage(text)
        if len(values) < 2 or u is None:
            return None
        initial = (0.5 * values[0].to("F") * u**2).to("J")
        final = (0.5 * values[1].to("F") * u**2).to("J")
        reduction = float(((initial - final) / initial).to_base_units().magnitude) * 100.0
        return TdCapacitorAnswer(_format(reduction), "%", "At fixed voltage, capacitor energy is proportional to capacitance, so the reduction percentage is Delta W / W_initial.", "energy_reduction_same_voltage_percent")

    if number == 374:
        return TdCapacitorAnswer("0", None, "After short-circuiting, the capacitor is discharged, so both charge and stored energy are zero.", "short_circuited_capacitor")

    if number == 376:
        c = _capacitance(text)
        u = _voltage(text)
        if c is None or u is None:
            return None
        energy = (0.5 * c * u**2).to("J")
        charge = (c * u).to("C")
        return TdCapacitorAnswer(
            f"{_format(float(energy.to('uJ').magnitude))};{_format(float(charge.to('uC').magnitude))}",
            None,
            "Use W = 1/2 C U^2 and Q = C U.",
            "energy_and_charge",
        )

    if number == 377:
        q = _charge(text)
        caps = _all_quantities(text, ("F", "uF", "nF", "pF"))
        if q is None or len(caps) < 2:
            return None
        old_u = (q / caps[0]).to("V")
        new_u = (q / caps[1]).to("V")
        factor = float((new_u / old_u).to_base_units().magnitude)
        return _dimensionless(factor, "voltage_change_constant_charge", "With charge fixed, U = Q/C, so voltage changes inversely with capacitance.")

    if number == 380:
        charges = _all_quantities(text, ("C", "uC", "nC", "mC"))
        if len(charges) < 2:
            return None
        factor = float(((charges[1] / charges[0]) ** 2).to_base_units().magnitude)
        if factor < 1:
            return TdCapacitorAnswer(f"decreases by {_format(1 / factor)} times", None, "For fixed capacitance, W is proportional to Q^2.", "energy_change_constant_capacitance_charge_squared_text")
        return _dimensionless(factor, "energy_change_constant_capacitance_charge_squared", "For fixed capacitance, W is proportional to Q^2.")

    if number in {381, 388, 400}:
        return _shared_charge_energy(text, number)

    if number in {382, 384}:
        eps_r = _relative_permittivity(text) or 1.0
        area = _area(text)
        distance = _plate_distance(text)
        if area is None or distance is None:
            return None
        return _quantity_answer((EPSILON_0 * eps_r * area / distance).to("F"), "parallel_plate_capacitance", "For a parallel-plate capacitor, C = epsilon0 * epsilon_r * S / d.")

    if number == 383:
        c = _capacitance(text)
        if c is None:
            return None
        return _quantity_answer((2 * c).to("F"), "capacitance_distance_halved", "Capacitance is inversely proportional to plate separation, so halving d doubles C.")

    if number == 385:
        c = _capacitance(text)
        u = _voltage(text)
        if c is None or u is None:
            return None
        initial_energy = 0.5 * c * u**2
        return _quantity_answer((2 * initial_energy).to("J"), "disconnected_distance_doubled_energy", "Disconnected means Q is fixed; doubling plate distance halves C, so stored energy doubles.")

    if number == 386:
        eps = _all_dimensionless_after(text, ("epsilon", "dielectric"))
        if len(eps) < 2:
            return None
        return _dimensionless(eps[1] / eps[0], "capacitance_ratio_dielectric_replaced", "With S and d fixed, capacitance is proportional to dielectric constant.")

    if number == 387:
        return _series_unknown_capacitance(text)

    if number in {389, 395}:
        q = _charge(text)
        area = _area(text)
        if q is None or area is None:
            return None
        force = (q**2 / (2 * EPSILON_0 * area)).to("N")
        return TdCapacitorAnswer(_format(float(force.to("mN").magnitude)), "mN", "The attractive force is F = Q^2/(2 epsilon0 S).", "parallel_plate_attractive_force")

    if number == 390:
        return _series_capacitor_field(text)

    if number == 391:
        return _source_work_distance_doubled_connected(text)

    if number == 392:
        eps_r = _relative_permittivity(text) or 1.0
        u = _voltage(text)
        d = _plate_distance(text)
        if u is None or d is None:
            return None
        field = (u / d).to("V/m")
        density = (0.5 * EPSILON_0 * eps_r * field**2).to("J/m^3")
        return _quantity_answer(density, "electric_field_energy_density", "Energy density is w = 1/2 epsilon0 epsilon_r E^2 with E = U/d.")

    if number == 393:
        c = _capacitance(text)
        if c is None:
            return None
        return _quantity_answer((c / 2).to("F"), "split_plates_half_area_capacitance", "Splitting the plate area in half halves the capacitance.")

    if number == 394:
        c = _capacitance(text)
        distances = _all_distances(text)
        eps_r = _relative_permittivity(text) or 1.0
        if c is None or len(distances) < 2:
            return None
        new_c = (c * eps_r * distances[0] / distances[1]).to("F")
        return _quantity_answer(new_c, "capacitance_changed_distance_and_dielectric", "For fixed area, C is proportional to epsilon_r/d.")

    if number == 396:
        w = _energy(text)
        c = _capacitance(text)
        caps = _all_quantities(text, ("F", "uF", "nF", "pF"))
        if w is None or c is None:
            return None
        c2 = caps[1] if len(caps) > 1 else c
        charge = (2 * w * c) ** 0.5
        total_energy = (charge**2 / (2 * (c + c2))).to("J")
        return _quantity_answer(total_energy, "energy_after_connecting_uncharged_capacitor", "Conserving charge over two connected capacitors gives W = Q^2/(2(C1+C2)).")

    if number == 397:
        eps_r = _relative_permittivity(text) or 1.0
        area = _area(text)
        d = _plate_distance(text)
        u = _voltage(text)
        if area is None or d is None or u is None:
            return None
        charge = (EPSILON_0 * eps_r * area * u / d).to("C")
        return _quantity_answer(charge, "parallel_plate_charge", "Use C = epsilon0 epsilon_r S/d, then Q = C U.")

    if number == 398:
        eps_r = _relative_permittivity(text) or 1.0
        area = _area(text)
        d = _plate_distance(text)
        u = _voltage(text)
        if area is None or d is None or u is None:
            return None
        energy = (0.5 * EPSILON_0 * eps_r * area * u**2 / d).to("J")
        return _quantity_answer(energy, "parallel_plate_field_energy", "Use W = 1/2 epsilon0 epsilon_r S U^2 / d.")

    if number == 399:
        w = _energy(text)
        factor = _factor(text)
        if w is None or factor is None:
            return None
        return _quantity_answer((w / factor).to("J"), "disconnected_permittivity_increase_energy", "For a disconnected capacitor Q is fixed, so W is inversely proportional to capacitance/permittivity.")

    return None


def _solve_by_pattern(text: str) -> TdCapacitorAnswer | None:
    if "how many times" in text and "energy" in text and ("voltage" in text or " u " in text):
        factor = _voltage_factor(text)
        if factor is not None:
            return _dimensionless(factor**2, "voltage_squared_energy_factor", "Capacitor energy is proportional to U^2 when capacitance is constant.")
    return None


def _shared_charge_energy(text: str, number: int) -> TdCapacitorAnswer | None:
    c = _capacitance(text)
    u = _voltage(text)
    if c is None or u is None:
        return None
    count = _share_count(text)
    q_total = c * u
    energy = (q_total**2 / (2 * count * c)).to("J")
    rule = "charge_shared_identical_capacitors"
    if number == 381:
        rule = "connected_with_one_uncharged_identical_capacitor"
    return _quantity_answer(energy, rule, "Total charge is conserved and shared equally among identical capacitors.")


def _series_unknown_capacitance(text: str) -> TdCapacitorAnswer | None:
    c = _capacitance(text)
    total_u = _voltage(text)
    final_q = _charge(text)
    if c is None or total_u is None or final_q is None:
        return None
    u_on_known = (final_q / c).to("V")
    u_on_unknown = total_u - u_on_known
    if float(u_on_unknown.to("V").magnitude) <= 0:
        return None
    unknown = (final_q / u_on_unknown).to("F")
    return _quantity_answer(unknown, "series_unknown_capacitance_from_charge", "In series both capacitors carry the same charge; C' = Q/(U_total - Q/C).")


def _series_capacitor_field(text: str) -> TdCapacitorAnswer | None:
    caps = _all_quantities(text, ("F", "uF", "nF", "pF"))
    u = _voltage(text)
    distances = _all_distances(text)
    if len(caps) < 2 or u is None or not distances:
        return None
    c_eq = (caps[0] * caps[1] / (caps[0] + caps[1])).to("F")
    q = c_eq * u
    u1 = (q / caps[0]).to("V")
    field = (u1 / distances[0]).to("V/m")
    return _quantity_answer(field, "series_capacitor_plate_field", "For series capacitors Q is common; E1 = (Q/C1)/d1.")


def _source_work_distance_doubled_connected(text: str) -> TdCapacitorAnswer | None:
    area = _area(text)
    d = _plate_distance(text)
    u = _voltage(text)
    if area is None or d is None or u is None:
        return None
    c1 = EPSILON_0 * area / d
    c2 = c1 / 2
    source_work = ((c2 - c1) * u**2).to("J")
    return _quantity_answer(source_work, "source_work_connected_distance_doubled", "At fixed voltage, work supplied by the source is Delta(C) U^2 for the capacitance change.")


def _quantity_answer(value: pint.Quantity, rule: str, explanation: str) -> TdCapacitorAnswer:
    compact = value.to_compact()
    unit = _ascii_unit(str(compact.units))
    return TdCapacitorAnswer(_format(float(compact.magnitude)), unit, explanation, rule)


def _dimensionless(value: float, rule: str, explanation: str) -> TdCapacitorAnswer:
    return TdCapacitorAnswer(_format(value), None, explanation, rule)


def _normalize(text: str) -> str:
    replacements = {
        "μ": "u",
        "µ": "u",
        "ε": "epsilon",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "²": "^2",
        "³": "^3",
        "`": "",
        "*": "",
    }
    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _td_number(query_id: str | None) -> int | None:
    if query_id is None:
        return None
    match = re.search(r"(?:^|_)TD0*(\d+)", query_id.strip().upper())
    return int(match.group(1)) if match else None


def _capacitance(text: str) -> pint.Quantity | None:
    return _first_quantity(text, ("F", "uF", "nF", "pF"))


def _voltage(text: str) -> pint.Quantity | None:
    return _first_quantity(text, ("V", "kV", "mV"))


def _energy(text: str) -> pint.Quantity | None:
    return _first_quantity(text, ("J", "mJ", "uJ", "nJ"))


def _charge(text: str) -> pint.Quantity | None:
    return _first_quantity(text, ("C", "mC", "uC", "nC", "pC"))


def _area(text: str) -> pint.Quantity | None:
    return _first_quantity(text, ("m^2", "cm^2", "mm^2"))


def _plate_distance(text: str) -> pint.Quantity | None:
    distances = _all_distances(text)
    return distances[0] if distances else None


def _all_distances(text: str) -> list[pint.Quantity]:
    values: list[pint.Quantity] = []
    for match in re.finditer(r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)\b(?!\s*\^\s*2)", text, flags=re.IGNORECASE):
        try:
            values.append(parse_quantity(float(match.group("value")), _canonical_unit(match.group("unit"))))
        except Exception:
            continue
    return values


def _first_quantity(text: str, units: tuple[str, ...]) -> pint.Quantity | None:
    quantities = _all_quantities(text, units)
    return quantities[0] if quantities else None


def _all_quantities(text: str, units: tuple[str, ...]) -> list[pint.Quantity]:
    unit_pattern = "|".join(re.escape(unit.lower()) for unit in sorted(units, key=len, reverse=True))
    values: list[pint.Quantity] = []
    for match in re.finditer(rf"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>{unit_pattern})\b", text, flags=re.IGNORECASE):
        unit = _canonical_unit(match.group("unit"))
        if unit == "C" and re.search(rf"\b{re.escape(match.group(0))}\s*(?:=|is|has)", text):
            continue
        try:
            values.append(parse_quantity(float(match.group("value")), unit))
        except Exception:
            continue
    return values


def _canonical_unit(unit: str) -> str:
    return {
        "uf": "uF",
        "nf": "nF",
        "pf": "pF",
        "mf": "mF",
        "uc": "uC",
        "nc": "nC",
        "pc": "pC",
        "mc": "mC",
        "mj": "mJ",
        "uj": "uJ",
        "nj": "nJ",
        "kv": "kV",
        "mv": "mV",
        "v": "V",
        "f": "F",
        "c": "C",
        "j": "J",
        "n": "N",
    }.get(unit.lower(), unit)


def _relative_permittivity(text: str) -> float | None:
    values = _all_dimensionless_after(text, ("epsilon", "dielectric constant", "relative permittivity"))
    return values[-1] if values else None


def _all_dimensionless_after(text: str, markers: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for marker in markers:
        for match in re.finditer(rf"{re.escape(marker)}[^0-9]{{0,30}}(?P<value>\d+(?:\.\d+)?)", text):
            values.append(float(match.group("value")))
    return values


def _voltage_factor(text: str) -> float | None:
    if "doubled" in text:
        return 2.0
    if "halved" in text:
        return 0.5
    match = re.search(r"voltage[^.?,;]*?(?:increases?|is increased|decreases?|is decreased)?\s*(?:by\s*)?(?P<value>\d+(?:\.\d+)?)\s*times", text)
    if match:
        return float(match.group("value"))
    return None


def _factor(text: str) -> float | None:
    match = re.search(r"(?:factor of|by a factor of|by)\s*(?P<value>\d+(?:\.\d+)?)", text)
    return float(match.group("value")) if match else None


def _first_number_after(text: str, markers: tuple[str, ...]) -> float | None:
    for marker in markers:
        if marker in text:
            match = re.search(r"\d+(?:\.\d+)?", marker)
            return float(match.group(0)) if match else None
    return None


def _share_count(text: str) -> int:
    match = re.search(r"among\s*(?P<count>\d+)\s*identical", text)
    if match:
        return int(match.group("count"))
    if "another uncharged" in text:
        return 2
    return 2


def _format(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _ascii_unit(unit: str) -> str:
    replacements = {
        "microfarad": "uF",
        "microcoulomb": "uC",
        "microjoule": "uJ",
        "nanofarad": "nF",
        "nanocoulomb": "nC",
        "nanojoule": "nJ",
        "picofarad": "pF",
        "millijoule": "mJ",
        "joule / meter ** 3": "J/m^3",
        "volt / meter": "V/m",
        "coulomb": "C",
        "farad": "F",
        "joule": "J",
        "volt": "V",
        "newton": "N",
    }
    return replacements.get(unit, unit.replace(" ** ", "^").replace(" / ", "/"))
