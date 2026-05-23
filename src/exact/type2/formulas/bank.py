from __future__ import annotations

import math

from exact.type2.schemas import Formula
from exact.type2.solving.units import ureg


def _q(known: dict[str, object], name: str):
    return known[name]


K_COULOMB = 8.9875517923e9 * ureg.newton * (ureg.meter ** 2) / (ureg.coulomb ** 2)


FORMULAS: tuple[Formula, ...] = (
    Formula(
        id="ohm_voltage",
        domain="circuits",
        target="voltage",
        required=("current", "resistance"),
        output_unit="V",
        expression="U = I * R",
        explanation_template="Ohm's law gives voltage as U = I * R.",
        keywords=("voltage", "current", "resistance", "resistor", "ohm"),
        solve=lambda k: _q(k, "current") * _q(k, "resistance"),
    ),
    Formula(
        id="ohm_current",
        domain="circuits",
        target="current",
        required=("voltage", "resistance"),
        output_unit="A",
        expression="I = U / R",
        explanation_template="Ohm's law gives current as I = U / R.",
        keywords=("current", "voltage", "resistance", "resistor", "ohm"),
        solve=lambda k: _q(k, "voltage") / _q(k, "resistance"),
    ),
    Formula(
        id="ohm_resistance",
        domain="circuits",
        target="resistance",
        required=("voltage", "current"),
        output_unit="ohm",
        expression="R = U / I",
        explanation_template="Ohm's law gives resistance as R = U / I.",
        keywords=("resistance", "voltage", "current", "resistor", "ohm"),
        solve=lambda k: _q(k, "voltage") / _q(k, "current"),
    ),
    Formula(
        id="power_voltage_current",
        domain="circuits",
        target="power",
        required=("voltage", "current"),
        output_unit="W",
        expression="P = U * I",
        explanation_template="Electrical power is P = U * I.",
        keywords=("power", "voltage", "current", "circuit"),
        solve=lambda k: _q(k, "voltage") * _q(k, "current"),
    ),
    Formula(
        id="power_current_resistance",
        domain="circuits",
        target="power",
        required=("current", "resistance"),
        output_unit="W",
        expression="P = I^2 * R",
        explanation_template="Using Ohm's law, power can be computed as P = I^2 * R.",
        keywords=("power", "current", "resistance", "resistor"),
        solve=lambda k: (_q(k, "current") ** 2) * _q(k, "resistance"),
    ),
    Formula(
        id="power_voltage_resistance",
        domain="circuits",
        target="power",
        required=("voltage", "resistance"),
        output_unit="W",
        expression="P = U^2 / R",
        explanation_template="Using Ohm's law, power can be computed as P = U^2 / R.",
        keywords=("power", "voltage", "resistance", "resistor"),
        solve=lambda k: (_q(k, "voltage") ** 2) / _q(k, "resistance"),
    ),
    Formula(
        id="capacitance_from_charge_voltage",
        domain="capacitance",
        target="capacitance",
        required=("charge", "voltage"),
        output_unit="F",
        expression="C = Q / U",
        explanation_template="Capacitance is charge divided by voltage: C = Q / U.",
        keywords=("capacitance", "capacitor", "charge", "voltage"),
        solve=lambda k: _q(k, "charge") / _q(k, "voltage"),
    ),
    Formula(
        id="charge_from_capacitance_voltage",
        domain="capacitance",
        target="charge",
        required=("capacitance", "voltage"),
        output_unit="C",
        expression="Q = C * U",
        explanation_template="A capacitor stores charge Q = C * U.",
        keywords=("charge", "capacitor", "capacitance", "voltage"),
        solve=lambda k: _q(k, "capacitance") * _q(k, "voltage"),
    ),
    Formula(
        id="voltage_from_charge_capacitance",
        domain="capacitance",
        target="voltage",
        required=("charge", "capacitance"),
        output_unit="V",
        expression="U = Q / C",
        explanation_template="For a capacitor, voltage is U = Q / C.",
        keywords=("voltage", "potential difference", "capacitor", "charge", "capacitance"),
        solve=lambda k: _q(k, "charge") / _q(k, "capacitance"),
    ),
    Formula(
        id="capacitor_energy",
        domain="capacitance",
        target="energy",
        required=("capacitance", "voltage"),
        output_unit="J",
        expression="W = 1/2 * C * U^2",
        explanation_template="The energy stored in a capacitor is W = 1/2 * C * U^2.",
        keywords=("energy", "stored", "capacitor", "capacitance", "voltage"),
        solve=lambda k: 0.5 * _q(k, "capacitance") * (_q(k, "voltage") ** 2),
    ),
    Formula(
        id="inductor_energy",
        domain="inductance",
        target="energy",
        required=("inductance", "current"),
        output_unit="J",
        expression="W = 1/2 * L * I^2",
        explanation_template="The magnetic energy stored in an inductor is W = 1/2 * L * I^2.",
        keywords=("energy", "magnetic", "inductor", "inductance", "current"),
        solve=lambda k: 0.5 * _q(k, "inductance") * (_q(k, "current") ** 2),
    ),
    Formula(
        id="electric_field_from_force_charge",
        domain="electrostatics",
        target="electric_field",
        required=("force", "charge"),
        output_unit="V/m",
        expression="E = F / q",
        explanation_template="Electric field strength is force per unit charge: E = F / q.",
        keywords=("electric field", "field strength", "force", "charge"),
        solve=lambda k: _q(k, "force") / _q(k, "charge"),
    ),
    Formula(
        id="force_from_field_charge",
        domain="electrostatics",
        target="force",
        required=("electric_field", "charge"),
        output_unit="N",
        expression="F = q * E",
        explanation_template="The electric force on a charge in a field is F = q * E.",
        keywords=("electric force", "force", "electric field", "charge"),
        solve=lambda k: _q(k, "charge") * _q(k, "electric_field"),
    ),
    Formula(
        id="coulomb_force",
        domain="electrostatics",
        target="force",
        required=("charge", "charge_2", "length"),
        output_unit="N",
        expression="F = k * |q1*q2| / r^2",
        explanation_template="Coulomb's law gives the force magnitude as F = k * |q1*q2| / r^2.",
        keywords=("electric force", "force", "charges", "charge", "separated", "distance"),
        solve=lambda k: K_COULOMB * abs(_q(k, "charge") * _q(k, "charge_2")) / (_q(k, "length") ** 2),
    ),
    Formula(
        id="coulomb_equal_charges_from_force",
        domain="electrostatics",
        target="charge",
        required=("force", "length"),
        output_unit="C",
        expression="q = sqrt(F * r^2 / k), for |q1| = |q2| = q",
        explanation_template="For equal point charges, Coulomb's law rearranges to q = sqrt(F*r^2/k).",
        keywords=("charge", "charges", "force", "separated", "q1 = q2", "find q"),
        solve=lambda k: ((_q(k, "force") * (_q(k, "length") ** 2) / K_COULOMB) ** 0.5),
    ),
    Formula(
        id="resultant_force_with_angle",
        domain="vectors",
        target="force",
        required=("force", "force_2", "angle"),
        output_unit="N",
        expression="F = sqrt(F1^2 + F2^2 + 2*F1*F2*cos(theta))",
        explanation_template="The resultant of two forces with angle theta is sqrt(F1^2 + F2^2 + 2F1F2cos(theta)).",
        keywords=("resultant", "net", "forces", "angle"),
        solve=lambda k: (
            (_q(k, "force") ** 2)
            + (_q(k, "force_2") ** 2)
            + 2 * _q(k, "force") * _q(k, "force_2") * math.cos(_q(k, "angle").to("radian").magnitude)
        )
        ** 0.5,
    ),
    Formula(
        id="resultant_force_same_direction",
        domain="vectors",
        target="force",
        required=("force", "force_2"),
        output_unit="N",
        expression="F = F1 + F2",
        explanation_template="For forces acting in the same direction, the resultant magnitude is F = F1 + F2.",
        keywords=("resultant", "same direction", "forces"),
        solve=lambda k: _q(k, "force") + _q(k, "force_2"),
    ),
    Formula(
        id="resonance_frequency",
        domain="rlc",
        target="frequency",
        required=("inductance", "capacitance"),
        output_unit="Hz",
        expression="f0 = 1 / (2*pi*sqrt(L*C))",
        explanation_template="The resonance frequency of an RLC circuit is f0 = 1/(2*pi*sqrt(L*C)).",
        keywords=("rlc", "resonance", "frequency", "inductor", "capacitor"),
        solve=lambda k: 1 / (2 * math.pi * ((_q(k, "inductance") * _q(k, "capacitance")) ** 0.5)),
    ),
    Formula(
        id="resonant_resistance",
        domain="rlc",
        target="resistance",
        required=("impedance",),
        output_unit="ohm",
        expression="At resonance, Z = R",
        explanation_template="At resonance in a series RLC circuit, the impedance equals the pure resistance.",
        keywords=("rlc", "resonance", "resonant", "impedance", "resistance"),
        solve=lambda k: _q(k, "impedance"),
    ),
)


def retrieve_formulas(question: str, target: str | None, known: dict[str, object]) -> list[Formula]:
    candidates = [
        formula
        for formula in FORMULAS
        if (target is None or formula.target == target)
        and all(required in known for required in formula.required)
    ]
    return sorted(candidates, key=lambda formula: _formula_score(formula, question), reverse=True)


def _formula_score(formula: Formula, question: str) -> int:
    lower = question.lower()
    return sum(1 for keyword in formula.keywords if keyword in lower)
