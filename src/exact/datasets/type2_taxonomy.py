from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Type2TaxonomyLabel:
    solver_family: str
    solve_method: str
    question_type: str


def classify_type2_taxonomy(question: str, cot: str = "", unit: str = "") -> Type2TaxonomyLabel:
    """Assign solver-strategy and physics-concept labels to a Type 2 question.

    The source CSV does not contain taxonomy columns, so these labels are inferred
    from the question text and explanation. Rules are intentionally deterministic:
    generated annotations can be reproduced exactly from the dataset.
    """

    del cot
    text = _normalize(question)
    answer_unit = unit.strip().lower()
    question_type = _classify_question_type(text, answer_unit)
    solve_method = _classify_solve_method(text, question_type)
    solver_family = _solver_family(solve_method)
    return Type2TaxonomyLabel(
        solver_family=solver_family,
        solve_method=solve_method,
        question_type=question_type,
    )


def _classify_solve_method(text: str, question_type: str) -> str:
    if question_type.startswith("conceptual_"):
        return "conceptual_reasoning"
    if question_type in {"equilibrium_zero_field", "equilibrium_zero_force", "resonance_condition"}:
        return "equilibrium_solve"
    if _has_geometry_markers(text):
        if question_type in {"vector_resultant_force", "vector_resultant_field"} or _has_any(
            text, "electric force", "net force", "electric field", "field strength"
        ):
            return "geometry_vector_graph"
        if _has_any(text, "area", "volume", "radius", "diameter"):
            return "geometry_formula"
    graph_method = _graph_solve_method(text, question_type)
    if graph_method is not None:
        return graph_method
    if question_type in {
        "vector_resultant_force",
        "vector_resultant_field",
        "superposition_force",
        "superposition_field",
    }:
        return "vector_superposition"
    if question_type in {
        "charge_from_force",
        "distance_from_force",
        "field_from_force",
        "capacitance_from_charge_voltage",
        "voltage_from_capacitor_energy",
        "resistance_from_ohm",
        "current_from_ohm",
        "voltage_from_ohm",
        "time_from_energy_power",
        "mass_from_heat",
    }:
        return "inverse_formula"
    if _has_multi_step_formula_markers(text):
        return "multi_step_formula"
    return "direct_formula"


def _solver_family(solve_method: str) -> str:
    if solve_method in {"direct_formula", "inverse_formula", "multi_step_formula"}:
        return "formula_executor"
    if solve_method in {"vector_superposition", "geometry_vector_graph"}:
        return "vector_solver"
    if solve_method in {
        "electrostatic_force_graph",
        "circuit_network_graph",
        "capacitor_network_graph",
    }:
        return "graph_solver"
    if solve_method == "equilibrium_solve":
        return "constraint_solver"
    if solve_method == "conceptual_reasoning":
        return "text_solver"
    return "formula_executor"


def _graph_solve_method(text: str, question_type: str) -> str | None:
    if question_type == "capacitor_network" or _has_capacitor_network_markers(text):
        return "capacitor_network_graph"
    if question_type == "resistor_network" or _has_circuit_network_markers(text):
        return "circuit_network_graph"
    if question_type in {
        "coulomb_force",
        "vector_resultant_force",
        "superposition_force",
        "electrostatics_other",
    } and _has_any(text, "q1", "q2", "q3", "test charge", "third charge", "force acting"):
        if _has_any(text, "midpoint", "straight line", "collinear", "between", "third charge", "test charge"):
            return "electrostatic_force_graph"
    if _has_any(text, "equivalent", "connected together"):
        if _has_any(text, "capacitor", "capacitance"):
            return "capacitor_network_graph"
        return "circuit_network_graph"
    return None


def _classify_question_type(text: str, unit: str) -> str:
    if _is_conceptual(text):
        return _conceptual_type(text)

    electrostatic = _has_any(
        text,
        "point charge",
        "point charges",
        "electric charge",
        "electric charges",
        "test charge",
        "coulomb",
        "electric field",
        "field strength",
        "electric potential",
        "potential energy",
        "electric force",
        "q1",
        "q2",
        "q3",
    )
    if electrostatic and not _has_any(text, "capacitor", "capacitance"):
        return _electrostatics_type(text)

    if _has_any(text, "capacitor", "capacitance", "parallel-plate", "dielectric"):
        return _capacitance_type(text)
    if _has_any(text, "inductor", "inductance", "lc circuit", "oscillation circuit"):
        return _inductance_type(text)
    if _has_any(text, "magnetic field", "magnetic flux", "transformer", "solenoid"):
        return _magnetism_type(text)
    if _has_any(text, "resistor", "resistance", "current", "voltage", "circuit", "ohm", "lamp"):
        return _circuit_type(text)
    if _has_any(text, "heat", "temperature", "specific heat", "calorific", "thermal"):
        return _thermal_type(text)
    if _has_any(text, "speed", "velocity", "distance", "time", "acceleration", "range"):
        return _mechanics_type(text, unit)
    if _has_any(text, "pressure", "density", "volume", "area", "radius", "diameter"):
        return _materials_geometry_type(text, unit)
    if _has_any(text, "wave", "wavelength", "frequency", "period", "sound"):
        return _wave_type(text)
    if unit in {"n", "newton"}:
        return "force"
    if unit in {"j", "joule"}:
        return "energy"
    if unit in {"v", "volt"}:
        return "voltage"
    if unit in {"a", "ampere"}:
        return "current"
    return "other_physics"


def _electrostatics_type(text: str) -> str:
    if _has_any(text, "zero", "equilibrium", "balanced") and _has_any(
        text, "electric field", "field strength"
    ):
        return "equilibrium_zero_field"
    if _has_any(text, "zero", "equilibrium", "balanced") and _has_any(text, "force", "electric force"):
        return "equilibrium_zero_force"
    if _has_any(text, "find q", "calculate q", "determine q") and _has_any(text, "force"):
        return "charge_from_force"
    if _asks_for_distance_from_force(text):
        return "distance_from_force"
    if _has_any(text, "potential energy"):
        return "electric_potential_energy"
    if _has_any(text, "electric potential", "potential at"):
        return "electric_potential"
    if _has_any(text, "field from force", "electric field") and _has_any(text, "force per unit charge"):
        return "field_from_force"
    if _has_any(text, "electric field", "field strength", "field intensity"):
        if _has_geometry_markers(text):
            return "vector_resultant_field"
        if _has_any(text, "net", "resultant", "due to these two", "caused by these two", "two point charges"):
            if _has_any(text, "vector", "magnitude", "perpendicular", "triangle", "midpoint"):
                return "vector_resultant_field"
            return "superposition_field"
        return "electric_field"
    if _has_geometry_markers(text) and _has_any(text, "force", "electric force"):
        return "vector_resultant_force"
    if _has_any(text, "resultant", "net electric force", "net force", "magnitude of the electric force"):
        if _has_any(text, "two charges", "three charges", "q1", "q2", "q3"):
            return "vector_resultant_force"
        return "superposition_force"
    if _has_any(text, "force", "attract", "repel", "coulomb"):
        return "coulomb_force"
    return "electrostatics_other"


def _capacitance_type(text: str) -> str:
    if _has_any(text, "energy stored", "electric field energy", "stored energy"):
        return "capacitor_energy"
    if _has_capacitor_network_markers(text):
        return "capacitor_network"
    if _has_any(text, "dielectric", "relative permittivity", "immersed", "inserted"):
        return "dielectric_capacitor"
    if _has_any(text, "charge q", "stores q", "charge of the capacitor"):
        return "capacitor_charge"
    if _has_any(text, "voltage", "potential difference"):
        if _has_any(text, "find", "calculate", "determine") and _has_any(text, "capacitance c"):
            return "capacitance_from_charge_voltage"
        return "capacitor_voltage"
    if _has_any(text, "plate separation", "plate distance", "area", "radius"):
        return "parallel_plate_capacitance"
    return "capacitance"


def _circuit_type(text: str) -> str:
    if _has_any(text, "electrical energy", "energy consumed", "power consumption", "power dissipated"):
        return "electrical_energy_power"
    if _has_any(text, "power", "watt", "luminosity"):
        return "electric_power"
    if _has_circuit_network_markers(text):
        return "resistor_network"
    if _has_any(text, "resistance"):
        if _has_any(text, "find", "calculate", "determine") and _has_any(text, "current", "voltage"):
            return "resistance_from_ohm"
        return "resistance"
    if _has_any(text, "current"):
        return "current_from_ohm"
    if _has_any(text, "voltage", "potential difference"):
        return "voltage_from_ohm"
    return "dc_circuit"


def _inductance_type(text: str) -> str:
    if _has_any(text, "energy"):
        return "inductor_energy"
    if _has_any(text, "frequency", "period", "oscillation", "resonance"):
        return "lc_oscillation"
    return "inductance"


def _magnetism_type(text: str) -> str:
    if _has_any(text, "transformer", "turns", "primary", "secondary"):
        return "transformer"
    if _has_any(text, "magnetic flux"):
        return "magnetic_flux"
    if _has_any(text, "solenoid"):
        return "solenoid_magnetic_field"
    return "magnetic_field"


def _thermal_type(text: str) -> str:
    if _has_any(text, "specific heat", "temperature"):
        return "heat_capacity"
    if _has_any(text, "calorific", "fuel", "combustion"):
        return "fuel_combustion"
    return "thermal_energy"


def _mechanics_type(text: str, unit: str) -> str:
    if _has_any(text, "work", "kinetic energy", "potential energy", "power"):
        return "mechanical_work_energy"
    if _has_any(text, "acceleration"):
        return "acceleration"
    if unit in {"m/s", "km/h"} or _has_any(text, "speed", "velocity"):
        return "speed"
    if _has_any(text, "distance", "range"):
        return "distance"
    return "kinematics"


def _materials_geometry_type(text: str, unit: str) -> str:
    if _has_any(text, "pressure"):
        return "pressure"
    if _has_any(text, "density"):
        return "density"
    if unit in {"m^2", "cm^2"} or _has_any(text, "area"):
        return "area_geometry"
    if unit in {"m^3", "l"} or _has_any(text, "volume"):
        return "volume_geometry"
    return "geometry"


def _wave_type(text: str) -> str:
    if _has_any(text, "frequency"):
        return "wave_frequency"
    if _has_any(text, "wavelength"):
        return "wavelength"
    if _has_any(text, "period"):
        return "wave_period"
    return "wave_basics"


def _conceptual_type(text: str) -> str:
    if _has_any(text, "resonance"):
        return "resonance_condition"
    if _has_any(text, "unit of", "what is the unit"):
        return "conceptual_unit"
    if _has_any(text, "increase", "decrease", "larger", "smaller", "directly proportional"):
        return "conceptual_relationship"
    if _has_any(text, "where", "which", "when"):
        return "conceptual_selection"
    return "conceptual_explanation"


def _is_conceptual(text: str) -> bool:
    if _has_numeric_answer_intent(text):
        return False
    if _has_any(text, "what is the unit", "unit of", "shape of the graph"):
        return True
    if _has_any(text, "why ", "explain", "describe", "which ", "where ", "when ", "what happens"):
        return not _has_any(text, "calculate", "determine the value", "find the value")
    return False


def _has_numeric_answer_intent(text: str) -> bool:
    if _has_any(
        text,
        "calculate",
        "determine",
        "find",
        "what is the magnitude",
        "what is the value",
    ):
        return True
    return bool(
        re.search(
            r"\bwhat is\b.*\b(force|electric force|field|current|voltage|energy|capacitance|resistance|distance|mass|time|speed|power)\b",
            text,
        )
    )


def _normalize(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("×", "x").replace("−", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _has_geometry_markers(text: str) -> bool:
    if _has_any(text, "triangle", "vertices", "right-angled", "equilateral", "perpendicular bisector"):
        return True
    return bool(re.search(r"\b(ab|ac|bc|ca|cb)\s*=", text))


def _has_capacitor_network_markers(text: str) -> bool:
    if not _has_any(text, "capacitor", "capacitors", "capacitance"):
        return False
    return bool(
        _has_any(
            text,
            "equivalent capacitance",
            "capacitors in series",
            "capacitors in parallel",
            "two capacitors",
            "three capacitors",
            "connected together",
            "connected with",
            "connected in series",
            "connected in parallel",
            "terminals joined",
            "like-polarity terminals",
            "like-poled terminals",
            "like-signed terminals",
            "charge is equally shared",
            "distributed equally",
        )
    )


def _has_circuit_network_markers(text: str) -> bool:
    return bool(
        _has_any(
            text,
            "equivalent resistance",
            "resistors in series",
            "resistors in parallel",
            "lamps are connected in parallel",
            "connected in series",
            "connected in parallel",
            "circuit segment",
        )
    )


def _asks_for_distance_from_force(text: str) -> bool:
    if not _has_any(text, "force"):
        return False
    return bool(
        re.search(
            r"\b(find|determine|calculate|what is)\s+(the\s+)?"
            r"(distance|separation|separation distance|r)\b",
            text,
        )
        or re.search(r"\b(how far|at what distance)\b", text)
    )


def _has_multi_step_formula_markers(text: str) -> bool:
    return bool(
        re.search(
            r"\b(afterwards|then|immersed|inserted|disconnected|short-circuited|"
            r"split|replaced|doubled|double|tripled|quadrupled|halved|"
            r"increased by|decreased by|increases by|decreases by|"
            r"changes from|decreases from|increases from|kept constant|"
            r"terminal velocity|reaches maximum|reaches its maximum|"
            r"time-dependent|varies according to|voltage changes according to|"
            r"current changing according to|current varies according to|"
            r"frequency is doubled|frequency is tripled|frequency is quadrupled)\b",
            text,
        )
        or bool(re.search(r"\bif\b.*\b(frequency|voltage|current|flux|capacitance).*\b(doubled|tripled|quadrupled|halved|decreases|increases)\b", text))
    )
