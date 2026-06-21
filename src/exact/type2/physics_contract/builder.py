from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from exact.datasets.type2_taxonomy import classify_type2_taxonomy
from exact.type2.formulas.knowledge import RetrievedFormulaContext
from exact.type2.physics_contract.dimensions import canonical_dimension, expected_unit_for
from exact.type2.physics_contract.models import ContractKnown, PhysicsConstraint, PhysicsContract
from exact.type2.schemas import Extraction


def build_physics_contract(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
) -> PhysicsContract:
    target = canonical_dimension(extraction.target)
    label = classify_type2_taxonomy(
        extraction.normalized_question,
        unit=_formula_context_unit_hint(formula_context),
    )
    topic, subtopic = _topic_subtopic(label.question_type)
    knowns = {
        name: ContractKnown(
            name=name,
            value=quantity.value,
            dimension=canonical_dimension(name),
            unit=str(quantity.value.units) if quantity.value is not None else None,
            evidence=quantity.evidence,
        )
        for name, quantity in extraction.quantities.items()
    }
    constraints = tuple(_infer_constraints(extraction))
    formula_ids = tuple(formula_context.formula_ids)
    return PhysicsContract(
        target=target,
        knowns=knowns,
        unknowns=(target,) if target and target not in knowns else (),
        topic=topic,
        subtopic=subtopic,
        principle=_infer_principle(extraction, label.question_type, formula_context),
        expected_dimension=target,
        expected_unit=expected_unit_for(target) or _formula_context_unit_hint(formula_context) or None,
        constraints=constraints,
        formula_ids=formula_ids,
        confidence=_contract_confidence(target, formula_ids, constraints),
        diagnostics={
            "taxonomy_solver_family": label.solver_family,
            "taxonomy_solve_method": label.solve_method,
            "taxonomy_question_type": label.question_type,
        },
    )


def contract_to_prompt_dict(contract: PhysicsContract) -> dict[str, Any]:
    payload = asdict(contract)
    for item in payload["knowns"].values():
        value = item.get("value")
        if value is not None:
            item["value"] = str(value)
    return payload


def _topic_subtopic(question_type: str) -> tuple[str, str]:
    if any(token in question_type for token in ("capacitor", "capacitance")):
        return "electricity", "capacitors"
    if any(token in question_type for token in ("electric", "coulomb", "charge", "potential")):
        return "electricity", "electrostatics"
    if any(token in question_type for token in ("resistance", "current", "voltage", "circuit", "ohm")):
        return "electricity", "dc_circuits"
    if any(token in question_type for token in ("inductor", "magnetic", "transformer", "solenoid")):
        return "magnetism", question_type
    if any(token in question_type for token in ("heat", "thermal", "temperature")):
        return "thermal", "heat"
    if any(token in question_type for token in ("speed", "force", "work", "energy", "kinematics")):
        return "mechanics", question_type
    if any(token in question_type for token in ("wave", "frequency", "wavelength", "period")):
        return "waves", question_type
    return "physics", question_type or "unknown"


def _infer_principle(
    extraction: Extraction,
    question_type: str,
    formula_context: RetrievedFormulaContext,
) -> str:
    text = extraction.normalized_question.lower()
    if extraction.target == "electric_field" and "midpoint" in text:
        return "field_superposition"
    if extraction.target in {"energy", "stored_energy"} and "capacitor" in text:
        return "capacitor_energy"
    if "series" in text and "resistor" in text:
        return "series_resistance"
    if "parallel" in text and "resistor" in text:
        return "parallel_resistance"
    if formula_context.solution_plan:
        return "formula_selection_plan"
    return question_type or "direct_formula"


def _infer_constraints(extraction: Extraction) -> list[PhysicsConstraint]:
    text = extraction.normalized_question.lower()
    constraints: list[PhysicsConstraint] = []
    if "midpoint" in text:
        constraints.append(PhysicsConstraint("midpoint", tuple(_charge_variables(extraction)), "between"))
    if "series" in text:
        constraints.append(PhysicsConstraint("network", tuple(_resistance_variables(extraction)), "series"))
    if "parallel" in text:
        constraints.append(PhysicsConstraint("network", tuple(_resistance_variables(extraction)), "parallel"))
    if re.search(r"\bopposite\s+charges?\b|\bequal and opposite\b", text):
        constraints.append(PhysicsConstraint("charge_signs", tuple(_charge_variables(extraction)), "opposite"))
    if "same direction" in text:
        constraints.append(PhysicsConstraint("vector_direction", tuple(), "same"))
    if "opposite direction" in text:
        constraints.append(PhysicsConstraint("vector_direction", tuple(), "opposite"))
    return constraints


def _charge_variables(extraction: Extraction) -> list[str]:
    return [name for name in extraction.quantities if "charge" in name]


def _resistance_variables(extraction: Extraction) -> list[str]:
    return [name for name in extraction.quantities if "resistance" in name]


def _contract_confidence(target: str, formula_ids: tuple[str, ...], constraints: tuple[PhysicsConstraint, ...]) -> float:
    confidence = 0.35
    if target and target != "unknown":
        confidence += 0.25
    if formula_ids:
        confidence += 0.20
    if constraints:
        confidence += 0.10
    return min(confidence, 0.9)


def _formula_context_unit_hint(formula_context: RetrievedFormulaContext) -> str:
    for summary in formula_context.summaries:
        output = str(summary.get("output_unit") or summary.get("output") or "").strip()
        if output and output.lower() != "none":
            return output
    return ""

