from __future__ import annotations

from exact.type2.formulas.bank import FORMULAS
from exact.type2.schemas import FormulaEntry, FormulaRetrievalResult, PhysicsIR


def retrieve_formulas(ir: PhysicsIR, settings=None) -> FormulaRetrievalResult:
    candidates: list[tuple[int, FormulaEntry]] = []
    for formula in FORMULAS:
        domain_parts = list(formula.domain)
        entry = FormulaEntry(
            id=formula.id,
            name=formula.id,
            domain=domain_parts,
            expression=formula.expression,
            output=formula.output,
            required=list(formula.required),
            output_dimension=formula.output_dimension,
            output_unit=formula.output_unit,
            conditions=list(formula.conditions),
            explanation_template=formula.explanation_template,
            variables=dict(formula.variables),
            callable=formula.callable,
        )
        score = _score_formula(entry, ir)
        if score > 0:
            candidates.append((score, entry))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return FormulaRetrievalResult(
        retrieved=[entry for _, entry in candidates[: getattr(settings, "type2_formula_limit", 10)]],
        query_notes=[],
    )


def _score_formula(entry: FormulaEntry, ir: PhysicsIR) -> int:
    score = 0
    if ir.target.dimension and entry.output_dimension == ir.target.dimension:
        score += 5
    if ir.physics_domain and ir.physics_domain in entry.domain:
        score += 3
    overlap = len(set(entry.required) & set(ir.knowns))
    score += overlap * 2
    if ir.flags.get("requires_vector") and entry.is_vector_formula:
        score += 1
    if ir.flags.get("requires_network") and entry.is_network_formula:
        score += 1
    return score
