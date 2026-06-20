from __future__ import annotations

from exact.type2.schemas import Extraction, PhysicsIR, Type2Request


def build_physics_ir(request: Type2Request, extraction: Extraction) -> PhysicsIR:
    target = extraction.target
    if target is None:
        raise ValueError("Type2 extraction must define a target")
    knowns = {quantity.name: quantity for quantity in extraction.quantities}
    flags = {
        "requires_vector": bool(extraction.geometry.vector_required or target.wants_direction or extraction.vector_contribution_groups),
        "requires_network": extraction.physics_domain == "circuits" and any(token in extraction.question_text.lower() for token in ("parallel", "series")),
        "requires_formula_plan": extraction.question_kind != "conceptual",
        "requires_conceptual_reasoning": extraction.question_kind == "conceptual",
        "has_geometry": extraction.geometry.layout != "none",
        "has_multiple_sources": len(knowns) > 2,
        "has_angle": bool(extraction.geometry.angles_degrees),
        "has_missing_values": len(knowns) == 0,
    }
    return PhysicsIR(
        query_id=request.query_id,
        question_text=request.problem_text,
        question_kind=extraction.question_kind,
        physics_domain=extraction.physics_domain,
        knowns=knowns,
        target=target,
        geometry=extraction.geometry,
        relations=extraction.relations,
        vector_contribution_groups=extraction.vector_contribution_groups,
        flags=flags,
        notes=list(extraction.assumptions) + list(extraction.warnings),
    )
