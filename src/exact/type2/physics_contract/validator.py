from __future__ import annotations

from exact.type2.physics_contract.dimensions import expected_unit_for, units_compatible
from exact.type2.physics_contract.models import PhysicsContract, PhysicsContractValidation


def validate_physics_contract(contract: PhysicsContract) -> PhysicsContractValidation:
    errors: list[str] = []
    warnings: list[str] = []

    if not contract.target or contract.target == "unknown":
        errors.append("missing_target")
    if not contract.expected_dimension or contract.expected_dimension == "unknown":
        warnings.append("missing_expected_dimension")

    canonical_unit = expected_unit_for(contract.expected_dimension)
    if canonical_unit and contract.expected_unit and not units_compatible(contract.expected_unit, canonical_unit):
        errors.append(
            f"expected_unit_incompatible:{contract.expected_unit}:expected:{canonical_unit}"
        )
    if canonical_unit and not contract.expected_unit:
        warnings.append(f"missing_expected_unit:{canonical_unit}")

    if contract.target in contract.knowns:
        warnings.append("target_already_present_in_knowns")
    if not contract.knowns and contract.target != "unknown":
        warnings.append("no_known_quantities")

    if contract.principle == "field_superposition" and contract.target not in {"electric_field", "field_strength"}:
        errors.append("principle_target_mismatch:field_superposition")
    if contract.principle == "capacitor_energy" and contract.target not in {"energy", "stored_energy"}:
        errors.append("principle_target_mismatch:capacitor_energy")

    confidence_delta = 0.05 if not errors and contract.expected_unit else 0.0
    confidence_delta -= 0.05 * len(warnings)
    return PhysicsContractValidation(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        confidence_delta=confidence_delta,
    )

