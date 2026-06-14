from __future__ import annotations

from typing import Any

from exact.type2.contract.schemas import (
    ContractConstraint,
    ContractTarget,
    PhysicsSceneContract,
    ValidatedPhysicsScene,
)
from exact.type2.geometry.coordinate_builder import ContractCoordinateResult
from exact.type2.physics.electrostatics_vector_solver import (
    ElectrostaticsContractResult,
    solve_electrostatics_contract,
)
from exact.type2.solver_contract.models import SolverContract
from exact.type2.solving.units import ureg


def _as_meter_quantity(value: Any) -> Any | None:
    try:
        if hasattr(value, "to"):
            return value.to("m")
        return float(value) * ureg.meter
    except Exception:
        return None


def _normalize_coordinates(
    coords: dict[str, tuple[Any, Any]],
) -> tuple[dict[str, tuple[Any, Any]], bool]:
    normalized: dict[str, tuple[Any, Any]] = {}
    invalid = False
    for point, xy in coords.items():
        if xy is None or len(xy) != 2:
            invalid = True
            continue
        x, y = xy
        xq = _as_meter_quantity(x)
        yq = _as_meter_quantity(y)
        if xq is None or yq is None:
            invalid = True
            continue
        normalized[point] = (xq, yq)
    return normalized, invalid


def run_electrostatics_vector_solver(
    solver_contract: SolverContract,
    coords: dict[str, tuple[Any, Any]],
) -> ElectrostaticsContractResult:
    # 1. Map bodies
    charge_values = {}
    body_points = {}
    source_ids = []
    target_point = solver_contract.target.at or solver_contract.target.point
    target_body = solver_contract.target.body

    for b in solver_contract.bodies:
        if b.body_type == "charge" and b.value is not None:
            charge_values[b.id] = b.value
            if b.point:
                body_points[b.id] = b.point
            if b.role in {"source", "given"} or (b.role == "target" and solver_contract.target.quantity == "electric_force"):
                # If target is electric force, the target charge acts as a test charge but is not a source.
                if b.id != target_body:
                    source_ids.append(b.id)
            if b.role == "target" and not target_body:
                target_body = b.id
                if not target_point:
                    target_point = b.point

    # 2. Build legacy contract structures
    legacy_target = ContractTarget(
        quantity=solver_contract.target.quantity, # type: ignore
        at=target_point,
        point=target_point,
        body=target_body,
        output=solver_contract.target.output, # type: ignore
        unit=solver_contract.target.unit,
    )
    
    legacy_contract = PhysicsSceneContract(
        bodies=(), # Not deeply needed by the solver
        points=tuple(solver_contract.geometry.points.keys()),
        constraints=(),
        target=legacy_target,
        domain=solver_contract.domain,
        system_type="multi_charge_vector_field",
    )
    
    scene = ValidatedPhysicsScene(
        contract=legacy_contract,
        charge_values=charge_values,
        body_points=body_points,
        source_ids=tuple(source_ids),
        target_point=target_point,
        target_body=target_body,
    )
    
    normalized_coords, invalid_coords = _normalize_coordinates(coords)
    for x, y in normalized_coords.values():
        if not (hasattr(x, "to") and hasattr(y, "to")):
            invalid_coords = True
            break
    if invalid_coords:
        return ElectrostaticsContractResult(
            status="unsolved",
            answer="",
            unit=None,
            value=None,
            result_vector=None,
            contributions=(),
            diagnostics={
                "solver": "solver_contract_adapter",
                "status": "unsolved",
                "reason": "coordinate_unit_invalid",
                "fallback_recommended": True,
            },
            reason="coordinate_unit_invalid",
            missing=(),
        )

    # Coordinates format
    coord_result = ContractCoordinateResult(
        layout=solver_contract.geometry.family,
        coordinates=normalized_coords,
    )
    
    return solve_electrostatics_contract(scene, coord_result)
