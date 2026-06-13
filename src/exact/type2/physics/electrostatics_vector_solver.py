from __future__ import annotations

import math
from dataclasses import dataclass

import pint

from exact.type2.contract.schemas import ValidatedPhysicsScene
from exact.type2.geometry.coordinate_builder import ContractCoordinateResult
from exact.type2.solving.units import ureg


K_COULOMB = 8.9875517923e9 * ureg.newton * ureg.meter**2 / ureg.coulomb**2


@dataclass(frozen=True)
class VectorContribution:
    source: str
    vector: tuple[pint.Quantity, pint.Quantity]


@dataclass(frozen=True)
class ElectrostaticsContractResult:
    status: str
    answer: str
    unit: str | None
    value: pint.Quantity | None
    result_vector: tuple[pint.Quantity, pint.Quantity] | None
    contributions: tuple[VectorContribution, ...]
    diagnostics: dict
    reason: str | None = None
    missing: tuple[str, ...] = ()


def solve_electrostatics_contract(
    scene: ValidatedPhysicsScene,
    coordinates: ContractCoordinateResult,
) -> ElectrostaticsContractResult:
    target = scene.contract.target
    if target.quantity == "electric_field":
        return _solve_electric_field(scene, coordinates)
    if target.quantity == "electric_force":
        return _solve_electric_force(scene, coordinates)
    if target.quantity == "electric_potential":
        return _solve_electric_potential(scene, coordinates)
    if target.quantity == "zero_electric_field_location":
        return _solve_zero_field_location(scene, coordinates)
    return ElectrostaticsContractResult(
        status="unsolved",
        answer="",
        unit=None,
        value=None,
        result_vector=None,
        contributions=(),
        diagnostics={"solver": "deterministic_graph_electrostatics", "status": "unsolved"},
        reason=f"unsupported target quantity `{target.quantity}`",
    )


def _solve_electric_field(scene: ValidatedPhysicsScene, coordinates: ContractCoordinateResult) -> ElectrostaticsContractResult:
    target_point = scene.target_point
    if target_point is None or target_point not in coordinates.coordinates:
        return _missing_coordinates(target_point)
    target_x, target_y = coordinates.coordinates[target_point]
    net_x = 0 * ureg.newton / ureg.coulomb
    net_y = 0 * ureg.newton / ureg.coulomb
    contributions = []
    for source_id in scene.source_ids:
        source_point = scene.body_points[source_id]
        if source_point not in coordinates.coordinates:
            return _missing_coordinates(source_point)
        source_x, source_y = coordinates.coordinates[source_point]
        dx = target_x - source_x
        dy = target_y - source_y
        r = (dx**2 + dy**2) ** 0.5
        if float(r.to("m").magnitude) == 0:
            return ElectrostaticsContractResult(
                status="unsolved",
                answer="",
                unit=None,
                value=None,
                result_vector=None,
                contributions=(),
                diagnostics={"solver": "deterministic_graph_electrostatics", "status": "unsolved"},
                reason=f"source `{source_id}` occupies target point `{target_point}`",
                missing=(),
            )
        magnitude = K_COULOMB * scene.charge_values[source_id].to("C") / (r**2)
        vector = ((magnitude * dx / r).to("N/C"), (magnitude * dy / r).to("N/C"))
        net_x += vector[0]
        net_y += vector[1]
        contributions.append(VectorContribution(source_id, vector))
    return _solved(scene, coordinates, (net_x.to("N/C"), net_y.to("N/C")), tuple(contributions), "N/C")


def _solve_electric_force(scene: ValidatedPhysicsScene, coordinates: ContractCoordinateResult) -> ElectrostaticsContractResult:
    target_body = scene.target_body
    target_point = scene.target_point
    if target_body is None:
        return _unsolved("electric force target body cannot be resolved", ("target.body",))
    if target_point is None or target_point not in coordinates.coordinates:
        return _missing_coordinates(target_point)
    target_x, target_y = coordinates.coordinates[target_point]
    target_charge = scene.charge_values[target_body].to("C")
    net_x = 0 * ureg.newton
    net_y = 0 * ureg.newton
    contributions = []
    for source_id in scene.source_ids:
        if source_id == target_body:
            continue
        source_point = scene.body_points[source_id]
        if source_point not in coordinates.coordinates:
            return _missing_coordinates(source_point)
        source_x, source_y = coordinates.coordinates[source_point]
        dx = target_x - source_x
        dy = target_y - source_y
        r = (dx**2 + dy**2) ** 0.5
        if float(r.to("m").magnitude) == 0:
            return _unsolved(f"source `{source_id}` occupies target point `{target_point}`")
        magnitude = K_COULOMB * abs(scene.charge_values[source_id].to("C") * target_charge) / (r**2)
        direction = 1 if scene.charge_values[source_id].to("C").magnitude * target_charge.magnitude > 0 else -1
        vector = ((direction * magnitude * dx / r).to("N"), (direction * magnitude * dy / r).to("N"))
        net_x += vector[0]
        net_y += vector[1]
        contributions.append(VectorContribution(source_id, vector))
    return _solved(scene, coordinates, (net_x.to("N"), net_y.to("N")), tuple(contributions), "N")


def _solve_electric_potential(
    scene: ValidatedPhysicsScene,
    coordinates: ContractCoordinateResult,
) -> ElectrostaticsContractResult:
    target_point = scene.target_point
    if target_point is None or target_point not in coordinates.coordinates:
        return _missing_coordinates(target_point)
    target_x, target_y = coordinates.coordinates[target_point]
    potential = 0 * ureg.volt
    diagnostics_contributions = []
    for source_id in scene.source_ids:
        source_point = scene.body_points[source_id]
        if source_point not in coordinates.coordinates:
            return _missing_coordinates(source_point)
        source_x, source_y = coordinates.coordinates[source_point]
        dx = target_x - source_x
        dy = target_y - source_y
        r = (dx**2 + dy**2) ** 0.5
        if float(r.to("m").magnitude) == 0:
            return _unsolved(f"source `{source_id}` occupies target point `{target_point}`")
        contribution = (K_COULOMB * scene.charge_values[source_id].to("C") / r).to("V")
        potential += contribution
        diagnostics_contributions.append({"source": source_id, "potential": float(contribution.magnitude), "unit": "V"})
    answer = _format_number(float(potential.to("V").magnitude))
    diagnostics = _base_diagnostics(scene, coordinates) | {
        "selected_rule": "electric_potential_scalar_sum",
        "formula_chain": ["V(P) = Σ k q_i / |P - R_i|"],
        "scalar_contributions": diagnostics_contributions,
        "result": {"value": float(potential.to("V").magnitude), "unit": "V"},
    }
    return ElectrostaticsContractResult(
        status="solved",
        answer=answer,
        unit="V",
        value=potential.to("V"),
        result_vector=None,
        contributions=(),
        diagnostics=diagnostics,
    )


def _solve_zero_field_location(
    scene: ValidatedPhysicsScene,
    coordinates: ContractCoordinateResult,
) -> ElectrostaticsContractResult:
    target_point = scene.target_point or scene.contract.target.point
    line = _unknown_line(scene, target_point)
    if line is None:
        return _unsolved("zero-field target point is not constrained to a line", ("constraints.on_line",))
    left, right = line
    if left not in coordinates.coordinates or right not in coordinates.coordinates:
        return _unsolved("zero-field line endpoints have no resolved coordinates", (f"coordinates({left},{right})",))
    lx, ly = coordinates.coordinates[left]
    rx, ry = coordinates.coordinates[right]
    if abs(float(ly.to("m").magnitude - ry.to("m").magnitude)) > 1e-12:
        return _unsolved("zero-field location currently supports horizontal resolved lines only")
    roots = _solve_zero_field_roots_on_x_axis(scene, coordinates, float(lx.to("m").magnitude), float(rx.to("m").magnitude))
    if not roots:
        return _unsolved("no zero electric-field location found on declared line")
    x = _select_zero_field_root(roots, float(lx.to("m").magnitude), float(rx.to("m").magnitude))
    value = x * ureg.meter
    diagnostics = _base_diagnostics(scene, coordinates) | {
        "selected_rule": "zero_field_location_1d_vector_equation",
        "formula_chain": ["ΣE_x(x) = 0", "ΣE_y(x) = 0"],
        "unknown": target_point,
        "candidate_roots_m": roots,
        "result": {"position": x, "unit": "m", "relative_to": left},
    }
    return ElectrostaticsContractResult(
        status="solved",
        answer=_format_number(x),
        unit="m",
        value=value,
        result_vector=None,
        contributions=(),
        diagnostics=diagnostics,
    )


def _solved(
    scene: ValidatedPhysicsScene,
    coordinates: ContractCoordinateResult,
    vector: tuple[pint.Quantity, pint.Quantity],
    contributions: tuple[VectorContribution, ...],
    unit: str,
) -> ElectrostaticsContractResult:
    magnitude = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
    output = scene.contract.target.output
    answer = _format_number(float(magnitude.to(unit).magnitude))
    if output == "vector":
        answer = f"({_format_number(float(vector[0].to(unit).magnitude))}, {_format_number(float(vector[1].to(unit).magnitude))})"
    diagnostics = {
        **_base_diagnostics(scene, coordinates),
        "selected_rule": "electric_field_vector_sum" if unit == "N/C" else "electric_force_vector_sum",
        "formula_chain": [
            "E(P) = Σ k q_i (P - R_i) / |P - R_i|^3"
            if unit == "N/C"
            else "F(P) = q0 * E(P)",
        ],
        "status": "solved",
        "vector_contributions": [
            {
                "source": item.source,
                "field_vector" if unit == "N/C" else "force_vector": [
                    float(item.vector[0].to(unit).magnitude),
                    float(item.vector[1].to(unit).magnitude),
                ],
                "unit": unit,
            }
            for item in contributions
        ],
        "result": {
            "vector": [float(vector[0].to(unit).magnitude), float(vector[1].to(unit).magnitude)],
            "magnitude": float(magnitude.to(unit).magnitude),
            "direction": _direction_label(vector, unit),
            "unit": unit,
        },
    }
    return ElectrostaticsContractResult(
        status="solved",
        answer=answer,
        unit=unit,
        value=magnitude.to(unit),
        result_vector=vector,
        contributions=contributions,
        diagnostics=diagnostics,
    )


def _missing_coordinates(point: str | None) -> ElectrostaticsContractResult:
    missing = (f"coordinates({point})",) if point else ("target.coordinates",)
    return _unsolved(f"target point {point or '<unknown>'} has no resolved coordinates", missing)


def _unsolved(reason: str, missing: tuple[str, ...] = ()) -> ElectrostaticsContractResult:
    return ElectrostaticsContractResult(
        status="unsolved",
        answer="",
        unit=None,
        value=None,
        result_vector=None,
        contributions=(),
        diagnostics={
            "solver": "deterministic_graph_electrostatics",
            "status": "unsolved",
            "reason": reason,
            "missing": list(missing),
            "fallback_recommended": True,
        },
        reason=reason,
        missing=missing,
    )


def _base_diagnostics(scene: ValidatedPhysicsScene, coordinates: ContractCoordinateResult) -> dict:
    return {
        "solver": "symbolic_vector_electrostatics_solver",
        "status": "solved",
        "resolved_coordinates": {
            point: [float(x.to("m").magnitude), float(y.to("m").magnitude)]
            for point, (x, y) in coordinates.coordinates.items()
        },
        "layout": coordinates.layout,
        "contract_system_type": scene.contract.system_type,
    }


def _direction_label(vector: tuple[pint.Quantity, pint.Quantity], unit: str) -> str:
    x = float(vector[0].to(unit).magnitude)
    y = float(vector[1].to(unit).magnitude)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return "zero vector"
    angle = math.degrees(math.atan2(y, x))
    return f"{angle:.6g} degree from +x"


def _unknown_line(scene: ValidatedPhysicsScene, target_point: str | None) -> tuple[str, str] | None:
    if target_point is None:
        return None
    for constraint in scene.contract.constraints:
        if constraint.type == "on_line" and len(constraint.points) >= 3 and constraint.points[0] == target_point:
            return constraint.points[1], constraint.points[2]
    return None


def _solve_zero_field_roots_on_x_axis(
    scene: ValidatedPhysicsScene,
    coordinates: ContractCoordinateResult,
    left_x: float,
    right_x: float,
) -> list[float]:
    source_positions = []
    for source_id in scene.source_ids:
        source_point = scene.body_points[source_id]
        if source_point not in coordinates.coordinates:
            return []
        sx, sy = coordinates.coordinates[source_point]
        if abs(float(sy.to("m").magnitude)) > 1e-12:
            return []
        source_positions.append((float(sx.to("m").magnitude), float(scene.charge_values[source_id].to("C").magnitude)))
    if len(source_positions) < 2:
        return []

    def field_x(x: float) -> float:
        total = 0.0
        for sx, q in source_positions:
            dx = x - sx
            if abs(dx) < 1e-12:
                return math.nan
            total += float(K_COULOMB.to("N*m^2/C^2").magnitude) * q * dx / abs(dx) ** 3
        return total

    span = max(abs(right_x - left_x), 1.0)
    singularities = sorted({sx for sx, _ in source_positions})
    bounds = [min(left_x, right_x) - 10 * span, *singularities, max(left_x, right_x) + 10 * span]
    roots: list[float] = []
    for lo, hi in zip(bounds, bounds[1:]):
        eps = max(1e-9, span * 1e-9)
        a = lo + eps
        b = hi - eps
        if a >= b:
            continue
        samples = [a + (b - a) * i / 128 for i in range(129)]
        previous_x = None
        previous_y = None
        for x in samples:
            y = field_x(x)
            if not math.isfinite(y):
                previous_x = None
                previous_y = None
                continue
            if abs(y) < 1e-6:
                roots.append(x)
            if previous_x is not None and previous_y is not None and previous_y * y < 0:
                roots.append(_bisect(field_x, previous_x, x))
            previous_x = x
            previous_y = y
    unique: list[float] = []
    for root in roots:
        if all(abs(root - existing) > max(1e-6, span * 1e-6) for existing in unique):
            unique.append(root)
    return sorted(unique)


def _bisect(func, lo: float, hi: float) -> float:
    flo = func(lo)
    for _ in range(100):
        mid = (lo + hi) / 2
        fmid = func(mid)
        if abs(fmid) < 1e-9:
            return mid
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return (lo + hi) / 2


def _select_zero_field_root(roots: list[float], left_x: float, right_x: float) -> float:
    inside = [root for root in roots if min(left_x, right_x) <= root <= max(left_x, right_x)]
    if inside:
        return inside[0]
    return min(roots, key=lambda root: min(abs(root - left_x), abs(root - right_x)))


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
