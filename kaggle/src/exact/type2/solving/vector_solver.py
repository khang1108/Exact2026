from __future__ import annotations

import re
from dataclasses import dataclass, replace

import pint

from exact.type2.contract.validate_contract import validate_contract
from exact.type2.geometry.coordinate_builder import build_contract_coordinates
from exact.type2.geometry_coordinates import CoordinateBuildResult, build_coordinates
from exact.type2.geometry_model import Body, GeometrySpec, TargetSpec
from exact.type2.geometry_extractor import extract_geometry_spec
from exact.type2.llm_parser.extract_contract import extract_contract
from exact.type2.physics.electrostatics_vector_solver import solve_electrostatics_contract
from exact.type2.schemas import Extraction, Type2SolveResult, Verification
from exact.type2.solving.units import ureg


K_COULOMB = 8.9875517923e9 * ureg.newton * ureg.meter**2 / ureg.coulomb**2
G_GRAVITATIONAL = 6.67430e-11 * ureg.newton * ureg.meter**2 / (ureg.kilogram**2)


@dataclass(frozen=True)
class VectorSolveState:
    spec: GeometrySpec
    coordinates: CoordinateBuildResult


def solve_geometry_vector_problem(extraction: Extraction) -> Type2SolveResult | None:
    symmetry_result = _solve_symbolic_symmetry_case(extraction)
    if symmetry_result is not None:
        return symmetry_result

    contract_result = _solve_from_contract(extraction)
    if contract_result is not None:
        return contract_result

    spec = extract_geometry_spec(extraction)
    if spec is None or spec.target_quantity not in {"force", "electric_field"}:
        return None
    if spec.target_quantity == "electric_field" and spec.target_body is None:
        target_point = _field_target_point(extraction, spec)
        if target_point is not None:
            virtual_target = Body(
                id="__field_target__",
                kind="charge",
                role="target",
                value=None,
                point=target_point,
                evidence=f"electric field at {target_point}",
            )
            spec = replace(
                spec,
                bodies={**spec.bodies, virtual_target.id: virtual_target},
                target=TargetSpec(body=virtual_target.id, quantity=spec.target_quantity, output=spec.target.output),
            )
    coordinates = build_coordinates(spec)
    if coordinates is None:
        return None
    state = VectorSolveState(spec=spec, coordinates=coordinates)
    if spec.target.output == "direction":
        return _solve_direction(extraction, state)
    if spec.target_quantity == "force":
        return _solve_net_force(extraction, state)
    if spec.target_quantity == "electric_field":
        return _solve_net_electric_field(extraction, state)
    return None


def _solve_from_contract(extraction: Extraction) -> Type2SolveResult | None:
    contract = extract_contract(extraction)
    if contract is None:
        return None
    scene, issue = validate_contract(contract)
    if issue is not None or scene is None:
        return None
    coordinates = build_contract_coordinates(scene)
    if coordinates is None:
        return None
    solved = solve_electrostatics_contract(scene, coordinates)
    if solved.status != "solved":
        return None
    quantity = scene.contract.target.quantity
    return Type2SolveResult(
        answer=solved.answer,
        unit=solved.unit,
        value=solved.value,
        formula=None,
        extraction=extraction,
        verification=Verification(True, f"Solved by validated contract graph solver `{coordinates.layout}`."),
        cot=[
            "Parsed a formal physics scene contract.",
            "Validated sources, target, units, references, and geometry constraints.",
            "Built coordinates from contract constraints without reading raw problem text.",
            "Summed signed electrostatic vectors deterministically.",
        ],
        premises=[
            f"contract_target={quantity}",
            f"layout={coordinates.layout}",
            f"diagnostics={solved.diagnostics}",
        ],
        confidence=0.95,
        error=None,
    )


def _solve_symbolic_symmetry_case(extraction: Extraction) -> Type2SolveResult | None:
    lower = extraction.normalized_question.lower()
    if extraction.target != "force":
        return None
    if not (
        "equal magnitude" in lower
        and "same sign" in lower
        and "midpoint" in lower
        and ("q3" in lower or "third point charge" in lower)
        and ("q1" in lower and "q2" in lower)
    ):
        return None

    return Type2SolveResult(
        answer="0",
        unit="N",
        value=0 * ureg.newton,
        formula=None,
        extraction=extraction,
        verification=Verification(True, "Solved by midpoint symmetry: equal same-sign source charges exert equal and opposite forces."),
        cot=[
            "Detected two equal same-sign source charges with the target charge at their midpoint.",
            "The two Coulomb force magnitudes on the midpoint charge are equal.",
            "The forces point in opposite directions along the same line, so the vector sum is zero.",
        ],
        premises=[
            "q1 and q2 have equal magnitude and the same sign.",
            "q3 is at the midpoint of the segment connecting q1 and q2.",
        ],
        confidence=0.94,
        error=None,
    )


def _solve_net_force(extraction: Extraction, state: VectorSolveState) -> Type2SolveResult | None:
    spec = state.spec
    if spec.target_body is None or spec.target_body not in spec.bodies:
        return None
    target = spec.bodies[spec.target_body]
    if target.point not in state.coordinates.coordinates:
        return None

    if target.kind == "charge":
        return _solve_coulomb_force(extraction, state, target)
    if target.kind == "mass":
        return _solve_gravitational_force(extraction, state, target)
    return None


def _solve_coulomb_force(
    extraction: Extraction,
    state: VectorSolveState,
    target: Body,
) -> Type2SolveResult | None:
    if target.value is None:
        return None
    target_x, target_y = state.coordinates.coordinates[target.point]
    net_fx = 0 * ureg.newton
    net_fy = 0 * ureg.newton
    cot = [
        "Extracted a GeometrySpec with charge nodes, target role, and edge constraints.",
        f"Built `{state.coordinates.layout}` coordinates deterministically.",
        f"Identified `{target.name}` at point {target.point} as the target charge.",
    ]

    source_count = 0
    expected_sources = _expected_source_count(state.spec, state.coordinates, target)
    for source in state.spec.bodies.values():
        if source.name == target.name or source.kind != "charge" or source.value is None:
            continue
        if source.point not in state.coordinates.coordinates:
            continue
        source_x, source_y = state.coordinates.coordinates[source.point]
        dx = target_x - source_x
        dy = target_y - source_y
        r = (dx**2 + dy**2) ** 0.5
        if float(r.to("m").magnitude) == 0:
            return None
        ux = dx / r
        uy = dy / r
        magnitude = K_COULOMB * abs(source.value.to("C") * target.value.to("C")) / (r**2)
        direction = 1 if source.value.to("C").magnitude * target.value.to("C").magnitude > 0 else -1
        net_fx += (direction * magnitude * ux).to("N")
        net_fy += (direction * magnitude * uy).to("N")
        source_count += 1
        cot.append(
            f"Added Coulomb force from {source.name}@{source.point}: "
            f"r={r.to('m')}, sign={'repulsive' if direction > 0 else 'attractive'}."
        )

    if source_count == 0:
        return None
    if source_count < expected_sources:
        return None
    magnitude = (net_fx**2 + net_fy**2) ** 0.5
    answer = _format_number(float(magnitude.to("N").magnitude))
    return Type2SolveResult(
        answer=answer,
        unit="N",
        value=magnitude.to("N"),
        formula=None,
        extraction=extraction,
        verification=_verify_vector_answer(
            target_quantity="force",
            unit="N",
            source_count=source_count,
            expected_sources=expected_sources,
            layout=state.coordinates.layout,
        ),
        cot=[*cot, "Summed vector components and returned the net force magnitude."],
        premises=[
            f"layout={state.coordinates.layout}",
            *[
                f"{body.name}@{body.point}={body.value.to('C')}"
                for body in state.spec.bodies.values()
                if body.kind == "charge" and body.value is not None
            ],
        ],
        confidence=0.93,
        error=None,
    )


def _solve_net_electric_field(extraction: Extraction, state: VectorSolveState) -> Type2SolveResult | None:
    spec = state.spec
    if spec.target_body is None or spec.target_body not in spec.bodies:
        return None
    target = spec.bodies[spec.target_body]
    if target.point is None or target.point not in state.coordinates.coordinates:
        return None

    target_x, target_y = state.coordinates.coordinates[target.point]
    net_ex = 0 * ureg.newton / ureg.coulomb
    net_ey = 0 * ureg.newton / ureg.coulomb
    cot = [
        "Extracted a GeometrySpec with source charges, field target point, and edge constraints.",
        f"Built `{state.coordinates.layout}` coordinates deterministically.",
        f"Identified point {target.point} as the electric-field target.",
    ]

    source_count = 0
    expected_sources = _expected_field_source_count(state.spec, state.coordinates, target.point)
    for source in state.spec.bodies.values():
        if source.name == target.name or source.kind != "charge" or source.value is None:
            continue
        if source.point is None or source.point not in state.coordinates.coordinates or source.point == target.point:
            continue
        source_x, source_y = state.coordinates.coordinates[source.point]
        dx = target_x - source_x
        dy = target_y - source_y
        r = (dx**2 + dy**2) ** 0.5
        if float(r.to("m").magnitude) == 0:
            return None
        magnitude = K_COULOMB * source.value.to("C") / (r**2)
        net_ex += (magnitude * dx / r).to("N/C")
        net_ey += (magnitude * dy / r).to("N/C")
        source_count += 1
        cot.append(f"Added electric field from {source.name}@{source.point}: r={r.to('m')}.")

    if source_count == 0:
        return None
    if source_count < expected_sources:
        return None
    magnitude = (net_ex**2 + net_ey**2) ** 0.5
    answer = _format_number(float(magnitude.to("N/C").magnitude))
    return Type2SolveResult(
        answer=answer,
        unit="N/C",
        value=magnitude.to("N/C"),
        formula=None,
        extraction=extraction,
        verification=_verify_vector_answer(
            target_quantity="electric_field",
            unit="N/C",
            source_count=source_count,
            expected_sources=expected_sources,
            layout=state.coordinates.layout,
        ),
        cot=[*cot, "Summed electric-field vector components and returned the net field magnitude."],
        premises=[
            f"layout={state.coordinates.layout}",
            *[
                f"{body.name}@{body.point}={body.value.to('C')}"
                for body in state.spec.bodies.values()
                if body.kind == "charge" and body.value is not None
            ],
        ],
        confidence=0.93,
        error=None,
    )


def _solve_gravitational_force(
    extraction: Extraction,
    state: VectorSolveState,
    target: Body,
) -> Type2SolveResult | None:
    if target.value is None:
        return None
    target_x, target_y = state.coordinates.coordinates[target.point]
    net_fx = 0 * ureg.newton
    net_fy = 0 * ureg.newton
    source_count = 0
    for source in state.spec.bodies.values():
        if source.name == target.name or source.kind != "mass" or source.value is None:
            continue
        if source.point not in state.coordinates.coordinates:
            continue
        source_x, source_y = state.coordinates.coordinates[source.point]
        dx = source_x - target_x
        dy = source_y - target_y
        r = (dx**2 + dy**2) ** 0.5
        if float(r.to("m").magnitude) == 0:
            return None
        magnitude = G_GRAVITATIONAL * target.value.to("kg") * source.value.to("kg") / (r**2)
        net_fx += (magnitude * dx / r).to("N")
        net_fy += (magnitude * dy / r).to("N")
        source_count += 1
    if source_count == 0:
        return None
    magnitude = (net_fx**2 + net_fy**2) ** 0.5
    return Type2SolveResult(
        answer=_format_number(float(magnitude.to("N").magnitude)),
        unit="N",
        value=magnitude.to("N"),
        formula=None,
        extraction=extraction,
        verification=Verification(True, f"Solved by gravitational geometry vector solver `{state.coordinates.layout}`."),
        cot=[
            "Extracted a GeometrySpec with mass nodes, target role, and edge constraints.",
            f"Built `{state.coordinates.layout}` coordinates deterministically.",
            "Summed gravitational vector components and returned the net force magnitude.",
        ],
        premises=[f"layout={state.coordinates.layout}"],
        confidence=0.9,
        error=None,
    )


def _solve_direction(extraction: Extraction, state: VectorSolveState) -> Type2SolveResult | None:
    spec = state.spec
    if spec.target_body is None or spec.target_body not in spec.bodies:
        return None
    target = spec.bodies[spec.target_body]
    if target.kind != "charge" or target.value is None or target.point not in state.coordinates.coordinates:
        return None

    target_x, target_y = state.coordinates.coordinates[target.point]
    net_fx = 0 * ureg.newton
    net_fy = 0 * ureg.newton
    for source in spec.bodies.values():
        if source.name == target.name or source.kind != "charge" or source.value is None:
            continue
        if source.point not in state.coordinates.coordinates:
            continue
        source_x, source_y = state.coordinates.coordinates[source.point]
        dx = target_x - source_x
        dy = target_y - source_y
        r = (dx**2 + dy**2) ** 0.5
        if float(r.to("m").magnitude) == 0:
            return None
        magnitude = K_COULOMB * abs(source.value.to("C") * target.value.to("C")) / (r**2)
        direction = 1 if source.value.to("C").magnitude * target.value.to("C").magnitude > 0 else -1
        net_fx += (direction * magnitude * dx / r).to("N")
        net_fy += (direction * magnitude * dy / r).to("N")

    answer = _nearest_body_direction(spec, state.coordinates, target, net_fx, net_fy)
    if answer is None:
        return None
    return Type2SolveResult(
        answer=answer,
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(True, "Solved vector direction from deterministic coordinates."),
        cot=[
            "Built coordinates from GeometrySpec.",
            "Summed Coulomb force components.",
            "Mapped the net vector direction to the nearest source body.",
        ],
        premises=[f"layout={state.coordinates.layout}"],
        confidence=0.86,
        error=None,
    )


def _nearest_body_direction(
    spec: GeometrySpec,
    coordinates: CoordinateBuildResult,
    target: Body,
    net_fx: pint.Quantity,
    net_fy: pint.Quantity,
) -> str | None:
    fx = float(net_fx.to("N").magnitude)
    fy = float(net_fy.to("N").magnitude)
    scale = (fx * fx + fy * fy) ** 0.5
    if scale <= 1e-12:
        return "no net direction"
    target_x, target_y = coordinates.coordinates[target.point]
    best: tuple[float, str] | None = None
    for source in spec.bodies.values():
        if source.id == target.id or source.point not in coordinates.coordinates:
            continue
        sx, sy = coordinates.coordinates[source.point]
        dx = float((sx - target_x).to("m").magnitude)
        dy = float((sy - target_y).to("m").magnitude)
        norm = (dx * dx + dy * dy) ** 0.5
        if norm <= 1e-12:
            continue
        score = (fx * dx + fy * dy) / (scale * norm)
        if best is None or score > best[0]:
            best = (score, source.id)
    if best is None:
        return None
    return f"toward {best[1]}"


def _verify_vector_answer(
    *,
    target_quantity: str,
    unit: str,
    source_count: int,
    expected_sources: int,
    layout: str,
) -> Verification:
    expected_unit = {"force": "N", "electric_field": "N/C"}.get(target_quantity)
    if expected_unit is None:
        return Verification(False, f"Vector solver returned unsupported target `{target_quantity}`.")
    if unit != expected_unit:
        return Verification(False, f"{target_quantity} result must use {expected_unit}, got `{unit}`.")
    if source_count < expected_sources:
        return Verification(
            False,
            f"Only included {source_count}/{expected_sources} source bodies in vector sum.",
        )
    return Verification(True, f"Solved by geometry vector solver `{layout}` with all source vectors included.")


def _expected_source_count(
    spec: GeometrySpec,
    coordinates: CoordinateBuildResult,
    target: Body,
) -> int:
    count = 0
    for body in spec.bodies.values():
        if body.id == target.id:
            continue
        if body.kind == target.kind and body.point in coordinates.coordinates:
            count += 1
    return max(1, count)


def _expected_field_source_count(
    spec: GeometrySpec,
    coordinates: CoordinateBuildResult,
    target_point: str,
) -> int:
    count = 0
    for body in spec.bodies.values():
        if body.kind != "charge" or body.value is None:
            continue
        if body.point is None or body.point == target_point:
            continue
        if body.point in coordinates.coordinates:
            count += 1
    return max(1, count)


def _field_target_point(extraction: Extraction, spec: GeometrySpec) -> str | None:
    text = extraction.normalized_question
    patterns = (
        r"(?:electric field(?: strength| intensity)?|field strength).*?\bat\s+(?:point\s+|vertex\s+)?(?P<point>[A-Z])\b",
        r"\bat\s+(?:point\s+|vertex\s+)?(?P<point>[A-Z])\b.*?(?:electric field(?: strength| intensity)?|field strength)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group("point").upper()

    occupied = {body.point for body in spec.bodies.values() if body.point}
    candidates = sorted(point for point in spec.points if point not in occupied)
    if len(candidates) == 1:
        return candidates[0]
    lower = text.lower()
    if "midpoint" in lower and "M" in spec.points:
        return "M"
    if ("center" in lower or "centre" in lower) and "O" in spec.points:
        return "O"
    return None


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
