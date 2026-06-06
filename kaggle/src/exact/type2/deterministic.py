from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Callable

import pint

from exact.type2.capacitor_contract import (
    _contract_solver_should_own,
    _solve_validated_capacitor,
    extract_capacitor_contract,
    validate_capacitor_contract,
)
from exact.type2.circuits.parser import extract_circuit_contract
from exact.type2.circuits.router import solve_circuit_contract
from exact.type2.circuits.validator import validate_contract as validate_circuit_contract
from exact.type2.contract.validate_contract import validate_contract as validate_physics_contract
from exact.type2.electromagnetism.parser import extract_electromagnetism_contract
from exact.type2.electromagnetism.router import solve_electromagnetism_contract
from exact.type2.electromagnetism.validator import validate_contract as validate_em_contract
from exact.type2.geometry.coordinate_builder import build_contract_coordinates
from exact.type2.llm_parser.extract_contract import extract_contract as extract_physics_contract
from exact.type2.measurement.parser import extract_measurement_contract
from exact.type2.measurement.router import solve_measurement_contract
from exact.type2.measurement.validator import validate_contract as validate_measurement_contract
from exact.type2.physics.electrostatics_vector_solver import solve_electrostatics_contract
from exact.type2.schemas import Extraction, Type2SolveResult, Verification


@dataclass(frozen=True)
class CandidateSolverDiagnostics:
    solver: str
    eligible: bool
    confidence: float
    requirements_met: tuple[str, ...]
    missing: tuple[str, ...]
    reasons: tuple[str, ...]
    unsupported_cases: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeterministicRoute:
    domain: str | None
    system_type: str | None
    candidate_solvers: tuple[CandidateSolverDiagnostics, ...]
    selected_solver: str | None
    failure_reason: str | None
    missing: tuple[str, ...]


@dataclass(frozen=True)
class DeterministicStageResult:
    result: Type2SolveResult | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _DomainAdapter:
    name: str
    solver: str
    extractor: Callable[[Extraction], Any | None]
    validator: Callable[[Any], tuple[Any | None, Any | None]]
    runner: Callable[[Any, Any], dict[str, Any]]
    supported_system_types: tuple[str, ...]
    supported_targets: tuple[str, ...]
    unsupported_cases: tuple[str, ...] = ()


def run_deterministic_stage(extraction: Extraction) -> DeterministicStageResult:
    """Run deterministic routing and conditional solving before LLM PoT fallback."""

    routes: list[DeterministicRoute] = []
    contracts: list[dict[str, Any]] = []
    for adapter in _adapters():
        contract = adapter.extractor(extraction)
        if contract is None:
            continue
        contracts.append(_contract_summary(contract))
        route, validated = _route_contract(adapter, contract)
        routes.append(route)
        if route.selected_solver is None or validated is None:
            continue

        solver_output = _normalize_solver_output(adapter, adapter.runner(contract, validated))
        failure = _solver_failure(adapter.solver, solver_output)
        if failure is not None:
            route_dict = _route_dict(route)
            route_dict["solver_result"] = failure
            return DeterministicStageResult(
                result=None,
                diagnostics=_stage_diagnostics(routes, contracts, route_dict),
            )

        validation = _validate_answer(adapter, contract, solver_output)
        route_dict = _route_dict(route)
        route_dict["solver_result"] = solver_output
        route_dict["answer_validation"] = validation
        if not validation["valid"]:
            route_dict["deterministic_failure"] = _standard_failure(
                adapter.solver,
                validation["reason"],
                validation.get("missing", ()),
            )
            return DeterministicStageResult(
                result=None,
                diagnostics=_stage_diagnostics(routes, contracts, route_dict),
            )

        result = _type2_result_from_solver_output(extraction, adapter, solver_output, validation)
        diagnostics = _stage_diagnostics(routes, contracts, route_dict)
        return DeterministicStageResult(
            result=replace(result, routing_diagnostics=diagnostics),
            diagnostics=diagnostics,
        )

    if routes:
        return DeterministicStageResult(
            result=None,
            diagnostics=_stage_diagnostics(routes, contracts, None),
        )
    return DeterministicStageResult(
        result=None,
        diagnostics={
            "stage": "deterministic_routing",
            "status": "unsolved",
            "routes": [],
            "contracts": contracts,
            "failure": _standard_failure(
                "deterministic_router",
                "no formal contract could be extracted",
                ("formal_contract",),
            ),
        },
    )


def merge_routing_diagnostics(
    deterministic: dict[str, Any],
    fallback: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(deterministic)
    if fallback is not None:
        merged["fallback"] = fallback
    return merged


def _adapters() -> tuple[_DomainAdapter, ...]:
    return (
        _DomainAdapter(
            name="electrostatics",
            solver="deterministic_electrostatics_vector_solver",
            extractor=extract_physics_contract,
            validator=validate_physics_contract,
            runner=_run_electrostatics,
            supported_system_types=("multi_charge_vector_field",),
            supported_targets=("electric_field", "electric_force", "electric_potential", "zero_electric_field_location"),
            unsupported_cases=("unresolved coordinates", "unsupported symbolic vector outputs"),
        ),
        _DomainAdapter(
            name="circuits",
            solver="deterministic_circuit_solver",
            extractor=extract_circuit_contract,
            validator=validate_circuit_contract,
            runner=lambda contract, _validated: solve_circuit_contract(contract),
            supported_system_types=(
                "single_resistor",
                "dc_resistor_network",
                "energy_consumption",
                "series_ac_circuit",
                "series_rlc_circuit",
                "ideal_transformer",
            ),
            supported_targets=(
                "current",
                "voltage",
                "resistance",
                "power",
                "component_current",
                "component_voltage",
                "component_power",
                "total_current",
                "total_power",
                "equivalent_resistance",
                "electrical_energy",
                "inductive_reactance",
                "capacitive_reactance",
                "impedance_magnitude",
                "phase_angle",
                "power_factor",
                "current_rms",
                "active_power",
                "secondary_voltage",
                "primary_voltage",
                "secondary_current",
                "primary_current",
                "transformer_type",
            ),
        ),
        _DomainAdapter(
            name="capacitors",
            solver="deterministic_capacitor_solver",
            extractor=extract_capacitor_contract,
            validator=validate_capacitor_contract,
            runner=_run_capacitor,
            supported_system_types=(
                "parallel_plate_capacitor",
                "single_capacitor",
                "series_capacitors",
                "parallel_capacitors",
                "connected_precharged_capacitors",
            ),
            supported_targets=(
                "capacitance",
                "charge",
                "stored_energy",
                "relative_permittivity",
                "maximum_charge_before_breakdown",
                "breakdown_voltage",
                "final_voltage",
                "voltage_across_capacitor",
                "equivalent_capacitance",
            ),
            unsupported_cases=("ambiguous connected-capacitor polarity",),
        ),
        _DomainAdapter(
            name="electromagnetism",
            solver="deterministic_electromagnetism_solver",
            extractor=extract_electromagnetism_contract,
            validator=validate_em_contract,
            runner=lambda contract, _validated: solve_electromagnetism_contract(contract),
            supported_system_types=(
                "ideal_lc_oscillator",
                "series_rlc_circuit",
                "ac_reactance",
                "long_solenoid",
                "electromagnetic_induction",
                "ideal_transformer",
            ),
            supported_targets=("*",),
            unsupported_cases=("direction targets without orientation convention",),
        ),
        _DomainAdapter(
            name="measurement_error",
            solver="deterministic_measurement_error_solver",
            extractor=extract_measurement_contract,
            validator=validate_measurement_contract,
            runner=lambda contract, _validated: solve_measurement_contract(contract),
            supported_system_types=(
                "least_count_error",
                "true_vs_measured_error",
                "direct_uncertainty",
                "repeated_measurement",
                "propagation",
                "circuit_lab_propagation",
            ),
            supported_targets=(
                "absolute_error",
                "relative_error",
                "percentage_error",
                "value",
                "result_with_uncertainty",
            ),
            unsupported_cases=("missing least-count convention", "missing denominator policy"),
        ),
    )


def _route_contract(adapter: _DomainAdapter, contract: Any) -> tuple[DeterministicRoute, Any | None]:
    domain = getattr(contract, "domain", None)
    system_type = getattr(contract, "system_type", None)
    target = getattr(contract, "target", None)
    target_quantity = getattr(target, "quantity", None)
    if target_quantity is None and hasattr(target, "quantities"):
        target_quantity = ",".join(getattr(target, "quantities") or ())

    requirements_met: list[str] = []
    missing: list[str] = []
    reasons: list[str] = []
    validated = None

    if domain != adapter.name:
        missing.append("domain")
        reasons.append(f"domain `{domain}` is not supported by {adapter.solver}")
    else:
        requirements_met.append("domain")
    if system_type not in adapter.supported_system_types:
        missing.append("system_type")
        reasons.append(f"system_type `{system_type}` is unsupported")
    else:
        requirements_met.append("system_type")
    if adapter.supported_targets != ("*",) and not _target_supported(target_quantity, adapter.supported_targets):
        missing.append("target.quantity")
        reasons.append(f"target quantity `{target_quantity}` is unsupported")
    else:
        requirements_met.append("target.quantity")

    if not reasons:
        validated, issue = adapter.validator(contract)
        if issue is not None or validated is None:
            issue_missing = tuple(getattr(issue, "missing", ()) or ())
            missing.extend(issue_missing)
            reasons.append(getattr(issue, "reason", "contract validation failed"))
        else:
            requirements_met.extend(("contract_validation", "units_normalized", "required_fields"))

    eligible = not reasons and validated is not None
    candidate = CandidateSolverDiagnostics(
        solver=adapter.solver,
        eligible=eligible,
        confidence=0.95 if eligible else 0.0,
        requirements_met=tuple(dict.fromkeys(requirements_met)),
        missing=tuple(dict.fromkeys(missing)),
        reasons=tuple(reasons),
        unsupported_cases=adapter.unsupported_cases,
    )
    route = DeterministicRoute(
        domain=domain,
        system_type=system_type,
        candidate_solvers=(candidate,),
        selected_solver=adapter.solver if eligible else None,
        failure_reason=None if eligible else "; ".join(reasons),
        missing=candidate.missing,
    )
    return route, validated


def _run_electrostatics(_contract: Any, validated: Any) -> dict[str, Any]:
    coordinates = build_contract_coordinates(validated)
    if coordinates is None:
        return _standard_failure(
            "deterministic_electrostatics_vector_solver",
            "geometry constraints could not resolve coordinates",
            ("geometry.constraints", "coordinates"),
        )
    solved = solve_electrostatics_contract(validated, coordinates)
    if solved.status != "solved":
        return _standard_failure(
            "deterministic_electrostatics_vector_solver",
            solved.reason or "electrostatics solver returned unsolved",
            solved.missing,
        )
    return {
        "solver": "deterministic_electrostatics_vector_solver",
        "status": "solved",
        "answer": solved.answer,
        "unit": solved.unit,
        "value": solved.value,
        "diagnostics": solved.diagnostics,
    }


def _run_capacitor(contract: Any, validated: Any) -> dict[str, Any]:
    if not _contract_solver_should_own(contract):
        return _standard_failure(
            "deterministic_capacitor_solver",
            "capacitor contract is outside deterministic ownership policy",
            ("supported_target",),
        )
    solved = _solve_validated_capacitor(validated)
    if solved is None:
        return _standard_failure(
            "deterministic_capacitor_solver",
            "no deterministic capacitor rule matched",
            ("solver_rule",),
        )
    answer, unit, value, diagnostics = solved
    return {
        "solver": "deterministic_capacitor_solver",
        "status": "solved",
        "answer": answer,
        "unit": unit,
        "value": value,
        "diagnostics": diagnostics,
    }


def _solver_failure(solver: str, output: dict[str, Any]) -> dict[str, Any] | None:
    if output.get("status") == "solved":
        return None
    return _standard_failure(
        solver,
        str(output.get("reason") or "deterministic solver returned unsolved"),
        tuple(output.get("missing") or ()),
    )


def _normalize_solver_output(adapter: _DomainAdapter, output: dict[str, Any]) -> dict[str, Any]:
    if output.get("status") != "solved" or "answer" in output:
        return output
    result = output.get("result")
    if not isinstance(result, dict) or not result:
        return output

    item = result if "value" in result and "unit" in result else result[_primary_result_key(result)]
    normalized = dict(output)
    normalized["answer"] = _answer_from_result_item(item)
    normalized["unit"] = _unit_from_result_item(item)
    normalized["value"] = None
    normalized["diagnostics"] = {
        "solver": adapter.solver,
        "selected_rule": output.get("selected_rule"),
        "result": result,
        "normalized_inputs": output.get("normalized_inputs", {}),
        "intermediate_values": output.get("intermediate_values", {}),
    }
    return normalized


def _primary_result_key(result: dict[str, Any]) -> str:
    if "value" in result:
        return "value"
    return next(iter(result))


def _answer_from_result_item(item: Any) -> str:
    if isinstance(item, dict):
        if "text" in item:
            return str(item["text"])
        if "value" in item:
            return _format_number(float(item["value"]))
    return str(item)


def _unit_from_result_item(item: Any) -> str | None:
    if isinstance(item, dict):
        unit = item.get("unit")
        if unit in {"dimensionless", "categorical", "boolean", "conceptual"}:
            return None
        return unit
    return None


def _validate_answer(adapter: _DomainAdapter, contract: Any, output: dict[str, Any]) -> dict[str, Any]:
    target = getattr(contract, "target", None)
    expected_unit = _expected_unit(target)
    actual_unit = output.get("unit")
    if output.get("status") != "solved":
        return {"valid": False, "reason": "solver status is not solved", "missing": ()}
    if not output.get("answer") and output.get("value") is None:
        return {"valid": False, "reason": "solver produced no answer", "missing": ("answer",)}
    if expected_unit is not None and actual_unit is not None and not _units_compatible(actual_unit, expected_unit):
        return {
            "valid": False,
            "reason": f"output unit `{actual_unit}` does not match target unit `{expected_unit}`",
            "missing": ("output.unit",),
        }
    if expected_unit is not None and actual_unit is None and expected_unit not in {"dimensionless", "categorical", "boolean", "conceptual"}:
        return {"valid": False, "reason": "numeric target has no output unit", "missing": ("output.unit",)}
    if _direction_requested(target) and not _answer_has_direction(output):
        return {"valid": False, "reason": "direction was requested but not returned", "missing": ("direction",)}
    return {
        "valid": True,
        "reason": "answer matches contract target, unit, output type, and scope",
        "target_quantity": _target_quantity(target),
        "expected_unit": expected_unit,
        "actual_unit": actual_unit,
        "solver": adapter.solver,
    }


def _type2_result_from_solver_output(
    extraction: Extraction,
    adapter: _DomainAdapter,
    output: dict[str, Any],
    validation: dict[str, Any],
) -> Type2SolveResult:
    value = output.get("value")
    if not isinstance(value, pint.Quantity):
        value = None
    return Type2SolveResult(
        answer=str(output.get("answer") or ""),
        unit=output.get("unit"),
        value=value,
        formula=None,
        extraction=extraction,
        verification=Verification(True, validation["reason"]),
        cot=[
            "Extracted a formal contract before deterministic solving.",
            "Validated the contract fields, required units, and target scope.",
            f"Selected `{adapter.solver}` through deterministic routing.",
            "Validated the deterministic answer before accepting it.",
        ],
        premises=[f"diagnostics={_jsonable(output.get('diagnostics', {}))}"],
        confidence=0.94,
        error=None,
    )


def _stage_diagnostics(
    routes: list[DeterministicRoute],
    contracts: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> dict[str, Any]:
    status = (
        "solved"
        if selected
        and selected.get("solver_result", {}).get("status") == "solved"
        and selected.get("answer_validation", {}).get("valid") is True
        else "unsolved"
    )
    diagnostics = {
        "stage": "deterministic_routing",
        "status": status,
        "contracts": contracts,
        "routes": [_route_dict(route) for route in routes],
        "selected_route": selected,
    }
    if status != "solved":
        diagnostics["failure"] = _standard_failure(
            "deterministic_router",
            "no eligible deterministic solver produced a validated answer",
            tuple(item for route in routes for item in route.missing) or ("selected_solver",),
        )
    return _jsonable(diagnostics)


def _route_dict(route: DeterministicRoute) -> dict[str, Any]:
    return {
        "domain": route.domain,
        "system_type": route.system_type,
        "candidate_solvers": [asdict(candidate) for candidate in route.candidate_solvers],
        "selected_solver": route.selected_solver,
        "failure_reason": route.failure_reason,
        "missing": list(route.missing),
    }


def _standard_failure(solver: str, reason: str, missing: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    return {
        "solver": solver,
        "status": "unsolved",
        "reason": reason,
        "missing": list(missing),
        "fallback_recommended": True,
    }


def _contract_summary(contract: Any) -> dict[str, Any]:
    target = getattr(contract, "target", None)
    return {
        "domain": getattr(contract, "domain", None),
        "system_type": getattr(contract, "system_type", None),
        "target": _jsonable(target),
        "parse_confidence": getattr(contract, "parse_confidence", None),
        "unresolved": list(getattr(contract, "unresolved", ()) or ()),
    }


def _target_supported(target_quantity: str | None, supported: tuple[str, ...]) -> bool:
    if target_quantity is None:
        return False
    parts = target_quantity.split(",")
    return all(part in supported for part in parts)


def _target_quantity(target: Any) -> str | None:
    if target is None:
        return None
    if hasattr(target, "quantity"):
        return getattr(target, "quantity")
    if hasattr(target, "quantities"):
        return ",".join(getattr(target, "quantities") or ())
    return None


def _expected_unit(target: Any) -> str | None:
    if target is None:
        return None
    unit = getattr(target, "unit", None)
    if unit:
        return unit
    quantity = _target_quantity(target)
    defaults = {
        "electric_field": "N/C",
        "electric_force": "N",
        "electric_potential": "V",
        "charge": "C",
        "maximum_charge_before_breakdown": "C",
        "final_voltage": "V",
        "breakdown_voltage": "V",
        "capacitance": "F",
        "stored_energy": "J",
        "relative_permittivity": "dimensionless",
    }
    return defaults.get(quantity)


def _direction_requested(target: Any) -> bool:
    return getattr(target, "output", None) in {"direction", "magnitude_direction", "vector"}


def _answer_has_direction(output: dict[str, Any]) -> bool:
    diagnostics = output.get("diagnostics") or {}
    return bool(diagnostics.get("result_vector") or diagnostics.get("direction") or output.get("direction"))


def _units_compatible(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    compatible = {
        "N/C": {"N/C", "V/m"},
        "V/m": {"N/C", "V/m"},
        "dimensionless": {"dimensionless", None},
        "categorical": {"categorical", None},
        "boolean": {"boolean", None},
    }
    if actual in compatible and expected in compatible[actual]:
        return True
    return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, pint.Quantity):
        return {"value": float(value.magnitude), "unit": str(value.units)}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"
