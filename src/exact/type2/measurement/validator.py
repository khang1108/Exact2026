from __future__ import annotations

from typing import Any

from exact.type2.measurement.quantity_registry import SUPPORTED_AST_OPS, SUPPORTED_SYSTEM_TYPES, SUPPORTED_TARGETS
from exact.type2.measurement.schemas import MeasurementContract, MeasurementValidationIssue, ValidatedMeasurementContract


def validate_contract(contract: MeasurementContract) -> tuple[ValidatedMeasurementContract | None, MeasurementValidationIssue | None]:
    if contract.domain != "measurement_error":
        return None, MeasurementValidationIssue("domain is not measurement_error", ("domain",))
    if contract.system_type not in SUPPORTED_SYSTEM_TYPES:
        return None, MeasurementValidationIssue("system_type is unsupported", ("system_type",))
    if not contract.target.quantities:
        return None, MeasurementValidationIssue("target.quantities is empty", ("target.quantities",))
    if not contract.target.of:
        return None, MeasurementValidationIssue("target.of is missing", ("target.of",))
    unsupported = [q for q in contract.target.quantities if q not in SUPPORTED_TARGETS]
    if unsupported:
        return None, MeasurementValidationIssue("multi-answer target includes unsupported quantity", tuple(unsupported))
    if contract.system_type == "least_count_error":
        if "least_count" not in contract.instrument:
            return None, MeasurementValidationIssue("least-count target is missing instrument least_count", ("instrument.least_count",))
        if contract.error_policy.get("least_count_rule") not in {"full", "half"}:
            return None, MeasurementValidationIssue(
                "least_count_rule is required to decide whether absolute error is LC or LC/2",
                ("error_policy.least_count_rule",),
            )
    if contract.system_type == "true_vs_measured_error":
        if contract.true_value is None or contract.measured_value is None:
            return None, MeasurementValidationIssue("true_vs_measured requires true_value and measured_value", ("true_value", "measured_value"))
    if contract.system_type == "direct_uncertainty":
        issue = _validate_uncertainties(contract)
        if issue:
            return None, issue
    if contract.system_type == "repeated_measurement":
        if not contract.measurements:
            return None, MeasurementValidationIssue("repeated_measurement requires measurements", ("measurements",))
        if not contract.error_model.get("mean_error_definition"):
            return None, MeasurementValidationIssue("mean_error_definition is missing", ("error_model.mean_error_definition",))
    if contract.system_type in {"propagation", "circuit_lab_propagation"}:
        if not contract.derived_quantity.get("formula_ast"):
            return None, MeasurementValidationIssue("formula/formula_ast is missing for propagation", ("derived_quantity.formula_ast",))
        issue = _validate_ast(contract.derived_quantity["formula_ast"], {q.symbol for q in contract.measured_quantities.values()})
        if issue:
            return None, issue
        issue = _validate_uncertainties(contract)
        if issue:
            return None, issue
    if any(q in contract.target.quantities for q in {"relative_error", "percentage_error"}):
        if contract.error_model.get("denominator_policy") not in {"measured_value", "true_value", "mean_value", "specified"}:
            return None, MeasurementValidationIssue("denominator_policy is missing for relative/percentage error", ("error_model.denominator_policy",))
    if contract.rounding_policy.get("mode") == "explicit" and not any(
        key in contract.rounding_policy for key in {"decimal_places", "significant_figures", "percentage_decimal_places"}
    ):
        return None, MeasurementValidationIssue("rounding policy is explicit but malformed", ("rounding_policy",))
    return ValidatedMeasurementContract(contract), None


def _validate_uncertainties(contract: MeasurementContract) -> MeasurementValidationIssue | None:
    needs_error = any(q in contract.target.quantities for q in {"absolute_error", "relative_error", "percentage_error", "result_with_uncertainty"})
    if not needs_error:
        return None
    for name, quantity in contract.measured_quantities.items():
        if quantity.absolute_uncertainty is None and quantity.relative_uncertainty is None and quantity.percentage_uncertainty is None:
            return MeasurementValidationIssue("uncertainty is missing when error target requires it", (f"measured_quantities.{name}.absolute_uncertainty",))
    return None


def _validate_ast(ast: Any, symbols: set[str]) -> MeasurementValidationIssue | None:
    if isinstance(ast, str):
        if ast not in symbols:
            return MeasurementValidationIssue(f"formula variable {ast} is missing from measured_quantities", (f"measured_quantities.{ast}",))
        return None
    if isinstance(ast, (int, float)):
        return None
    op = ast.get("op")
    if op not in SUPPORTED_AST_OPS:
        return MeasurementValidationIssue("formula contains unsupported operations", (f"derived_quantity.formula_ast.{op}",))
    for arg in ast.get("args", ()):
        issue = _validate_ast(arg, symbols)
        if issue:
            return issue
    return None

