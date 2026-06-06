from __future__ import annotations

from exact.type2.measurement.schemas import (
    MeasuredQuantity,
    MeasurementContract,
    MeasurementEvidence,
    MeasurementQuantity,
    MeasurementTarget,
    Uncertainty,
)
from exact.type2.schemas import Extraction


def extract_measurement_contract(extraction: Extraction) -> MeasurementContract | None:
    text = extraction.normalized_question.lower()
    if "least count" in text:
        return _least_count(extraction)
    if any(marker in text for marker in ("true value", "actual value", "accepted value")) and "measured" in text:
        return _true_vs_measured(extraction)
    if "±" in extraction.normalized_question or "+/-" in text or "absolute error" in text:
        if any(marker in text for marker in ("r =", "resistance", "p =", "power", "percentage error")) and {"voltage", "current"} <= set(extraction.quantities):
            return _circuit_lab(extraction)
        return _direct_uncertainty(extraction)
    return None


def _least_count(extraction: Extraction) -> MeasurementContract:
    lc = extraction.quantities.get("least_count") or extraction.quantities.get("length") or next(iter(extraction.quantities.values()))
    text = extraction.normalized_question.lower()
    rule = "half" if "half" in text else "full" if "equals least count" in text or "absolute error equals" in text else None
    return MeasurementContract(
        system_type="least_count_error",
        target=MeasurementTarget(("absolute_error",), extraction.target or "measurement", str(lc.value.units)),
        instrument={"least_count": MeasurementQuantity(float(lc.value.magnitude), str(lc.value.units), lc.evidence)},
        error_policy={"least_count_rule": rule} if rule else {},
        rounding_policy={"mode": "none"},
        parse_confidence=0.65,
        evidence=(MeasurementEvidence("least count", {"system_type": "least_count_error"}),),
    )


def _true_vs_measured(extraction: Extraction) -> MeasurementContract | None:
    true_q = extraction.quantities.get("true_value") or extraction.quantities.get("actual_value")
    measured_q = extraction.quantities.get("measured_value")
    if true_q is None or measured_q is None:
        return None
    return MeasurementContract(
        system_type="true_vs_measured_error",
        target=MeasurementTarget(("absolute_error", "relative_error", "percentage_error"), extraction.target or "measurement"),
        true_value=MeasurementQuantity(float(true_q.value.magnitude), str(true_q.value.units), true_q.evidence),
        measured_value=MeasurementQuantity(float(measured_q.value.magnitude), str(measured_q.value.units), measured_q.evidence),
        error_model={"method": "true_vs_measured", "denominator_policy": "true_value"},
        rounding_policy={"mode": "none"},
        parse_confidence=0.7,
    )


def _direct_uncertainty(extraction: Extraction) -> MeasurementContract | None:
    measured = _measured_with_uncertainty(extraction)
    if measured is None:
        return None
    return MeasurementContract(
        system_type="direct_uncertainty",
        target=MeasurementTarget(("relative_error", "percentage_error"), extraction.target or "measurement"),
        measured_quantities={"x": measured},
        error_model={"method": "direct", "denominator_policy": "measured_value"},
        rounding_policy={"mode": "none"},
        parse_confidence=0.65,
    )


def _circuit_lab(extraction: Extraction) -> MeasurementContract | None:
    voltage = extraction.quantities.get("voltage")
    current = extraction.quantities.get("current")
    du = extraction.quantities.get("voltage_uncertainty") or extraction.quantities.get("uncertainty")
    di = extraction.quantities.get("current_uncertainty") or extraction.quantities.get("uncertainty_2")
    if voltage is None or current is None or du is None or di is None:
        return None
    target_of = "power" if (extraction.target or "").lower() == "power" else "resistance"
    ast = {"op": "mul", "args": ["U", "I"]} if target_of == "power" else {"op": "div", "args": ["U", "I"]}
    unit = "W" if target_of == "power" else "ohm"
    return MeasurementContract(
        system_type="circuit_lab_propagation",
        target=MeasurementTarget(("value", "absolute_error", "relative_error", "percentage_error", "result_with_uncertainty"), target_of, unit),
        measured_quantities={
            "voltage": MeasuredQuantity("U", float(voltage.value.magnitude), str(voltage.value.units), voltage.evidence, Uncertainty(float(du.value.magnitude), str(du.value.units), "given")),
            "current": MeasuredQuantity("I", float(current.value.magnitude), str(current.value.units), current.evidence, Uncertainty(float(di.value.magnitude), str(di.value.units), "given")),
        },
        derived_quantity={"id": target_of, "symbol": "P" if target_of == "power" else "R", "formula": "P = U * I" if target_of == "power" else "R = U / I", "formula_ast": ast, "unit": unit},
        error_model={"method": "first_order_relative_propagation", "denominator_policy": "measured_value"},
        rounding_policy={"mode": "none"},
        parse_confidence=0.72,
    )


def _measured_with_uncertainty(extraction: Extraction) -> MeasuredQuantity | None:
    value_q = extraction.quantities.get("length") or extraction.quantities.get("value") or extraction.quantities.get("measurement")
    uncertainty_q = extraction.quantities.get("absolute_error") or extraction.quantities.get("uncertainty")
    if value_q is None or uncertainty_q is None:
        return None
    return MeasuredQuantity(
        "x",
        float(value_q.value.magnitude),
        str(value_q.value.units),
        value_q.evidence,
        Uncertainty(float(uncertainty_q.value.magnitude), str(uncertainty_q.value.units), "given"),
    )

