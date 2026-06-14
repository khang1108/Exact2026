from __future__ import annotations

import math
from dataclasses import dataclass

from exact.type2.schemas import Verification
from exact.type2.solving.units import parse_quantity


OUTPUT_SANITY_ERROR = "type2_output_sanity_failed"
# Backward-compatible name for older call sites and reports.
POT_VERIFICATION_ERROR = OUTPUT_SANITY_ERROR


@dataclass(frozen=True)
class OutputSanityResult:
    verification: Verification
    answer: str
    unit: str | None
    value: object | None
    error: str | None = None


def verify_output_sanity(
    ans: object | None,
    ans_unit: str | None,
    formula_ids_used: list[str],
    allowed_formula_ids: tuple[str, ...],
    magnitude_target: bool = False,
) -> OutputSanityResult:
    if ans is None:
        return _reject("PoT execution did not produce ans.")

    try:
        magnitude = float(ans)
    except (TypeError, ValueError):
        return _reject(f"PoT ans `{ans}` is not numeric.")

    if not math.isfinite(magnitude):
        return _reject("PoT ans is not finite.")

    if magnitude_target:
        magnitude = abs(magnitude)

    unit = (ans_unit or "").strip() or None
    if unit is None:
        return _reject("PoT execution did not produce ans_unit.")

    try:
        value = parse_quantity(magnitude, unit)
        value.to_base_units()
    except Exception as exc:
        return _reject(f"PoT ans_unit `{unit}` is not parseable by Pint: {exc}")

    allowed = set(allowed_formula_ids)
    invented = [formula_id for formula_id in formula_ids_used if formula_id not in allowed]
    if invented:
        return _reject(f"PoT used formula IDs outside retrieved context: {invented}")

    return OutputSanityResult(
        verification=Verification(True, "Output sanity passed numeric, unit, and formula-id checks."),
        answer=_format_number(magnitude),
        unit=unit,
        value=value,
        error=None,
    )


def verify_pot_execution(
    ans: object | None,
    ans_unit: str | None,
    formula_ids_used: list[str],
    allowed_formula_ids: tuple[str, ...],
    magnitude_target: bool = False,
) -> OutputSanityResult:
    return verify_output_sanity(
        ans,
        ans_unit,
        formula_ids_used,
        allowed_formula_ids,
        magnitude_target=magnitude_target,
    )


def _reject(reason: str) -> OutputSanityResult:
    return OutputSanityResult(
        verification=Verification(False, reason),
        answer="",
        unit=None,
        value=None,
        error=OUTPUT_SANITY_ERROR,
    )


# Backward-compatible type alias.
PotVerificationResult = OutputSanityResult


def _format_number(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"
