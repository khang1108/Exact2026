#!/usr/bin/env python
"""Evaluate Type 2 prediction files that include gold_answer and gold_unit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exact.scripts.config_utils import load_toml_config
from exact.type2.solving.units import parse_quantity


DEFAULT_CONFIG = Path("configs/type2_dataset_run.example.toml")


@dataclass(frozen=True)
class EvalRow:
    id: str | None
    answer: str
    unit: str | None
    gold_answer: str | None
    gold_unit: str | None
    status: str
    numeric_ok: bool | None
    unit_ok: bool | None
    relative_error: float | None
    absolute_error: float | None
    error: str | None


def main() -> None:
    args = parse_args()
    config = load_toml_config(args.config) if args.config.exists() else {}
    eval_cfg = config.get("evaluation", {})

    relative_tolerance = float(args.relative_tolerance or eval_cfg.get("relative_tolerance", 0.02))
    absolute_tolerance = float(args.absolute_tolerance or eval_cfg.get("absolute_tolerance", 1e-9))
    case_sensitive_text = bool(eval_cfg.get("case_sensitive_text", False))
    report_path = Path(args.report or eval_cfg.get("report_path", "artifacts/reports/type2_eval_report.json"))
    errors_path = Path(args.errors or eval_cfg.get("errors_path", "artifacts/reports/type2_eval_errors.csv"))

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("Prediction file must contain a top-level predictions list")

    rows = [
        evaluate_prediction(
            pred,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            case_sensitive_text=case_sensitive_text,
        )
        for pred in predictions
    ]
    summary = summarize(rows, relative_tolerance, absolute_tolerance)

    report = {
        "source": str(args.predictions),
        "count": len(rows),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "summary": summary,
        "rows": [asdict(row) for row in rows],
        "routing_logs": _build_routing_logs(predictions, rows),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_errors_csv(errors_path, rows)

    print_summary(summary)
    print(f"wrote report to {report_path}")
    print(f"wrote error rows to {errors_path}")


def evaluate_prediction(
    pred: dict[str, Any],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    case_sensitive_text: bool,
) -> EvalRow:
    answer = str(pred.get("answer") or "").strip()
    unit = _clean_unit(pred.get("unit"))
    gold_answer = _clean_text(pred.get("gold_answer"))
    gold_unit = _clean_unit(pred.get("gold_unit"))
    runtime_error = _clean_text(pred.get("error"))
    if _is_conceptual_prediction(pred):
        status = "pipeline_error" if runtime_error else "conceptual_only"
        return _row(pred, answer, unit, gold_answer, gold_unit, status, None, None, None, None)

    if gold_answer is None or gold_answer == "":
        return _row(pred, answer, unit, gold_answer, gold_unit, "missing_gold", None, None, None, None)

    pred_number = parse_number(answer)
    gold_number = parse_number(gold_answer)
    if pred_number is not None and gold_number is not None:
        return evaluate_numeric(
            pred,
            answer,
            unit,
            gold_answer,
            gold_unit,
            pred_number,
            gold_number,
            relative_tolerance,
            absolute_tolerance,
            runtime_error,
        )

    normalized_answer = answer if case_sensitive_text else answer.lower()
    normalized_gold = gold_answer if case_sensitive_text else gold_answer.lower()
    ok = normalized_answer == normalized_gold
    status = "correct_text" if ok else "wrong_text"
    if runtime_error and not ok:
        status = "pipeline_error"
    return _row(pred, answer, unit, gold_answer, gold_unit, status, None, None, None, None)


def evaluate_numeric(
    pred: dict[str, Any],
    answer: str,
    unit: str | None,
    gold_answer: str,
    gold_unit: str | None,
    pred_number: float,
    gold_number: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    runtime_error: str | None,
) -> EvalRow:
    pred_value = pred_number
    gold_value = gold_number
    unit_ok: bool | None = None

    if unit and gold_unit:
        try:
            converted = parse_quantity(pred_number, unit).to(gold_unit)
            pred_value = float(converted.magnitude)
            unit_ok = True
        except Exception:
            unit_ok = False

    absolute_error = abs(pred_value - gold_value)
    denominator = max(abs(gold_value), absolute_tolerance)
    relative_error = absolute_error / denominator
    numeric_ok = (
        absolute_error <= absolute_tolerance
        or relative_error <= relative_tolerance
        or _rounded_to_gold_precision_ok(gold_answer, pred_value, gold_value)
    )
    ok = numeric_ok and unit_ok is not False

    if ok:
        status = "correct_numeric"
    elif runtime_error:
        status = "pipeline_error"
    elif unit_ok is False:
        status = "unit_mismatch"
    else:
        status = "wrong_numeric"

    return _row(
        pred,
        answer,
        unit,
        gold_answer,
        gold_unit,
        status,
        numeric_ok,
        unit_ok,
        relative_error,
        absolute_error,
    )


def summarize(
    rows: list[EvalRow],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row.status.startswith("correct") for row in rows)
    numeric_rows = [row for row in rows if row.numeric_ok is not None]
    numeric_correct = sum(row.numeric_ok and row.unit_ok is not False for row in numeric_rows)
    pipeline_errors = sum(row.status == "pipeline_error" for row in rows)
    conceptual_only = sum(row.status == "conceptual_only" for row in rows)
    unit_mismatches = sum(row.status == "unit_mismatch" for row in rows)
    missing_gold = sum(row.status == "missing_gold" for row in rows)
    scored_total = total - missing_gold - conceptual_only
    wrong = total - correct - missing_gold - conceptual_only

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "missing_gold": missing_gold,
        "conceptual_only": conceptual_only,
        "accuracy": _safe_ratio(correct, scored_total),
        "numeric_total": len(numeric_rows),
        "numeric_correct": numeric_correct,
        "numeric_accuracy": _safe_ratio(numeric_correct, len(numeric_rows)),
        "pipeline_errors": pipeline_errors,
        "unit_mismatches": unit_mismatches,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "by_status": _count_by_status(rows),
    }


def _build_routing_logs(predictions: list[dict[str, Any]], rows: list[EvalRow]) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for pred, row in zip(predictions, rows, strict=False):
        routing = pred.get("routing_log") or pred.get("routing_diagnostics")
        if not isinstance(routing, dict):
            continue
        log = dict(routing)
        log["correct"] = row.status.startswith("correct")
        log["evaluation_status"] = row.status
        logs.append(log)
    return logs


def write_errors_csv(path: Path, rows: list[EvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()) if rows else ["id"])
        writer.writeheader()
        for row in rows:
            if not row.status.startswith("correct"):
                writer.writerow(asdict(row))


def parse_number(text: str) -> float | None:
    normalized = _normalize_numeric_text(text)
    if not normalized:
        return None

    expression = _numeric_expression(normalized)
    if expression is not None:
        try:
            value = float(eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt}))
        except (ArithmeticError, NameError, SyntaxError, TypeError, ValueError):
            value = None
        if value is not None and math.isfinite(value):
            return value

    match = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?", normalized, re.IGNORECASE)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _normalize_numeric_text(text: str) -> str:
    normalized = (
        text.strip()
        .replace(",", ".")
        .replace("×", "x")
        .replace("Ã—", "x")
        .replace("−", "-")
        .replace("âˆ’", "-")
        .replace("\\sqrt", "sqrt")
        .replace("^", "**")
        .replace(" ", "")
    )
    normalized = re.sub(r"sqrt\{([^{}]+)\}", r"sqrt(\1)", normalized)
    normalized = re.sub(r"sqrt([0-9.]+)", r"sqrt(\1)", normalized)
    normalized = re.sub(r"(?<=\d)(?=sqrt\()", "*", normalized)
    return normalized


def _numeric_expression(normalized: str) -> str | None:
    expression = re.sub(
        r"x10(?:\*\*)?(?P<exp>[-+]?\d+)",
        r"*10**\g<exp>",
        normalized,
        flags=re.IGNORECASE,
    )
    expression = re.sub(
        r"(?P<base>(?:\d+(?:\.\d+)?|\.\d+))e(?P<exp>[-+]?\d+)",
        r"\g<base>*10**\g<exp>",
        expression,
        flags=re.IGNORECASE,
    )
    if not re.fullmatch(r"[-+*/().0-9sqrt]+", expression):
        return None
    if not re.search(r"\d", expression):
        return None
    return expression


def _rounded_to_gold_precision_ok(gold_answer: str, pred_value: float, gold_value: float) -> bool:
    """Accept values that round to the explicit decimal precision shown in gold."""
    normalized = (
        _normalize_numeric_text(gold_answer)
        .replace("Ã—", "x")
        .replace("âˆ’", "-")
    )
    match = re.search(
        r"[-+]?\d+\.(?P<decimals>\d+)(?P<scientific>x10(?:\*\*)?(?P<x_exp>[-+]?\d+)|e(?P<e_exp>[-+]?\d+))?",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return False
    decimal_places = len(match.group("decimals"))
    if match.group("scientific"):
        exponent = int(match.group("x_exp") or match.group("e_exp"))
        scale = 10**exponent
        return round(pred_value / scale, decimal_places) == round(gold_value / scale, decimal_places)
    return round(pred_value, decimal_places) == round(gold_value, decimal_places)


def print_summary(summary: dict[str, Any]) -> None:
    print(f"total: {summary['total']}")
    print(f"accuracy: {summary['accuracy']:.3f}")
    print(f"numeric_accuracy: {summary['numeric_accuracy']:.3f}")
    print(f"pipeline_errors: {summary['pipeline_errors']}")
    print(f"conceptual_only: {summary['conceptual_only']}")
    print(f"unit_mismatches: {summary['unit_mismatches']}")
    print(f"by_status: {summary['by_status']}")


def _row(
    pred: dict[str, Any],
    answer: str,
    unit: str | None,
    gold_answer: str | None,
    gold_unit: str | None,
    status: str,
    numeric_ok: bool | None,
    unit_ok: bool | None,
    relative_error: float | None,
    absolute_error: float | None,
) -> EvalRow:
    return EvalRow(
        id=pred.get("id"),
        answer=answer,
        unit=unit,
        gold_answer=gold_answer,
        gold_unit=gold_unit,
        status=status,
        numeric_ok=numeric_ok,
        unit_ok=unit_ok,
        relative_error=relative_error,
        absolute_error=absolute_error,
        error=pred.get("error"),
    )


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def _clean_unit(value: Any) -> str | None:
    text = _clean_text(value)
    if not text or text in {"-", "—"}:
        return None
    return text.replace("μ", "u")


def _is_conceptual_prediction(pred: dict[str, Any]) -> bool:
    question_type = _clean_text(pred.get("question_type"))
    type2_kind = _clean_text(pred.get("type2_kind"))
    if question_type == "open_ended" or type2_kind == "conceptual":
        return True
    if type2_kind != "mixed":
        return False

    answer = _clean_text(pred.get("answer")) or ""
    gold_answer = _clean_text(pred.get("gold_answer")) or ""
    return parse_number(answer) is None or parse_number(gold_answer) is None


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _count_by_status(rows: list[EvalRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--errors", type=Path, default=None)
    parser.add_argument("--relative-tolerance", type=float, default=None)
    parser.add_argument("--absolute-tolerance", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
