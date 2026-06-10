#!/usr/bin/env python
"""Evaluate Type 1 prediction files that include gold_answer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class EvalRow:
    id: str | None
    answer: str
    gold_answer: str | None
    question_type: str | None
    status: str
    error: str | None


def main() -> None:
    args = parse_args()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("Prediction file must contain a top-level predictions list")

    rows = [evaluate_prediction(pred, case_sensitive=args.case_sensitive) for pred in predictions]
    summary = summarize(rows)

    report = {
        "source": str(args.predictions),
        "count": len(rows),
        "case_sensitive": args.case_sensitive,
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_errors_csv(args.errors, rows)

    print_summary(summary)
    print(f"wrote report to {args.report}")
    print(f"wrote error rows to {args.errors}")


def evaluate_prediction(pred: dict[str, Any], *, case_sensitive: bool) -> EvalRow:
    answer = _clean_text(pred.get("answer")) or ""
    gold_answer = _clean_text(pred.get("gold_answer"))
    runtime_error = _clean_text(pred.get("error"))
    question_type = _clean_text(pred.get("question_type"))

    if gold_answer is None or gold_answer == "":
        return _row(pred, answer, gold_answer, question_type, "missing_gold", runtime_error)

    normalized_answer = _normalize_answer(answer, case_sensitive=case_sensitive)
    normalized_gold = _normalize_answer(gold_answer, case_sensitive=case_sensitive)
    ok = normalized_answer == normalized_gold

    if ok:
        status = "correct"
    elif runtime_error and not answer:
        status = "pipeline_error"
    else:
        status = "wrong"

    return _row(pred, answer, gold_answer, question_type, status, runtime_error)


def summarize(rows: list[EvalRow]) -> dict[str, Any]:
    total = len(rows)
    missing_gold = sum(row.status == "missing_gold" for row in rows)
    scored_total = total - missing_gold
    correct = sum(row.status == "correct" for row in rows)
    pipeline_errors = sum(row.status == "pipeline_error" for row in rows)
    runtime_diagnostics = sum(bool(row.error) for row in rows)
    wrong = scored_total - correct

    return {
        "total": total,
        "scored_total": scored_total,
        "correct": correct,
        "wrong": wrong,
        "missing_gold": missing_gold,
        "accuracy": _safe_ratio(correct, scored_total),
        "pipeline_errors": pipeline_errors,
        "runtime_diagnostics": runtime_diagnostics,
        "by_status": _count_by(rows, "status"),
        "by_question_type": _accuracy_by_question_type(rows),
    }


def write_errors_csv(path: Path, rows: list[EvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()) if rows else ["id"])
        writer.writeheader()
        for row in rows:
            if row.status != "correct":
                writer.writerow(asdict(row))


def print_summary(summary: dict[str, Any]) -> None:
    print(f"total: {summary['total']}")
    print(f"scored_total: {summary['scored_total']}")
    print(f"accuracy: {summary['accuracy']:.3f}")
    print(f"pipeline_errors: {summary['pipeline_errors']}")
    print(f"runtime_diagnostics: {summary['runtime_diagnostics']}")
    print(f"by_status: {summary['by_status']}")
    print(f"by_question_type: {summary['by_question_type']}")


def _row(
    pred: dict[str, Any],
    answer: str,
    gold_answer: str | None,
    question_type: str | None,
    status: str,
    error: str | None,
) -> EvalRow:
    return EvalRow(
        id=pred.get("id"),
        answer=answer,
        gold_answer=gold_answer,
        question_type=question_type,
        status=status,
        error=error,
    )


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def _normalize_answer(value: str, *, case_sensitive: bool) -> str:
    normalized = value if case_sensitive else value.lower()
    if normalized.lower() in {"unknown", "uncertain"}:
        return "uncertain"
    return normalized


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _count_by(rows: list[EvalRow], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(getattr(row, field_name) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _accuracy_by_question_type(rows: list[EvalRow]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EvalRow]] = {}
    for row in rows:
        if row.status == "missing_gold":
            continue
        grouped.setdefault(row.question_type or "unknown", []).append(row)

    output: dict[str, dict[str, Any]] = {}
    for question_type, group in sorted(grouped.items()):
        correct = sum(row.status == "correct" for row in group)
        output[question_type] = {
            "total": len(group),
            "correct": correct,
            "accuracy": _safe_ratio(correct, len(group)),
            "pipeline_errors": sum(row.status == "pipeline_error" for row in group),
            "runtime_diagnostics": sum(bool(row.error) for row in group),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reports/type1_eval_report.json"),
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=Path("artifacts/reports/type1_eval_errors.csv"),
    )
    parser.add_argument("--case-sensitive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
