from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from exact.scripts.evaluate_type2_predictions import (
    evaluate_prediction,
    summarize,
    write_errors_csv,
)


def configure_incremental_type2_prompt_log(settings: Any, artifact_dir: Path) -> Any:
    """Return settings that append PoT prompts beside incremental artifacts."""

    prompt_path = artifact_dir / "type2_pot_prompts.jsonl"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.touch(exist_ok=True)
    if hasattr(settings, "model_copy"):
        return settings.model_copy(
            update={
                "type2_debug_log_pot_prompts": True,
                "type2_debug_pot_prompt_log_path": str(prompt_path),
            }
        )
    return settings


def write_incremental_type2_artifacts(
    *,
    artifact_dir: Path,
    predictions_filename: str,
    predictions: list[dict[str, Any]],
    config_path: Path | str,
    source_path: Path | str,
    requested_count: int,
    offset: int,
    run_elapsed_seconds: float,
    interrupted: bool,
    relative_tolerance: float,
    absolute_tolerance: float,
    case_sensitive_text: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write all benchmark artifacts after each solved item.

    The files intentionally match the Kaggle benchmark artifact names:
    predictions JSON, report JSON, errors CSV, PoT prompts JSONL, and routing JSONL.
    """

    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = artifact_dir / predictions_filename
    stem = predictions_path.stem
    report_path = artifact_dir / f"type2_{stem}_report.json"
    errors_path = artifact_dir / f"type2_{stem}_errors.csv"
    routing_path = artifact_dir / "type2_routing_logs.jsonl"
    prompt_path = artifact_dir / "type2_pot_prompts.jsonl"
    prompt_path.touch(exist_ok=True)

    rows = [
        evaluate_prediction(
            pred,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            case_sensitive_text=case_sensitive_text,
        )
        for pred in predictions
    ]
    _attach_running_evaluation(predictions, rows)
    summary = summarize(rows, relative_tolerance, absolute_tolerance)
    routing_logs = _build_routing_logs(predictions, rows)

    prediction_payload = {
        **(metadata or {}),
        "config": str(config_path),
        "source": str(source_path),
        "count": len(predictions),
        "requested_count": requested_count,
        "offset": offset,
        "interrupted": interrupted,
        "run_elapsed_seconds": round(run_elapsed_seconds, 6),
        "format": "exact_type2_incremental_predictions",
        "summary": summary | {"average_elapsed_seconds": _average_elapsed_seconds(predictions)},
        "predictions": predictions,
    }
    _atomic_write_text(
        predictions_path,
        json.dumps(prediction_payload, ensure_ascii=False, indent=2) + "\n",
    )

    report = {
        "source": str(predictions_path),
        "count": len(rows),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "summary": summary,
        "rows": [asdict(row) for row in rows],
        "routing_logs": routing_logs,
    }
    _atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_errors_csv(errors_path, rows)
    _atomic_write_text(
        routing_path,
        "".join(json.dumps(log, ensure_ascii=False) + "\n" for log in routing_logs),
    )


def _build_routing_logs(predictions: list[dict[str, Any]], rows) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for pred, row in zip(predictions, rows, strict=False):
        routing = pred.get("routing_log") or pred.get("routing_diagnostics")
        if not isinstance(routing, dict):
            continue
        log = dict(routing)
        log["correct"] = row.status.startswith("correct")
        log["evaluation_status"] = row.status
        if isinstance(pred.get("running_summary"), dict):
            log["running_summary"] = pred["running_summary"]
        logs.append(log)
    return logs


def _attach_running_evaluation(predictions: list[dict[str, Any]], rows) -> None:
    stats = {
        "total": 0,
        "correct": 0,
        "wrong": 0,
        "pipeline_errors": 0,
        "missing_gold": 0,
        "conceptual_only": 0,
        "numeric_total": 0,
        "numeric_correct": 0,
    }
    for pred, row in zip(predictions, rows, strict=False):
        _update_stats(stats, row)
        summary = _running_summary(stats)
        pred["evaluation"] = {
            "status": row.status,
            "correct": row.status.startswith("correct"),
            "numeric_ok": row.numeric_ok,
            "unit_ok": row.unit_ok,
            "relative_error": row.relative_error,
            "absolute_error": row.absolute_error,
        }
        pred["running_summary"] = summary
        routing = pred.get("routing_log") or pred.get("routing_diagnostics")
        if isinstance(routing, dict):
            routing["correct"] = row.status.startswith("correct")
            routing["evaluation_status"] = row.status
            routing["running_summary"] = summary
            pred["routing_log"] = routing


def _update_stats(stats: dict[str, Any], row) -> None:
    stats["total"] += 1
    if row.status.startswith("correct"):
        stats["correct"] += 1
    elif row.status == "missing_gold":
        stats["missing_gold"] += 1
    elif row.status == "conceptual_only":
        stats["conceptual_only"] += 1
    else:
        stats["wrong"] += 1
    if row.status == "pipeline_error":
        stats["pipeline_errors"] += 1
    if row.numeric_ok is not None:
        stats["numeric_total"] += 1
        if row.numeric_ok and row.unit_ok is not False:
            stats["numeric_correct"] += 1


def _running_summary(stats: dict[str, Any]) -> dict[str, Any]:
    denominator = stats["total"] - stats["missing_gold"] - stats["conceptual_only"]
    numeric_total = stats["numeric_total"]
    return dict(stats) | {
        "accuracy": stats["correct"] / denominator if denominator > 0 else 0.0,
        "numeric_accuracy": stats["numeric_correct"] / numeric_total if numeric_total > 0 else 0.0,
    }


def _average_elapsed_seconds(predictions: list[dict[str, Any]]) -> float:
    values = [float(prediction["elapsed_seconds"]) for prediction in predictions if "elapsed_seconds" in prediction]
    return sum(values) / len(values) if values else 0.0


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)
