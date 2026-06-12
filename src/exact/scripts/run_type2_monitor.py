#!/usr/bin/env python
"""Run Type 2 predictions with live correctness counters."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exact.datasets.dataset import ExactDataset, LoadedExample
from exact.datasets.schemas import PredictionResponse, QuestionType, TaskType
from exact.llm_client import has_json_llm_client_config
from exact.scripts.config_utils import build_settings_from_config, load_toml_config
from exact.scripts.evaluate_type2_predictions import evaluate_prediction
from exact.type2.pipeline import run_type2_pipeline, set_generate_final_explanation


DEFAULT_CONFIG = Path("configs/type2_dataset_run.example.toml")


@dataclass
class MonitorStats:
    total: int = 0
    correct: int = 0
    wrong: int = 0
    pipeline_errors: int = 0
    missing_gold: int = 0
    conceptual_only: int = 0
    numeric_total: int = 0
    numeric_correct: int = 0

    @property
    def accuracy(self) -> float:
        denominator = self.total - self.missing_gold - self.conceptual_only
        return self.correct / denominator if denominator > 0 else 0.0

    @property
    def numeric_accuracy(self) -> float:
        return self.numeric_correct / self.numeric_total if self.numeric_total > 0 else 0.0


def main() -> None:
    args = parse_args()
    config = load_toml_config(args.config)
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    type2_cfg = config.get("type2_pipeline", {})
    eval_cfg = config.get("evaluation", {})

    input_path = args.input or dataset_cfg.get("input", "")
    dataset = ExactDataset.from_file(input_path, skip_invalid=bool(dataset_cfg.get("skip_invalid", False)))
    examples = list(dataset.filter_type2())
    offset = args.offset if args.offset is not None else int(dataset_cfg.get("offset", 0) or 0)
    limit = args.limit if args.limit is not None else dataset_cfg.get("limit")
    examples = _slice_examples(examples, offset=offset, limit=limit)

    settings = build_settings_from_config(config)
    _require_real_llm_settings(settings)
    set_generate_final_explanation(bool(type2_cfg.get("generate_final_explanation", False)))
    relative_tolerance = float(eval_cfg.get("relative_tolerance", 0.02))
    absolute_tolerance = float(eval_cfg.get("absolute_tolerance", 1e-9))
    case_sensitive_text = bool(eval_cfg.get("case_sensitive_text", False))

    output_path = args.output or Path(output_cfg.get("path", "artifacts/predictions/type2/type2_monitor.json"))
    routing_log_path = args.routing_log or _optional_path(output_cfg.get("routing_log_path"))
    predictions: list[dict[str, Any]] = []
    stats = MonitorStats()
    run_started = time.perf_counter()
    interrupted = False

    try:
        for index, example in enumerate(examples, start=1):
            item_started = time.perf_counter()
            prediction = _predict(example, settings)
            prediction["elapsed_seconds"] = round(time.perf_counter() - item_started, 6)
            predictions.append(prediction)
            row = evaluate_prediction(
                prediction,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                case_sensitive_text=case_sensitive_text,
            )
            _update_prediction_routing_log(prediction, row.status)
            _update_stats(stats, row.status, row.numeric_ok)
            _print_progress(index, len(examples), example, prediction, row.status, stats)
    except KeyboardInterrupt:
        interrupted = True
        print("\nreceived Ctrl+C; writing partial predictions before exit...", flush=True)

    _write_output(
        output_path,
        config_path=args.config,
        source_path=dataset.source_path,
        predictions=predictions,
        stats=stats,
        run_elapsed_seconds=round(time.perf_counter() - run_started, 6),
        interrupted=interrupted,
        requested_count=len(examples),
        offset=offset,
    )
    if routing_log_path is not None:
        _write_routing_log(routing_log_path, predictions)
        print(f"wrote routing logs to {routing_log_path}")
    suffix = "partial " if interrupted else ""
    print(f"wrote {suffix}{len(predictions)} predictions to {output_path}")
    if interrupted:
        raise SystemExit(130)


def _predict(example: LoadedExample, settings) -> dict[str, Any]:
    try:
        response = run_type2_pipeline(example.request, settings=settings)
    except Exception as exc:
        response = PredictionResponse(
            id=example.request.id,
            task_type=TaskType.TYPE2_PHYSICS,
            question_type=QuestionType.NUMERICAL,
            answer="",
            explanation=f"Prediction failed: {exc}",
            fol=None,
            cot=["The Type 2 monitor caught an exception for this item."],
            premises=[],
            confidence=0.0,
            unit=None,
            error=str(exc),
        )
    prediction = response.model_dump(mode="json")
    if response.routing_diagnostics is not None:
        prediction["routing_log"] = response.routing_diagnostics
    prediction["gold_answer"] = example.gold_answer
    prediction["gold_unit"] = example.gold_unit
    return prediction


def _require_real_llm_settings(settings) -> None:
    if not has_json_llm_client_config(settings):
        raise ValueError(
            "Type 2 runtime requires a real LLM backend. Configure [llm].backend, model, and credentials."
        )


def _update_stats(stats: MonitorStats, status: str, numeric_ok: bool | None) -> None:
    stats.total += 1
    if status.startswith("correct"):
        stats.correct += 1
    elif status == "missing_gold":
        stats.missing_gold += 1
    elif status == "conceptual_only":
        stats.conceptual_only += 1
    else:
        stats.wrong += 1
    if status == "pipeline_error":
        stats.pipeline_errors += 1
    if numeric_ok is not None:
        stats.numeric_total += 1
        if numeric_ok:
            stats.numeric_correct += 1


def _print_progress(
    index: int,
    total: int,
    example: LoadedExample,
    prediction: dict[str, Any],
    status: str,
    stats: MonitorStats,
) -> None:
    print(
        "processed {index}/{total} id={id} status={status} "
        "answer={answer} unit={unit} gold={gold_answer} {gold_unit} "
        "elapsed={elapsed:.2f}s correct={correct} wrong={wrong} errors={errors} acc={acc:.3f}".format(
            index=index,
            total=total,
            id=example.request.id,
            status=status,
            answer=prediction.get("answer"),
            unit=prediction.get("unit"),
            gold_answer=example.gold_answer,
            gold_unit=example.gold_unit or "",
            elapsed=float(prediction.get("elapsed_seconds") or 0.0),
            correct=stats.correct,
            wrong=stats.wrong,
            errors=stats.pipeline_errors,
            acc=stats.accuracy,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--routing-log", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    return parser.parse_args()


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _update_prediction_routing_log(prediction: dict[str, Any], status: str) -> None:
    routing = prediction.get("routing_log") or prediction.get("routing_diagnostics")
    if not isinstance(routing, dict):
        return
    routing["correct"] = status.startswith("correct")
    routing["evaluation_status"] = status
    prediction["routing_log"] = routing


def _write_routing_log(path: Path, predictions: list[dict[str, Any]]) -> None:
    logs = [
        prediction["routing_log"]
        for prediction in predictions
        if isinstance(prediction.get("routing_log"), dict)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        lines = [json.dumps(log, ensure_ascii=False) for log in logs]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return
    payload = {"count": len(logs), "format": "exact_type2_routing_logs", "routing_logs": logs}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _slice_examples(
    examples: list[LoadedExample],
    *,
    offset: int,
    limit: int | None,
) -> list[LoadedExample]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    sliced = examples[offset:]
    if limit is not None:
        sliced = sliced[: int(limit)]
    return sliced


def _write_output(
    output_path: Path,
    *,
    config_path: Path,
    source_path: Path,
    predictions: list[dict[str, Any]],
    stats: MonitorStats,
    run_elapsed_seconds: float,
    interrupted: bool,
    requested_count: int,
    offset: int,
) -> None:
    output = {
        "config": str(config_path),
        "source": str(source_path),
        "count": len(predictions),
        "requested_count": requested_count,
        "offset": offset,
        "interrupted": interrupted,
        "run_elapsed_seconds": run_elapsed_seconds,
        "format": "exact_type2_monitor_predictions",
        "summary": asdict(stats) | {
            "accuracy": stats.accuracy,
            "numeric_accuracy": stats.numeric_accuracy,
            "average_elapsed_seconds": _average_elapsed_seconds(predictions),
        },
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _average_elapsed_seconds(predictions: list[dict[str, Any]]) -> float:
    values = [float(prediction["elapsed_seconds"]) for prediction in predictions if "elapsed_seconds" in prediction]
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
