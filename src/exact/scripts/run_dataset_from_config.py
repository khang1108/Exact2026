#!/usr/bin/env python
"""Run Type 2 dataset predictions from a TOML run config."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exact.datasets.dataset import ExactDataset, LoadedExample
from exact.datasets.schemas import PredictionResponse, TaskType, to_official_response
from exact.llm_client import has_json_llm_client_config
from exact.scripts.config_utils import (
    build_settings_from_config,
    load_toml_config,
    settings_for_disabled_llm,
)
from exact.scripts.evaluate_type2_predictions import EvalRow, evaluate_prediction
from exact.type2.pipeline import run_type2_pipeline
from exact.type2.pipeline import set_generate_final_explanation


DEFAULT_CONFIG = Path("configs/type2_dataset_run.example.toml")


@dataclass
class RunningStats:
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
    pipeline_cfg = config.get("pipeline", {})
    type2_pipeline_cfg = config.get("type2_pipeline", {})
    llm_cfg = config.get("llm", {})
    eval_cfg = config.get("evaluation", {})

    input_path = args.input or dataset_cfg.get("input", "")
    dataset = ExactDataset.from_file(
        input_path,
        skip_invalid=bool(dataset_cfg.get("skip_invalid", False)),
    )
    dataset = dataset.filter_type2()

    limit = args.limit if args.limit is not None else dataset_cfg.get("limit")
    examples = list(dataset)
    if limit is not None:
        examples = examples[: int(limit)]

    settings = build_settings_from_config(config)
    if bool(llm_cfg.get("require_llm", False)):
        _require_real_type2_llm(settings)
    disabled_settings = settings_for_disabled_llm(settings)
    use_type2_llm_fallback = bool(pipeline_cfg.get("use_type2_llm_fallback", True))
    fail_fast = bool(pipeline_cfg.get("fail_fast", False))
    set_generate_final_explanation(bool(type2_pipeline_cfg.get("generate_final_explanation", True)))
    include_gold = bool(output_cfg.get("include_gold", True))
    include_raw = bool(output_cfg.get("include_raw", False))
    progress_every = int(output_cfg.get("progress_every", 100) or 0)
    relative_tolerance = float(eval_cfg.get("relative_tolerance", 0.02))
    absolute_tolerance = float(eval_cfg.get("absolute_tolerance", 1e-9))
    case_sensitive_text = bool(eval_cfg.get("case_sensitive_text", False))
    answer_types = eval_cfg.get("answer_types")
    output_path = resolve_output_path(
        args.output,
        output_cfg.get("path"),
    )
    routing_log_path = _optional_path(output_cfg.get("routing_log_path"))

    predictions: list[dict[str, Any]] = []
    running_stats = RunningStats()

    for index, example in enumerate(examples, start=1):
        try:
            response = run_type2_pipeline(
                example.request,
                settings=settings if use_type2_llm_fallback else disabled_settings,
            )
        except Exception as exc:
            if fail_fast:
                raise
            response = _error_response(example, exc)

        prediction = response.model_dump(mode="json")
        if response.routing_diagnostics is not None:
            prediction["routing_log"] = response.routing_diagnostics
        prediction["official"] = to_official_response(response)
        if include_gold:
            prediction["gold_answer"] = example.gold_answer
            prediction["gold_unit"] = example.gold_unit
        if include_raw:
            prediction["raw"] = example.raw

        if include_gold:
            row = evaluate_prediction(
                prediction,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                case_sensitive_text=case_sensitive_text,
                answer_types=set(answer_types) if isinstance(answer_types, list) else None,
            )
            _update_running_stats(running_stats, row)
            _attach_evaluation_metadata(prediction, row, running_stats)
        predictions.append(prediction)

        if progress_every and index % progress_every == 0:
            print(
                "processed {index}/{total} correct={correct} wrong={wrong} acc={acc:.3f}".format(
                    index=index,
                    total=len(examples),
                    correct=running_stats.correct,
                    wrong=running_stats.wrong,
                    acc=running_stats.accuracy,
                )
            )

    output = {
        "config": str(args.config),
        "source": str(dataset.source_path),
        "count": len(predictions),
        "format": "exact_predictions",
        "summary": _running_summary(running_stats) if include_gold else None,
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(predictions)} predictions to {output_path}")
    if routing_log_path is not None:
        _write_routing_log(routing_log_path, predictions)
        print(f"wrote routing logs to {routing_log_path}")


def resolve_output_path(
    cli_output: Path | None,
    configured_output: str | None,
) -> Path:
    if cli_output is not None:
        return cli_output
    if configured_output:
        return Path(configured_output)

    return Path("artifacts/predictions/type2/config_run.json")


def _error_response(
    example: LoadedExample,
    exc: Exception,
) -> PredictionResponse:
    return PredictionResponse(
        id=example.request.id,
        task_type=TaskType.TYPE2_PHYSICS,
        answer="",
        explanation=f"Prediction failed: {exc}",
        fol=None,
        cot=["The configured dataset runner caught an exception for this item."],
        premises=[],
        confidence=0.0,
        error=str(exc),
    )


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


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


def _update_running_stats(stats: RunningStats, row: EvalRow) -> None:
    stats.total += 1
    if row.status.startswith("correct"):
        stats.correct += 1
    elif row.status == "missing_gold":
        stats.missing_gold += 1
    elif row.status == "conceptual_only":
        stats.conceptual_only += 1
    else:
        stats.wrong += 1
    if row.status == "pipeline_error":
        stats.pipeline_errors += 1
    if row.numeric_ok is not None:
        stats.numeric_total += 1
        if row.numeric_ok and row.unit_ok is not False:
            stats.numeric_correct += 1


def _attach_evaluation_metadata(
    prediction: dict[str, Any],
    row: EvalRow,
    stats: RunningStats,
) -> None:
    running_summary = _running_summary(stats)
    prediction["evaluation"] = {
        "status": row.status,
        "correct": row.status.startswith("correct"),
        "numeric_ok": row.numeric_ok,
        "unit_ok": row.unit_ok,
        "relative_error": row.relative_error,
        "absolute_error": row.absolute_error,
    }
    prediction["running_summary"] = running_summary

    routing = prediction.get("routing_log") or prediction.get("routing_diagnostics")
    if isinstance(routing, dict):
        routing["correct"] = row.status.startswith("correct")
        routing["evaluation_status"] = row.status
        routing["running_summary"] = running_summary
        prediction["routing_log"] = routing


def _running_summary(stats: RunningStats) -> dict[str, Any]:
    return asdict(stats) | {
        "accuracy": stats.accuracy,
        "numeric_accuracy": stats.numeric_accuracy,
    }


def _require_real_type2_llm(settings) -> None:
    if not has_json_llm_client_config(settings):
        raise ValueError(
            "Type 2 runtime requires a real LLM backend. Configure [llm].backend, model, and credentials."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
