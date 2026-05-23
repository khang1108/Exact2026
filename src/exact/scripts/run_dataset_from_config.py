#!/usr/bin/env python
"""Run EXACT dataset predictions from a TOML run config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exact.datasets.dataset import ExactDataset, LoadedExample
from exact.datasets.schemas import PredictionResponse, TaskType, to_official_response
from exact.llm_client import build_json_client_from_settings
from exact.logic.llm_translator import JsonLLMClient
from exact.logic.pipeline import run_type1_pipeline
from exact.router.task_router import TaskRouter
from exact.scripts.config_utils import (
    build_settings_from_config,
    load_toml_config,
    settings_for_disabled_llm,
)
from exact.type2.pipeline import run_type2_pipeline


DEFAULT_CONFIG = Path("configs/type2_dataset_run.example.toml")


def main() -> None:
    args = parse_args()
    config = load_toml_config(args.config)
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    pipeline_cfg = config.get("pipeline", {})

    input_path = args.input or dataset_cfg.get("input", "")
    dataset = ExactDataset.from_file(
        input_path,
        skip_invalid=bool(dataset_cfg.get("skip_invalid", False)),
    )
    task_filter = args.task_filter or str(dataset_cfg.get("task_filter", "all"))
    dataset = _filter_dataset(dataset, task_filter)

    limit = args.limit if args.limit is not None else dataset_cfg.get("limit")
    examples = list(dataset)
    if limit is not None:
        examples = examples[: int(limit)]

    settings = build_settings_from_config(config)
    disabled_settings = settings_for_disabled_llm(settings)
    use_type1_llm = bool(pipeline_cfg.get("use_type1_llm", True))
    use_type2_llm_fallback = bool(pipeline_cfg.get("use_type2_llm_fallback", True))
    fail_fast = bool(pipeline_cfg.get("fail_fast", False))
    include_gold = bool(output_cfg.get("include_gold", True))
    include_raw = bool(output_cfg.get("include_raw", False))
    progress_every = int(output_cfg.get("progress_every", 100) or 0)
    output_path = resolve_output_path(
        args.output,
        output_cfg.get("path"),
        task_filter,
    )

    translator_client: JsonLLMClient | None = None
    if use_type1_llm:
        translator_client = build_json_client_from_settings(settings)

    router = TaskRouter()
    predictions: list[dict[str, Any]] = []

    for index, example in enumerate(examples, start=1):
        route = router.route(example.request)
        try:
            if route.task_type == TaskType.TYPE1_LOGIC:
                response = run_type1_pipeline(
                    example.request,
                    translator_client=translator_client if use_type1_llm else None,
                    settings=settings if use_type1_llm else disabled_settings,
                )
            elif route.task_type == TaskType.TYPE2_PHYSICS:
                response = run_type2_pipeline(
                    example.request,
                    settings=settings if use_type2_llm_fallback else disabled_settings,
                )
            else:
                raise ValueError(f"Unsupported task type: {route.task_type}")
        except Exception as exc:
            if fail_fast:
                raise
            response = _error_response(example, route.task_type, exc)

        prediction = response.model_dump(mode="json")
        prediction["route_reason"] = route.reason
        prediction["official"] = to_official_response(response)
        if include_gold:
            prediction["gold_answer"] = example.gold_answer
            prediction["gold_unit"] = example.gold_unit
        if include_raw:
            prediction["raw"] = example.raw
        predictions.append(prediction)

        if progress_every and index % progress_every == 0:
            print(f"processed {index}/{len(examples)}")

    output = {
        "config": str(args.config),
        "source": str(dataset.source_path),
        "count": len(predictions),
        "format": "exact_predictions",
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(predictions)} predictions to {output_path}")


def _filter_dataset(dataset: ExactDataset, task_filter: str) -> ExactDataset:
    task_filter = task_filter.strip().lower()
    if task_filter == "type1":
        return dataset.filter_type1()
    if task_filter == "type2":
        return dataset.filter_type2()
    if task_filter != "all":
        raise ValueError("dataset.task_filter must be one of: all, type1, type2")
    return dataset


def resolve_output_path(
    cli_output: Path | None,
    configured_output: str | None,
    task_filter: str,
) -> Path:
    if cli_output is not None:
        return cli_output
    if configured_output:
        return Path(configured_output)

    task_filter = task_filter.strip().lower()
    if task_filter == "type1":
        return Path("artifacts/predictions/type1/config_run.json")
    if task_filter == "type2":
        return Path("artifacts/predictions/type2/config_run.json")
    return Path("artifacts/predictions/mixed/config_run.json")


def _error_response(
    example: LoadedExample,
    task_type: TaskType,
    exc: Exception,
) -> PredictionResponse:
    return PredictionResponse(
        id=example.request.id,
        task_type=task_type,
        answer="",
        explanation=f"Prediction failed: {exc}",
        fol=None,
        cot=["The configured dataset runner caught an exception for this item."],
        premises=[],
        confidence=0.0,
        error=str(exc),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-filter", choices=["all", "type1", "type2"], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
