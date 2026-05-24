#!/usr/bin/env python
"""Run the full EXACT prediction flow over a JSON batch.

This mirrors the production shape:
input instance -> PredictionRequest -> TaskRouter -> Type-specific pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exact.config import get_settings
from exact.datasets.schemas import PredictionRequest, TaskType, to_official_response
from exact.llm_client import build_json_client_from_settings
from exact.logger import setup_logging
from exact.logic.llm_translator import JsonLLMClient
from exact.logic.pipeline import run_type1_pipeline
from exact.router.task_router import TaskRouter
from exact.type2.pipeline import run_type2_pipeline


DEFAULT_INPUT = Path("src/exact/datasets/exact/Logic_Based_Educational_Queries_inference.json")
DEFAULT_OUTPUT = Path("artifacts/predictions/predictions.json")


def main() -> None:
    args = parse_args()
    if args.no_llm and args.require_llm:
        raise ValueError("--no-llm and --require-llm cannot be used together")

    instances = load_instances(args.input)
    if args.limit is not None:
        instances = instances[: args.limit]

    router = TaskRouter()
    settings = get_settings()
    setup_logging(
        level=settings.log_level,
        log_file=args.log_file,
        json_logs=settings.json_logs,
    )
    logger = logging.getLogger(__name__)

    translator_client: JsonLLMClient | None = (
        None if args.no_llm else build_json_client_from_settings(settings)
    )
    logger.info(
        "prediction runner: "
        f"provider={settings.llm_provider}, "
        f"model={settings.llm_model}, "
        f"llm_enabled={translator_client is not None}, "
        f"require_llm={args.require_llm}, "
        f"max_tokens={settings.llm_max_tokens}, "
        f"instances={len(instances)}"
    )
    predictions: list[dict[str, Any]] = []

    for index, instance in enumerate(instances, start=1):
        request = PredictionRequest.model_validate(instance)
        route = router.route(request)
        logger.info(
            "processing %s/%s id=%s route=%s",
            index,
            len(instances),
            request.id,
            route.task_type.value,
        )

        if route.task_type == TaskType.TYPE1_LOGIC:
            response = run_type1_pipeline(
                request,
                translator_client=translator_client,
                settings=settings,
                allow_heuristic_fallback=not args.require_llm,
                question_type=route.question_type,
            )
        elif route.task_type == TaskType.TYPE2_PHYSICS:
            response = run_type2_pipeline(request)
        else:
            raise ValueError(f"Unsupported task type: {route.task_type}")

        prediction = response.model_dump(mode="json")
        prediction["route_reason"] = route.reason
        prediction["official"] = to_official_response(response)
        predictions.append(prediction)

        if args.progress_every and index % args.progress_every == 0:
            status = f"processed {index}/{len(instances)} answer={response.answer}"
            if response.error:
                status += f" error={_shorten(response.error)}"
            logger.info(status)

    output = {
        "source": str(args.input),
        "count": len(predictions),
        "format": "exact_predictions",
        "predictions": predictions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s predictions to %s", len(predictions), args.output)


def load_instances(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        instances = payload
    elif isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        instances = payload["instances"]
    else:
        raise ValueError(f"Expected a list or a top-level 'instances' list in {path}")

    if not all(isinstance(instance, dict) for instance in instances):
        raise ValueError(f"All instances in {path} must be JSON objects")

    return instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--log-file", type=Path, default=Path("outputs/logs/run_predictions.log"))
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM translation fallback.")
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail if LLM translation is unavailable or invalid; do not use heuristic parser.",
    )
    return parser.parse_args()


def _shorten(text: str, max_len: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


if __name__ == "__main__":
    main()
