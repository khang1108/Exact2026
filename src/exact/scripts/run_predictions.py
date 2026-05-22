#!/usr/bin/env python
"""Run the full EXACT prediction flow over a JSON batch.

This mirrors the production shape:
input instance -> PredictionRequest -> TaskRouter -> Type-specific pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exact.config import get_settings
from exact.datasets.schemas import PredictionRequest, TaskType, to_official_response
from exact.llm_client import build_json_client_from_settings
from exact.logic.llm_translator import JsonLLMClient
from exact.logic.pipeline import run_type1_pipeline
from exact.router.task_router import TaskRouter
from exact.type2.pipeline import run_type2_pipeline


DEFAULT_INPUT = Path("src/exact/datasets/exact/Logic_Based_Educational_Queries_inference.json")
DEFAULT_OUTPUT = Path("artifacts/predictions/predictions.json")


def main() -> None:
    args = parse_args()
    instances = load_instances(args.input)
    if args.limit is not None:
        instances = instances[: args.limit]

    router = TaskRouter()
    settings = get_settings()
    translator_client: JsonLLMClient | None = (
        None if args.no_llm else build_json_client_from_settings(settings)
    )
    predictions: list[dict[str, Any]] = []

    for index, instance in enumerate(instances, start=1):
        request = PredictionRequest.model_validate(instance)
        route = router.route(request)

        if route.task_type == TaskType.TYPE1_LOGIC:
            response = run_type1_pipeline(
                request,
                translator_client=translator_client,
                settings=settings,
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
            print(f"processed {index}/{len(instances)}")

    output = {
        "source": str(args.input),
        "count": len(predictions),
        "format": "exact_predictions",
        "predictions": predictions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(predictions)} predictions to {args.output}")


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
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM translation fallback.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
