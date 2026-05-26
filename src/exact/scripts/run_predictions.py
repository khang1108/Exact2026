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
from exact.common.schemas import PredictionRequest, TaskType, to_official_response
from exact.llm_client import build_json_client_from_settings
from exact.logger import setup_logging
from exact.logic.llm_translator import JsonLLMClient
from exact.logic.pipeline import run_type1_pipeline
from exact.router.task_router import TaskRouter
from exact.scripts.config_utils import settings_for_disabled_llm
from exact.type2.pipeline import run_type2_pipeline


DEFAULT_INPUT = Path("src/exact/datasets/exact/Logic_Based_Educational_Queries_inference.json")
DEFAULT_OUTPUT = Path("artifacts/predictions/type1/predictions.json")


def main() -> None:
    """Run offline predictions from CLI arguments."""

    args = parse_args()
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

    translator_client: JsonLLMClient | None = build_json_client_from_settings(settings)
    logger.info(
        "prediction runner: "
        f"provider={settings.llm_provider}, "
        f"model={settings.llm_model}, "
        f"llm_enabled={translator_client is not None}, "
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
            if translator_client is None:
                logger.error("Type 1 request %s cannot run without a JSON LLM client", request.id)
                raise RuntimeError(
                    "Type 1 prediction requires a JSON LLM client. Configure EXACT_LLM_BASE_URL "
                    "or EXACT_LLM_PROVIDER=local."
                )
            response = run_type1_pipeline(
                request,
                translator_client=translator_client,
                settings=settings,
                question_type=route.question_type,
            )
        elif route.task_type == TaskType.TYPE2_PHYSICS:
            response = run_type2_pipeline(request, settings=settings)
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
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s predictions to %s", len(predictions), args.output)


def load_instances(path: Path) -> list[dict[str, Any]]:
    """Load a JSON prediction batch from a list or `instances` object."""

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
    """Parse command-line arguments for the prediction runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--log-file", type=Path, default=Path("outputs/logs/run_predictions.log"))
    return parser.parse_args()


def _shorten(text: str, max_len: int = 180) -> str:
    """Collapse long log snippets to a single bounded line."""

    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


if __name__ == "__main__":
    main()
