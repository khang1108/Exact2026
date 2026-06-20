#!/usr/bin/env python
"""Run the current Type 2 pipeline against a dataset using a vLLM config.

Supports deterministic slicing via ``--offset`` / ``--limit`` and optional
randomized sampling via ``--random on`` with ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exact.datasets.dataset import ExactDataset
from exact.datasets.schemas import to_official_response
from exact.scripts.config_utils import build_settings_from_config, load_toml_config
from exact.type2.pipeline import run_type2_pipeline, set_generate_final_explanation


DEFAULT_CONFIG = Path("configs/type2_dataset_run.example.toml")


def main() -> None:
    args = parse_args()
    config = load_toml_config(args.config)
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    llm_cfg = config.get("llm", {})
    type2_cfg = config.get("type2_pipeline", {})

    input_path = args.input or Path(str(dataset_cfg.get("input", "")).strip())
    if not str(input_path).strip():
        raise ValueError("No dataset input path provided. Set [dataset].input or pass --input.")

    dataset = ExactDataset.from_file(
        input_path,
        skip_invalid=bool(dataset_cfg.get("skip_invalid", False)),
    ).filter_type2()
    examples = list(dataset)

    settings = build_settings_from_config(config)
    require_llm = bool(llm_cfg.get("require_llm", False))
    if require_llm and not settings.llm_base_url:
        raise ValueError(
            "This runner is configured to require a real vLLM-backed LLM config. Set [llm].backend='vllm', base_url, and model, or set [llm].require_llm=false."
        )

    set_generate_final_explanation(bool(type2_cfg.get("generate_final_explanation", True)))
    selected_examples = select_examples(
        examples,
        limit=args.limit if args.limit is not None else dataset_cfg.get("limit"),
        offset=args.offset,
        random_mode=args.random,
        seed=args.seed,
    )

    output_path = args.output or Path(str(output_cfg.get("path") or "artifacts/predictions/type2/type2_vllm_run.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, Any]] = []
    total = len(selected_examples)
    for index, example in enumerate(selected_examples, start=1):
        response = run_type2_pipeline(example.request, settings=settings)
        item = response.model_dump(mode="json")
        item["official"] = to_official_response(response)
        item["gold_answer"] = example.gold_answer
        item["gold_unit"] = example.gold_unit
        predictions.append(item)
        print(f"processed {index}/{total} id={example.request.id}", flush=True)

    payload = {
        "config": str(args.config),
        "source": str(dataset.source_path),
        "count": len(predictions),
        "offset": args.offset,
        "limit": args.limit if args.limit is not None else dataset_cfg.get("limit"),
        "random": args.random,
        "seed": args.seed,
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(predictions)} predictions to {output_path}")


def select_examples(
    examples,
    *,
    limit: int | None,
    offset: int,
    random_mode: str,
    seed: int,
):
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if limit is not None and int(limit) < 0:
        raise ValueError("limit must be >= 0")

    working = list(examples)
    if random_mode == "on":
        rng = random.Random(seed)
        rng.shuffle(working)
        offset = 0

    sliced = working[offset:]
    if limit is not None:
        sliced = sliced[: int(limit)]
    return sliced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--random", choices=["on", "off"], default="off")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
