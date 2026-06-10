#!/usr/bin/env python
"""Download/cache a Hugging Face model declared in a TOML run config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exact.scripts.config_utils import load_toml_config


DEFAULT_CONFIG = Path("configs/type2_dataset_run.example.toml")


def main() -> None:
    args = parse_args()
    config = load_toml_config(args.config)
    llm = config.get("llm", {})
    pull = config.get("model_pull", {})

    model_id = args.model or str(pull.get("model") or llm.get("model") or "").strip()
    if not model_id:
        raise ValueError("No model set. Configure model_pull.model or pass --model.")

    backend = str(pull.get("backend") or llm.get("backend") or "none").strip().lower()
    if backend not in {"transformers", "huggingface"} and not args.force:
        raise ValueError(
            "Model pull is intended for Hugging Face model ids. "
            "Use --force if you still want to download this model id."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install dependencies with requirements.txt."
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": model_id,
        "revision": str(pull.get("revision") or "main"),
    }
    cache_dir = str(pull.get("cache_dir") or "").strip()
    local_dir = str(pull.get("local_dir") or "").strip()
    allow_patterns = pull.get("allow_patterns") or None
    ignore_patterns = pull.get("ignore_patterns") or None

    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if local_dir:
        kwargs["local_dir"] = local_dir
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns

    path = snapshot_download(**kwargs)
    print(f"downloaded {model_id} to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
