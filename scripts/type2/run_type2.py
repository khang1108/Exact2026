#!/usr/bin/env python3
"""Run EXACT Type 2 with a single configurable runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    default_config = root_dir / "configs" / "type2_dataset_run.example.toml"
    default_input = root_dir / "src" / "exact" / "datasets" / "exact" / "type2_physics_questions.csv"
    default_output = root_dir / "artifacts" / "predictions" / "type2" / "type2_run.json"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--backend",
        choices=["vllm"],
        default="vllm",
        help="Self-hosted LLM backend.",
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--routing-log", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--python-exe",
        type=Path,
        default=None,
        help="Python interpreter that has the project dependencies installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_text = build_config_text(args)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".toml",
        prefix="exact_type2_run_",
        encoding="utf-8",
        delete=False,
    ) as config_file:
        config_file.write(config_text)
        config_path = Path(config_file.name)

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root_dir / "src")
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        env.setdefault("TQDM_DISABLE", "0")
        env.setdefault("TRANSFORMERS_VERBOSITY", "info")
        python_exe = args.python_exe or python_executable(root_dir)
        cmd = [
            str(python_exe),
            "-u",
            "-m",
            "exact.scripts.run_type2_monitor",
            "--config",
            str(config_path),
            "--input",
            str(args.input),
            "--limit",
            str(args.limit),
            "--offset",
            str(args.offset),
            "--output",
            str(args.output),
        ]
        print(f"Type 2 backend:          {args.backend}", flush=True)
        print(f"Temporary config:        {config_path}", flush=True)
        print(f"Output:                  {args.output}", flush=True)
        if args.routing_log is not None:
            print(f"Routing log:             {args.routing_log}", flush=True)
        subprocess.run(cmd, env=env, check=True)
    finally:
        try:
            config_path.unlink()
        except OSError:
            pass


def build_config_text(args: argparse.Namespace) -> str:
    with args.config.open("rb") as file:
        config = tomllib.load(file)

    dataset_cfg = dict(config.get("dataset", {}))
    dataset_cfg["input"] = str(args.input)
    dataset_cfg["limit"] = args.limit
    dataset_cfg["task_filter"] = "type2"

    output_cfg = dict(config.get("output", {}))
    output_cfg["path"] = str(args.output)
    if args.routing_log is not None:
        output_cfg["routing_log_path"] = str(args.routing_log)

    llm_cfg = build_llm_config(args.backend)
    llm_cfg = apply_cli_overrides(llm_cfg, args)
    validate_llm_config(llm_cfg)

    sections: dict[str, dict[str, Any]] = {
        "dataset": dataset_cfg,
        "output": output_cfg,
        "llm": llm_cfg,
        "type2_pipeline": dict(config.get("type2_pipeline", {})),
        "evaluation": dict(config.get("evaluation", {})),
    }
    if "model_pull" in config:
        sections["model_pull"] = dict(config["model_pull"])
    return dump_toml(sections)


def build_llm_config(backend: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "backend": "vllm",
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "max_retries": DEFAULT_MAX_RETRIES,
    }


def apply_cli_overrides(llm_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(llm_cfg)
    if args.model is not None:
        updated["model"] = args.model
    if args.base_url is not None:
        updated["base_url"] = args.base_url
    if args.api_key is not None:
        updated["api_key"] = args.api_key
        updated.pop("api_key_env", None)
    if args.api_key_env is not None:
        updated["api_key_env"] = args.api_key_env
        updated.pop("api_key", None)
    updated["temperature"] = args.temperature
    updated["top_p"] = args.top_p
    updated["timeout_seconds"] = args.timeout_seconds
    updated["max_retries"] = args.max_retries
    return updated


def validate_llm_config(llm_cfg: dict[str, Any]) -> None:
    backend = str(llm_cfg.get("backend") or "").strip().lower()
    model = str(llm_cfg.get("model") or "").strip()
    base_url = str(llm_cfg.get("base_url") or "").strip()
    if not model:
        raise ValueError("This backend requires a model. Pass --model or use a preset with a default model.")
    if backend != "vllm":
        raise ValueError("Only the self-hosted vllm backend is supported.")
    if not base_url:
        raise ValueError("vllm requires --base-url.")


def python_executable(root_dir: Path) -> Path:
    if os.name == "nt":
        return Path(sys.executable)
    venv_python = root_dir / "venv" / "bin" / "python"
    return venv_python if venv_python.exists() else Path(sys.executable)


def dump_toml(sections: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


if __name__ == "__main__":
    main()
