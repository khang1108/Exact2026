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


DEFAULT_CLOUDFLARE_BASE_URL = "https://exact-llm-api.duchoaiduong100.workers.dev/v1"
DEFAULT_CLOUDFLARE_MODEL = "@cf/openai/gpt-oss-120b"
DEFAULT_CLOUDFLARE_API_KEY = "exact2026"
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
        choices=[
            "cloudflare",
            "groq",
            "ollama",
            "transformers",
            "huggingface",
            "openai_compatible",
        ],
        required=True,
        help="LLM client/backend preset for this run.",
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
    parser.add_argument("--device-map", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
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
        python_exe = args.python_exe or python_executable(root_dir)
        cmd = [
            str(python_exe),
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
        print(f"Type 2 backend:          {args.backend}")
        print(f"Temporary config:        {config_path}")
        print(f"Output:                  {args.output}")
        if args.routing_log is not None:
            print(f"Routing log:             {args.routing_log}")
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
    preset: dict[str, Any] = {
        "enabled": True,
        "backend": backend,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "max_retries": DEFAULT_MAX_RETRIES,
    }
    if backend == "cloudflare":
        preset["backend"] = "openai_compatible"
        preset["base_url"] = env_or_dotenv("EXACT_CLOUDFLARE_LLM_BASE_URL", DEFAULT_CLOUDFLARE_BASE_URL)
        preset["model"] = env_or_dotenv("EXACT_CLOUDFLARE_LLM_MODEL", DEFAULT_CLOUDFLARE_MODEL)
        preset["api_key"] = env_or_dotenv("EXACT_CLOUDFLARE_LLM_API_KEY", DEFAULT_CLOUDFLARE_API_KEY)
    elif backend == "groq":
        preset["backend"] = "groq"
        preset["model"] = "llama-3.1-8b-instant"
        preset["api_key_env"] = "GROQ_API_KEY"
    elif backend == "ollama":
        preset["backend"] = "ollama"
        preset["model"] = "qwen3.5:4b-cloud"
        preset["base_url"] = "http://127.0.0.1:11434/v1"
        preset["api_key"] = "ollama"
    elif backend == "transformers":
        preset["backend"] = "transformers"
        preset["model"] = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        preset["device_map"] = "auto"
        preset["torch_dtype"] = "bfloat16"
        preset["local_files_only"] = False
        preset["trust_remote_code"] = False
    elif backend == "huggingface":
        preset["backend"] = "huggingface"
        preset["api_key_env"] = "HF_TOKEN"
    elif backend == "openai_compatible":
        preset["backend"] = "openai_compatible"
    return preset


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
    if args.device_map is not None:
        updated["device_map"] = args.device_map
    if args.torch_dtype is not None:
        updated["torch_dtype"] = args.torch_dtype
    if args.local_files_only:
        updated["local_files_only"] = True
    if args.trust_remote_code:
        updated["trust_remote_code"] = True
    return updated


def validate_llm_config(llm_cfg: dict[str, Any]) -> None:
    backend = str(llm_cfg.get("backend") or "").strip().lower()
    model = str(llm_cfg.get("model") or "").strip()
    base_url = str(llm_cfg.get("base_url") or "").strip()
    if not model:
        raise ValueError("This backend requires a model. Pass --model or use a preset with a default model.")
    if backend == "openai_compatible" and not base_url:
        raise ValueError("openai_compatible requires --base-url or the Cloudflare preset.")


def python_executable(root_dir: Path) -> Path:
    if os.name == "nt":
        return Path(sys.executable)
    venv_python = root_dir / "venv" / "bin" / "python"
    return venv_python if venv_python.exists() else Path(sys.executable)


def env_or_dotenv(name: str, default: str) -> str:
    value = os.getenv(name)
    if value:
        return value

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return default

    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, raw_value = text.split("=", 1)
        if key.strip() != name:
            continue
        parsed = raw_value.strip()
        if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {'"', "'"}:
            parsed = parsed[1:-1]
        return parsed or default
    return default


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
