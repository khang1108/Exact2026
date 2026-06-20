#!/usr/bin/env python
"""Run the Type 2 no-LLM test bundle in one command.

This wrapper intentionally delegates to the existing runner/evaluator modules
so their JSON formats stay identical.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = Path("configs/type2_dataset_run.example.toml")
DEFAULT_OUTPUT = Path("artifacts/predictions/type2/no_llm_test.json")
DEFAULT_REPORT = Path("artifacts/reports/no_llm_test_report.json")
DEFAULT_ERRORS = Path("artifacts/reports/no_llm_test_errors.csv")
DEFAULT_ANSWER_TYPES = "numeric_with_unit"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    env = _build_env(project_root)

    if not args.skip_pytest:
        _run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(args.tests),
                "-q",
            ],
            cwd=project_root,
            env=env,
            label="pytest",
        )

    monitor_command = [
        sys.executable,
        "-m",
        "exact.scripts.run_type2_monitor",
        "--config",
        str(args.config),
        "--limit",
        str(args.limit),
        "--output",
        str(args.output),
    ]
    if args.input is not None:
        monitor_command.extend(["--input", str(args.input)])
    if args.offset is not None:
        monitor_command.extend(["--offset", str(args.offset)])

    _run(
        monitor_command,
        cwd=project_root,
        env=env,
        label="type2 predictions",
    )

    _run(
        [
            sys.executable,
            "-m",
            "exact.scripts.evaluate_type2_predictions",
            str(args.output),
            "--config",
            str(args.config),
            "--report",
            str(args.report),
            "--errors",
            str(args.errors),
            "--answer-types",
            str(args.answer_types),
        ],
        cwd=project_root,
        env=env,
        label="type2 evaluation",
    )

    print("", flush=True)
    print("Type 2 no-LLM test bundle completed.", flush=True)
    print(f"Predictions: {args.output}", flush=True)
    print(f"Report:      {args.report}", flush=True)
    print(f"Errors CSV:  {args.errors}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--answer-types", type=str, default=DEFAULT_ANSWER_TYPES)
    parser.add_argument("--tests", type=Path, default=Path("tests"))
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Only run predictions and evaluation.",
    )
    return parser.parse_args()


def _build_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(project_root / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path
        if not current_pythonpath
        else os.pathsep.join([src_path, current_pythonpath])
    )
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Make the bundle deterministic-only even when a local .env contains stale
    # or placeholder model settings.
    env["EXACT_LLM_ENABLED"] = "false"
    return env


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    label: str,
) -> None:
    print(f"\n==> Running {label}", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
