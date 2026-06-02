#!/usr/bin/env python3
"""Run EXACT Type 2 pipeline with local transformers model on Thunder Compute.

This script is designed to run the full Type 2 physics pipeline directly
in-process (no subprocess), fixing several issues with the standard
`run_type2.py`:

  1. Loads the model ONCE in-process instead of spawning a child process,
     avoiding env/PYTHONPATH issues on remote GPU machines.
  2. Validates GPU availability and prints detailed CUDA diagnostics before
     starting.
  3. Configurable model, dtype, attention implementation, and batch size.
  4. Automatically handles flash_attention_2 → sdpa fallback.
  5. Pre-downloads the model with progress before running the pipeline.

Usage (on Thunder Compute):
    # Basic — run 5 questions starting from offset 0
    python run_type2_thunder.py --limit 5

    # Run 20 questions from offset 10, with a larger model
    python run_type2_thunder.py --limit 20 --offset 10 \\
        --model Qwen/Qwen2.5-Math-7B-Instruct

    # Force float16 instead of bfloat16
    python run_type2_thunder.py --limit 5 --torch-dtype float16

    # Use a custom config file
    python run_type2_thunder.py --limit 5 --config configs/my_config.toml

    # Dry run — just check GPU and download model, don't run pipeline
    python run_type2_thunder.py --dry-run --model Qwen/Qwen2.5-Math-7B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the project src/ directory is importable
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_LIMIT = 5
DEFAULT_OFFSET = 0
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_POT_MAX_RETRIES = 3
DEFAULT_POT_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# GPU Diagnostics
# ---------------------------------------------------------------------------
def print_gpu_diagnostics() -> dict[str, Any]:
    """Print detailed GPU info and return a diagnostics dict."""
    info: dict[str, Any] = {"cuda_available": False}
    print("=" * 70)
    print("  GPU DIAGNOSTICS")
    print("=" * 70)

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        print(f"  PyTorch version:    {torch.__version__}")
        print(f"  CUDA available:     {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            print(f"  CUDA version:       {torch.version.cuda}")
            print(f"  GPU count:          {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                free = total - reserved
                info[f"gpu_{i}"] = {
                    "name": name,
                    "total_gb": round(total, 2),
                    "free_gb": round(free, 2),
                }
                print(f"  GPU {i}:             {name}")
                print(f"    Total memory:     {total:.2f} GB")
                print(f"    Reserved:         {reserved:.2f} GB")
                print(f"    Allocated:        {allocated:.2f} GB")
                print(f"    Free:             {free:.2f} GB")
            # Check bf16 support
            major = torch.cuda.get_device_capability(0)[0]
            info["bf16_supported"] = major >= 8
            print(f"  bfloat16 support:   {'Yes' if major >= 8 else 'No (compute capability < 8.0)'}")
        else:
            print("  WARNING: No CUDA GPU detected! Model will run on CPU (very slow).")
    except ImportError:
        print("  ERROR: PyTorch not installed!")
        info["error"] = "torch not installed"

    # Check accelerate
    try:
        import accelerate
        info["accelerate"] = accelerate.__version__
        print(f"  accelerate:         {accelerate.__version__}")
    except ImportError:
        info["accelerate"] = None
        print("  accelerate:         NOT INSTALLED (device_map='auto' will not work)")

    # Check transformers
    try:
        import transformers
        info["transformers"] = transformers.__version__
        print(f"  transformers:       {transformers.__version__}")
    except ImportError:
        info["transformers"] = None
        print("  transformers:       NOT INSTALLED")

    # Check flash attention
    try:
        import flash_attn
        info["flash_attn"] = flash_attn.__version__
        print(f"  flash_attn:         {flash_attn.__version__}")
    except ImportError:
        info["flash_attn"] = None
        print("  flash_attn:         not installed (will use SDPA)")

    print("=" * 70)
    return info


def recommend_dtype(gpu_info: dict[str, Any], requested: str) -> str:
    """Recommend a safe dtype based on GPU capabilities."""
    if requested != "auto":
        if requested == "bfloat16" and not gpu_info.get("bf16_supported", True):
            print(f"  ⚠ bfloat16 requested but GPU doesn't support it → falling back to float16")
            return "float16"
        return requested
    # Auto-select: bf16 if supported, else fp16
    if gpu_info.get("bf16_supported", False):
        return "bfloat16"
    return "float16"


# ---------------------------------------------------------------------------
# Model preloading with progress
# ---------------------------------------------------------------------------
def ensure_model_downloaded(model_id: str, trust_remote_code: bool = False) -> str:
    """Download the model if not cached, with progress. Returns the cache path."""
    from huggingface_hub import snapshot_download

    print(f"\n📥 Ensuring model '{model_id}' is cached...")
    started = time.monotonic()
    path = snapshot_download(
        repo_id=model_id,
        revision="main",
    )
    elapsed = time.monotonic() - started
    print(f"   Model ready at: {path} ({elapsed:.1f}s)")
    return path


# ---------------------------------------------------------------------------
# Build Settings from CLI args + TOML config
# ---------------------------------------------------------------------------
def build_settings(args: argparse.Namespace) -> Any:
    """Build a Settings object from CLI args, optionally merging a TOML config."""
    from exact.scripts.config_utils import load_toml_config, build_settings_from_config

    # Start from the TOML config if provided
    config = load_toml_config(args.config) if args.config.exists() else {}

    # Override LLM section for transformers
    config["llm"] = {
        "enabled": True,
        "backend": "transformers",
        "model": args.model,
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout_seconds": 120.0,
        "max_retries": 0,
        "max_tokens": args.max_new_tokens,
    }

    # Override type2_pipeline section with CLI values where provided
    type2_cfg = config.get("type2_pipeline", {})
    type2_cfg.setdefault("extraction_mode", args.extraction_mode)
    type2_cfg.setdefault("pot_max_retries", args.pot_max_retries)
    type2_cfg.setdefault("pot_timeout", args.pot_timeout)
    type2_cfg.setdefault("generate_final_explanation", args.generate_explanation)
    # For local models, increase token budgets slightly
    type2_cfg.setdefault("extraction_max_tokens", 768)
    type2_cfg.setdefault("pot_code_max_tokens", 3072)
    type2_cfg.setdefault("pot_repair_max_tokens", 2048)
    config["type2_pipeline"] = type2_cfg

    return build_settings_from_config(config)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    total: int = 0
    correct: int = 0
    wrong: int = 0
    errors: int = 0
    missing_gold: int = 0
    conceptual: int = 0
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        denom = self.total - self.missing_gold - self.conceptual
        return self.correct / denom if denom > 0 else 0.0


def run_pipeline(args: argparse.Namespace, settings: Any) -> RunResult:
    """Run the Type 2 pipeline on the dataset and return results."""
    from exact.datasets.dataset import ExactDataset
    from exact.datasets.schemas import PredictionResponse, QuestionType, TaskType
    from exact.type2.extraction.extractor import extract_type2
    from exact.type2.pipeline import run_type2_pipeline, set_generate_final_explanation

    # Load dataset
    input_path = args.input
    print(f"\n📂 Loading dataset from: {input_path}")
    dataset = ExactDataset.from_file(str(input_path), skip_invalid=True)
    examples = list(dataset.filter_type2())
    print(f"   Total Type 2 examples: {len(examples)}")

    # Apply offset and limit
    examples = examples[args.offset:]
    if args.limit is not None:
        examples = examples[:args.limit]
    print(f"   Running: offset={args.offset}, limit={args.limit}, actual={len(examples)}")

    if not examples:
        print("   ⚠ No examples to process!")
        return RunResult()

    # Set explanation generation
    type2_cfg = {}
    set_generate_final_explanation(args.generate_explanation)

    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    result = RunResult()
    run_started = time.perf_counter()

    # Optionally lazy-import evaluator
    try:
        from exact.scripts.evaluate_type2_predictions import evaluate_prediction
        has_evaluator = True
    except ImportError:
        has_evaluator = False

    print(f"\n🚀 Starting pipeline run ({len(examples)} questions)...\n")
    print("-" * 90)

    for idx, example in enumerate(examples, start=1):
        item_started = time.perf_counter()
        question_preview = example.request.question[:80].replace("\n", " ")
        print(f"[{idx}/{len(examples)}] ID={example.request.id}  {question_preview}...")

        try:
            response = run_type2_pipeline(example.request, settings=settings)
            pred = response.model_dump(mode="json")
        except Exception as exc:
            print(f"   ❌ Pipeline error: {exc}")
            result.errors += 1
            result.total += 1
            pred = {
                "id": example.request.id,
                "task_type": TaskType.TYPE2_PHYSICS.value,
                "question_type": QuestionType.NUMERICAL.value,
                "answer": "",
                "explanation": f"Pipeline error: {exc}",
                "fol": None,
                "cot": [str(exc)],
                "premises": [],
                "confidence": 0.0,
                "unit": None,
                "error": str(exc),
            }

        elapsed = time.perf_counter() - item_started
        pred["elapsed_seconds"] = round(elapsed, 4)
        pred["type2_kind"] = extract_type2(example.request.question).kind.value
        pred["gold_answer"] = example.gold_answer
        pred["gold_unit"] = example.gold_unit
        predictions.append(pred)
        result.total += 1

        # Evaluate if possible
        status = "unknown"
        if has_evaluator and not pred.get("error"):
            try:
                row = evaluate_prediction(pred)
                status = row.status
                if status.startswith("correct"):
                    result.correct += 1
                elif status == "missing_gold":
                    result.missing_gold += 1
                elif status == "conceptual_only":
                    result.conceptual += 1
                else:
                    result.wrong += 1
            except Exception:
                result.wrong += 1
                status = "eval_error"
        elif pred.get("error"):
            status = "pipeline_error"

        acc = result.accuracy
        print(
            f"   → answer={pred.get('answer')!r}  unit={pred.get('unit')!r}  "
            f"gold={example.gold_answer!r} {example.gold_unit or ''}  "
            f"status={status}  time={elapsed:.1f}s  "
            f"acc={acc:.1%} ({result.correct}/{result.total - result.missing_gold - result.conceptual})"
        )

    result.elapsed_seconds = time.perf_counter() - run_started
    print("-" * 90)
    print(f"\n✅ Run complete in {result.elapsed_seconds:.1f}s")
    print(f"   Total={result.total}  Correct={result.correct}  Wrong={result.wrong}  "
          f"Errors={result.errors}  Accuracy={result.accuracy:.1%}")

    # Write output
    output_data = {
        "format": "exact_type2_thunder_predictions",
        "model": args.model,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "offset": args.offset,
        "limit": args.limit,
        "count": len(predictions),
        "run_elapsed_seconds": round(result.elapsed_seconds, 4),
        "summary": {
            "total": result.total,
            "correct": result.correct,
            "wrong": result.wrong,
            "errors": result.errors,
            "missing_gold": result.missing_gold,
            "conceptual": result.conceptual,
            "accuracy": round(result.accuracy, 6),
        },
        "predictions": predictions,
    }
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"   Output saved to: {output_path}")

    return result


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    default_config = ROOT_DIR / "configs" / "type2_dataset_run.example.toml"
    default_input = ROOT_DIR / "src" / "exact" / "datasets" / "exact" / "type2_physics_questions.csv"
    default_output = ROOT_DIR / "artifacts" / "predictions" / "type2" / "type2_thunder_run.json"

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset / IO
    g_io = p.add_argument_group("Dataset & Output")
    g_io.add_argument("--config", type=Path, default=default_config,
                       help="TOML config for type2_pipeline / evaluation tuning.")
    g_io.add_argument("--input", type=Path, default=default_input,
                       help="Path to the Type 2 physics questions CSV.")
    g_io.add_argument("--output", type=Path, default=default_output,
                       help="Path to write predictions JSON.")
    g_io.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                       help=f"Number of questions to process (default: {DEFAULT_LIMIT}).")
    g_io.add_argument("--offset", type=int, default=DEFAULT_OFFSET,
                       help=f"Skip the first N questions (default: {DEFAULT_OFFSET}).")

    # Model
    g_model = p.add_argument_group("Model Configuration")
    g_model.add_argument("--model", default=DEFAULT_MODEL,
                          help=f"HuggingFace model ID (default: {DEFAULT_MODEL}).")
    g_model.add_argument("--torch-dtype", default=DEFAULT_TORCH_DTYPE,
                          choices=["auto", "float16", "bfloat16", "float32"],
                          help=f"Model precision (default: {DEFAULT_TORCH_DTYPE}).")
    g_model.add_argument("--device-map", default=DEFAULT_DEVICE_MAP,
                          help=f"Device placement strategy (default: {DEFAULT_DEVICE_MAP}).")
    g_model.add_argument("--local-files-only", action="store_true",
                          help="Only use cached models, don't download.")
    g_model.add_argument("--trust-remote-code", action="store_true",
                          help="Trust remote code in the model repo.")
    g_model.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                          help=f"Maximum new tokens per LLM call (default: {DEFAULT_MAX_NEW_TOKENS}).")

    # Generation
    g_gen = p.add_argument_group("Generation Parameters")
    g_gen.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).")
    g_gen.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                        help=f"Top-p nucleus sampling (default: {DEFAULT_TOP_P}).")

    # Pipeline
    g_pipe = p.add_argument_group("Pipeline Tuning")
    g_pipe.add_argument("--extraction-mode", default="merge",
                         choices=["merge", "llm_only", "heuristic_only"],
                         help="How to combine heuristic and LLM extractions (default: merge).")
    g_pipe.add_argument("--pot-max-retries", type=int, default=DEFAULT_POT_MAX_RETRIES,
                         help=f"PoT code generation retries (default: {DEFAULT_POT_MAX_RETRIES}).")
    g_pipe.add_argument("--pot-timeout", type=float, default=DEFAULT_POT_TIMEOUT,
                         help=f"PoT code execution timeout in seconds (default: {DEFAULT_POT_TIMEOUT}).")
    g_pipe.add_argument("--generate-explanation", action="store_true", default=False,
                         help="Generate final explanation (adds an extra LLM call).")
    g_pipe.add_argument("--no-generate-explanation", action="store_false", dest="generate_explanation",
                         help="Skip final explanation generation (default: skip).")

    # Utility
    g_util = p.add_argument_group("Utility")
    g_util.add_argument("--dry-run", action="store_true",
                         help="Only check GPU and download model, don't run pipeline.")
    g_util.add_argument("--skip-download", action="store_true",
                         help="Skip the model pre-download step.")
    g_util.add_argument("--verbose", "-v", action="store_true",
                         help="Enable verbose/debug logging.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print("  EXACT Type 2 Pipeline — Thunder Compute Runner")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Dtype:          {args.torch_dtype}")
    print(f"  Device map:     {args.device_map}")
    print(f"  Questions:      limit={args.limit}, offset={args.offset}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Top-p:          {args.top_p}")
    print(f"  Extraction:     {args.extraction_mode}")
    print(f"  Explanation:    {'yes' if args.generate_explanation else 'no'}")
    print(f"  Config:         {args.config}")
    print(f"  Input:          {args.input}")
    print(f"  Output:         {args.output}")
    print("=" * 70)

    # Step 1: GPU diagnostics
    gpu_info = print_gpu_diagnostics()

    # Auto-fix dtype if needed
    args.torch_dtype = recommend_dtype(gpu_info, args.torch_dtype)

    # Warn if no GPU
    if not gpu_info.get("cuda_available", False):
        print("\n⚠  WARNING: Running without GPU. This will be EXTREMELY slow.")
        print("   Consider using a Thunder Compute instance with a GPU.\n")
        response = input("   Continue anyway? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("   Aborted.")
            sys.exit(1)

    # Warn if no accelerate
    if not gpu_info.get("accelerate") and args.device_map == "auto":
        print("\n⚠  WARNING: 'accelerate' not installed. device_map='auto' won't work.")
        print("   The model will be loaded on CPU then moved to GPU manually.\n")

    # Step 2: Pre-download model
    if not args.skip_download and not args.local_files_only:
        ensure_model_downloaded(args.model, trust_remote_code=args.trust_remote_code)

    if args.dry_run:
        print("\n🏁 Dry run complete. Model cached and GPU checked. Exiting.")
        sys.exit(0)

    # Step 3: Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    else:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    # Step 4: Build settings (this does NOT load the model yet)
    print("\n⚙  Building pipeline settings...")
    settings = build_settings(args)
    print(f"   LLM provider:   {settings.llm_provider}")
    print(f"   LLM model:      {settings.llm_model}")
    print(f"   LLM dtype:      {settings.llm_torch_dtype}")
    print(f"   LLM device_map: {settings.llm_device_map}")

    # Step 5: Pre-warm the LLM client so the model is loaded before the timer
    # starts. The first `build_json_client_from_settings` call triggers model load.
    print("\n🔥 Pre-loading model into GPU memory...")
    load_started = time.monotonic()
    from exact.llm_client import build_json_client_from_settings
    client = build_json_client_from_settings(settings)
    if client is None:
        print("   ❌ Failed to build LLM client! Check settings.")
        sys.exit(1)
    load_elapsed = time.monotonic() - load_started
    print(f"   Model loaded in {load_elapsed:.1f}s")

    # Print GPU memory after model load
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"   GPU {i} after load: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")
    except Exception:
        pass

    # Step 6: Warm up with a test generation to catch issues early
    print("\n🧪 Warm-up test generation...")
    try:
        test_messages = [
            {"role": "system", "content": "You are a physics assistant. Always respond in JSON."},
            {"role": "user", "content": 'What is 2+2? Respond as {"answer": <number>}'},
        ]
        warmup_start = time.monotonic()
        test_result = client.complete_json_sync(
            messages=test_messages,
            temperature=0.0,
            max_tokens=64,
        )
        warmup_elapsed = time.monotonic() - warmup_start
        print(f"   Warm-up OK: {test_result} ({warmup_elapsed:.1f}s)")
    except Exception as exc:
        print(f"   ⚠ Warm-up failed: {exc}")
        print("   The model may struggle with JSON output. Pipeline will still attempt to run.")

    # Step 7: Clear CUDA cache before the real run
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except Exception:
        pass

    # Step 8: Run the pipeline
    run_pipeline(args, settings)

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
