# EXACT 2026 — Change Log (branch: main)

## 2026-06-02T21:42:00+07:00 — Thunder Compute Type 2 Runner

**Prompt**: Viết script chạy pipeline type 2 với model transformers trên Thunder Compute, configurable params.

**Changes**:
- Created `scripts/type2/run_type2_thunder.py` — self-contained in-process runner that fixes issues with `run_type2.py`:
  - Loads model directly in-process (no subprocess spawn → no PYTHONPATH/env issues)
  - GPU diagnostics with CUDA memory, dtype compatibility, accelerate checks
  - Auto dtype fallback (bfloat16 → float16 if GPU doesn't support bf16)
  - Model pre-download with progress
  - Warm-up test generation before real run
  - Detailed per-question progress with live accuracy tracking
  - Configurable: `--limit`, `--offset`, `--model`, `--torch-dtype`, `--device-map`, `--temperature`, `--top-p`, `--extraction-mode`, `--pot-max-retries`, `--pot-timeout`, `--generate-explanation`, `--dry-run`
- Created `configs/type2_thunder.toml` — optimized config for local transformers:
  - `deterministic_first = true` (try symbolic solver before LLM)
  - `generate_final_explanation = false` (skip extra LLM call)
  - Reduced token budgets for smaller models

**Files created**:
- `scripts/type2/run_type2_thunder.py`
- `configs/type2_thunder.toml`

**Architecture impact**: New entry point for local GPU runs. No changes to existing pipeline code or config structure. Uses existing `build_settings_from_config()` and `build_json_client_from_settings()`.
