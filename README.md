# EXACT 2026 Type 2 Pipeline

This repository currently contains the Type 2 physics pipeline for the EXACT
2026 challenge. The previous Type 1 logic implementation has been removed so it
can be rebuilt from scratch.

## Request Flow

```text
POST /predict
    -> PredictionRequest
    -> Type 2 domain routing
    -> deterministic solver when eligible
    -> LLM Program-of-Thought fallback when needed
    -> PredictionResponse
```

## Layout

- `src/exact/app/`: FastAPI service.
- `src/exact/common/`: shared request and response schemas.
- `src/exact/datasets/`: challenge dataset loading and normalization.
- `src/exact/type2/`: Type 2 extraction, routing, deterministic solvers, and fallbacks.
- `src/exact/scripts/`: Type 2 dataset and evaluation CLIs.
- `scripts/type2/`: Type 2 execution entry points.
- `artifacts/`: generated predictions and reports.

The preserved logic dataset files remain under `src/exact/datasets/exact/` for
future Type 1 redevelopment, but there is no active Type 1 runtime or routing.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,dev]"
bash scripts/serve_exact_api.sh
```

Health check: `GET http://localhost:8080/health`

Committee-facing vLLM model discovery is publicly proxied through the EXACT API
without exposing the private vLLM API key:

```bash
curl https://api.iamphuckhang.dev/v1/models
bash scripts/check_committee_models.sh
```

## Type 2 Dataset Runs

```bash
cp configs/type2_dataset_run.example.toml configs/type2_local.toml
./venv/bin/python scripts/type2/run_type2.py --backend vllm --base-url http://YOUR_VM:8000/v1 --config configs/type2_local.toml
```

Evaluate predictions:

```bash
PYTHONPATH=src python -m exact.scripts.evaluate_type2_predictions \
  artifacts/predictions/type2/type2_config_smoke.json \
  --config configs/type2_local.toml
```

## Tests

```bash
pytest
```
