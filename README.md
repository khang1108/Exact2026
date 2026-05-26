# EXACT 2026 system

Hybrid neuro-symbolic pipeline for the [EXACT 2026](https://ura.hcmut.edu.vn/exact) challenge (transparent educational QA).

## Layout

- `src/exact/` — core library (`exact.config`, pipelines, solvers)
- `app/` — FastAPI service (`uvicorn app.main:app`)
- `data/` — datasets, formula banks, few-shot pools
- `eval/` — offline metrics and runs
- `docs/` — challenge notes and experiment logs
- `paper/` — paper drafts

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e ".[api,dev]"
cp .env.example .env
PYTHONPATH=src uvicorn exact.app.main:app --host 0.0.0.0 --port 8080
```

Health check: `GET http://localhost:8080/health`

The API exposes:

- `GET /health`
- `POST /predict`
- `POST /batch`

Each prediction returns at least `answer`, `explanation`, `fol`, `cot`, `premises`,
and `confidence`, with local metadata such as `id`, `task_type`, `question_type`,
`unit`, and `error` included during development.

Configure a local OpenAI-compatible LLM server for real predictions:

```bash
export EXACT_LLM_BASE_URL=http://127.0.0.1:8001/v1
export EXACT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Type 1 is LLM-only: if no JSON LLM client is configured, the request fails with
a clear error instead of substituting a local parser.

Type 2 is currently a clean extension point. The API routes physics questions
there and returns a structured placeholder response until the team implements
the paper-backed physics pipeline.

Example Type 1 request:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": "t1_001",
    "premises-NL": [
      "If a curriculum is well-structured and has exercises, it enhances student engagement.",
      "The curriculum is well-structured.",
      "The curriculum has exercises."
    ],
    "question": "Does the curriculum enhance student engagement?"
  }'
```

Example Type 2 request:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": "t2_001",
    "question": "Calculate the current when U = 12 V and R = 6 ohm."
  }'
```

## Tests

```bash
pytest
```

## Type 2 Dataset Runs

```bash
cp configs/type2_dataset_run.example.toml configs/type2_local.toml
PYTHONPATH=src python -m exact.scripts.run_dataset_from_config --config configs/type2_local.toml
```

Evaluate a Type 2 prediction file with the tolerances configured in the same
TOML file:

```bash
PYTHONPATH=src python -m exact.scripts.evaluate_type2_predictions \
  artifacts/predictions/type2/type2_config_smoke.json \
  --config configs/type2_local.toml
```

Monitor a Type 2 run with live correct/wrong/error counters:

```bash
PYTHONPATH=src python -m exact.scripts.run_type2_monitor \
  --config configs/type2_local.toml
```

Useful settings in `configs/type2_local.toml`:

- `llm.backend = "transformers"` loads a Hugging Face model directly.
- `llm.backend = "ollama"` calls an Ollama OpenAI-compatible endpoint.
- `llm.backend = "groq"` calls Groq's OpenAI-compatible API.
- `llm.backend = "openai_compatible"` calls a local/cloud GPU server such as vLLM.
- `llm.backend = "huggingface"` calls Hugging Face's OpenAI-compatible router.
- `pipeline.use_type2_llm_fallback = true` enables Type 2 PoT-first solving:
  formula retrieval, LLM-generated Pint code, sandbox execution, PoT
  verification, and final LLM evidence generation.
- `type2_pipeline.generate_final_explanation = false` skips the final
  explanation/evidence LLM call for faster smoke runs.

For CPU smoke tests with transformers, start with a small model:

```toml
[llm]
enabled = true
backend = "transformers"
model = "Qwen/Qwen2.5-0.5B-Instruct"
device_map = "cpu"
torch_dtype = "float32"

[pipeline]
use_type1_llm = true
use_type2_llm_fallback = true
```

To download/cache the configured Hugging Face model first:

```bash
PYTHONPATH=src python -m exact.scripts.pull_model --config configs/type2_local.toml
```

For Ollama:

```toml
[llm]
enabled = true
backend = "ollama"
model = "qwen2.5:0.5b"
base_url = "http://127.0.0.1:11434/v1"
```

For a cloud or LAN GPU server exposing an OpenAI-compatible API:

```toml
[llm]
enabled = true
backend = "openai_compatible"
model = "Qwen/Qwen2.5-7B-Instruct"
base_url = "http://YOUR_SERVER:8000/v1"
api_key = "EMPTY"
```

For Groq:

```toml
[llm]
enabled = true
backend = "groq"
model = "llama-3.1-8b-instant"
api_key_env = "GROQ_API_KEY"
```
