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

Without `EXACT_LLM_BASE_URL`, the pipelines return conservative fallback
responses instead of fake model answers.

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
