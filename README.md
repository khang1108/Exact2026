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
PYTHONPATH=src:. uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Health check: `GET http://localhost:8080/health`

## Tests

```bash
pytest
```
