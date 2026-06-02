# EXACT 2026 — Architecture Overview (branch: main)

## Project Purpose
EXACT 2026 is a competitive AI system for solving educational questions:
- **Type 1**: Logic-based queries (FOL translation → Z3/symbolic solving)
- **Type 2**: Physics numerical questions (extraction → formula retrieval → PoT code → answer)

## Tech Stack
- Python 3.11+, Pydantic v2, pydantic-settings
- PyTorch, HuggingFace Transformers, Accelerate
- OpenAI-compatible LLM clients (Cloudflare, Groq, Ollama, HuggingFace, local transformers)
- FastAPI + Uvicorn (API server)
- Z3, SymPy, Pint (symbolic + unit computation)

## Directory Structure
```
Exact2026/
├── configs/                     # TOML run configs
│   ├── type2_dataset_run.example.toml
│   └── type2_thunder.toml       # ← NEW: Thunder Compute optimized config
├── scripts/
│   └── type2/
│       ├── run_type2.py         # Original runner (subprocess-based)
│       └── run_type2_thunder.py # ← NEW: In-process Thunder Compute runner
├── src/exact/
│   ├── config.py                # Settings (pydantic-settings, env-based)
│   ├── llm_client.py            # LLM clients: OpenAI-compatible, LocalClient, LocalJsonClient
│   ├── type2/
│   │   ├── pipeline.py          # run_type2_pipeline() — main orchestrator
│   │   ├── extraction/          # Heuristic + LLM extractors
│   │   ├── formulas/            # Formula bank, retrieval, reranking
│   │   └── solving/             # PoT solver, unit handling
│   ├── scripts/
│   │   ├── run_type2_monitor.py # CLI runner with live eval stats
│   │   ├── config_utils.py      # TOML → Settings builder
│   │   ├── pull_model.py        # HF model downloader
│   │   └── run_predictions.py   # Generic prediction runner
│   └── datasets/                # Dataset loaders
├── artifacts/                   # Output predictions, reports
├── requirements.txt
└── pyproject.toml
```

## LLM Backend Architecture
1. **BaseJsonLLMClient** (ABC) → `complete_json_sync()` / `complete_json()`
2. **OpenAICompatibleJsonClient** → HTTP calls to OpenAI-compatible APIs
3. **LLMClient** → alias for OpenAICompatibleJsonClient
4. **LocalClient** (BaseTextLLMClient) → HF Transformers `AutoModelForCausalLM`
5. **LocalJsonClient** (BaseJsonLLMClient) → wraps LocalClient, parses JSON from text output

## Config Flow
```
CLI args → run_type2.py → temp TOML → run_type2_monitor.py
             ↓
         build_config_text() → TOML sections
             ↓
         config_utils.build_settings_from_config() → Settings object
             ↓
         build_json_client_from_settings() → LocalJsonClient (for transformers)
```

## Type 2 Pipeline Flow
```
Question → extract_type2() (heuristic)
         → parse_with_llm() (LLM extraction)
         → merge extractions
         → retrieve_formula_context()
         → solve_with_pot() (LLM code gen + exec)
         → PredictionResponse
```
