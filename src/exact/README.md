# `exact` Package

The active package currently implements the EXACT Type 2 physics pipeline.
The previous Type 1 logic implementation and symbolic solver packages have
been removed for a clean rewrite.

## Structure

```text
src/exact/
├── app/          # FastAPI routes and application entry point
├── common/       # Shared request and response contracts
├── datasets/     # Dataset loading and normalization
├── prompts/      # Shared prompt templates
├── scripts/      # Type 2 run and evaluation helpers
├── type2/        # Type 2 extraction, routing, solvers, and fallbacks
├── config.py     # Runtime settings
├── llm_client.py # LLM client implementations
└── logger.py     # Logging setup
```

## Boundaries

- `datasets/` normalizes raw files but does not solve questions.
- `app/` validates requests and invokes the Type 2 pipeline.
- `type2/` owns task-specific extraction, routing, solving, and explanations.
- `common/` contains only contracts used across package boundaries.
