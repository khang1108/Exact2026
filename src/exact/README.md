# `exact` package

Core Python package for the EXACT 2026 neuro-symbolic reasoning system.

This package contains reusable application code. Scripts, notebooks, raw data,
and experiment artifacts should live outside this package.

## Purpose

The system follows a controlled neuro-symbolic pipeline:

```text
raw dataset
    |
    v
data loader
    |
    v
normalized schemas
    |
    +--> baselines
    +--> logic translation
    +--> symbolic inference
    +--> self-refinement
    +--> evaluation
```

The main design rule is:

> Downstream modules should depend on typed schemas, not raw JSON/CSV formats.

## Package layout

```text
src/exact/
  config.py                    # Runtime settings loaded from environment variables
  logger.py                    # Application logging setup

  app/                         # API or application entrypoints

  datasets/
    schemas.py                 # Shared data contracts for examples and outputs
    loader.py                  # Raw dataset readers and normalizers

  baselines/                   # Baseline systems and reference experiments

  prompts/                     # Prompt templates for generation and refinement

  translation/
    logic_program_generator.py # Converts natural language tasks into logic programs

  solvers/
    base.py                    # Solver interface
    logic/                     # Logic solvers such as Z3/Prover9-style backends
    physics/                   # Physics solver integration boundary

  inference/
    engine.py                  # Executes generated programs through solvers
    result.py                  # Inference result models

  refinement/
    self_refiner.py            # Repairs failed logic programs using feedback

  evaluation/
    metrics.py                 # Metric functions
    evaluator.py               # Offline evaluation orchestration
```

Some directories above may be introduced gradually as the implementation grows.
Keep the package structure aligned with the pipeline stages.

## Module responsibilities

### `config.py`

Owns runtime configuration.

Examples:

- dataset version
- data and artifact paths
- LLM provider/model settings
- solver timeouts
- logging level

It should not load datasets, call models, run solvers, or compute metrics.

### `logger.py`

Owns application-wide logging setup.

Call logging setup once from an entrypoint such as a script, API server, or
notebook bootstrap. Library modules should use:

```python
import logging

logger = logging.getLogger(__name__)
```

### `datasets/schemas.py`

Defines typed contracts shared across the pipeline.

Schemas make the rest of the system independent from raw dataset details such
as CSV column names or JSON keys like `premises-NL`.

### `datasets/loader.py`

Converts raw EXACT dataset files into normalized schema objects.

Expected behavior:

```text
EXACT Type 1 JSON -> normalized logic examples
EXACT Type 2 CSV  -> normalized physics examples
```

The loader should not call LLMs, run symbolic solvers, or evaluate accuracy.

### `translation/`

Converts natural language problems into formal or semi-formal logic programs.

This stage may use LLM prompts, but it should save structured outputs that can
be inspected and reused by later stages.

### `solvers/`

Contains deterministic reasoning backends.

Solvers should expose a small interface so inference code does not depend on
one specific backend implementation.

### `inference/`

Coordinates execution of generated logic programs through solvers.

This layer owns:

- selecting the appropriate solver
- catching parse/execution failures
- returning structured inference results
- preserving failure status for refinement and evaluation

### `refinement/`

Repairs failed generated programs using solver feedback.

This should be a separate stage, not hidden inside evaluation or solver code.

### `evaluation/`

Scores saved predictions and inference outputs.

Evaluation code should be deterministic and should not call LLMs.

## Design rules

1. Keep raw dataset formats inside `datasets/loader.py`.
2. Pass typed schema objects between pipeline stages.
3. Keep prompting, inference, refinement, and evaluation separate.
4. Do not use notebooks as source code.
5. Do not use `print()` in package code; use module loggers.
6. Do not put experiment artifacts inside `src/exact/`.
7. Prefer small modules with one clear responsibility.
8. Add abstractions only after at least two real implementations need them.

## Import style

Use absolute imports inside the package:

```python
from exact.datasets.schemas import Example
from exact.datasets.loader import ExactDatasetLoader
```

Avoid importing from raw source-root modules such as:

```python
from config import settings
from logger import setup_logging
```

## Testing expectations

Each important module should have focused tests:

```text
tests/
  test_config.py
  test_loader.py
  test_metrics.py
  test_inference.py
```

Loader tests should verify that raw dataset files become valid normalized
examples. Evaluation tests should use fixed predictions and must not call LLMs.

## Relationship to Logic-LLM

This project follows the same high-level pattern as Logic-LLM:

```text
LLM generation -> symbolic inference -> self-refinement -> evaluation
```

However, this package uses clearer module boundaries and typed schemas so the
pipeline can grow into a maintainable research system.
