# EXACT 2026 Type 2 Architecture Document

## Overview
This document describes the architecture of the EXACT 2026 challenge's Type 2 physics pipeline under the `test-type2_batch_pot` branch.

## Request Flow
```text
POST /predict
    -> PredictionRequest
    -> Type 2 question kind classification (domain routing) -> numerical / conceptual / mixed
    -> deterministic solver when eligible
    -> LLM Formula Selector (selects relevant formulas from the retrieved context)
    -> PoT Solver with robust retry loop (tries code execution & repair up to 10 times, uses pre-trained knowledge, loops until code runs successfully and produces any valid non-empty numeric or text value)
    -> Final explanation/Cot construction without oracle verification checks
    -> PredictionResponse
```

## Directory Layout
- `src/exact/app/`: FastAPI service endpoints and request routing.
- `src/exact/common/`: Shared request and response Pydantic schemas.
- `src/exact/datasets/`: Challenge dataset loading and taxonomy classification.
- `src/exact/type2/`:
  - `domains/router.py`: Domain routing logic classifying questions into `numerical`, `conceptual`, and `mixed`.
  - `extraction/llm_structured.py`: LLM schema validations, prompt templates, and classification functions.
  - `solving/pot_solver.py`: Program-of-Thought (PoT) execution, repair loop, formula selection integration, and output formatting.
  - `pipeline.py`: Coordinates routing, extraction, deterministic checks, and the solver stages.

## Key Branch Architecture Changes
1. **Simplified Router**: Modified the LLM domain router to classify questions strictly into three modes: `numerical`, `conceptual`, and `mixed`.
2. **Formula Selector Integration**: Integrated `select_formula_ids` in `solve_with_pot` so the LLM selects a subset of formula IDs before writing the program code.
3. **Robust PoT Solver Loop**: Expanded the repair loop to retry up to `max(10, max_retries)`. Instructed the prompt templates to leverage the model's pre-trained physics knowledge in addition to context formulas. The LLM is allowed to self-compose the answer as either a numeric value or a qualitative text string, and the execution checks and sandbox output validation have been relaxed to support both formats.
4. **Verification Bypass**: Disabled unit sanity verifier and physics oracle verification checks at the end of code execution, directly accepting successful sandbox executions.
5. **LLM Output Normalization**: Added raw input normalization layer `_normalize_formula_choice_raw` to prevent validation errors in `FormulaChoiceSpec` when the LLM outputs a single string instead of a string list for list-type fields.
