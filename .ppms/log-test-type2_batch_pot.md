# EXACT 2026 Type 2 Changelog

## [2026-06-13T00:52:00+07:00]
- **Prompt Summary**: Allow the LLM to self-compose either numeric or text answers for 'ans' in the PoT program, adjusting prompt templates and python execution success checks to accept both formats.
- **Modified Files**:
  - `src/exact/type2/extraction/llm_structured.py`: Updated `_build_pot_messages` and `_build_repair_messages` to instruct the LLM that `ans` can be a numeric value or a qualitative text string.
  - `src/exact/type2/solving/pot_solver.py`: Modified `_is_execution_successful` and `_verify_or_accept_execution` to accept qualitative text answers and bypass strict float check for non-numeric answers.
- **Architecture Impact**:
  - The Program-of-Thought pipeline is generalized to allow qualitative/textual reasoning output directly from the sandbox execution, in addition to pure numeric magnitudes.

## [2026-06-13T00:48:00+07:00]
- **Prompt Summary**: Adjust LLM domain router to only route to conceptual, numerical, and mixed kinds. Have the LLM select formula IDs from the retrieved formulas, activate a loop to run PoT code until a numeric result is computed using the LLM's pre-trained knowledge, and disable final verification checks.
- **Modified Files**:
  - `src/exact/type2/domains/router.py`: Changed valid domains to `numerical`, `conceptual`, `mixed` and updated the prompt and heuristic.
  - `src/exact/type2/pipeline.py`: Restored and updated pipelines to support and pass `domain_hint` to override extraction kind.
  - `src/exact/type2/solving/pot_solver.py`: Added LLM formula selection step using `select_formula_ids`, extended the repair loop to run up to 10+ retries checking for valid numeric outcomes, and bypassed final unit/oracle verification checks.
  - `src/exact/type2/extraction/llm_structured.py`: Updated system messages/prompts to instruct the LLM to use its pre-trained knowledge when solving or repairing PoT programs.
  - `tests/test_type2_llm_domain_routing.py`: Updated mock domains and routing assertions to match new taxonomy (numerical/conceptual).
- **Created Files**:
  - `.ppms/architecture-test-type2_batch_pot.md`: Core architecture documentation for the branch.
  - `.ppms/log-test-type2_batch_pot.md`: Chronological changelog for the branch.
- **Architecture Impact**:
  - Question domain classification is simplified from subfields (LD, TD, etc.) to general question modes (numerical, conceptual, mixed).
  - Program-of-Thought (PoT) flow now incorporates a strict formula-selection phase and a robust 10+ attempt repair loop to guarantee a numeric answer before completion, without requiring post-hoc unit/oracle verification.
