"""LLM-backed NL→logic translation for Type 1."""

from exact.logic.translation.llm_translator import (
    JsonLLMClient,
    clear_formula_premise_cache,
    translate_formula_goals_with_llm,
    translate_formula_premises_only_with_llm,
    translate_mcq_options_with_llm,
    translate_premises_only_with_llm,
    translate_problem_with_llm,
    translate_query_only_with_llm,
)

__all__ = [
    "JsonLLMClient",
    "clear_formula_premise_cache",
    "translate_formula_goals_with_llm",
    "translate_formula_premises_only_with_llm",
    "translate_mcq_options_with_llm",
    "translate_premises_only_with_llm",
    "translate_problem_with_llm",
    "translate_query_only_with_llm",
]
