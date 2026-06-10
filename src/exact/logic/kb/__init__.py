"""Knowledge-base construction and premise caching."""

from exact.logic.kb.kb import (
    LLM_PARSER_VERSION,
    KnowledgeBase,
    build_kb_candidates_from_premises,
    build_kb_from_parsed_premises,
    build_kb_from_premises,
    clear_kb_cache,
    get_or_build_kb,
    get_or_build_kb_candidates,
    hash_premises,
)

__all__ = [
    "LLM_PARSER_VERSION",
    "KnowledgeBase",
    "build_kb_candidates_from_premises",
    "build_kb_from_parsed_premises",
    "build_kb_from_premises",
    "clear_kb_cache",
    "get_or_build_kb",
    "get_or_build_kb_candidates",
    "hash_premises",
]
