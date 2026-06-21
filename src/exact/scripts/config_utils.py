from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from exact.config import Settings, get_settings, validate_self_hosted_model_url


ROOT_DIR = Path(__file__).resolve().parents[3]


def load_toml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as file:
        return tomllib.load(file)


def build_settings_from_config(config: dict[str, Any]) -> Settings:
    llm = config.get("llm", {})
    type2_pipeline = config.get("type2_pipeline", {})
    settings = _base_settings_for_config(llm)
    type2_updates = _type2_settings_updates(type2_pipeline, settings)

    backend = str(llm.get("backend", "none")).strip().lower()
    enabled = bool(llm.get("enabled", backend != "none"))
    if not enabled or backend == "none":
        return _settings_without_llm(settings).model_copy(update=type2_updates)

    raw_base_url = str(llm.get("base_url") or "").strip()
    base_url = validate_self_hosted_model_url(raw_base_url) if raw_base_url else None
    api_key = _resolve_api_key(llm)
    if backend not in ("vllm", "transformers"):
        raise ValueError("Only backend='vllm' or 'transformers' is supported")
    if backend == "vllm" and not base_url:
        raise ValueError("llm.base_url is required for the self-hosted vLLM server")
    api_key = api_key or "EMPTY"

    updates: dict[str, Any] = {
        "llm_enabled": True,
        "llm_backend": backend,
        "llm_model": str(llm.get("model") or settings.llm_model),
        "llm_base_url": base_url,
        "llm_api_key": SecretStr(api_key) if api_key else None,
        "llm_max_tokens": int(llm.get("max_tokens", settings.llm_max_tokens)),
        "llm_temperature": float(llm.get("temperature", settings.llm_temperature)),
        "llm_top_p": float(llm.get("top_p", settings.llm_top_p)),
        "llm_timeout_seconds": float(llm.get("timeout_seconds", settings.llm_timeout_seconds)),
        "llm_max_retries": int(llm.get("max_retries", settings.llm_max_retries)),
        **type2_updates,
    }

    return settings.model_copy(update=updates)


def _base_settings_for_config(llm: dict[str, Any]) -> Settings:
    backend = str(llm.get("backend", "none")).strip().lower()
    enabled = bool(llm.get("enabled", backend != "none"))
    if enabled and backend != "none":
        return get_settings()

    # When the runner is configured for deterministic-only execution, avoid
    # loading a potentially invalid external LLM endpoint from environment.
    return Settings(llm_base_url=None, llm_api_key=None)


def _type2_settings_updates(type2_pipeline: dict[str, Any], settings: Settings) -> dict[str, Any]:
    extraction_mode = str(
        type2_pipeline.get("extraction_mode", settings.type2_extraction_mode)
    ).strip().lower()
    if extraction_mode == "llm_preferred":
        extraction_mode = "merge"
    if extraction_mode not in {"merge", "llm_only", "heuristic_only"}:
        raise ValueError(
            "type2_pipeline.extraction_mode must be one of: merge, llm_only, heuristic_only"
        )

    return {
        "type2_use_llm_domain_routing": bool(
            type2_pipeline.get(
                "use_llm_domain_routing",
                settings.type2_use_llm_domain_routing,
            )
        ),
        "type2_use_llm_question_kind_routing": bool(
            type2_pipeline.get(
                "use_llm_question_kind_routing",
                settings.type2_use_llm_question_kind_routing,
            )
        ),
        "type2_use_recovery_loop": bool(
            type2_pipeline.get("use_recovery_loop", settings.type2_use_recovery_loop)
        ),
        "type2_extraction_mode": extraction_mode,
        "type2_use_extraction_verifier": bool(
            type2_pipeline.get("use_extraction_verifier", settings.type2_use_extraction_verifier)
        ),
        "type2_use_llm_formula_selection": bool(
            type2_pipeline.get("use_llm_formula_selection", settings.type2_use_llm_formula_selection)
        ),
        "type2_use_formula_bank": bool(
            type2_pipeline.get("use_formula_bank", settings.type2_use_formula_bank)
        ),
        "type2_use_unit_verifier": bool(
            type2_pipeline.get("use_unit_verifier", settings.type2_use_unit_verifier)
        ),
        "type2_force_llm_formula_selection": bool(
            type2_pipeline.get("force_llm_formula_selection", settings.type2_force_llm_formula_selection)
        ),
        "type2_use_concept_bank": bool(
            type2_pipeline.get("use_concept_bank", settings.type2_use_concept_bank)
        ),
        "type2_use_pot_solver": bool(
            type2_pipeline.get("use_pot_solver", settings.type2_use_pot_solver)
        ),
        "type2_deterministic_first": bool(
            type2_pipeline.get("deterministic_first", settings.type2_deterministic_first)
        ),
        "type2_use_executable_fallback": bool(
            type2_pipeline.get("use_executable_fallback", settings.type2_use_executable_fallback)
        ),
        "type2_pot_max_retries": int(
            type2_pipeline.get("pot_max_retries", settings.type2_pot_max_retries)
        ),
        "type2_recovery_loop_max_attempts": int(
            type2_pipeline.get(
                "recovery_loop_max_attempts",
                settings.type2_recovery_loop_max_attempts,
            )
        ),
        "type2_pot_batch_size": int(
            type2_pipeline.get("pot_batch_size", settings.type2_pot_batch_size)
        ),
        "type2_pot_batch_temperature": float(
            type2_pipeline.get(
                "pot_batch_temperature",
                settings.type2_pot_batch_temperature,
            )
        ),
        "type2_formula_limit": int(
            type2_pipeline.get("formula_limit", settings.type2_formula_limit)
        ),
        "type2_rerank_limit": int(
            type2_pipeline.get("rerank_limit", settings.type2_rerank_limit)
        ),
        "type2_generate_explanation": bool(
            type2_pipeline.get("generate_final_explanation", settings.type2_generate_explanation)
        ),
        "type2_pot_timeout": float(
            type2_pipeline.get("pot_timeout", settings.type2_pot_timeout)
        ),
        "type2_extraction_max_tokens": int(
            type2_pipeline.get("extraction_max_tokens", settings.type2_extraction_max_tokens)
        ),
        "type2_domain_routing_max_tokens": int(
            type2_pipeline.get(
                "domain_routing_max_tokens",
                settings.type2_domain_routing_max_tokens,
            )
        ),
        "type2_question_kind_max_tokens": int(
            type2_pipeline.get(
                "question_kind_max_tokens",
                settings.type2_question_kind_max_tokens,
            )
        ),
        "type2_recovery_loop_max_tokens": int(
            type2_pipeline.get(
                "recovery_loop_max_tokens",
                settings.type2_recovery_loop_max_tokens,
            )
        ),
        "type2_formula_selection_max_tokens": int(
            type2_pipeline.get(
                "formula_selection_max_tokens",
                settings.type2_formula_selection_max_tokens,
            )
        ),
        "type2_conceptual_max_tokens": int(
            type2_pipeline.get("conceptual_max_tokens", settings.type2_conceptual_max_tokens)
        ),
        "type2_pot_code_max_tokens": int(
            type2_pipeline.get("pot_code_max_tokens", settings.type2_pot_code_max_tokens)
        ),
        "type2_pot_repair_max_tokens": int(
            type2_pipeline.get("pot_repair_max_tokens", settings.type2_pot_repair_max_tokens)
        ),
        "type2_final_explanation_max_tokens": int(
            type2_pipeline.get(
                "final_explanation_max_tokens",
                settings.type2_final_explanation_max_tokens,
            )
        ),
        "type2_debug_log_pot_prompts": bool(
            type2_pipeline.get(
                "debug_log_pot_prompts",
                settings.type2_debug_log_pot_prompts,
            )
        ),
        "type2_debug_pot_prompt_log_path": str(
            type2_pipeline.get(
                "debug_pot_prompt_log_path",
                settings.type2_debug_pot_prompt_log_path,
            )
        ),
    }


def settings_for_disabled_llm(settings: Settings | None = None) -> Settings:
    return _settings_without_llm(settings or get_settings())


def _settings_without_llm(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "llm_enabled": False,
            "llm_backend": "none",
            "llm_base_url": None,
            "llm_api_key": None,
        }
    )


def _resolve_api_key(llm: dict[str, Any]) -> str | None:
    api_key = str(llm.get("api_key") or "").strip()
    if api_key:
        return api_key

    env_name = str(llm.get("api_key_env") or "").strip()
    if env_name:
        return os.getenv(env_name) or _dotenv_value(env_name)

    return None


def _dotenv_value(name: str, path: Path | None = None) -> str | None:
    env_path = path or ROOT_DIR / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value or None
    return None
