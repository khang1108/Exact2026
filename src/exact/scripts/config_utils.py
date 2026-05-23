from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from exact.config import Settings, get_settings


def load_toml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as file:
        return tomllib.load(file)


def build_settings_from_config(config: dict[str, Any]) -> Settings:
    settings = get_settings()
    llm = config.get("llm", {})
    pipeline = config.get("pipeline", {})

    backend = str(llm.get("backend", "none")).strip().lower()
    enabled = bool(llm.get("enabled", backend != "none"))
    if not enabled or backend == "none":
        return _settings_without_llm(settings)

    base_url = str(llm.get("base_url") or "").strip() or None
    api_key = _resolve_api_key(llm)
    provider = "openai"

    if backend == "transformers":
        provider = "local"
        base_url = None
        api_key = None
    elif backend == "ollama":
        base_url = base_url or "http://127.0.0.1:11434/v1"
        api_key = api_key or "ollama"
    elif backend == "huggingface":
        base_url = base_url or "https://router.huggingface.co/v1"
        api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    elif backend == "openai_compatible":
        if not base_url:
            raise ValueError("llm.base_url is required for backend='openai_compatible'")
        api_key = api_key or "EMPTY"
    else:
        raise ValueError(f"Unsupported llm.backend: {backend}")

    updates: dict[str, Any] = {
        "llm_provider": provider,
        "llm_model": str(llm.get("model") or settings.llm_model),
        "llm_base_url": base_url,
        "llm_api_key": SecretStr(api_key) if api_key else None,
        "llm_max_tokens": int(llm.get("max_tokens", settings.llm_max_tokens)),
        "llm_temperature": float(llm.get("temperature", settings.llm_temperature)),
        "llm_top_p": float(llm.get("top_p", settings.llm_top_p)),
        "llm_timeout_seconds": float(llm.get("timeout_seconds", settings.llm_timeout_seconds)),
        "llm_max_retries": int(llm.get("max_retries", settings.llm_max_retries)),
        "llm_device_map": _optional_string(llm.get("device_map", settings.llm_device_map)),
        "llm_torch_dtype": str(llm.get("torch_dtype", settings.llm_torch_dtype)),
        "llm_local_files_only": bool(llm.get("local_files_only", settings.llm_local_files_only)),
        "llm_trust_remote_code": bool(llm.get("trust_remote_code", settings.llm_trust_remote_code)),
        "mock_llm": False,
    }

    if not bool(pipeline.get("use_type1_llm", True)) and not bool(
        pipeline.get("use_type2_llm_fallback", True)
    ):
        updates["mock_llm"] = True

    return settings.model_copy(update=updates)


def settings_for_disabled_llm(settings: Settings | None = None) -> Settings:
    return _settings_without_llm(settings or get_settings())


def _settings_without_llm(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "llm_provider": "openai",
            "llm_base_url": None,
            "llm_api_key": None,
            "mock_llm": True,
        }
    )


def _resolve_api_key(llm: dict[str, Any]) -> str | None:
    api_key = str(llm.get("api_key") or "").strip()
    if api_key:
        return api_key

    env_name = str(llm.get("api_key_env") or "").strip()
    if env_name:
        return os.getenv(env_name)

    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text
