from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
PACKAGE_DIR = SRC_DIR / "exact"

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        env_prefix="EXACT_",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Literal["local", "dev", "test", "prod"] = "local"

    data_dir: Path = PACKAGE_DIR / "datasets"
    exact_dataset_dir: Path = PACKAGE_DIR / "datasets" / "exact"
    type2_path: Path = exact_dataset_dir / "type2_physics_questions.csv"

    artifacts_dir: Path = ROOT_DIR / "artifacts"
    predictions_dir: Path = artifacts_dir / "predictions"
    reports_dir: Path = artifacts_dir / "reports"
    splits_dir: Path = artifacts_dir / "splits"
    normalized_data_path: Path = artifacts_dir / "normalized_dataset.jsonl"

    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.70, "dev": 0.15, "test": 0.15}
    )
    default_seed: int = 42

    llm_provider: Literal["openai", "anthropic", "groq", "ollama", "local"] = "local"
    llm_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        validation_alias=AliasChoices("EXACT_LLM_MODEL", "EXACT_MODEL_ID"),
    )
    math_model_id: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_max_tokens: int = Field(default=2048, ge=1, validation_alias="EXACT_MAX_NEW_TOKENS")
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices("EXACT_LLM_TEMPERATURE", "EXACT_TEMPERATURE"),
    )
    llm_top_p: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        validation_alias=AliasChoices("EXACT_LLM_TOP_P", "EXACT_TOP_P"),
    )
    llm_timeout_seconds: float = Field(default=55.0, gt=0)
    llm_max_retries: int = Field(default=0, ge=0, validation_alias="EXACT_MAX_RETRIES")
    llm_device_map: str | None = Field(default="auto", validation_alias="EXACT_LLM_DEVICE_MAP")
    llm_torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="float16",
        validation_alias="EXACT_LLM_TORCH_DTYPE",
    )
    llm_local_files_only: bool = Field(default=False, validation_alias="EXACT_LLM_LOCAL_FILES_ONLY")
    llm_trust_remote_code: bool = Field(default=False, validation_alias="EXACT_LLM_TRUST_REMOTE_CODE")

    # Type 2 Physics Pipeline Settings
    type2_extraction_mode: Literal["merge", "llm_only", "heuristic_only"] = Field(
        default="merge",
        validation_alias="EXACT_TYPE2_EXTRACTION_MODE",
    )
    type2_use_extraction_verifier: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_EXTRACTION_VERIFIER",
    )
    type2_use_llm_formula_selection: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_LLM_FORMULA_SELECTION",
    )
    type2_use_formula_bank: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_FORMULA_BANK",
    )
    type2_use_unit_verifier: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_UNIT_VERIFIER",
    )
    type2_force_llm_formula_selection: bool = Field(
        default=False,
        validation_alias="EXACT_TYPE2_FORCE_LLM_FORMULA_SELECTION",
    )
    type2_use_concept_bank: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_CONCEPT_BANK",
    )
    type2_use_pot_solver: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_POT_SOLVER",
    )
    type2_deterministic_first: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_DETERMINISTIC_FIRST",
    )
    type2_use_executable_fallback: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_EXECUTABLE_FALLBACK",
    )
    type2_pot_max_retries: int = Field(default=3, ge=0, validation_alias="EXACT_TYPE2_POT_MAX_RETRIES")
    type2_formula_limit: int = Field(default=24, ge=1, validation_alias="EXACT_TYPE2_FORMULA_LIMIT")
    type2_rerank_limit: int = Field(default=12, ge=1, validation_alias="EXACT_TYPE2_RERANK_LIMIT")
    type2_generate_explanation: bool = Field(default=True, validation_alias="EXACT_TYPE2_GENERATE_EXPLANATION")
    type2_pot_timeout: float = Field(default=10.0, gt=0.0, validation_alias="EXACT_TYPE2_POT_TIMEOUT")
    type2_extraction_max_tokens: int = Field(
        default=768,
        ge=1,
        validation_alias="EXACT_TYPE2_EXTRACTION_MAX_TOKENS",
    )
    type2_formula_selection_max_tokens: int = Field(
        default=768,
        ge=1,
        validation_alias="EXACT_TYPE2_FORMULA_SELECTION_MAX_TOKENS",
    )
    type2_conceptual_max_tokens: int = Field(
        default=1024,
        ge=1,
        validation_alias="EXACT_TYPE2_CONCEPTUAL_MAX_TOKENS",
    )
    type2_pot_code_max_tokens: int = Field(
        default=3072,
        ge=1,
        validation_alias="EXACT_TYPE2_POT_CODE_MAX_TOKENS",
    )
    type2_pot_repair_max_tokens: int = Field(
        default=2048,
        ge=1,
        validation_alias="EXACT_TYPE2_POT_REPAIR_MAX_TOKENS",
    )
    type2_final_explanation_max_tokens: int = Field(
        default=768,
        ge=1,
        validation_alias="EXACT_TYPE2_FINAL_EXPLANATION_MAX_TOKENS",
    )
    type2_debug_log_pot_prompts: bool = Field(
        default=False,
        validation_alias="EXACT_TYPE2_DEBUG_LOG_POT_PROMPTS",
    )
    type2_debug_pot_prompt_log_path: str = Field(
        default="artifacts/debug/type2_pot_prompts.jsonl",
        validation_alias="EXACT_TYPE2_DEBUG_POT_PROMPT_LOG_PATH",
    )

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8080, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_logs: bool = False

    def ensure_artifact_dirs(self) -> None:
        for directory in [
            self.artifacts_dir,
            self.predictions_dir,
            self.reports_dir,
            self.splits_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
