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
    type1_path: Path = exact_dataset_dir / "Logic_Based_Educational_Queries.json"
    type2_path: Path = exact_dataset_dir / "Physics_Problems_Text_Only.csv"

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
    mock_llm: bool = Field(
        default=False,
        validation_alias=AliasChoices("EXACT_MOCK_LLM", "MOCK_LLM"),
    )
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
    type1_translation_samples: int = Field(
        # Changed from 3 → 1: the new one-shot formula translator (translate_problem_with_llm)
        # covers premises + query + options in a single call, eliminating vocabulary drift.
        # Sampling is only used in the repair/fallback path, not on every request.
        default=1,
        ge=1,
        validation_alias="EXACT_TYPE1_TRANSLATION_SAMPLES",
    )
    type1_sampling_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        validation_alias="EXACT_TYPE1_SAMPLING_TEMPERATURE",
    )
    type1_enable_cot_fallback: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE1_ENABLE_COT_FALLBACK",
    )
    type1_add_contrapositives: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE1_ADD_CONTRAPOSITIVES",
    )
    type1_use_z3_fallback: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE1_USE_Z3_FALLBACK",
    )
    type1_use_formula_z3: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE1_USE_FORMULA_Z3",
    )
    type1_enable_legacy_fallback: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE1_ENABLE_LEGACY_FALLBACK",
    )
    # Guided JSON decoding passes the formula JSON schema to vLLM so the model
    # is constrained to valid formula-tree JSON from the first token.  Requires
    # vLLM >= 0.6.x with lm-format-enforcer.  Default True because the
    # competition vLLM server is a custom build that supports guided_json.
    type1_use_guided_json: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE1_USE_GUIDED_JSON",
    )
    # Split premise translation from goal translation and cache premise results.
    # When True, each request makes two smaller LLM calls instead of one large call:
    #   Call 1 (cached): premises only  → predicate dict + FormulaItems
    #   Call 2 (fast):   query/options  → goal FormulaItems using predicate dict
    # This eliminates vocabulary drift across calls (predicate dict is shared) and
    # makes subsequent questions in the same premise group nearly free (cache hit).
    # Expected impact: 14-premise groups go from ~55s to ~42s (first Q) / ~8s (later Qs).
    type1_formula_cache_premises: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE1_FORMULA_CACHE_PREMISES",
    )
    # Soft deadline for a single Type 1 request (seconds).  When the remaining
    # time is less than this threshold the pipeline skips the LLM repair/fallback
    # call and returns the best symbolic answer available, preventing ReadTimeout.
    type1_soft_deadline_s: float = Field(
        default=45.0,
        gt=0.0,
        validation_alias="EXACT_TYPE1_SOFT_DEADLINE_S",
    )
    # Per-call timeout for LLM requests. Set high enough that a large one-shot
    # formula translation (14 premises + 4 options, ~2800 tokens) can complete.
    # Retries are disabled (max_retries=0) because a retry after timeout wastes
    # the remaining request budget: 2 × 55s = 110s > 60s hard cap.
    # The func_timeout backstop in run_type1_pipeline provides the real safety net.
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
    type2_pot_max_retries: int = Field(default=3, ge=0, validation_alias="EXACT_TYPE2_POT_MAX_RETRIES")
    type2_formula_limit: int = Field(default=24, ge=1, validation_alias="EXACT_TYPE2_FORMULA_LIMIT")
    type2_rerank_limit: int = Field(default=12, ge=1, validation_alias="EXACT_TYPE2_RERANK_LIMIT")
    type2_generate_explanation: bool = Field(default=True, validation_alias="EXACT_TYPE2_GENERATE_EXPLANATION")
    type2_pot_timeout: float = Field(default=10.0, gt=0.0, validation_alias="EXACT_TYPE2_POT_TIMEOUT")

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
