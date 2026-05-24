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

    llm_provider: Literal["openai", "anthropic", "local"] = "local"
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
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, validation_alias="EXACT_MAX_RETRIES")

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