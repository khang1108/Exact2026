from __future__ import annotations

from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
PACKAGE_DIR = SRC_DIR / "exact"


def validate_self_hosted_model_url(value: str) -> str:
    """Return a normalized model URL or reject endpoints outside local infrastructure.

    Model inference may target the same machine, a private network address, or
    an internal service-discovery hostname. Public DNS names and public IP
    addresses are rejected so an inference client cannot accidentally call a
    third-party model API.
    """

    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    hostname = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("model endpoint must be an http(s) URL with a hostname")

    lowered_hostname = hostname.lower()
    if lowered_hostname == "localhost":
        return normalized

    try:
        address = ip_address(lowered_hostname)
    except ValueError:
        is_internal_name = "." not in lowered_hostname or lowered_hostname.endswith(
            (".internal", ".local", ".svc", ".cluster.local")
        )
        if is_internal_name:
            return normalized
    else:
        if address.is_private or address.is_loopback or address.is_link_local:
            return normalized

    raise ValueError(
        "model endpoint must use localhost, a private IP, or an internal service hostname"
    )


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

    llm_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        validation_alias=AliasChoices("EXACT_LLM_MODEL", "EXACT_MODEL_ID"),
    )
    math_model_id: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    llm_backend: Literal["vllm", "transformers", "none"] = "vllm"
    llm_enabled: bool = Field(default=False, validation_alias="EXACT_LLM_ENABLED")
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

    # Dedicated Type 1 parser vLLM service. These settings remain separate from
    # the general LLM endpoint because parser traffic has different models,
    # token limits, and concurrency requirements.
    type1_parser_base_url: str | None = None
    type1_parser_model: str = "type1-parser"
    type1_parser_api_key: SecretStr | None = None
    type1_parser_concurrency: int = Field(default=32, ge=1)
    type1_parser_timeout_seconds: float = Field(default=30.0, gt=0)
    type1_parser_max_retries: int = Field(default=2, ge=0)
    type1_parser_max_tokens: int = Field(default=512, ge=1)
    type1_uncertain_token: str = Field(
        default="Uncertain",
        validation_alias="EXACT_TYPE1_UNCERTAIN_TOKEN",
    )
    type1_translator: Literal["decompose", "single_pass", "hybrid"] = Field(
        default="decompose",
        validation_alias="EXACT_TYPE1_TRANSLATOR",
    )
    type1_single_pass_max_refines: int = Field(
        default=1,
        ge=0,
        validation_alias="EXACT_TYPE1_SINGLE_PASS_MAX_REFINES",
    )
    type1_verify_refine: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE1_VERIFY_REFINE",
    )
    type1_self_consistency_samples: int = Field(
        default=1,
        ge=1,
        validation_alias="EXACT_TYPE1_SELF_CONSISTENCY_SAMPLES",
    )
    type1_self_consistency_temperature: float = Field(
        default=0.7,
        ge=0.0,
        validation_alias="EXACT_TYPE1_SELF_CONSISTENCY_TEMPERATURE",
    )
    # Hard end-to-end deadline for one Type 1 request. If the parse → solve →
    # refine pipeline exceeds this, the request returns "Uncertain" instead of
    # running unbounded. Keep it below any upstream proxy/tunnel timeout.
    type1_request_deadline_seconds: float = Field(
        default=55.0, gt=0, validation_alias="EXACT_TYPE1_DEADLINE_SECONDS"
    )

    # Type 2 Physics Pipeline Settings
    type2_use_llm_domain_routing: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_LLM_DOMAIN_ROUTING",
    )
    type2_use_llm_question_kind_routing: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_LLM_QUESTION_KIND_ROUTING",
    )
    type2_use_recovery_loop: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_RECOVERY_LOOP",
    )
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
    type2_use_formula_ontology: bool = Field(
        default=True,
        validation_alias="EXACT_TYPE2_USE_FORMULA_ONTOLOGY",
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
    type2_pot_max_retries: int = Field(
        default=3, ge=0, validation_alias="EXACT_TYPE2_POT_MAX_RETRIES"
    )
    type2_recovery_loop_max_attempts: int = Field(
        default=2,
        ge=1,
        validation_alias="EXACT_TYPE2_RECOVERY_LOOP_MAX_ATTEMPTS",
    )
    type2_pot_batch_size: int = Field(
        default=1,
        ge=1,
        validation_alias="EXACT_TYPE2_POT_BATCH_SIZE",
    )
    type2_pot_batch_temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        validation_alias="EXACT_TYPE2_POT_BATCH_TEMPERATURE",
    )
    type2_formula_limit: int = Field(default=10, ge=1, validation_alias="EXACT_TYPE2_FORMULA_LIMIT")
    type2_rerank_limit: int = Field(default=6, ge=1, validation_alias="EXACT_TYPE2_RERANK_LIMIT")
    type2_generate_explanation: bool = Field(
        default=False, validation_alias="EXACT_TYPE2_GENERATE_EXPLANATION"
    )
    type2_pot_timeout: float = Field(
        default=10.0, gt=0.0, validation_alias="EXACT_TYPE2_POT_TIMEOUT"
    )
    type2_extraction_max_tokens: int = Field(
        default=512,
        ge=1,
        validation_alias="EXACT_TYPE2_EXTRACTION_MAX_TOKENS",
    )
    type2_domain_routing_max_tokens: int = Field(
        default=256,
        ge=1,
        validation_alias="EXACT_TYPE2_DOMAIN_ROUTING_MAX_TOKENS",
    )
    type2_question_kind_max_tokens: int = Field(
        default=128,
        ge=1,
        validation_alias="EXACT_TYPE2_QUESTION_KIND_MAX_TOKENS",
    )
    type2_recovery_loop_max_tokens: int = Field(
        default=1024,
        ge=1,
        validation_alias="EXACT_TYPE2_RECOVERY_LOOP_MAX_TOKENS",
    )
    type2_formula_selection_max_tokens: int = Field(
        default=512,
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
        default=512,
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

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def empty_llm_url_means_disabled(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("type1_parser_base_url")
    @classmethod
    def validate_model_endpoint(cls, value: str | None) -> str | None:
        """Prevent model clients configured from environment variables from using public APIs."""

        return validate_self_hosted_model_url(value) if value is not None else None

    @model_validator(mode="after")
    def configure_optional_llm(self) -> "Settings":
        """Keep Type 2 deterministic-only startup independent from LLM config.

        A local developer may have stale or placeholder LLM settings in ``.env``.
        Unless ``EXACT_LLM_ENABLED=true`` is explicit, those settings are ignored
        and all LLM-dependent Type 2 stages are disabled.
        """

        if not self.llm_enabled:
            self.llm_base_url = None
            self.llm_api_key = None
            self.type2_use_llm_domain_routing = False
            self.type2_use_llm_question_kind_routing = False
            self.type2_use_llm_formula_selection = False
            self.type2_force_llm_formula_selection = False
            self.type2_use_recovery_loop = False
            if self.type2_extraction_mode == "llm_only":
                self.type2_extraction_mode = "heuristic_only"
            return self

        if self.llm_backend == "vllm":
            if self.llm_base_url is None:
                raise ValueError(
                    "EXACT_LLM_BASE_URL is required when EXACT_LLM_ENABLED=true and backend is vllm"
                )
            self.llm_base_url = validate_self_hosted_model_url(self.llm_base_url)
        return self

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
    """Load and cache validated application settings on first runtime use."""

    return Settings()
