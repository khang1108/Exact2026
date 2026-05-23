from __future__ import annotations

from exact.llm_client import (
    GroqClient,
    LLMClient,
    _failed_generation_from_payload,
    _parse_jsonish_object,
    _parse_payload_text,
    build_json_client_from_settings,
)
from exact.scripts.config_utils import _dotenv_value, build_settings_from_config
from exact.scripts.run_type2_monitor import _slice_examples


def test_build_settings_from_config_supports_groq_backend(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    config = {
        "llm": {
            "backend": "groq",
            "enabled": True,
            "model": "llama-3.1-8b-instant",
        },
        "pipeline": {"use_type1_llm": True, "use_type2_llm_fallback": True},
    }

    settings = build_settings_from_config(config)

    assert settings.llm_provider == "groq"
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "groq-test-key"
    assert settings.llm_model == "llama-3.1-8b-instant"


def test_build_json_client_from_settings_returns_groq_client(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    config = {
        "llm": {
            "backend": "groq",
            "enabled": True,
            "model": "llama-3.1-8b-instant",
        },
        "pipeline": {"use_type1_llm": True, "use_type2_llm_fallback": True},
    }

    settings = build_settings_from_config(config)
    client = build_json_client_from_settings(settings)

    assert isinstance(client, GroqClient)


def test_slice_examples_supports_offset_and_limit() -> None:
    examples = ["a", "b", "c", "d"]

    assert _slice_examples(examples, offset=1, limit=2) == ["b", "c"]
    assert _slice_examples(examples, offset=2, limit=None) == ["c", "d"]


def test_dotenv_value_reads_unexported_api_key(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('GROQ_API_KEY="groq-dotenv-key"\n', encoding="utf-8")

    assert _dotenv_value("GROQ_API_KEY", env_file) == "groq-dotenv-key"


def test_parse_jsonish_object_recovers_triple_quoted_code_field() -> None:
    text = """{
  "code": \"\"\"
import pint
ans = 1
ans_unit = "N"
\"\"\"
}"""

    parsed = _parse_jsonish_object(text)

    assert parsed is not None
    assert "import pint" in parsed["code"]
    assert 'ans_unit = "N"' in parsed["code"]


def test_parse_payload_text_recovers_groq_failed_generation_payload() -> None:
    failed_generation = """{
  "code": \"\"\"
ans = 1
ans_unit = "N"
\"\"\"
}"""
    text = (
        "Error code: 400 - "
        + repr({"error": {"code": "json_validate_failed", "failed_generation": failed_generation}})
    )

    payloads = _parse_payload_text(text.split(" - ", 1)[1])
    failed = _failed_generation_from_payload(payloads[0])
    recovered = _parse_jsonish_object(failed)

    assert recovered is not None
    assert recovered["code"].strip().startswith("ans = 1")


def test_complete_json_sync_closes_async_client() -> None:
    class DummyAsyncOpenAI:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class DummyLLMClient(LLMClient):
        def __init__(self) -> None:
            self.client = DummyAsyncOpenAI()
            self.model = "dummy"

        async def complete_json(self, messages, temperature=0.0, max_tokens=2048):
            return {"ok": True}

    client = DummyLLMClient()

    assert client.complete_json_sync([]) == {"ok": True}
    assert client.client.closed is True
