from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from exact.config import settings


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleLLMClient(LLMClient):
    """Minimal client for local OpenAI-compatible servers such as vLLM or Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])


def get_default_llm_client() -> LLMClient:
    if not settings.llm_base_url:
        raise RuntimeError(
            "No local LLM endpoint configured. Set EXACT_LLM_BASE_URL to an "
            "OpenAI-compatible server such as vLLM, llama.cpp, or Ollama."
        )

    return OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        top_p=settings.llm_top_p,
    )
