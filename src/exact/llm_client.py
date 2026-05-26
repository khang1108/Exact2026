"""
Một module cung cấp một lớp để có thể tạo một LLMClient dùng để sinh các câu trả lời từ LLM
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Iterable
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from exact.config import Settings, get_settings
from exact.logger import get_logger

logger = get_logger(__name__)


class BaseJsonLLMClient(ABC):
    """Common contract for clients that return JSON objects from chat messages."""

    @abstractmethod
    def complete_json_sync(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Synchronously return a JSON object for command-line scripts and sync routes."""

    async def complete_json(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Async adapter for sync-first clients."""

        return await asyncio.to_thread(
            self.complete_json_sync,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def complete_as(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> BaseModel:
        """Return a JSON response validated as a Pydantic model."""

        data = await self.complete_json(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"LLM output does not match schema: {exc}") from exc


class BaseTextLLMClient(ABC):
    """Common contract for clients that return plain generated text."""

    @abstractmethod
    def complete(self, messages: list[dict], max_new_tokens: int = 2048) -> str:
        """Return generated text for chat messages."""


class OpenAICompatibleJsonClient(BaseJsonLLMClient):
    """
    JSON client for OpenAI-compatible chat completion APIs.

    Args:
        api_key: Khóa API để xác thực với dịch vụ LLM.
        base_url: URL cơ sở của dịch vụ LLM.
        model: Tên mô hình LLM để sử dụng.
        timeout: Thời gian chờ tối đa cho mỗi yêu cầu.
        max_retries: Số lần thử lại tối đa khi yêu cầu thất bại.
    """
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def complete_json(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Gọi API của LLM để sinh ra một câu trả lời dưới dạng JSON.

        Args:
            messages: Danh sách các tin nhắn để gửi đến LLM.
            temperature: Nhiệt độ cho quá trình sinh.
            max_tokens: Số lượng token tối đa cho phép.

        Returns:
            Một dictionary chứa kết quả từ LLM.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        try:
            return _parse_json_object(text)
        except ValueError as exc:
            raise ValueError(f"LLM returned invalid JSON with finish_reason={choice.finish_reason}: {text}") from exc

    def complete_json_sync(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Synchronous wrapper for command-line scripts and FastAPI sync routes.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.complete_json(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
        raise RuntimeError("complete_json_sync cannot run inside an active event loop")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "LLMClient":
        settings = settings or get_settings()
        api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "EMPTY"
        return cls(
            api_key=api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )


class LLMClient(OpenAICompatibleJsonClient):
    """Backward-compatible name for the OpenAI-compatible JSON client."""


class LocalClient(BaseTextLLMClient):
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local tokenizer for %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {"dtype": _preferred_torch_dtype()}
        if importlib.util.find_spec("accelerate") is not None:
            model_kwargs["device_map"] = "auto"

        logger.info("Loading local model %s with %s", model_name, model_kwargs)
        started_at = time.monotonic()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if "device_map" not in model_kwargs and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        logger.info("Loaded local model %s in %.1fs", model_name, time.monotonic() - started_at)

    def complete(self, messages: list[dict], max_new_tokens: int = 2048) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        logger.info("Tokenizing local LLM prompt")
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        logger.info("Starting local generation: input_tokens=%s, max_new_tokens=%s", input_length, max_new_tokens)
        started_at = time.monotonic()

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = outputs[0][input_length:]
        logger.info(
            "Finished local generation: output_tokens=%s, elapsed=%.1fs",
            len(generated_tokens),
            time.monotonic() - started_at,
        )
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


class LocalJsonClient(BaseJsonLLMClient):
    """JSON adapter over the direct transformers-based LocalClient."""

    def __init__(self, model_name: str):
        self.client = LocalClient(model_name)

    def complete_json_sync(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        text = self.client.complete(
            messages=list(messages),
            max_new_tokens=max_tokens,
        )
        return _parse_json_object(text)


def build_json_client_from_settings(settings: Settings | None = None) -> BaseJsonLLMClient | None:
    """Build a JSON-producing LLM client from runtime settings.

    - `llm_base_url` set: call an OpenAI-compatible local/remote server.
    - `llm_provider=local`: load the model directly with transformers.
    - otherwise: return None and let the pipeline use heuristic fallback.
    """

    settings = settings or get_settings()
    if settings.mock_llm:
        return None
    if settings.llm_base_url:
        return LLMClient.from_settings(settings)
    if settings.llm_provider == "local":
        return LocalJsonClient(settings.llm_model)
    return None


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        raise ValueError(f"LLM output did not contain a JSON object: {text}")
    if end == -1 or end < start:
        raise ValueError(f"LLM output contained incomplete JSON; increase EXACT_MAX_NEW_TOKENS: {text}")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {text}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON output must be an object")
    return parsed


def _preferred_torch_dtype() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.float16
    return torch.float32
