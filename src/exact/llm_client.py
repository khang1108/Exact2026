"""
Một module cung cấp một lớp để có thể tạo một LLMClient dùng để sinh các câu trả lời từ LLM
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import inspect
import time
from abc import ABC, abstractmethod
from typing import Any, Iterable
from openai import APIStatusError, AsyncOpenAI
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
        # When provided, the server is asked to constrain its output to JSON
        # that matches this schema (vLLM guided_json / lm-format-enforcer).
        # Subclasses that do not support guided decoding should silently ignore
        # this parameter — the prompt-level constraints act as the fallback.
        json_schema: dict[str, Any] | None = None,
        # When the pipeline knows its remaining time budget it passes it here so
        # the HTTP client does not wait past the 60s hard cap.  None = use the
        # client's default self._timeout.
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        """Synchronously return a JSON object for command-line scripts and sync routes."""

    async def complete_json(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        """Async adapter for sync-first clients."""

        return await asyncio.to_thread(
            self.complete_json_sync,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            timeout_override=timeout_override,
        )

    async def complete_as(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
    ) -> BaseModel:
        """Return a JSON response validated as a Pydantic model."""

        data = await self.complete_json(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
        )
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"LLM output does not match schema: {exc}") from exc


class BaseTextLLMClient(ABC):
    """Common contract for clients that return plain generated text."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
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
        self.max_retries = max_retries
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._timeout = timeout
        # Keep AsyncOpenAI only for legacy _complete_json_and_close; actual calls use httpx.
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    async def complete_json(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        """
        Gọi API của LLM để sinh ra một câu trả lời dưới dạng JSON.
        Dùng httpx trực tiếp để tránh headers của OpenAI SDK bị Cloudflare WAF chặn.

        Khi json_schema được cung cấp, nó được truyền xuống vLLM dưới key "guided_json"
        (lm-format-enforcer).  Server phải hỗ trợ tính năng này; nếu không, payload
        vẫn hợp lệ vì "guided_json" là extra field bị bỏ qua bởi OpenAI-compatible servers.

        timeout_override: khi pipeline tính được budget còn lại, truyền vào đây để httpx
        không chờ quá thời gian còn lại (tránh trường hợp call bắt đầu lúc T=50s với
        self._timeout=55s, dẫn đến vượt hard cap 60s).
        """
        import httpx

        effective_timeout = timeout_override if timeout_override is not None else self._timeout
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        # guided_json constrains token sampling to only produce JSON matching the
        # schema, effectively replacing the retry-on-bad-JSON loop with a one-shot
        # guarantee.  Only sent when the caller opts in (settings.type1_use_guided_json).
        if json_schema is not None:
            payload["guided_json"] = json_schema
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"

        last_exc: Exception | None = None
        retry_number = 0
        attempts_remaining = self.max_retries + 1
        while attempts_remaining > 0:
            attempts_remaining -= 1
            try:
                async with httpx.AsyncClient(timeout=effective_timeout) as http:
                    resp = await http.post(url, json=payload, headers=headers)
                    if resp.status_code == 429 and attempts_remaining > 0:
                        await asyncio.sleep(2 ** retry_number)
                        retry_number += 1
                        continue
                    # 400 with guided_json: the schema was rejected by the server
                    # (recursive $ref not supported by lm-format-enforcer, or wrong
                    # vLLM version).  Drop guided_json and retry once without it so
                    # the formula path continues rather than failing entirely.
                    if resp.status_code == 400 and "guided_json" in payload:
                        logger.warning(
                            "vLLM returned 400 for guided_json request — "
                            "retrying without guided_json (schema may use unsupported features)"
                        )
                        payload = {k: v for k, v in payload.items() if k != "guided_json"}
                        # This compatibility fallback is independent of ordinary
                        # retries. Production keeps max_retries=0 to respect the
                        # 60-second cap, but still needs one unguided attempt.
                        attempts_remaining += 1
                        continue
                    resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
                finish_reason = choice.get("finish_reason", "")
                try:
                    return _parse_json_object(text)
                except ValueError as exc:
                    raise ValueError(
                        "LLM returned invalid JSON "
                        f"with finish_reason={finish_reason}: {_clip_text(text)}"
                    ) from exc
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if attempts_remaining > 0:
                    retry_number += 1
                    continue
                raise
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempts_remaining > 0:
                    retry_number += 1
                    continue
                raise
        raise last_exc or RuntimeError("LLM request exhausted attempts without a response")

    def complete_json_sync(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        """
        Synchronous wrapper for command-line scripts, FastAPI sync routes, and Jupyter notebooks.
        """
        import concurrent.futures

        coro = self.complete_json(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            timeout_override=timeout_override,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # Inside a running event loop (e.g. Jupyter) — run in a dedicated thread
        # that owns its own event loop so asyncio.run() works cleanly.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    async def _complete_json_and_close(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        try:
            return await self.complete_json(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        finally:
            close = getattr(self.client, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result

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
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | None = "auto",
        torch_dtype: str = "float16",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local tokenizer for %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_torch_dtype(torch_dtype),
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if device_map is not None:
            if importlib.util.find_spec("accelerate") is not None:
                model_kwargs["device_map"] = device_map
            else:
                logger.warning("Ignoring device_map=%s because accelerate is not installed", device_map)

        logger.info("Loading local model %s with %s", model_name, model_kwargs)
        started_at = time.monotonic()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except (ImportError, ValueError, RuntimeError) as exc:
            if not _should_retry_with_sdpa(exc):
                raise
            logger.warning(
                "flash_attention_2 unavailable for %s; retrying with sdpa: %s",
                model_name,
                exc,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                **model_kwargs,
            )
        if "device_map" not in model_kwargs and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        logger.info("Loaded local model %s in %.1fs", model_name, time.monotonic() - started_at)

    def complete(
        self,
        messages: list[dict],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        logger.info("Tokenizing local LLM prompt")
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        do_sample = temperature > 0.0
        logger.info(
            "Starting local generation: input_tokens=%s, max_new_tokens=%s, do_sample=%s, temperature=%s, top_p=%s",
            input_length,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
        )
        started_at = time.monotonic()

        generate_kwargs: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        outputs = self.model.generate(**generate_kwargs)

        generated_tokens = outputs[0][input_length:]
        logger.info(
            "Finished local generation: output_tokens=%s, elapsed=%.1fs",
            len(generated_tokens),
            time.monotonic() - started_at,
        )
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


class LocalJsonClient(BaseJsonLLMClient):
    """JSON adapter over the direct transformers-based LocalClient."""

    def __init__(
        self,
        model_name: str,
        *,
        top_p: float = 1.0,
        device_map: str | None = "auto",
        torch_dtype: str = "float16",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        self.client = LocalClient(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.top_p = top_p

    def complete_json_sync(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        # LocalClient uses HuggingFace Transformers directly and does not
        # support guided JSON decoding or per-call timeout override — both
        # parameters are accepted to satisfy the BaseJsonLLMClient interface
        # but are intentionally ignored here.
        json_schema: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        text = self.client.complete(
            messages=list(messages),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=self.top_p,
        )
        return _parse_json_object(text)


def build_json_client_from_settings(settings: Settings | None = None) -> BaseJsonLLMClient | None:
    """Build a JSON-producing LLM client from runtime settings.

    - `llm_base_url` set: call an OpenAI-compatible local/remote server.
    - `llm_provider=local`: load the model directly with transformers.
    - otherwise: return None so callers can fail clearly in LLM-only flows.
    """

    settings = settings or get_settings()
    if settings.mock_llm:
        return None
    if settings.llm_base_url:
        return LLMClient.from_settings(settings)
    if settings.llm_provider == "local":
        return LocalJsonClient(
            settings.llm_model,
            top_p=settings.llm_top_p,
            device_map=settings.llm_device_map,
            torch_dtype=settings.llm_torch_dtype,
            local_files_only=settings.llm_local_files_only,
            trust_remote_code=settings.llm_trust_remote_code,
        )
    return None


def _resolve_torch_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _should_retry_with_sdpa(exc: BaseException) -> bool:
    text = str(exc).lower()
    retry_markers = (
        "flash_attn",
        "flash attention",
        "flash_attention_2",
        "attn_implementation",
        "sm_75",
    )
    return isinstance(exc, (ImportError, ValueError)) or any(marker in text for marker in retry_markers)


def _escape_invalid_json_escapes(text: str) -> str:
    result = []
    i = 0
    n = len(text)
    in_string = False
    escaped = False
    while i < n:
        char = text[i]
        if in_string:
            if escaped:
                is_valid = False
                if char in ['"', '\\', 'n']:
                    is_valid = True
                elif char == 'u':
                    if i + 4 < n:
                        hex_part = text[i+1:i+5]
                        if all(c in '0123456789abcdefABCDEF' for c in hex_part):
                            is_valid = True
                if is_valid:
                    result.append(char)
                else:
                    if result and result[-1] == '\\':
                        result[-1] = '\\\\'
                    result.append(char)
                escaped = False
            else:
                if char == '\\':
                    escaped = True
                    result.append('\\')
                else:
                    if char == '"':
                        in_string = False
                    result.append(char)
        else:
            if char == '"':
                in_string = True
            result.append(char)
        i += 1
    return "".join(result)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = _escape_invalid_json_escapes(text)
    span = _find_first_json_object_span(text)
    start = -1 if span is None else span[0]
    end = -1 if span is None else span[1]
    if start == -1:
        raise ValueError(f"LLM output did not contain a JSON object: {_clip_text(text)}")
    if end == -1 or end < start:
        raise ValueError(
            "LLM output contained incomplete JSON; reduce prompt/output size or increase "
            f"EXACT_MAX_NEW_TOKENS. Output snippet: {_clip_text(text)}"
        )
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        snippet = _json_error_snippet(text[start : end + 1], exc.pos)
        raise ValueError(
            f"LLM returned invalid JSON at char {exc.pos}: {exc.msg}. Around error: {snippet}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON output must be an object")
    return parsed


def _find_first_json_object_span(text: str) -> tuple[int, int] | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, index

    return (start, -1)


def _clip_text(text: str, limit: int = 1200) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head} ... <truncated {len(text) - limit} chars> ... {tail}"


def _json_error_snippet(text: str, pos: int, radius: int = 500) -> str:
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return _clip_text(text[start:end], limit=radius * 2)
