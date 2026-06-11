"""JSON client for a self-hosted vLLM server."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError

from exact.config import Settings, get_settings, validate_self_hosted_model_url
from exact.logger import get_logger

logger = get_logger(__name__)

ChatMessage = dict[str, Any]


class VLLMJsonClient:
    """JSON client for a self-hosted vLLM chat-completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not base_url.strip():
            raise ValueError("A self-hosted vLLM base URL is required")
        if not model.strip():
            raise ValueError("A vLLM model name is required")

        self.model = model
        self.max_retries = max_retries
        self._api_key = api_key
        self._base_url = validate_self_hosted_model_url(base_url)
        self._timeout = timeout

    async def complete_json(
        self,
        messages: Iterable[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        """
        Return a JSON object from the configured self-hosted vLLM server.

        When provided, ``json_schema`` is sent as vLLM ``guided_json`` structured
        decoding. If the server rejects that schema, the request is retried once
        using prompt-level JSON constraints.
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
        # schema, effectively replacing retry-on-bad-JSON loops with a one-shot
        # structural guarantee.
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
        messages: Iterable[ChatMessage],
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

    async def complete_as(
        self,
        messages: Iterable[ChatMessage],
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
    ) -> BaseModel:
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


def has_json_llm_client_config(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return bool(settings.llm_base_url)


def build_json_client_from_settings(settings: Settings | None = None) -> VLLMJsonClient | None:
    """Build a client for the configured self-hosted vLLM server."""

    settings = settings or get_settings()
    if not has_json_llm_client_config(settings):
        return None

    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "EMPTY"
    return VLLMJsonClient(
        base_url=settings.llm_base_url or "",
        model=settings.llm_model,
        api_key=api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


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
