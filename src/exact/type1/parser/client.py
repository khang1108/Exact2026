"""Asynchronous client for the dedicated Type 1 parser vLLM service.

The parser model receives many short, independent sentence-parsing requests.
This client keeps one HTTP connection pool alive, limits concurrent requests,
and submits each expected Pydantic schema as a vLLM structured-output schema.
vLLM then continuously batches the concurrent requests on the GPU.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from exact.config import Settings, get_settings, validate_self_hosted_model_url

if TYPE_CHECKING:
    from exact.type1.parser.router import ParserRequest

ParserResult = TypeVar("ParserResult", bound=BaseModel)
ChatMessage = dict[str, Any]


class ParserClient:
    """Call a dedicated self-hosted vLLM parser model asynchronously.

    The client owns a persistent ``httpx.AsyncClient`` unless one is injected
    for testing. A semaphore applies application-level backpressure while vLLM
    performs continuous batching for the requests that reach the model server.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "type1-parser",
        *,
        api_key: str = "EMPTY",
        concurrency: int = 32,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        max_tokens: int = 512,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the parser endpoint, request limits, and connection pool."""

        if not base_url.strip():
            raise ValueError("ParserClient requires a vLLM base URL")
        if not model.strip():
            raise ValueError("ParserClient requires a served model name")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        normalized_base_url = validate_self_hosted_model_url(base_url)
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(concurrency)
        self._owns_http_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=normalized_base_url + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )

    async def parse_as(
        self,
        messages: Sequence[ChatMessage],
        schema: type[ParserResult],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ParserResult:
        """Parse one prompt and validate the structured response as ``schema``.

        The JSON schema is sent in ``response_format`` so vLLM stops once it has
        generated a complete valid object. This replaces client-side closing
        brace stopping criteria, which cannot run across an HTTP model server.
        """

        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        payload = self._build_payload(
            messages=messages,
            schema=schema,
            temperature=temperature,
            max_tokens=effective_max_tokens,
        )
        response_data = await self._post_with_retries(payload)

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("vLLM parser response has an invalid chat-completion shape") from exc

        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            raise ValueError(f"Parser output does not match {schema.__name__}: {exc}") from exc

    async def parse_many_as(
        self,
        message_batches: Iterable[Sequence[ChatMessage]],
        schema: type[ParserResult],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> list[ParserResult]:
        """Parse independent prompts concurrently while preserving input order.

        This is the method to use for the 14 premises of a Type 1 request.
        ``asyncio.gather`` fans requests out, the semaphore bounds pressure, and
        vLLM combines active requests into GPU batches.
        """

        return list(
            await asyncio.gather(
                *(
                    self.parse_as(
                        messages,
                        schema,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    for messages in message_batches
                )
            )
        )

    async def parse_request(self, request: "ParserRequest") -> BaseModel:
        """Execute one prompt-router request using its attached output schema."""

        return await self.parse_as(request.messages, request.schema)

    async def parse_requests(self, requests: Iterable["ParserRequest"]) -> list[BaseModel]:
        """Execute heterogeneous parser requests concurrently in input order.

        Unlike ``parse_many_as()``, each request may use a different system
        prompt and output schema, such as atomic and quantified premise parsers.
        """

        return list(await asyncio.gather(*(self.parse_request(request) for request in requests)))

    async def aclose(self) -> None:
        """Close the owned HTTP connection pool during application shutdown."""

        if self._owns_http_client:
            await self._http.aclose()

    async def __aenter__(self) -> "ParserClient":
        """Return this client for use in an asynchronous context manager."""

        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release owned HTTP connections when leaving an async context."""

        await self.aclose()

    def _build_payload(
        self,
        *,
        messages: Sequence[ChatMessage],
        schema: type[BaseModel],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build a vLLM chat request with Qwen thinking disabled and JSON constrained."""

        return {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
        }

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post one request with bounded retries for transient server failures."""

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._http.post("chat/completions", json=payload)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2**attempt)
                continue

            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
                return response.json()

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2**attempt)

        raise last_error or RuntimeError("Parser request exhausted retries")


def build_parser_client_from_settings(settings: Settings | None = None) -> ParserClient | None:
    """Build the dedicated parser client when its vLLM endpoint is configured.

    Returning ``None`` allows application startup and deterministic tests to run
    without a parser model server. Type 1 LLM flows should fail clearly if they
    require the client while this endpoint is absent.
    """

    settings = settings or get_settings()
    if not settings.type1_parser_base_url:
        return None

    api_key = (
        settings.type1_parser_api_key.get_secret_value()
        if settings.type1_parser_api_key
        else "EMPTY"
    )
    return ParserClient(
        base_url=settings.type1_parser_base_url,
        model=settings.type1_parser_model,
        api_key=api_key,
        concurrency=settings.type1_parser_concurrency,
        timeout_seconds=settings.type1_parser_timeout_seconds,
        max_retries=settings.type1_parser_max_retries,
        max_tokens=settings.type1_parser_max_tokens,
    )
