# Using `llm_client.py`

`exact.llm_client` is the shared JSON-only client for the project's self-hosted
[vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).
Use it when a prompt must return a JSON object, optionally validated against a
Pydantic model.

The client sends requests to:

```text
{EXACT_LLM_BASE_URL}/chat/completions
```

Therefore, the configured base URL normally includes `/v1`, for example
`http://127.0.0.1:8000/v1`.

## Configure the Client

Add the general LLM settings to the repository-level `.env` file:

```dotenv
EXACT_LLM_BASE_URL=http://127.0.0.1:8000/v1
EXACT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EXACT_LLM_API_KEY=exact-local-token
EXACT_LLM_TIMEOUT_SECONDS=55
EXACT_MAX_RETRIES=2
```

`EXACT_LLM_BASE_URL` must point to self-hosted infrastructure. The validation in
`exact.config` accepts localhost, private/link-local IP addresses, and internal
hostnames such as `vllm`, `vllm.internal`, or `vllm.default.svc`. Public model
API endpoints are intentionally rejected.

Build a client from these settings:

```python
from exact.llm_client import build_json_client_from_settings

client = build_json_client_from_settings()
if client is None:
    raise RuntimeError("EXACT_LLM_BASE_URL is not configured")
```

For a one-off client, construct it directly:

```python
from exact.llm_client import VLLMJsonClient

client = VLLMJsonClient(
    base_url="http://127.0.0.1:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key="exact-local-token",
    timeout=55.0,
    max_retries=2,
)
```

## Basic JSON Completion

Use `complete_json` from asynchronous application code:

```python
result = await client.complete_json(
    messages=[
        {"role": "system", "content": "Return JSON only."},
        {
            "role": "user",
            "content": 'Return {"answer": <number>, "unit": <string>} for: 2 + 3',
        },
    ],
    temperature=0.0,
    max_tokens=256,
)

print(result["answer"])
```

Use `complete_json_sync` from synchronous scripts, synchronous routes, or
notebooks:

```python
result = client.complete_json_sync(
    messages=[
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": 'Return {"topic": <string>} for: Ohm law'},
    ],
    max_tokens=128,
    timeout_override=20.0,
)
```

The sync wrapper also works when called inside a running event loop, such as a
Jupyter notebook. In async services, prefer `await complete_json(...)` so a
worker thread is not needed.

## Structured Output With Pydantic

`complete_as` validates the returned object and gives the caller a Pydantic
model:

```python
from pydantic import BaseModel, ConfigDict


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: float
    unit: str


answer = await client.complete_as(
    messages=[
        {"role": "system", "content": "Solve the problem and return JSON only."},
        {"role": "user", "content": "A 10 V source drives a 5 ohm resistor."},
    ],
    schema=Answer,
    json_schema=Answer.model_json_schema(),
    max_tokens=256,
)

print(answer.answer, answer.unit)
```

Passing `json_schema` enables vLLM guided JSON decoding. Passing `schema` to
`complete_as` performs Pydantic validation after the response is received.
These are separate operations: `complete_as` does **not** automatically send
`schema.model_json_schema()` to vLLM.

For synchronous code that needs a Pydantic object, validate the dictionary:

```python
answer = Answer.model_validate(
    client.complete_json_sync(
        messages=messages,
        json_schema=Answer.model_json_schema(),
        max_tokens=256,
    )
)
```

## Method Reference

| Method | Use case | Returns |
| --- | --- | --- |
| `await complete_json(...)` | Async application code | `dict[str, Any]` |
| `complete_json_sync(...)` | Scripts, sync routes, notebooks | `dict[str, Any]` |
| `await complete_as(...)` | Async call plus Pydantic validation | `BaseModel` instance |
| `build_json_client_from_settings(...)` | Construct from `Settings` or `.env` | Client, or `None` when no base URL is configured |
| `has_json_llm_client_config(...)` | Check whether a base URL is configured | `bool` |

Completion arguments:

- `messages`: iterable of OpenAI-style chat messages.
- `temperature`: sampling temperature; defaults to `0.0`.
- `max_tokens`: response token limit; defaults to `2048`.
- `json_schema`: optional JSON Schema sent to vLLM as `guided_json`.
- `timeout_override`: optional timeout for this call. Available on
  `complete_json` and `complete_json_sync`.

The builder uses configured connection values such as base URL, model, API key,
timeout, and retry count. It does not automatically apply
`settings.llm_temperature` or `settings.llm_max_tokens` to each completion.
Pass per-call generation values explicitly, as the Type 2 code does in
`type2/extraction/llm_structured.py`.

## Response and Retry Behavior

- Every request asks vLLM for a JSON object using
  `response_format={"type": "json_object"}`.
- When `json_schema` is supplied, the request also sends `guided_json`.
- If vLLM rejects `guided_json` with HTTP 400, the client logs a warning and
  retries once without guided decoding.
- HTTP 429 responses, other HTTP errors, transport failures, and timeouts are
  retried up to `max_retries`; the total normal attempt count is
  `max_retries + 1`.
- A 429 retry waits with exponential backoff. Other retryable failures are
  retried immediately.
- The parser extracts the first complete JSON object even if the model adds
  surrounding text. The top-level JSON value must be an object, not an array.

## Errors to Handle

Typical failures are:

- `ValueError` when configuration is empty/invalid, output contains no complete
  JSON object, JSON is malformed, or `complete_as` validation fails.
- `httpx.HTTPStatusError` after HTTP retries are exhausted.
- `httpx.TransportError` or `httpx.TimeoutException` after connection retries
  are exhausted.

Catch errors at the boundary that can recover or fall back:

```python
import httpx

try:
    result = await client.complete_json(messages, max_tokens=256)
except (httpx.HTTPError, ValueError) as exc:
    # Log, use a deterministic fallback, or return an application-level error.
    raise RuntimeError("LLM completion failed") from exc
```

## Test

Run the focused client and endpoint-validation tests:

```bash
pytest tests/test_vllm_client.py tests/test_self_hosted_model_endpoints.py
```
