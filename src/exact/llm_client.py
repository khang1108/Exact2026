"""
Một module cung cấp một lớp để có thể tạo một LLMClient dùng để sinh các câu trả lời từ LLM
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

from exact.config import Settings, get_settings

class LLMClient:
    """
    Một lớp duy nhất tạo LLMClient để gọi các API.

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

        text = response.choices[0].message.content or "{}"

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {text}") from exc

    async def complete_as(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> BaseModel:
        """
        Gọi API của LLM để sinh ra một câu trả lời và xác thực nó theo một schema Pydantic.
        """
        data = await self.complete_json(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"LLM output does not match schema: {exc}") from exc

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
        


class LocalClient:
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | None = "auto",
        torch_dtype: str = "float16",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs: dict[str, Any] = {
            "torch_dtype": _torch_dtype(torch_dtype),
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if device_map and device_map not in {"none", "cpu"}:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device_map == "cpu":
            self.model.to("cpu")

    def complete(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Truyền đích danh input_ids và attention_mask thay vì dùng **inputs
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # Gán pad_token_id bằng eos_token_id để khắc phục lỗi thiếu padding token
            pad_token_id=self.tokenizer.eos_token_id, 
        )

        # Decode tensor kết quả thành chuỗi văn bản (cắt bỏ phần prompt đầu vào)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response


class LocalJsonClient:
    """JSON adapter over the direct transformers-based LocalClient."""

    def __init__(
        self,
        model_name: str,
        *,
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


def build_json_client_from_settings(settings: Settings | None = None) -> Any | None:
    """Build a JSON-producing LLM client from runtime settings.

    - `llm_provider=local`: load the model directly with transformers.
    - `llm_base_url` set: call an OpenAI-compatible local/remote server.
    - otherwise: return None and let the pipeline use heuristic fallback.
    """

    settings = settings or get_settings()
    if settings.llm_provider == "local":
        return LocalJsonClient(
            settings.llm_model,
            device_map=settings.llm_device_map,
            torch_dtype=settings.llm_torch_dtype,
            local_files_only=settings.llm_local_files_only,
            trust_remote_code=settings.llm_trust_remote_code,
        )
    if settings.llm_base_url:
        return LLMClient.from_settings(settings)
    return None


def _torch_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM output did not contain a JSON object: {text}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON output must be an object")
    return parsed
