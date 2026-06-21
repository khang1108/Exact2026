"""Local Hugging Face transformers client for the EXACT pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError

# We import parse_json_object from llm_client
from exact.llm_client import _parse_json_object, _clip_text, ChatMessage

logger = logging.getLogger(__name__)


class TransformersJsonClient:
    """JSON client for a local Hugging Face transformers model."""

    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        if not model.strip():
            raise ValueError("A Hugging Face model name is required")
        
        self.model_name = model
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        self._model = None
        self._tokenizer = None
        self._load_lock = asyncio.Lock()

    def _load_model_sync(self):
        """Load model and tokenizer if not already loaded (sync)."""
        if self._model is not None:
            return
            
        logger.info(f"Loading transformers model: {self.model_name}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        dtype = torch.float16 if self.torch_dtype == "auto" else self.torch_dtype
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=dtype,
            trust_remote_code=False,
        )
        self._model.eval()

    async def _load_model_async(self):
        async with self._load_lock:
            if self._model is None:
                await asyncio.to_thread(self._load_model_sync)

    def _generate_sync(
        self,
        messages: list[ChatMessage],
        n: int,
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None,
    ) -> list[str]:
        self._load_model_sync()

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model not loaded")

        # N=1 is mainly supported by standard generate without complex beam search setups,
        # so we will duplicate the prompt n times in the batch.

        # We enforce a JSON instruction if needed, though Qwen handles it well.
        prepared_messages = [dict(message) for message in messages]
        if json_schema is not None:
            schema_str = json.dumps(json_schema)
            json_instruction = f" You must output your response as a valid JSON object matching this schema: {schema_str}"
            if prepared_messages and prepared_messages[0].get("role") == "system":
                prepared_messages[0] = {
                    **prepared_messages[0],
                    "content": str(prepared_messages[0].get("content") or "") + json_instruction,
                }
            else:
                prepared_messages = [
                    {"role": "system", "content": json_instruction},
                    *prepared_messages,
                ]

        prompt = self._tokenizer.apply_chat_template(
            prepared_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer([prompt] * n, return_tensors="pt").to(self._model.device)
        
        import torch
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        responses = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

    async def complete_json_batch(
        self,
        messages: Iterable[ChatMessage],
        n: int,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return one or more JSON objects from the local transformers model.
        """
        await self._load_model_async()
        
        messages_list = list(messages)
        
        responses = await asyncio.to_thread(
            self._generate_sync,
            messages=messages_list,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema
        )
        
        parsed_choices = []
        parse_errors = []
        for text in responses:
            try:
                parsed_choices.append(_parse_json_object(text))
            except ValueError as exc:
                parse_errors.append(f"parse_error: {_clip_text(text)}")
                if n == 1:
                    raise ValueError(f"LLM returned invalid JSON: {_clip_text(text)}") from exc
                    
        if parsed_choices:
            return parsed_choices
            
        raise ValueError("LLM returned no parseable JSON choices: " + "; ".join(parse_errors))

    async def complete_json(
        self,
        messages: Iterable[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        results = await self.complete_json_batch(
            messages=messages,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            timeout_override=timeout_override,
        )
        return results[0]

    def complete_json_batch_sync(
        self,
        messages: Iterable[ChatMessage],
        n: int,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Synchronous wrapper for batched JSON completions.
        """
        import concurrent.futures

        coro = self.complete_json_batch(
            messages=messages,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            timeout_override=timeout_override,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    def complete_json_sync(
        self,
        messages: Iterable[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_schema: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        """
        Synchronous wrapper for command-line scripts.
        """
        return self.complete_json_batch_sync(
            messages=messages,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            timeout_override=timeout_override,
        )[0]

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
