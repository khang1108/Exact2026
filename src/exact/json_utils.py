from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = _strip_code_fence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM output did not contain a JSON object: {text}")

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = json.loads(_escape_control_chars_inside_json_strings(candidate))

    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON output must be an object")
    return parsed


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json|python)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if stripped.startswith("```"):
        # Handle partial fences or assistant output that starts with a fence but
        # does not close cleanly.
        stripped = stripped.lstrip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.lower().startswith("python"):
            stripped = stripped[6:].strip()
    return stripped


def _escape_control_chars_inside_json_strings(text: str) -> str:
    """Escape raw control characters that appear inside quoted JSON strings.

    Some LLMs return JSON-like objects where newline characters inside string
    values are emitted literally instead of as ``\n``. Standard JSON parsers
    reject that output, so we normalize it lazily before retrying parsing.
    """

    result: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                result.append(char)
                escaped = False
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                continue
            if char == '"':
                result.append(char)
                in_string = False
                continue
            if char == "\n":
                result.append("\\n")
                continue
            if char == "\r":
                result.append("\\r")
                continue
            if char == "\t":
                result.append("\\t")
                continue
            if ord(char) < 0x20:
                result.append(f"\\u{ord(char):04x}")
                continue
            result.append(char)
            continue

        result.append(char)
        if char == '"':
            in_string = True
            escaped = False

    return "".join(result)
