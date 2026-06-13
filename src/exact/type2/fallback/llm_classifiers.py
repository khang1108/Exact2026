from __future__ import annotations

import json
from typing import Any

from exact.config import Settings
from exact.llm_client import build_json_client_from_settings
from exact.type2.schemas import Extraction, Type2SolveResult, Verification
from exact.type2.execution_policy import is_canonical_direction, is_short_text_answer


def solve_with_direction_classifier(
    extraction: Extraction,
    settings: Settings | None = None,
) -> Type2SolveResult:
    """Use a small LLM to classify the direction strictly into canonical labels."""
    prompt = f"""
You are a physics expert. Answer the following question with ONLY a single, exact directional label.
Do not output anything else. Do not output markdown or code blocks.

Canonical Labels:
- toward_q1
- toward_q2
- toward_positive_charge
- toward_negative_charge
- left
- right
- up
- down
- clockwise
- counterclockwise
- same_direction_as_larger_force
- opposite_direction_to_larger_force
- zero_no_direction
- unknown

Question: {extraction.normalized_question}
"""
    client = build_json_client_from_settings(settings)
    if client is None:
        answer = "unknown"
    else:
        messages = [{"role": "user", "content": prompt}]
        text = client.complete_text_sync(messages, temperature=0.0)
        answer = text.strip().lower()

    if not is_canonical_direction(answer):
        answer = "unknown"

    return Type2SolveResult(
        answer=answer,
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(ok=True, message="Directional classification fallback executed."),
        cot=["Directional classification fallback executed."],
        premises=[],
        confidence=1.0,
        error=None,
    )


def solve_with_conceptual_llm(
    extraction: Extraction,
    settings: Settings | None = None,
) -> Type2SolveResult:
    """Use a small LLM to output a qualitative conceptual answer in JSON format."""
    prompt = f"""
You are a physics expert. Answer the following conceptual question.
Output exactly a JSON object with this schema:
{{
  "answer": "short and direct answer",
  "confidence": 0.95,
  "reason": "short explanation"
}}
Do not output anything else, no markdown formatting outside of the JSON block.

Question: {extraction.normalized_question}
"""
    client = build_json_client_from_settings(settings)
    if client is None:
        text = ""
    else:
        messages = [{"role": "user", "content": prompt}]
        text = client.complete_text_sync(messages, temperature=0.0).strip()
    
    # Remove markdown code blocks if they exist
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-len("```")].strip()
        
    if not is_short_text_answer(text):
        return Type2SolveResult(
            answer="",
            unit=None,
            value=None,
            formula=None,
            extraction=extraction,
            verification=Verification(ok=False, message="Conceptual classification failed to produce valid JSON."),
            cot=["Conceptual classification failed."],
            premises=[],
            confidence=0.0,
            error="json_parse_error",
        )

    data = json.loads(text)
    return Type2SolveResult(
        answer=str(data["answer"]),
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(ok=True, message=data["reason"]),
        cot=[data["reason"]],
        premises=[],
        confidence=float(data.get("confidence", 1.0)),
        error=None,
    )
