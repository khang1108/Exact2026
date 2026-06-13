"""LLM-based answer head for Type 1 logic questions.

Called only when Z3 returns "Uncertain" — i.e., when the SMT solver cannot
prove or disprove any option from the parsed FOL.  The head receives the
original natural-language premises and question, so it is not affected by any
parser FOL errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exact.llm_client import VLLMJsonClient

# ---------------------------------------------------------------------------
# Guided-JSON schemas (keep arity minimal — vLLM enforcer is strict)
# ---------------------------------------------------------------------------

_YNU_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["Yes", "No", "Uncertain"]},
    },
    "required": ["answer"],
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_YNU = """\
You are a formal logic expert. Given a list of premises and a yes/no question, \
decide whether the answer is Yes, No, or Uncertain.

Rules:
- Yes   → the conclusion follows necessarily from the premises.
- No    → the conclusion contradicts the premises.
- Uncertain → the premises do not provide enough information to decide.

Return ONLY valid JSON: {"answer": "Yes" | "No" | "Uncertain"}"""

_SYSTEM_MCQ = """\
You are a formal logic expert. Given a list of premises and multiple-choice options, \
select the option that logically follows from the premises.

Rules:
- Pick the single option best supported by the premises.
- If no option clearly follows, pick the most plausible one.

Return ONLY valid JSON: {"answer": "<label>"}"""


# ---------------------------------------------------------------------------
# Head class
# ---------------------------------------------------------------------------

class Type1LLMHead:
    """Thin async wrapper that calls the main vLLM to resolve uncertain answers."""

    def __init__(self, client: VLLMJsonClient) -> None:
        self._client = client

    async def answer_ynu(self, premises: list[str], question: str) -> str:
        """Return 'Yes', 'No', or 'Uncertain' for a yes/no/uncertain question."""
        user = _fmt_ynu(premises, question)
        data = await self._client.complete_json(
            messages=[
                {"role": "system", "content": _SYSTEM_YNU},
                {"role": "user",   "content": user},
            ],
            temperature=0.0,
            max_tokens=32,
            json_schema=_YNU_SCHEMA,
        )
        answer = data.get("answer", "Uncertain")
        return answer if answer in ("Yes", "No", "Uncertain") else "Uncertain"

    async def answer_mcq(
        self,
        premises: list[str],
        question: str,
        options: dict[str, str],
    ) -> str:
        """Return the option label (A/B/C/D) or 'Uncertain'."""
        labels = sorted(options)
        schema: dict = {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": labels}},
            "required": ["answer"],
        }
        user = _fmt_mcq(premises, question, options)
        data = await self._client.complete_json(
            messages=[
                {"role": "system", "content": _SYSTEM_MCQ},
                {"role": "user",   "content": user},
            ],
            temperature=0.0,
            max_tokens=32,
            json_schema=schema,
        )
        answer = data.get("answer", "Uncertain")
        return answer if answer in labels else "Uncertain"


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def _fmt_premises(premises: list[str]) -> str:
    return "\n".join(f"{i}. {p}" for i, p in enumerate(premises, 1))


def _fmt_ynu(premises: list[str], question: str) -> str:
    return f"Premises:\n{_fmt_premises(premises)}\n\nQuestion: {question}"


def _fmt_mcq(premises: list[str], question: str, options: dict[str, str]) -> str:
    opts = "\n".join(f"{label}. {text}" for label, text in sorted(options.items()))
    return f"Premises:\n{_fmt_premises(premises)}\n\nQuestion: {question}\n\nOptions:\n{opts}"
