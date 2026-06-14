"""LLM adjudication fallback for Type 1 cases the symbolic pipeline cannot decide."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient

_UNCERTAIN_TOKEN = "Uncertain"


class Type1FallbackResult(BaseModel):
    """Structured answer returned by the fallback reasoner."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    explanation: str
    premises_used: list[int] = []  # 0-based indices of premises used


class Type1FallbackReasoner:
    """Answer from original natural language when parser drift blocks Z3."""

    def __init__(self, client: ParserClient) -> None:
        self.client = client

    async def answer(
        self,
        *,
        premises: list[str],
        question: str,
        option_labels: list[str],
        options: dict[str, str] | None = None,
    ) -> Type1FallbackResult:
        # Determine mode BEFORE appending "Uncertain" to allowed set
        is_open_ended = not option_labels  # no labels → free-form / wh-question
        allowed = [] if is_open_ended else list(dict.fromkeys([*option_labels, _UNCERTAIN_TOKEN]))

        option_text = "\n".join(
            f"{label}. {text}" for label, text in (options or {}).items()
        )
        numbered_premises = [f"{i}. {p}" for i, p in enumerate(premises, start=1)]
        user_parts = [
            "Premises:",
            *numbered_premises,
            "",
            "Question:",
            question,
        ]
        if option_text and option_text not in question:
            user_parts.extend(["", "Options:", option_text])
        if allowed:
            user_parts.extend(["", f"Allowed answers: {', '.join(allowed)}"])

        system_prompt = _SYSTEM_PROMPT_OPEN if is_open_ended else _SYSTEM_PROMPT
        result = await self.client.parse_as(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            Type1FallbackResult,
            max_tokens=256,
        )

        # Convert 1-based premise indices (shown to LLM) to 0-based
        zero_based = sorted(max(0, i - 1) for i in result.premises_used)

        if is_open_ended:
            # Open-ended / wh-question: return raw LLM answer as-is, no canonicalization
            return result.model_copy(update={"premises_used": zero_based})

        canonical = {
            label.casefold(): label
            for label in allowed
        }.get(result.answer.strip().casefold(), _UNCERTAIN_TOKEN)

        # If LLM still returned Uncertain but we have premises to reason from,
        # retry once with a stronger "best guess" instruction.
        if canonical == _UNCERTAIN_TOKEN and premises:
            canonical, explanation, zero_based = await self._force_concrete_answer(
                numbered_premises=numbered_premises,
                question=question,
                option_text=option_text,
                allowed=[a for a in allowed if a != _UNCERTAIN_TOKEN] or allowed,
                prior_explanation=result.explanation,
            )
            return Type1FallbackResult(
                answer=canonical,
                explanation=explanation,
                premises_used=zero_based,
            )

        return result.model_copy(update={"answer": canonical, "premises_used": zero_based})

    async def _force_concrete_answer(
        self,
        *,
        numbered_premises: list[str],
        question: str,
        option_text: str,
        allowed: list[str],
        prior_explanation: str,
    ) -> tuple[str, str, list[int]]:
        """Retry with an explicit instruction to commit to the best answer."""
        user_parts = [
            "Premises:",
            *numbered_premises,
            "",
            "Question:",
            question,
        ]
        if option_text:
            user_parts.extend(["", "Options:", option_text])
        user_parts.extend([
            "",
            f"Your previous reasoning: {prior_explanation}",
            "",
            f"Based on your reasoning above, select the single best answer from: {', '.join(allowed)}",
            "Do NOT answer Uncertain. Commit to the most likely answer.",
        ])
        result = await self.client.parse_as(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            Type1FallbackResult,
            max_tokens=192,
        )
        canonical = {
            label.casefold(): label for label in allowed
        }.get(result.answer.strip().casefold(), allowed[0] if allowed else result.answer)
        zero_based = sorted(max(0, i - 1) for i in result.premises_used)
        return canonical, result.explanation, zero_based


_SYSTEM_PROMPT = """
You are the final adjudicator for an educational natural-language logic task.
The symbolic solver could not decide because its parser may have represented
equivalent phrases with different predicates, constants, or arities.

Reason directly from the original premises and question:
- Treat clear paraphrases and coreferences as the same concept.
- Apply multi-step implication chains.
- Respect explicit negation.
- For a polar question, answer Yes if the claim follows, No if its negation
  follows, and Uncertain only when neither follows.
- For MCQ, select the single best option label.
- For "fewest premises", choose the option with the shortest explicit
  derivation from stated premises; do not use vacuous truth as a shortcut.
- For "strongest conclusion", prefer the most informative downstream
  conclusion justified by the complete implication chain.
- Return exactly one answer from the supplied Allowed answers.
- In premises_used, list the 1-based numbers of every premise you actually
  relied on to reach your answer (e.g. [1, 3] means premises 1 and 3).

Return JSON only with:
{"answer": "<allowed answer>", "explanation": "<brief justification>", "premises_used": [<1-based premise numbers>]}
""".strip()


_SYSTEM_PROMPT_OPEN = """
You are answering an open-ended question from a set of logical premises.

Rules:
- Read only from the given premises. Do not use outside knowledge.
- Answer concisely and precisely — numbers, names, or short phrases only.
- Do not answer Yes/No unless the question explicitly asks for Yes or No.
- In premises_used, list the 1-based numbers of every premise you actually
  relied on to reach your answer.

Return JSON only with:
{"answer": "<direct answer>", "explanation": "<one sentence justification>", "premises_used": [<1-based premise numbers>]}
""".strip()
