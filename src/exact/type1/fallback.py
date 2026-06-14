"""LLM adjudication fallback for Type 1 cases the symbolic pipeline cannot decide."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient


class Type1FallbackResult(BaseModel):
    """Structured answer returned by the fallback reasoner."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    explanation: str


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
        allowed = list(dict.fromkeys([*option_labels, "Uncertain"]))
        option_text = "\n".join(
            f"{label}. {text}" for label, text in (options or {}).items()
        )
        user_parts = [
            "Premises:",
            *[f"{index}. {premise}" for index, premise in enumerate(premises, start=1)],
            "",
            "Question:",
            question,
        ]
        if option_text and option_text not in question:
            user_parts.extend(["", "Options:", option_text])
        user_parts.extend(["", f"Allowed answers: {', '.join(allowed)}"])

        result = await self.client.parse_as(
            [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": "\n".join(user_parts),
                },
            ],
            Type1FallbackResult,
            max_tokens=256,
        )
        canonical = {
            label.casefold(): label
            for label in allowed
        }.get(result.answer.strip().casefold(), "Uncertain")
        return result.model_copy(update={"answer": canonical})


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

Return JSON only with:
{"answer": "<allowed answer>", "explanation": "<brief justification>"}
""".strip()
