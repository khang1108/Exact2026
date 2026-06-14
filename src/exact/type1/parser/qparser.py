"""Question classifier — one LLM call per question, no FOL.

``QParser`` mirrors ``PremiseFrameParser``: it sends each question to the parser
model and receives a ``QuestionFrameResult`` describing what kind of question it
is, which solver mode should answer it, how to read the word "can", and—for
polar questions—the single declarative claim to test.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from exact.type1.parser.schemas import QuestionFrameResult
from exact.type1.prompts import get_system_prompt_question_frame

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient

ChatMessage = dict[str, Any]


class QParser:
    """Classify each question into a QuestionFrameResult via one LLM call."""

    def __init__(self, client: ParserClient) -> None:
        self.client = client

    async def classify(self, question: str) -> QuestionFrameResult:
        """Classify a single question."""
        return (await self.classify_many([question]))[0]

    async def classify_many(self, questions: list[str]) -> list[QuestionFrameResult]:
        """Dispatch all questions concurrently; vLLM batches them on the GPU."""
        messages_batch = [self._messages(q) for q in questions]
        results = await self.client.parse_many_as(messages_batch, QuestionFrameResult)
        return [
            _correct_obvious_mode_misclassification(question, result)
            for question, result in zip(questions, results)
        ]

    def _messages(self, question: str) -> list[ChatMessage]:
        return [
            {"role": "system", "content": get_system_prompt_question_frame().strip()},
            {"role": "user", "content": f'Input: "{question}"'},
        ]


_PREMISE_SELECTION_RE = re.compile(
    r"\bwhich\b.{0,100}\b(?:premise|premises|set of premises|combination of premises)\b",
    re.IGNORECASE | re.DOTALL,
)


def _correct_obvious_mode_misclassification(
    question: str,
    result: QuestionFrameResult,
) -> QuestionFrameResult:
    """Correct premise-selection labels when options are conclusions/capabilities."""

    if (
        result.solver_mode == "premise_selection"
        and _PREMISE_SELECTION_RE.search(question) is None
    ):
        return result.model_copy(update={"solver_mode": "entailment"})
    return result
