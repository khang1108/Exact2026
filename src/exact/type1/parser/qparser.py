"""Question classifier — one LLM call per question, no FOL.

``QParser`` mirrors ``PremiseFrameParser``: it sends each question to the parser
model and receives a ``QuestionFrameResult`` describing what kind of question it
is, which solver mode should answer it, how to read the word "can", and—for
polar questions—the single declarative claim to test.
"""

from __future__ import annotations

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
        return await self.client.parse_many_as(messages_batch, QuestionFrameResult)

    def _messages(self, question: str) -> list[ChatMessage]:
        return [
            {"role": "system", "content": get_system_prompt_question_frame().strip()},
            {"role": "user", "content": f'Input: "{question}"'},
        ]
