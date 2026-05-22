from __future__ import annotations

import re

from exact.logger import get_logger
from dataclasses import dataclass
from exact.datasets.schemas import TaskType, PredictionRequest

logger = get_logger(__name__)

@dataclass(frozen=True)
class RouterDecision:
    task_type: TaskType
    reason: str

class TaskRouter:
    """
    A router class to determine the task type based on the input.

    Types:
        - Type 1: premise-NL + question
        - Type 2: question only
    """

    def route(
        self, 
        request: PredictionRequest,
    ) -> RouterDecision:
        if request.premises_nl:
            return RouterDecision(
                task_type=TaskType.TYPE1_LOGIC,
                reason="Has premises"
            )
        return RouterDecision(
            task_type=TaskType.TYPE2_PHYSICS,
            reason="No premises"
        )