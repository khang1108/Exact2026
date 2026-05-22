from dataclasses import dataclass

from exact.datasets.schemas import PredictionRequest, TaskType

@dataclass(frozen=True)
class RouteDecision:
    task_type: TaskType
    reason: str


# Backward-compatible alias for older imports.
RouterDecision = RouteDecision

class TaskRouter:
    def route(self, request: PredictionRequest) -> RouteDecision:
        if request.premises_nl:
            return RouteDecision(
                task_type=TaskType.TYPE1_LOGIC,
                reason="premises_nl present",
            )

        return RouteDecision(
            task_type=TaskType.TYPE2_PHYSICS,
            reason="premises_nl absent",
        )
