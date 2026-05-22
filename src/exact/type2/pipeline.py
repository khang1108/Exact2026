from __future__ import annotations

from exact.datasets.schemas import PredictionRequest, PredictionResponse, TaskType


class Type2Pipeline:
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        raise NotImplementedError("Type 2 physics pipeline is not implemented yet.")


def ensure_type2_request(request: PredictionRequest) -> None:
    if request.inferred_task_type is not TaskType.TYPE2_PHYSICS:
        raise ValueError("Type2Pipeline expects a physics request without premises.")
