"""
Một module dùng để định hướng khi được gọi API từ Endpoint để xác định xem là cần gọi loại nào
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from exact.datasets.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorInfo,
    PredictionRequest,
    PredictionResponse,
    TaskType
)

from exact.logger import get_logger
from exact.router.task_router import TaskRouter

api_router = APIRouter()
task_router = TaskRouter()

@api_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@api_router.get("/predict")
def predict(
    payload: PredictionRequest,
    request: Request
) -> PredictionResponse:
    pass