"""Compatibility exports for shared EXACT schemas.

New code should import these contracts from `exact.common.schemas`. This module
remains so older notebooks and scripts that import `exact.datasets.schemas` keep
working while dataset-specific parsing stays under `exact.datasets`.
"""

from __future__ import annotations

from exact.common.schemas import (
    AppBaseModel,
    BatchPredictionRequest,
    BatchPredictionResponse,
    EquationStep,
    ErrorInfo,
    InboundBaseModel,
    PredictionRequest,
    PredictionResponse,
    ProofStep,
    QuestionType,
    TaskType,
    Type1Evidence,
    Type2Evidence,
    to_official_response,
)

__all__ = [
    "AppBaseModel",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "EquationStep",
    "ErrorInfo",
    "InboundBaseModel",
    "PredictionRequest",
    "PredictionResponse",
    "ProofStep",
    "QuestionType",
    "TaskType",
    "Type1Evidence",
    "Type2Evidence",
    "to_official_response",
]
