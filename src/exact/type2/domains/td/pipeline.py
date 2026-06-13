from __future__ import annotations

from exact.common.schemas import PredictionRequest, PredictionResponse
from exact.config import Settings
from exact.type2.pipeline import run_generic_pipeline

def run_td_pipeline(
    request: PredictionRequest,
    settings: Settings | None = None,
) -> PredictionResponse:
    """Run the TD (capacitor/static) specific pipeline.
    
    Currently wraps the generic pipeline as the existing deterministic solvers
    are perfectly suited for TD tasks without modifications.
    """
    return run_generic_pipeline(request, settings)
