from __future__ import annotations

__all__ = ["run_type1_pipeline"]


def __getattr__(name: str):
    if name == "run_type1_pipeline":
        from exact.logic.pipeline import run_type1_pipeline

        return run_type1_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
