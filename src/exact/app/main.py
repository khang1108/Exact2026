from __future__ import annotations

from fastapi import FastAPI

from exact.app.router import api_router
from exact.logger import setup_logging


def create_app() -> FastAPI:
    setup_logging(level="INFO", log_file="outputs/logs/api.log")

    app = FastAPI(
        title="TraceQA EXACT 2026 API",
        version="0.1.0",
    )
    app.include_router(api_router)
    return app


app = create_app()
