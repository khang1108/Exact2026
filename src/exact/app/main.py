from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from exact.app.router import api_router
from exact.llm_client import build_json_client_from_settings
from exact.logger import get_logger, setup_logging
from exact.type1.parser import FOLParser, build_parser_client_from_settings
from exact.type1.refiner import Type1Refiner
from exact.type1.solvers import FOLSolver


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared model clients at startup and close them at shutdown.

    The parser client owns a persistent HTTP connection pool and concurrency
    semaphore. Keeping one instance on ``app.state`` allows every Type 1 request
    to reuse those resources instead of reopening connections per premise.
    """

    parser_client = build_parser_client_from_settings()
    app.state.type1_parser_client = parser_client
    app.state.type1_fol_parser = FOLParser(parser_client) if parser_client is not None else None
    app.state.type1_solver = FOLSolver()
    llm_client = build_json_client_from_settings()
    app.state.type1_refiner = Type1Refiner(llm_client) if llm_client is not None else None
    try:
        yield
    finally:
        parser_client = app.state.type1_parser_client
        if parser_client is not None:
            await parser_client.aclose()


def create_app() -> FastAPI:
    """Create the EXACT API with shared model-client lifecycle management."""

    setup_logging(level="INFO", log_file="outputs/logs/api.log")

    app = FastAPI(
        title="TraceQA EXACT 2026 API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def _log_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # uvicorn's error logger has propagate=False, so its 500 traceback never
        # reaches api.log. Log it through the app logger (which does) so failures
        # are diagnosable from the same file as the rest of the app.
        get_logger("exact.app").error(
            f"Unhandled exception on {request.method} {request.url.path}: {exc!r}",
            exc_info=True,
        )
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    app.include_router(api_router)
    return app


app = create_app()
