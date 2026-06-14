from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from exact.app.router import api_router
from exact.logger import setup_logging
from exact.type1.fallback import Type1FallbackReasoner
from exact.type1.parser import (
    PremiseParser,
    QuestionSideParser,
    build_parser_client_from_settings,
)
from exact.type1.solvers import FOLSolver


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared model clients at startup and close them at shutdown."""

    parser_client = build_parser_client_from_settings()
    app.state.type1_parser_client = parser_client
    app.state.type1_premise_parser = (
        PremiseParser.from_parser_client(parser_client) if parser_client is not None else None
    )
    app.state.type1_question_parser = (
        QuestionSideParser.from_parser_client(parser_client) if parser_client is not None else None
    )
    app.state.type1_solver = FOLSolver()
    app.state.type1_fallback_reasoner = (
        Type1FallbackReasoner(parser_client) if parser_client is not None else None
    )
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
    app.include_router(api_router)
    return app


app = create_app()
