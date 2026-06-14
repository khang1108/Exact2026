from __future__ import annotations

import asyncio
from types import SimpleNamespace

from exact.app import router
from exact.common.schemas import (
    PredictionRequest,
    PredictionResponse,
    TaskType,
    UnifiedPredictionRequest,
)
from exact.type1.pipeline import _normalize_options


def _response(task_type: TaskType, request_id: str | None) -> PredictionResponse:
    return PredictionResponse(
        id=request_id,
        task_type=task_type,
        answer="ok",
        explanation="test",
    )


def test_unified_schema_accepts_public_and_legacy_field_names() -> None:
    public = UnifiedPredictionRequest.model_validate(
        {
            "query_id": "T1_0001",
            "type": "type1",
            "query": "Is Student A eligible for graduation?",
            "premises": [" Student A completed 118 credits. "],
            "options": ["Yes", "No", "Uncertain"],
        }
    )
    legacy = UnifiedPredictionRequest.model_validate(
        {"id": "T2_0001", "type": "type2", "question": "Calculate the voltage."}
    )

    assert public.id == "T1_0001"
    assert public.question == "Is Student A eligible for graduation?"
    assert public.premises == ["Student A completed 118 credits."]
    assert legacy.query_id == "T2_0001"


def test_openapi_schema_uses_unified_public_field_names() -> None:
    schema = UnifiedPredictionRequest.model_json_schema()
    properties = schema["properties"]

    assert "query_id" in properties
    assert "query" in properties
    assert "id" not in properties
    assert "question" not in properties
    assert set(schema["required"]) == {"query_id", "query", "type"}


def test_ynu_labels_are_not_treated_as_mcq_conclusions() -> None:
    assert _normalize_options(["Yes", "No", "Uncertain"]) == {}
    assert _normalize_options(["Yes", "No", "Unknown"]) == {}
    assert _normalize_options(["First conclusion", "Second conclusion"]) == {
        "A": "First conclusion",
        "B": "Second conclusion",
    }


def test_predict_routes_explicit_type1_and_type2(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    async def fake_type1(payload, premise_parser, question_parser, solver):
        calls.append(("type1", payload.query_id))
        return _response(TaskType.TYPE1_LOGIC, payload.query_id)

    def fake_type2(payload):
        calls.append(("type2", payload.query_id))
        return _response(TaskType.TYPE2_PHYSICS, payload.query_id)

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(router, "run_type1_pipeline", fake_type1)
    monkeypatch.setattr(router, "run_type2_pipeline", fake_type2)
    monkeypatch.setattr(router.asyncio, "to_thread", run_inline)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                type1_premise_parser=object(),
                type1_question_parser=object(),
                type1_solver=object(),
            )
        )
    )

    async def scenario() -> None:
        type1 = UnifiedPredictionRequest(
            query_id="T1_0001",
            type="type1",
            query="Is Student A eligible?",
            premises=["Student A completed 118 credits."],
            options=["Yes", "No", "Uncertain"],
        )
        type2 = UnifiedPredictionRequest(
            query_id="T2_0001",
            type="type2",
            query="Calculate the voltage.",
        )

        assert (await router.predict(type1, request)).task_type == TaskType.TYPE1_LOGIC
        assert (await router.predict(type2, request)).task_type == TaskType.TYPE2_PHYSICS

    asyncio.run(scenario())
    assert calls == [("type1", "T1_0001"), ("type2", "T2_0001")]
