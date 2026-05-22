from __future__ import annotations

from exact.datasets.schemas import PredictionRequest, PredictionResponse, TaskType, to_official_response
from exact.router.task_router import TaskRouter
from exact.type1.pipeline import run_type1_pipeline
from exact.type2.pipeline import run_type2_pipeline


class FakeType1Client:
    def generate(self, prompt: str) -> str:
        return """
        {
          "answer": "yes",
          "explanation": "P1 and P2 support the conclusion.",
          "fol": null,
          "cot": ["Use P1", "Use P2"],
          "premises": ["P1", "P2"],
          "confidence": 0.9
        }
        """


def test_prediction_request_alias_and_router() -> None:
    request = PredictionRequest.model_validate(
        {
            "id": "t1",
            "premises-NL": ["If A then B.", "A."],
            "question": "Does B hold?",
        }
    )

    decision = TaskRouter().route(request)

    assert request.premises_nl == ["If A then B.", "A."]
    assert request.inferred_task_type == TaskType.TYPE1_LOGIC
    assert decision.task_type == TaskType.TYPE1_LOGIC


def test_official_response_contains_exact_fields() -> None:
    response = PredictionResponse(answer="Yes", explanation="Because P1 applies.", confidence=0.8)

    assert to_official_response(response) == {
        "answer": "Yes",
        "explanation": "Because P1 applies.",
        "fol": None,
        "cot": None,
        "premises": None,
        "confidence": 0.8,
    }


def test_type1_pipeline_normalizes_structured_model_output() -> None:
    request = PredictionRequest.model_validate(
        {
            "id": "logic_1",
            "premises-NL": ["If A then B.", "A."],
            "question": "Does B hold?",
        }
    )

    response = run_type1_pipeline(request, llm_client=FakeType1Client())

    assert response.answer == "Yes"
    assert response.task_type == TaskType.TYPE1_LOGIC
    assert response.premises == ["P1", "P2"]
    assert response.confidence == 0.9


def test_type2_placeholder_keeps_response_contract_stable() -> None:
    request = PredictionRequest.model_validate(
        {
            "id": "physics_1",
            "question": "Calculate the current when U = 12 V and R = 6 ohm.",
        }
    )
    response = run_type2_pipeline(request)

    assert response.task_type == TaskType.TYPE2_PHYSICS
    assert response.answer == ""
    assert response.confidence == 0.0
    assert response.error == "type2_pipeline_not_implemented"
