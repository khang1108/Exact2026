import string

import pytest
from fastapi.testclient import TestClient

from exact.app.main import create_app


REQUIRED_KEYS = {
    "query_id",
    "answer",
    "unit",
    "explanation",
    "premises_used",
    "reasoning",
}


def assert_official_response_shape(data):
    assert isinstance(data, list)
    assert len(data) == 1
    item = data[0]
    assert isinstance(item, dict)
    assert REQUIRED_KEYS.issubset(item.keys())
    return item


def assert_type2_contract(item, query_id: str):
    assert item["query_id"] == query_id
    assert isinstance(item["answer"], str)
    assert isinstance(item["unit"], str)
    assert isinstance(item["explanation"], str)
    assert item["explanation"].strip()
    assert item["premises_used"] == []
    assert item["reasoning"] is None or isinstance(item["reasoning"], dict)


def assert_type1_contract(item, query_id: str, options: list[str]):
    assert item["query_id"] == query_id
    assert isinstance(item["answer"], str)
    assert item["answer"] in options
    assert item["unit"] == ""
    assert isinstance(item["explanation"], str)
    assert item["explanation"].strip()
    assert isinstance(item["premises_used"], list)
    assert all(isinstance(index, int) for index in item["premises_used"])
    assert item["reasoning"] is None or isinstance(item["reasoning"], dict)


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_predict_type2_response_shape(client: TestClient):
    payload = {
        "query_id": "T2_0001",
        "type": "type2",
        "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12 V battery. Find the total current.",
        "premises": [],
        "options": [],
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    item = assert_official_response_shape(response.json())
    assert_type2_contract(item, payload["query_id"])


def test_predict_type2_answer_unit_separation(client: TestClient):
    payload = {
        "query_id": "T2_0002",
        "type": "type2",
        "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12 V battery. Find the total current.",
        "premises": [],
        "options": [],
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    item = assert_official_response_shape(response.json())
    assert_type2_contract(item, payload["query_id"])
    assert item["unit"]
    assert all(char in string.printable for char in item["unit"])
    assert item["unit"].isascii()
    assert item["unit"] not in {"Ω", "μF"}
    assert item["unit"] not in item["answer"]


def test_predict_type1_response_shape_if_available(client: TestClient):
    payload = {
        "query_id": "T1_0001",
        "type": "type1",
        "query": "Is Student A eligible for graduation?",
        "premises": [
            "A student with at least 120 credits is eligible.",
            "Student A has 118 credits.",
        ],
        "options": ["Yes", "No", "Uncertain"],
    }

    response = client.post("/predict", json=payload)

    if response.status_code == 503:
        pytest.skip("Type 1 parser model service is not configured in test environment")

    assert response.status_code == 200
    item = assert_official_response_shape(response.json())
    assert_type1_contract(item, payload["query_id"], payload["options"])
