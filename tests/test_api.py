from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_placeholder():
    body = {
        "id": "t1",
        "question": "Sample?",
        "premises": ["If A then B.", "A."],
    }
    r = client.post("/query", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data and "explanation" in data
    assert data["reasoning_type"] == "logic"
