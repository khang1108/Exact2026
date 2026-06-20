import csv
from pathlib import Path

from exact.common.schemas import PredictionRequest
from exact.type2.pipeline import run_type2_pipeline


TD_DATASET = Path("src/exact/datasets/exact/type2_physics_questions_TD.csv")


def _late_td_rows():
    with TD_DATASET.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            query_id = row["id"]
            if query_id.startswith("TD") and query_id[2:].isdigit():
                number = int(query_id[2:])
                if 367 <= number <= 400:
                    yield query_id, row["question"]


def test_td367_to_td400_have_deterministic_answers_without_pipeline_error():
    failures = []
    for query_id, question in _late_td_rows():
        response = run_type2_pipeline(
            PredictionRequest(query_id=query_id, type="type2", question=question)
        )
        if response.error or not str(response.answer).strip():
            failures.append((query_id, response.error, response.answer))

    assert failures == []
