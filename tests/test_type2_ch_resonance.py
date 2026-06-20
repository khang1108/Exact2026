import csv
import math
from pathlib import Path

from exact.common.schemas import PredictionRequest
from exact.config import Settings
from exact.scripts.evaluate_type2_predictions import evaluate_prediction
from exact.type2.pipeline import run_type2_pipeline


CH_DATASET = Path("src/exact/datasets/exact/type2_physics_questions_CH.csv")


def _no_llm_settings() -> Settings:
    return Settings(
        llm_enabled=False,
        type2_use_llm_domain_routing=False,
        type2_use_llm_question_kind_routing=False,
        type2_extraction_mode="heuristic_only",
        type2_use_recovery_loop=False,
        type2_use_llm_formula_selection=False,
        type2_generate_explanation=False,
    )


def _ch_rows_in_requested_range():
    with CH_DATASET.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            query_id = row["id"].strip()
            if query_id.startswith("CH") and query_id[2:].isdigit():
                number = int(query_id[2:])
                if 1 <= number <= 154:
                    yield row


def test_existing_ch001_to_ch154_rows_use_deterministic_pipeline_and_match_dataset():
    settings = _no_llm_settings()
    failures = []

    for row in _ch_rows_in_requested_range():
        response = run_type2_pipeline(
            PredictionRequest(query_id=row["id"], type="type2", question=row["question"]),
            settings=settings,
        )
        route = (response.routing_diagnostics or {}).get("type2_domain_route", {})
        if response.error or not response.answer or route.get("domain") != "CH":
            failures.append((row["id"], "route_or_error", response.error, response.answer, route))
            continue
        if _normalize_unit(response.unit) != _normalize_unit(row["unit"]):
            failures.append((row["id"], "unit", row["unit"], response.unit))
            continue
        if not _answers_match(row["answer"], response.answer):
            failures.append((row["id"], "answer", row["answer"], response.answer))

    assert failures == []


def test_existing_ch001_to_ch154_rows_route_by_question_without_query_id():
    settings = _no_llm_settings()
    failures = []

    for row in _ch_rows_in_requested_range():
        response = run_type2_pipeline(
            PredictionRequest(type="type2", question=row["question"]),
            settings=settings,
        )
        route = (response.routing_diagnostics or {}).get("type2_domain_route", {})
        prediction = response.model_dump(mode="json")
        prediction["gold_answer"] = row["answer"]
        prediction["gold_unit"] = row["unit"]
        evaluation = evaluate_prediction(
            prediction,
            relative_tolerance=0.02,
            absolute_tolerance=1e-9,
            case_sensitive_text=False,
        )
        if route.get("domain") != "CH" or not evaluation.status.startswith("correct"):
            failures.append(
                (
                    row["id"],
                    route,
                    response.answer,
                    response.unit,
                    row["answer"],
                    row["unit"],
                    evaluation.status,
                )
            )

    assert failures == []


def _normalize_unit(unit: str | None) -> str:
    return (unit or "").replace("Ω", "ohm").replace("μ", "u")


def _answers_match(expected: str, actual: str) -> bool:
    if "π" in expected or "π" in actual:
        return expected == actual
    expected_value = _to_float(expected)
    actual_value = _to_float(actual)
    return math.isclose(
        actual_value,
        expected_value,
        rel_tol=0.005,
        abs_tol=0.02,
    )


def _to_float(value: str) -> float:
    normalized = value.replace("× 10^-", "e-").replace("x 10^-", "e-")
    return float(normalized)
