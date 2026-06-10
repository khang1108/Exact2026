from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

SUPPORTED_RECORD_EXTENSIONS = {".json", ".jsonl"}

QUESTION_KEYS = ["question", "query", "problem"]
ANSWER_KEYS = ["answer", "answers", "gold_answer", "label"]
UNIT_KEYS = ["unit", "gold_unit"]


def read_json(path: str | Path) -> Any:
    file_path = Path(path)
    _ensure_file(file_path)

    if file_path.suffix != ".json":
        raise ValueError(f"Expected .json file, got {file_path.suffix}: {file_path}")

    return json.loads(file_path.read_text(encoding="utf-8"))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    file_path = Path(path)
    _ensure_file(file_path)

    if file_path.suffix != ".jsonl":
        raise ValueError(f"Expected .jsonl file, got {file_path.suffix}: {file_path}")

    with file_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {file_path}:{line_number}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at {file_path}:{line_number}, "
                    f"got {type(record).__name__}"
                )

            yield record


def load_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    _ensure_file(file_path)

    if file_path.suffix not in SUPPORTED_RECORD_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {file_path.suffix}. "
            f"Expected one of {sorted(SUPPORTED_RECORD_EXTENSIONS)}"
        )

    if file_path.suffix == ".jsonl":
        return list(iter_jsonl(file_path))

    data = read_json(file_path)
    records = _extract_records(data, file_path)

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Expected object at index {index} in {file_path}, "
                f"got {type(record).__name__}"
            )

    return records


def load_physics_dataset(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    _ensure_file(file_path)

    if file_path.suffix != ".csv":
        raise ValueError(f"Expected .csv file, got {file_path.suffix}: {file_path}")

    rows: list[dict[str, Any]] = []

    with file_path.open(newline="", encoding="utf-8") as file:
        for record in csv.DictReader(file):
            example_id = str(record.get("id", "")).strip()
            answer = str(record.get("answer", "")).strip()
            unit = str(record.get("unit", "")).strip()
            prefix = physics_id_prefix(example_id)
            answer_type = detect_physics_answer_type(answer, unit)

            rows.append(
                {
                    "id": f"physics_{example_id}",
                    "group_id": f"physics_{example_id}",
                    "task_type": "physics",
                    "question_type": "physics",
                    "question": str(record.get("question", "")).strip(),
                    "gold_answer": answer,
                    "gold_unit": unit,
                    "gold_explanation": str(record.get("cot", "")).strip(),
                    "source_path": str(file_path),
                    "id_prefix": prefix,
                    "answer_type": answer_type,
                    "stratify_label": f"physics::{prefix}::{answer_type}",
                    "gold_solver_family": str(record.get("solver_family", "")).strip() or None,
                    "gold_or_dataset_method": str(record.get("solve_method", "")).strip() or None,
                    "gold_formula_family": str(record.get("question_type", "")).strip() or None,
                }
            )

    return pd.DataFrame(rows)


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    record = dict(raw)

    question = _first_present(record, QUESTION_KEYS)
    answer = _first_present(record, ANSWER_KEYS)
    unit = _first_present(record, UNIT_KEYS)

    normalized: dict[str, Any] = {
        "id": record.get("id") or record.get("qid") or record.get("uid"),
        "question": question,
    }

    if answer is not None:
        normalized["answer"] = _normalize_answer(answer)
    if unit is not None:
        normalized["unit"] = str(unit)

    normalized["_raw"] = record
    return normalized


def detect_physics_answer_type(answer: str, unit: str) -> str:
    answer = str(answer or "").strip()
    unit = str(unit or "").strip()

    if not answer:
        return "missing_answer"

    has_digit = any(char.isdigit() for char in answer)
    if has_digit:
        return "numeric_with_unit" if unit and unit not in {"-", "—"} else "numeric_no_unit"
    return "conceptual"


def physics_id_prefix(example_id: str) -> str:
    prefix = ""
    for char in str(example_id):
        if char.isalpha():
            prefix += char
        else:
            break
    return prefix or "unknown"


def _extract_records(data: Any, source_path: Path) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("instances"), list):
        return data["instances"]
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Unsupported JSON structure in {source_path}: {type(data).__name__}")


def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected file path, got directory: {path}")
