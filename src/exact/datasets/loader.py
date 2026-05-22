from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

SUPPORTED_RECORD_EXTENSIONS = {".json", ".jsonl"}

PREMISES_KEYS = [
    "premises-NL",
    "premises_NL",
    "premises_nl",
    "premises",
    "context",
]
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


def load_exact_dataset(type1_path: str | Path, type2_path: str | Path) -> pd.DataFrame:
    logic_df = load_logic_dataset(Path(type1_path))
    physics_df = load_physics_dataset(Path(type2_path))
    return pd.concat([logic_df, physics_df], ignore_index=True)


def load_logic_dataset(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    records = load_records(file_path)
    rows: list[dict[str, Any]] = []

    for group_index, record in enumerate(records):
        rows.extend(_flatten_logic_record(record, group_index, file_path))

    return pd.DataFrame(rows)


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
                    "premises_nl": [],
                    "premises_fol": [],
                    "support_idx": [],
                    "gold_answer": answer,
                    "gold_unit": unit,
                    "gold_explanation": str(record.get("cot", "")).strip(),
                    "source_path": str(file_path),
                    "id_prefix": prefix,
                    "answer_type": answer_type,
                    "stratify_label": f"physics::{prefix}::{answer_type}",
                }
            )

    return pd.DataFrame(rows)


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    record = dict(raw)

    question = _first_present(record, QUESTION_KEYS)
    premises = _first_present(record, PREMISES_KEYS)
    answer = _first_present(record, ANSWER_KEYS)
    unit = _first_present(record, UNIT_KEYS)

    normalized: dict[str, Any] = {
        "id": record.get("id") or record.get("qid") or record.get("uid"),
        "question": question,
    }

    if premises is not None:
        normalized["premises-NL"] = _normalize_premises(premises)
    if answer is not None:
        normalized["answer"] = _normalize_answer(answer)
    if unit is not None:
        normalized["unit"] = str(unit)

    normalized["_raw"] = record
    return normalized


def detect_logic_question_type(question: str, answer: str) -> str:
    question = question or ""
    answer = str(answer or "").strip()

    if "\nA." in question or answer in {"A", "B", "C", "D"}:
        return "multiple_choice"
    if answer in {"Yes", "No", "Unknown", "Uncertain", "False"}:
        return "yes_no_unknown"
    return "other"


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


def assign_group_splits(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    if abs(sum(ratios.values()) - 1.0) >= 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)
    split_by_group: dict[str, str] = {}
    group_table = df[[group_col, stratify_col]].drop_duplicates(subset=[group_col]).reset_index(drop=True)

    for _, label_groups in group_table.groupby(stratify_col):
        groups = label_groups[group_col].tolist()
        rng.shuffle(groups)

        n_groups = len(groups)
        n_train = int(round(n_groups * ratios["train"]))
        n_dev = int(round(n_groups * ratios["dev"]))

        for group in groups[:n_train]:
            split_by_group[group] = "train"
        for group in groups[n_train : n_train + n_dev]:
            split_by_group[group] = "dev"
        for group in groups[n_train + n_dev :]:
            split_by_group[group] = "test"

    output = df.copy()
    output["split"] = output[group_col].map(split_by_group)
    return output


def _flatten_logic_record(
    record: dict[str, Any],
    group_index: int,
    source_path: Path,
) -> list[dict[str, Any]]:
    premises_nl = record.get("premises-NL", [])
    premises_fol = record.get("premises-FOL", [])
    support_indices = record.get("idx", [])
    questions = record.get("questions", [])
    answers = record.get("answers", [])
    explanations = record.get("explanation", [])
    rows: list[dict[str, Any]] = []

    for question_index, question in enumerate(questions):
        answer = _get_indexed_value(answers, question_index, "")
        explanation = _get_indexed_value(explanations, question_index, "")
        support_idx = _get_indexed_value(support_indices, question_index, [])
        question_type = detect_logic_question_type(question, answer)

        rows.append(
            {
                "id": f"logic_{group_index:04d}_{question_index:02d}",
                "group_id": f"logic_{group_index:04d}",
                "task_type": "logic",
                "question_type": question_type,
                "question": question,
                "premises_nl": premises_nl,
                "premises_fol": premises_fol,
                "support_idx": support_idx,
                "gold_answer": str(answer).strip(),
                "gold_unit": "",
                "gold_explanation": explanation,
                "source_path": str(source_path),
                "stratify_label": f"logic::{question_type}",
            }
        )

    return rows


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


def _normalize_premises(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]

    raise ValueError(
        f"Invalid premises format: expected list[str] or str, got {type(value).__name__}"
    )


def _normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _get_indexed_value(values: Any, index: int, default: Any) -> Any:
    if isinstance(values, list) and index < len(values):
        return values[index]
    return default


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected file path, got directory: {path}")
