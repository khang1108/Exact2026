import json
import random
from pathlib import Path
import pandas as pd

def detect_logic_question_type(question: str, answer: str) -> str:
    """Classify Type 1 questions into broad answer-format groups."""
    question = question or ""
    answer = str(answer or "").strip()

    if "\nA." in question or answer in {"A", "B", "C", "D"}:
        return "multiple_choice"
    if answer in {"Yes", "No", "Unknown", "Uncertain", "False"}:
        return "yes_no_unknown"
    return "other"

def detect_physics_answer_type(answer: str, unit: str) -> str:
    """Classify Type 2 answers by whether they look numeric, conceptual, or missing."""
    answer = str(answer or "").strip()
    unit = str(unit or "").strip()

    if not answer:
        return "missing_answer"

    has_digit = any(ch.isdigit() for ch in answer)
    if has_digit:
        return "numeric_with_unit" if unit and unit not in {"-", "—"} else "numeric_no_unit"
    return "conceptual"

def physics_id_prefix(example_id: str) -> str:
    """Return the alphabetic prefix of a physics ID, e.g. TD401 -> TD."""
    prefix = ""
    for ch in str(example_id):
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix or "unknown"

def load_logic_dataset(path: Path) -> pd.DataFrame:
    """Load and flatten the Type 1 logic dataset into one row per question."""
    raw_records = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    for group_index, record in enumerate(raw_records):
        premises_nl = record.get("premises-NL", [])
        premises_fol = record.get("premises-FOL", [])
        support_indices = record.get("idx", [])
        questions = record.get("questions", [])
        answers = record.get("answers", [])
        explanations = record.get("explanation", [])

        for question_index, question in enumerate(questions):
            answer = answers[question_index] if question_index < len(answers) else ""
            explanation = explanations[question_index] if question_index < len(explanations) else ""
            support_idx = support_indices[question_index] if question_index < len(support_indices) else []
            question_type = detect_logic_question_type(question, answer)

            rows.append({
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
                "source_path": str(path),
                "stratify_label": f"logic::{question_type}",
            })

    return pd.DataFrame(rows)

def load_physics_dataset(path: Path) -> pd.DataFrame:
    """Load the Type 2 physics CSV dataset into the normalized row format."""
    raw_df = pd.read_csv(path).fillna("")
    rows = []

    for _, record in raw_df.iterrows():
        example_id = str(record.get("id", "")).strip()
        answer = str(record.get("answer", "")).strip()
        unit = str(record.get("unit", "")).strip()
        prefix = physics_id_prefix(example_id)
        answer_type = detect_physics_answer_type(answer, unit)

        rows.append({
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
            "source_path": str(path),
            "id_prefix": prefix,
            "answer_type": answer_type,
            "stratify_label": f"physics::{prefix}::{answer_type}",
        })

    return pd.DataFrame(rows)

def assign_group_splits(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """Assign train/dev/test labels while keeping every group in only one split."""
    assert abs(sum(ratios.values()) - 1.0) < 1e-9, "Split ratios must sum to 1.0"
    rng = random.Random(seed)
    split_by_group: dict[str, str] = {}

    group_table = df[[group_col, stratify_col]].drop_duplicates(subset=[group_col]).reset_index(drop=True)

    for _, label_groups in group_table.groupby(stratify_col):
        groups = label_groups[group_col].tolist()
        rng.shuffle(groups)

        n = len(groups)
        n_train = int(round(n * ratios["train"]))
        n_dev = int(round(n * ratios["dev"]))

        train_groups = groups[:n_train]
        dev_groups = groups[n_train : n_train + n_dev]
        test_groups = groups[n_train + n_dev :]

        for group in train_groups:
            split_by_group[group] = "train"
        for group in dev_groups:
            split_by_group[group] = "dev"
        for group in test_groups:
            split_by_group[group] = "test"

    output = df.copy()
    output["split"] = output[group_col].map(split_by_group)
    return output
