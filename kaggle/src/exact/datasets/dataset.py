from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from pydantic import ValidationError

from exact.datasets.loader import load_logic_dataset, load_physics_dataset, load_records, normalize_record
from exact.common.schemas import PredictionRequest
from exact.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedExample:
    """Validated prediction request plus optional gold labels for evaluation."""

    request: PredictionRequest
    gold_answer: str | None = None
    gold_unit: str | None = None
    raw: dict[str, Any] | None = None


class ExactDataset:
    """Map-style dataset for validated EXACT prediction requests."""

    def __init__(self, examples: list[LoadedExample], source_path: Path | None = None):
        self.examples = examples
        self.source_path = source_path

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        skip_invalid: bool = False,
    ) -> "ExactDataset":
        file_path = Path(path)
        rows = _load_rows(file_path)
        examples: list[LoadedExample] = []

        for index, row in enumerate(rows):
            try:
                request = PredictionRequest.model_validate(_to_prediction_payload(row))
                examples.append(
                    LoadedExample(
                        request=request,
                        gold_answer=row.get("gold_answer") or row.get("answer"),
                        gold_unit=row.get("gold_unit") or row.get("unit"),
                        raw=row.get("_raw") or row,
                    )
                )
            except (ValidationError, ValueError) as exc:
                message = f"Invalid record at {file_path}:{index}: {exc}"
                if skip_invalid:
                    logger.warning(message)
                    continue
                raise ValueError(message) from exc

        logger.info("Loaded %d examples from %s", len(examples), file_path)
        return cls(examples=examples, source_path=file_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LoadedExample:
        return self.examples[index]

    def __iter__(self) -> Iterator[LoadedExample]:
        return iter(self.examples)

    def head(self, n: int = 5) -> list[LoadedExample]:
        return self.examples[:n]

    def filter_type1(self) -> "ExactDataset":
        return ExactDataset(
            examples=[ex for ex in self.examples if ex.request.inferred_task_type.value == "type1_logic"],
            source_path=self.source_path,
        )

    def filter_type2(self) -> "ExactDataset":
        return ExactDataset(
            examples=[ex for ex in self.examples if ex.request.inferred_task_type.value == "type2_physics"],
            source_path=self.source_path,
        )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        return load_physics_dataset(path).to_dict(orient="records")

    if path.name == "Logic_Based_Educational_Queries.json":
        return load_logic_dataset(path).to_dict(orient="records")

    return [normalize_record(record) for record in load_records(path)]


def _to_prediction_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": row.get("id"),
        "question": row.get("question"),
    }

    premises = row.get("premises_nl") or row.get("premises-NL")
    if premises:
        payload["premises-NL"] = premises

    for key in ("gold_solver_family", "gold_or_dataset_method", "gold_formula_family"):
        if row.get(key):
            payload[key] = row[key]

    return payload
