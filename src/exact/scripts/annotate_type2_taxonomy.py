from __future__ import annotations

import argparse
import csv
from pathlib import Path

from exact.datasets.type2_taxonomy import classify_type2_taxonomy


DEFAULT_INPUT = Path("src/exact/datasets/exact/type2_physics_questions.csv")
DEFAULT_OUTPUT = Path("artifacts/datasets/type2_physics_questions_annotated.csv")


def annotate_type2_csv(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")

        fieldnames = [*reader.fieldnames]
        for column in ("solver_family", "solve_method", "question_type"):
            if column not in fieldnames:
                fieldnames.append(column)

        rows_written = 0
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                label = classify_type2_taxonomy(
                    question=row.get("question", ""),
                    cot=row.get("cot", ""),
                    unit=row.get("unit", ""),
                )
                row["solver_family"] = label.solver_family
                row["solve_method"] = label.solve_method
                row["question_type"] = label.question_type
                writer.writerow(row)
                rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate Type 2 CSV with solver and physics taxonomy.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    count = annotate_type2_csv(args.input, args.output)
    print(f"Annotated {count} rows -> {args.output}")


if __name__ == "__main__":
    main()
