"""Audit typed IR compatibility against the released EXACT Type 1 premises."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from exact.logic.fol_parser import parse_fol
from exact.logic.ir import FormulaItem, TranslatedProblem
from exact.symbolic_solvers.z3_prop import build_theory, theory_status

DEFAULT_DATASET = (
    Path(__file__).parents[1] / "datasets" / "exact" / "Logic_Based_Educational_Queries.json"
)


def audit_type1_ir_coverage(
    dataset_path: Path = DEFAULT_DATASET,
    *,
    check_sat: bool = False,
    timeout_ms: int = 250,
) -> dict[str, Any]:
    """Parse and encode every released premise, optionally checking group SAT."""

    records = json.loads(dataset_path.read_text())
    errors: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    premise_count = 0

    for record_index, record in enumerate(records):
        items: list[FormulaItem] = []
        for premise_index, premise in enumerate(record["premises-FOL"]):
            premise_count += 1
            try:
                items.append(
                    FormulaItem(
                        formula=parse_fol(premise),
                        source_idx=premise_index,
                        text=premise,
                        role="premise",
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "record_index": record_index,
                        "premise_index": premise_index,
                        "premise": premise,
                        "stage": "parse",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if len(items) != len(record["premises-FOL"]):
            continue

        try:
            constraints, _ = build_theory(TranslatedProblem({}, tuple(items), ()))
            if check_sat:
                status_counts[theory_status(constraints, timeout_ms=timeout_ms)] += 1
        except Exception as exc:
            errors.append(
                {
                    "record_index": record_index,
                    "premise_index": None,
                    "premise": None,
                    "stage": "encode",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "dataset": str(dataset_path),
        "records": len(records),
        "premises": premise_count,
        "errors": errors,
        "theory_statuses": dict(status_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--check-sat", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=250)
    args = parser.parse_args()

    result = audit_type1_ir_coverage(
        args.dataset,
        check_sat=args.check_sat,
        timeout_ms=args.timeout_ms,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
