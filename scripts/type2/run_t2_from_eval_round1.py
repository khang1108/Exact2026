#!/usr/bin/env python3
"""Extract Type 2 questions from an eval-round1 log file and run the local Type 2 pipeline on them.

Usage:
    python scripts/type2/run_t2_from_eval_round1.py
    python scripts/type2/run_t2_from_eval_round1.py --eval tmp/exact_eval_round1_Prompt2Win.json
    python scripts/type2/run_t2_from_eval_round1.py --limit 1   # smoke test on the first T2 question
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exact.common.schemas import PredictionRequest  # noqa: E402
from exact.config import Settings, get_settings  # noqa: E402
from exact.scripts.evaluate_type2_predictions import (  # noqa: E402
    classify_answer_type,
    evaluate_prediction,
    summarize,
)
from exact.type2.pipeline import run_type2_pipeline  # noqa: E402

DEFAULT_EVAL = ROOT / "tmp" / "exact_eval_round1_Prompt2Win.json"
DEFAULT_OUTPUT = ROOT / "tmp" / "t2_pipeline_run_round1.json"
RELATIVE_TOLERANCE = 0.02
ABSOLUTE_TOLERANCE = 1e-9


def build_deterministic_settings() -> Settings:
    """Return settings with the LLM disabled so the pipeline runs deterministic-first."""
    base = get_settings()
    return base.model_copy(
        update={
            "llm_base_url": None,
            "llm_api_key": None,
            "type2_use_llm_domain_routing": False,
            "type2_use_llm_question_kind_routing": False,
            "type2_use_recovery_loop": False,
        }
    )


def extract_t2_logs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    logs = payload.get("logs") or []
    return [log for log in logs if str(log.get("query_id", "")).startswith("T2")]


def domain_solver(diagnostics: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(diagnostics, dict):
        return "", ""
    route = diagnostics.get("type2_domain_route") or {}
    domain = str(route.get("domain") or diagnostics.get("domain") or "")
    solver = str(diagnostics.get("solver") or "")
    if not solver:
        fallback = diagnostics.get("fallback") or {}
        solver = str(fallback.get("actual_solver") or fallback.get("predicted_solver_family") or "")
    return domain, solver
def run_one(log: dict[str, Any], settings: Settings) -> dict[str, Any]:
    qid = str(log.get("query_id"))
    request_payload = log.get("request_payload") or {}
    query = request_payload.get("query") or request_payload.get("question") or ""
    premises = request_payload.get("premises") or []

    expected = log.get("expected") or {}
    gold_answer = expected.get("answer")
    gold_unit = expected.get("unit")

    round1 = log.get("model_response") or {}
    round1_answer = round1.get("answer")
    round1_unit = round1.get("unit")
    round1_ok = bool(log.get("ok"))
    round1_status = log.get("status")

    request = PredictionRequest(query_id=qid, type="type2", question=query, premises=premises)
    response = run_type2_pipeline(request, settings=settings)
    dumped = response.model_dump(mode="json")
    diagnostics = dumped.get("routing_diagnostics")
    domain, solver = domain_solver(diagnostics)

    prediction = {
        "query_id": qid,
        "answer": dumped.get("answer") or "",
        "unit": dumped.get("unit") or "",
        "gold_answer": gold_answer,
        "gold_unit": gold_unit,
        "error": dumped.get("error"),
        "question_type": dumped.get("question_type"),
    }
    row = evaluate_prediction(
        prediction,
        relative_tolerance=RELATIVE_TOLERANCE,
        absolute_tolerance=ABSOLUTE_TOLERANCE,
        case_sensitive_text=False,
    )
    return {
        "query_id": qid,
        "query": query,
        "gold_answer": gold_answer,
        "gold_unit": gold_unit,
        "answer_type": classify_answer_type(gold_answer, gold_unit),
        "our_answer": prediction["answer"],
        "our_unit": prediction["unit"],
        "our_status": row.status,
        "our_numeric_ok": row.numeric_ok,
        "our_unit_ok": row.unit_ok,
        "our_relative_error": row.relative_error,
        "our_error": prediction["error"],
        "our_domain": domain,
        "our_solver": solver,
        "our_explanation": dumped.get("explanation"),
        "round1_answer": round1_answer,
        "round1_unit": round1_unit,
        "round1_ok": round1_ok,
        "round1_status": round1_status,
    }
def _compact(text: str) -> str:
    return " ".join(str(text).split())


def print_table(results: list[dict[str, Any]]) -> None:
    header = f"{'ID':<9} {'Gold':<18} {'Our answer':<20} {'St':<16} {'R1':<4} {'Domain':<8} {'Solver'}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        gold = _compact(f"{r['gold_answer']} {r['gold_unit'] or ''}")
        ours = _compact(f"{r['our_answer']} {r['our_unit'] or ''}")
        r1 = "ok" if r["round1_ok"] else "X"
        print(
            f"{r['query_id']:<9} {gold:<18} {ours:<20} {r['our_status']:<16} {r1:<4} "
            f"{r['our_domain']:<8} {r['our_solver']}"
        )


def print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    our_correct = sum(1 for r in results if r["our_status"].startswith("correct"))
    our_numeric = [r for r in results if r["our_numeric_ok"] is not None]
    our_numeric_correct = sum(
        1 for r in our_numeric if r["our_numeric_ok"] and r["our_unit_ok"] is not False
    )
    r1_correct = sum(1 for r in results if r["round1_ok"])
    our_errors = sum(1 for r in results if r["our_error"])

    print()
    print("=" * 62)
    print(f"T2 questions extracted from eval-round1 : {total}")
    print(f"Our pipeline correct                    : {our_correct}/{total}")
    if our_numeric:
        print(
            f"Our pipeline numeric accuracy           : {our_numeric_correct}/{len(our_numeric)} "
            f"({our_numeric_correct / len(our_numeric):.1%})"
        )
    print(f"Our pipeline errors                     : {our_errors}")
    print(f"Round-1 (Prompt2Win) correct            : {r1_correct}/{total}")

    both = sum(1 for r in results if r["our_status"].startswith("correct") and r["round1_ok"])
    only_us = sum(1 for r in results if r["our_status"].startswith("correct") and not r["round1_ok"])
    only_r1 = sum(1 for r in results if not r["our_status"].startswith("correct") and r["round1_ok"])
    print()
    print(f"Both correct       : {both}")
    print(f"Only our pipeline  : {only_us}")
    print(f"Only round-1       : {only_r1}")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N T2 questions (0 = all).")
    args = parser.parse_args()

    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    t2_logs = extract_t2_logs(payload)
    if args.limit and args.limit > 0:
        t2_logs = t2_logs[: args.limit]

    print(f"Source : {args.eval}")
    print(f"T2 logs: {len(t2_logs)}")

    settings = build_deterministic_settings()
    print(f"LLM configured: {bool(settings.llm_base_url)} -> deterministic-first mode")

    results: list[dict[str, Any]] = []
    for log in t2_logs:
        result = run_one(log, settings)
        results.append(result)
        marker = "OK" if result["our_status"].startswith("correct") else "!!"
        print(
            f"[{marker}] {result['query_id']}: gold={result['gold_answer']} {result['gold_unit'] or ''} "
            f"| ours={result['our_answer']} {result['our_unit'] or ''} ({result['our_status']})"
        )

    eval_rows = [
        evaluate_prediction(
            {
                "query_id": r["query_id"],
                "answer": r["our_answer"],
                "unit": r["our_unit"],
                "gold_answer": r["gold_answer"],
                "gold_unit": r["gold_unit"],
                "error": r["our_error"],
                "question_type": "open_ended" if r["answer_type"] == "textual_conceptual" else "numerical",
            },
            relative_tolerance=RELATIVE_TOLERANCE,
            absolute_tolerance=ABSOLUTE_TOLERANCE,
            case_sensitive_text=False,
        )
        for r in results
    ]
    summary = summarize(eval_rows, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)

    report = {
        "source": str(args.eval),
        "mode": "deterministic_first_llm_disabled",
        "count": len(results),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print_table(results)
    print_summary(results)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()


