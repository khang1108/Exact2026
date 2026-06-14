"""Evaluate a prefix of the EXACT Type 1 dataset through the public API."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

_OPTION_LINE = re.compile(r"^\s*([A-E])[.)]\s+(.+)$", re.MULTILINE)
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "src/exact/datasets/exact/Logic_Based_Educational_Queries.json"


def _samples(limit: int | None) -> list[dict[str, Any]]:
    groups = json.loads(DATASET.read_text())
    samples: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        for question_index, (question, gold) in enumerate(
            zip(group["questions"], group["answers"])
        ):
            samples.append(
                {
                    "query_id": f"T1_{group_index:04d}_{question_index:02d}",
                    "type": "type1",
                    "query": question,
                    "premises": group["premises-NL"],
                    "_gold": gold,
                    "_is_mcq": bool(_OPTION_LINE.search(question)),
                }
            )
    return samples[:limit] if limit is not None else samples


async def _evaluate(
    endpoint: str,
    samples: list[dict[str, Any]],
    concurrency: int,
    timeout: float,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def call(sample: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    endpoint,
                    json={key: value for key, value in sample.items() if not key.startswith("_")},
                )
                response.raise_for_status()
                return {
                    **sample,
                    "_response": response.json(),
                    "_latency": time.perf_counter() - started,
                }

        return list(await asyncio.gather(*(call(sample) for sample in samples)))


def _report(results: list[dict[str, Any]]) -> None:
    correct = [
        result
        for result in results
        if str(result["_response"]["answer"]).upper() == str(result["_gold"]).upper()
    ]
    mcq = [result for result in results if result["_is_mcq"]]
    polar = [result for result in results if not result["_is_mcq"]]

    def score(rows: list[dict[str, Any]]) -> str:
        count = sum(
            str(row["_response"]["answer"]).upper() == str(row["_gold"]).upper()
            for row in rows
        )
        return f"{count / len(rows):.1%} ({count}/{len(rows)})" if rows else "n/a"

    print(f"Overall: {score(results)}")
    print(f"MCQ:     {score(mcq)}")
    print(f"Polar:   {score(polar)}")
    print("Answers:", dict(Counter(row["_response"]["answer"] for row in results)))
    print(
        "Causes:",
        dict(
            Counter(
                (row["_response"].get("routing_diagnostics") or {}).get("uncertainty_cause")
                for row in results
            )
        ),
    )
    latencies = [row["_latency"] for row in results]
    print(
        f"Latency: mean={statistics.mean(latencies):.2f}s "
        f"p50={statistics.median(latencies):.2f}s max={max(latencies):.2f}s"
    )
    for row in results:
        response = row["_response"]
        marker = "PASS" if row in correct else "FAIL"
        cause = (response.get("routing_diagnostics") or {}).get("uncertainty_cause")
        print(
            f"{marker} {row['query_id']} pred={response['answer']!r} "
            f"gold={row['_gold']!r} cause={cause}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="https://api.iamphuckhang.dev/predict")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = asyncio.run(
        _evaluate(args.endpoint, _samples(args.limit), args.concurrency, args.timeout)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
    _report(results)


if __name__ == "__main__":
    main()
