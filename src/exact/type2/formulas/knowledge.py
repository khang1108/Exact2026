from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from exact.type2.formulas.bank import FORMULAS, formula_summary
from exact.type2.schemas import Extraction


JSON_BANK_PATH = Path(__file__).with_name("circuits_and_electrostatics_bank.json")


@dataclass(frozen=True)
class RetrievedFormulaContext:
    formula_ids: tuple[str, ...]
    context: str
    summaries: list[dict[str, Any]]


def canonicalize_formula_ids(
    formula_ids_used: list[str],
    summaries: list[dict[str, Any]],
) -> list[str]:
    allowed = {str(summary.get("id")): summary for summary in summaries if summary.get("id")}
    canonicalized: list[str] = []
    for formula_id in formula_ids_used:
        candidate = str(formula_id).strip()
        if not candidate:
            continue
        if candidate in allowed:
            canonicalized.append(candidate)
            continue
        mapped = _match_formula_id(candidate, summaries)
        if mapped is not None and mapped not in canonicalized:
            canonicalized.append(mapped)
    return canonicalized


def retrieve_formula_context(
    question: str,
    extraction: Extraction | None = None,
    limit: int = 8,
) -> RetrievedFormulaContext:
    executable = [_executable_summary(formula) for formula in FORMULAS]
    knowledge = _load_json_formula_summaries()
    all_summaries = executable + knowledge

    query = " ".join(
        part
        for part in [
            question,
            extraction.target if extraction else None,
            " ".join(extraction.quantities) if extraction else None,
        ]
        if part
    )
    ranked = sorted(
        all_summaries,
        key=lambda item: _score_summary(
            item,
            query,
            target=extraction.target if extraction else None,
            known=set(extraction.quantities) if extraction else set(),
        ),
        reverse=True,
    )
    selected = ranked[:limit]
    return RetrievedFormulaContext(
        formula_ids=tuple(str(item["id"]) for item in selected),
        context=_format_context(selected),
        summaries=selected,
    )


def _executable_summary(formula) -> dict[str, Any]:
    summary = formula_summary(formula)
    summary["source"] = "executable"
    summary["title"] = formula.id
    summary["latex"] = formula.expression
    return summary


@lru_cache
def _load_json_formula_summaries() -> list[dict[str, Any]]:
    if not JSON_BANK_PATH.exists():
        return []
    rows = json.loads(JSON_BANK_PATH.read_text(encoding="utf-8"))
    summaries: list[dict[str, Any]] = []
    for row in rows:
        fields = tuple(row.get("Fields") or ())
        variables = {
            str(item.get("Symbol")): str(item.get("Meaning"))
            for item in row.get("Variables", [])
            if item.get("Symbol")
        }
        summaries.append(
            {
                "id": str(row.get("Reference_ID")),
                "source": "knowledge_json",
                "title": str(row.get("Title") or row.get("Reference_ID")),
                "domain": fields[0] if fields else "physics",
                "subfield": fields[1] if len(fields) > 1 else fields[0] if fields else "physics",
                "target": None,
                "required": tuple(variables),
                "output_unit": None,
                "expression": str(row.get("LaTeX_Formula") or ""),
                "latex": str(row.get("LaTeX_Formula") or ""),
                "variables": variables,
                "conditions": (str(row.get("Explanation") or ""),),
                "common_mistakes": (),
            }
        )
    return summaries


def _format_context(summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in summaries:
        variables = item.get("variables") or {}
        conditions = item.get("conditions") or ()
        lines.append(
            "- {id} [{source}; {domain}/{subfield}]: {expression}; "
            "variables={variables}; conditions={conditions}".format(
                id=item.get("id"),
                source=item.get("source"),
                domain=item.get("domain"),
                subfield=item.get("subfield"),
                expression=item.get("expression") or item.get("latex"),
                variables=variables,
                conditions=conditions,
            )
        )
    return "\n".join(lines)


def _score_summary(
    item: dict[str, Any],
    query: str,
    *,
    target: str | None = None,
    known: set[str] | None = None,
) -> tuple[int, int, int, int]:
    query_tokens = set(_tokens(query))
    text = " ".join(
        str(value)
        for key, value in item.items()
        if key not in {"source"}
    )
    item_tokens = set(_tokens(text))
    overlap = len(query_tokens & item_tokens)
    known = known or set()
    required = set(str(value) for value in item.get("required") or ())
    target_bonus = 6 if target and item.get("target") == target else 0
    required_bonus = len(required & known)
    exact_bonus = 4 if target_bonus and required and required <= known else 0
    executable_bonus = 2 if item.get("source") == "executable" else 0
    return (target_bonus + exact_bonus, required_bonus, overlap, executable_bonus)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


def _match_formula_id(candidate: str, summaries: list[dict[str, Any]]) -> str | None:
    candidate_tokens = set(_tokens(candidate))
    candidate_lower = candidate.lower()
    if not candidate_tokens:
        return None

    best_id: str | None = None
    best_score = 0

    for summary in summaries:
        summary_id = str(summary.get("id") or "").strip()
        if not summary_id:
            continue
        if summary_id.lower() in candidate.lower() or candidate.lower() in summary_id.lower():
            return summary_id

        summary_text = _summary_text(summary)
        summary_lower = summary_text.lower()

        if "coulomb" in candidate_lower and "coulomb" in summary_lower:
            return summary_id
        if any(token in candidate_lower for token in ("pythagorean", "vector addition", "resultant")) and (
            "resultant" in summary_lower or "vector" in summary_lower or "sqrt" in summary_lower
        ):
            return summary_id
        if "equilateral" in candidate_lower and (
            "equilateral" in summary_lower or "sqrt(3)" in summary_lower or "60" in summary_lower
        ):
            return summary_id
        if "midpoint" in candidate_lower and (
            "midpoint" in summary_lower or "opposite" in summary_lower or "equal charges" in summary_lower
        ):
            return summary_id

        summary_tokens = set(_tokens(summary_text))
        score = len(candidate_tokens & summary_tokens)
        if score > best_score:
            best_score = score
            best_id = summary_id

    if best_score >= 1:
        return best_id
    return None


def _summary_text(summary: dict[str, Any]) -> str:
    return " ".join(
        str(value)
        for key, value in summary.items()
        if key not in {"source"}
    )
