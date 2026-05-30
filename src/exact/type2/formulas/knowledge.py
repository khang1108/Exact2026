from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from exact.config import Settings
from exact.config import PACKAGE_DIR
from exact.type2.extraction.llm_structured import select_formula_ids
from exact.type2.formulas.bank import FORMULAS, formula_summary
from exact.type2.schemas import Extraction


JSON_BANK_PATH = PACKAGE_DIR / "datasets" / "exact" / "circuits_and_electrostatics_bank.json"
REGISTRY_BANK_PATH = PACKAGE_DIR / "datasets" / "exact" / "physics_formulas_registry.json"


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
    limit: int = 24,
    settings: Settings | None = None,
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
    selected = _rerank_with_llm(question, extraction, selected, settings=settings)
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
    summaries: list[dict[str, Any]] = []

    if JSON_BANK_PATH.exists():
        rows = json.loads(JSON_BANK_PATH.read_text(encoding="utf-8"))
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

    if REGISTRY_BANK_PATH.exists():
        rows = json.loads(REGISTRY_BANK_PATH.read_text(encoding="utf-8"))
        for row in rows:
            formula_text = row.get("formula", {}).get("en", "")
            target_val = None
            if "=" in formula_text:
                lhs = formula_text.split("=")[0].strip()
                target_val = lhs.replace("[", "").replace("]", "")

            variables = {
                str(item.get("en_symbol")): str(item.get("en_name"))
                for item in row.get("symbol_map", [])
                if item.get("en_symbol")
            }
            required_vars = tuple(v for v in variables if v != target_val)
            summaries.append(
                {
                    "id": str(row.get("key")),
                    "source": "registry_json",
                    "title": f"Formula registry {row.get('key')[:8]}",
                    "domain": "physics",
                    "subfield": "physics",
                    "target": target_val,
                    "required": required_vars,
                    "output_unit": None,
                    "expression": formula_text,
                    "latex": formula_text,
                    "variables": variables,
                    "conditions": (f"Formula registry entry {row.get('key')}",),
                    "common_mistakes": (),
                }
            )

    return summaries


def _format_context(summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in summaries:
        variables = _compact_mapping(item.get("variables") or {})
        conditions = _compact_sequence(item.get("conditions") or ())
        lines.append(
            "- {id} [{source}; {domain}/{subfield}; target={target}; required={required}; output={output}]: "
            "{expression}; vars={variables}; conditions={conditions}".format(
                id=item.get("id"),
                source=item.get("source"),
                domain=item.get("domain"),
                subfield=item.get("subfield"),
                target=item.get("target"),
                required=tuple(item.get("required") or ()),
                output=item.get("output_unit"),
                expression=item.get("expression") or item.get("latex"),
                variables=variables,
                conditions=conditions,
            )
        )
    return "\n".join(lines)


def _compact_mapping(mapping: dict[str, Any]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in mapping.items()) or "-"


def _compact_sequence(values) -> str:
    return "; ".join(str(value) for value in values if str(value).strip()) or "-"


def _rerank_with_llm(
    question: str,
    extraction: Extraction | None,
    ranked: list[dict[str, Any]],
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    if settings is None or settings.mock_llm or len(ranked) < 3:
        return ranked

    top_score = _score_summary(
        ranked[0],
        question,
        target=extraction.target if extraction else None,
        known=set(extraction.quantities) if extraction else set(),
    )
    second_score = _score_summary(
        ranked[1],
        question,
        target=extraction.target if extraction else None,
        known=set(extraction.quantities) if extraction else set(),
    )
    if extraction is not None and extraction.target is not None and top_score[0] >= 6 and top_score > second_score:
        return ranked
    if top_score[0] >= 10 and top_score > second_score:
        return ranked

    candidate_summaries = ranked[:12]
    selection = select_formula_ids(
        question,
        _build_extraction_summary(extraction),
        candidate_summaries,
        settings=settings,
    )
    if selection is None or not selection.formula_ids:
        return ranked

    selected_ids = [str(formula_id) for formula_id in selection.formula_ids]
    selected_lookup = {formula_id: index for index, formula_id in enumerate(selected_ids)}
    promoted = [item for item in ranked if item.get("id") in selected_lookup]
    promoted.sort(key=lambda item: selected_lookup[str(item.get("id"))])
    remainder = [item for item in ranked if item not in promoted]
    return promoted + remainder


def _build_extraction_summary(extraction: Extraction | None) -> str:
    if extraction is None:
        return "kind=unknown; target=None; quantities=[]; notes=[]"
    quantities = [
        f"{name}={quantity.value} ({quantity.evidence})"
        for name, quantity in extraction.quantities.items()
    ]
    notes = [note for note in extraction.notes if note.strip()]
    return (
        f"kind={extraction.kind.value}; target={extraction.target}; "
        f"quantities={quantities or []}; notes={notes or []}"
    )


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

    # Geometry match bonus
    geom_keywords = {"equilateral", "midpoint", "bisector", "isosceles", "perpendicular", "triangle"}
    query_geom = query_tokens & geom_keywords
    geom_bonus = 0
    if query_geom:
        item_geom = item_tokens & geom_keywords
        geom_bonus = 15 * len(query_geom & item_geom)

    return (target_bonus + exact_bonus + geom_bonus, required_bonus, overlap, executable_bonus)


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
