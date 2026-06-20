from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from exact.config import Settings
from exact.config import PACKAGE_DIR
from exact.llm_client import has_json_llm_client_config
from exact.type2.extraction.llm_structured import select_formula_ids
from exact.type2.formulas.bank import FORMULAS, formula_summary
from exact.type2.schemas import Extraction


JSON_BANK_PATH = (
    PACKAGE_DIR / "datasets" / "exact" / "type2_circuits_electrostatics_formula_bank.json"
)
REGISTRY_BANK_PATH = PACKAGE_DIR / "datasets" / "exact" / "type2_physics_formula_registry.json"


@dataclass(frozen=True)
class RetrievedFormulaContext:
    formula_ids: tuple[str, ...]
    context: str
    summaries: list[dict[str, Any]]
    solution_plan: tuple[str, ...] = ()
    missing_variables: tuple[str, ...] = ()
    selector_confidence: float | None = None
    selector_notes: tuple[str, ...] = ()


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
    include_knowledge_bank: bool = False,
) -> RetrievedFormulaContext:
    executable = [_executable_summary(formula) for formula in FORMULAS]
    knowledge = (
        _load_json_formula_summaries()
        if include_knowledge_bank and (settings is None or settings.type2_use_formula_bank)
        else []
    )
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
    ranked, selection = _rerank_with_llm(question, extraction, ranked, settings=settings)
    selected = ranked[:limit]
    return RetrievedFormulaContext(
        formula_ids=tuple(str(item["id"]) for item in selected),
        context=_format_context(selected, selection=selection),
        summaries=selected,
        solution_plan=tuple(selection.solution_plan) if selection is not None else (),
        missing_variables=tuple(selection.missing_variables) if selection is not None else (),
        selector_confidence=selection.confidence if selection is not None else None,
        selector_notes=tuple(selection.notes) if selection is not None else (),
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


def _format_context(summaries: list[dict[str, Any]], selection=None) -> str:
    lines: list[str] = []
    if selection is not None and selection.solution_plan:
        lines.append("Selector solution plan:")
        lines.extend(f"- {step}" for step in selection.solution_plan)
        lines.append("")
    for item in summaries:
        variables = _compact_mapping(item.get("variables") or {}, limit=220)
        conditions = _compact_sequence(item.get("conditions") or (), limit=220)
        expression = _clip_text(str(item.get("expression") or item.get("latex") or ""), 260)
        lines.append(
            "- {id} [{source}; {domain}/{subfield}; target={target}; output={output}]: "
            "{expression}; vars={variables}; conditions={conditions}".format(
                id=item.get("id"),
                source=item.get("source"),
                domain=item.get("domain"),
                subfield=item.get("subfield"),
                target=item.get("target"),
                output=item.get("output_unit"),
                expression=expression,
                variables=variables,
                conditions=conditions,
            )
        )
    return "\n".join(lines)


def _compact_mapping(mapping: dict[str, Any], *, limit: int) -> str:
    return _clip_text(", ".join(f"{key}:{value}" for key, value in mapping.items()) or "-", limit)


def _compact_sequence(values, *, limit: int) -> str:
    return _clip_text("; ".join(str(value) for value in values if str(value).strip()) or "-", limit)


def _clip_text(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _rerank_with_llm(
    question: str,
    extraction: Extraction | None,
    ranked: list[dict[str, Any]],
    settings: Settings | None = None,
) -> tuple[list[dict[str, Any]], Any | None]:
    if settings is None or not has_json_llm_client_config(settings) or len(ranked) < 3:
        return ranked, None
    if not settings.type2_use_llm_formula_selection:
        return ranked, None
    if extraction is not None and extraction.kind.value == "conceptual":
        return ranked, None

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
    if not settings.type2_force_llm_formula_selection:
        if extraction is not None and extraction.target is not None and top_score[0] >= 6 and top_score > second_score:
            return ranked, None
        if top_score[0] >= 10 and top_score > second_score:
            return ranked, None

    rerank_limit = settings.type2_rerank_limit if settings else 12
    candidate_summaries = ranked[:rerank_limit]
    try:
        selection = select_formula_ids(
            question,
            _build_extraction_summary(extraction),
            candidate_summaries,
            settings=settings,
        )
    except Exception:
        return ranked, None
    if selection is None or not selection.formula_ids:
        return ranked, None

    selected_ids = [str(formula_id) for formula_id in selection.formula_ids]
    selected_lookup = {formula_id: index for index, formula_id in enumerate(selected_ids)}
    promoted = [item for item in ranked if item.get("id") in selected_lookup]
    promoted.sort(key=lambda item: selected_lookup[str(item.get("id"))])
    remainder = [item for item in ranked if item not in promoted]
    return promoted + remainder, selection


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
