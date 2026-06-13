"""Self-refinement module for the Type 1 logic pipeline.

When Z3 cannot prove or disprove a conclusion ("Uncertain"), the refiner reads
all (NL, FOL) pairs together using a 7B reasoning LLM and returns NL rewrites
that unify the vocabulary — making the same concept use the same predicate and
the same entity name across sentences (plus fixing structural errors like
tautologies and constant-where-variable). The caller re-parses those rewrites
with the 1.7B parser and retries Z3.

Architecture: LLM = translator only; solver = decision maker.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from exact.type1.prompts import get_system_prompt_refiner

if TYPE_CHECKING:
    from exact.llm_client import VLLMJsonClient

_REFINE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "rephrased": {"type": "string"},
                },
                "required": ["id", "rephrased"],
            },
        }
    },
    "required": ["corrections"],
}


class Type1Refiner:
    """Ask a 7B LLM to rewrite NL so the whole batch shares one FOL vocabulary."""

    def __init__(self, client: VLLMJsonClient) -> None:
        self._client = client

    async def refine(self, items: list[dict[str, str]]) -> dict[str, str]:
        """
        items: [{"id": "premise-1", "nl": "...", "fol": "..."}, ...]
        Returns: {"premise-3": "rephrased NL", "A": "rephrased NL", ...}
        Empty dict if the refiner call fails or finds nothing to fix.
        """
        if not items:
            return {}

        user = _format_items(items)
        try:
            data = await self._client.complete_json(
                messages=[
                    {"role": "system", "content": get_system_prompt_refiner()},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=1024,
                json_schema=_REFINE_SCHEMA,
            )
        except Exception as exc:
            import traceback
            return {"_error": f"LLM client exception: {exc}\n{traceback.format_exc()}"}

        corrections = data.get("corrections") or []
        originals = {item["id"]: item["nl"] for item in items}
        return {
            c["id"]: c["rephrased"]
            for c in corrections
            if isinstance(c, dict)
            and c.get("id")
            and c.get("rephrased")
            and _is_safe_premise_rewrite(
                c["id"],
                originals.get(c["id"], ""),
                c["rephrased"],
            )
        }


def _format_items(items: list[dict[str, str]]) -> str:
    lines = ["NL → FOL translation pairs for the full problem:\n"]
    for item in items:
        lines.append(f"[{item['id']}]")
        lines.append(f"NL:  {item['nl']}")
        lines.append(f"FOL: {item['fol']}")
        lines.append("")
    lines.append(
        "Unify the vocabulary: rewrite the NL of any sentence whose predicate or "
        "entity names are inconsistent with the rest, and fix structural errors."
    )
    return "\n".join(lines)


_EXPLICIT_VARIABLE_RE = re.compile(r"\b(?:x|y|z)\d*\b")
_NEGATION_RE = re.compile(r"\b(?:not|no|never|cannot|without|lacks?|doesn't|isn't)\b", re.I)


def _is_safe_premise_rewrite(item_id: str, original: str, rephrased: str) -> bool:
    """Reject rewrites that can silently change the problem's logical contract."""

    if not all(isinstance(value, str) for value in (item_id, original, rephrased)):
        return False
    if not re.fullmatch(r"premise-\d+", item_id):
        return False
    original = original.strip()
    rephrased = rephrased.strip()
    if not original or not rephrased or original == rephrased:
        return False
    if _EXPLICIT_VARIABLE_RE.search(rephrased) and not _EXPLICIT_VARIABLE_RE.search(original):
        return False
    if bool(_NEGATION_RE.search(original)) != bool(_NEGATION_RE.search(rephrased)):
        return False
    ratio = len(rephrased.split()) / max(1, len(original.split()))
    if not 0.5 <= ratio <= 2.0:
        return False

    original_tokens = set(re.findall(r"[a-z0-9]+", original.lower()))
    rephrased_tokens = set(re.findall(r"[a-z0-9]+", rephrased.lower()))
    union = original_tokens | rephrased_tokens
    overlap = len(original_tokens & rephrased_tokens) / len(union) if union else 0.0
    return overlap >= 0.45
