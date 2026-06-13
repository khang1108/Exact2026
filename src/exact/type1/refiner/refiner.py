"""Self-refinement module for the Type 1 logic pipeline.

When Z3 cannot prove or disprove a conclusion ("Uncertain"), the refiner
inspects each (NL, FOL) pair using a 7B reasoning LLM and returns clearer NL
rephrasing for sentences it judges as structurally wrong.  The caller then
re-parses those rephrasings with the 1.7B parser and retries Z3.

Architecture: LLM = translator only; solver = decision maker.
"""

from __future__ import annotations

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
    """Ask a 7B LLM to rephrase NL sentences whose FOL translation is structurally wrong."""

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
        except Exception:
            return {}

        corrections = data.get("corrections") or []
        return {
            c["id"]: c["rephrased"]
            for c in corrections
            if isinstance(c, dict) and c.get("id") and c.get("rephrased")
        }


def _format_items(items: list[dict[str, str]]) -> str:
    lines = ["NL → FOL translation pairs for review:\n"]
    for item in items:
        lines.append(f"[{item['id']}]")
        lines.append(f"NL:  {item['nl']}")
        lines.append(f"FOL: {item['fol']}")
        lines.append("")
    lines.append("Identify which translations are wrong and provide clearer rephrasing.")
    return "\n".join(lines)
