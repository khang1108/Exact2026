"""LLM predicate reconciliation — collapse independently-named synonyms.

Premises are parsed one at a time, so the same relation can surface under
different predicate names (``PriorityDeliveryStatus`` vs
``HasPriorityDeliveryStatus``). Z3 treats those as distinct symbols and the
proof chain silently breaks. Rather than hard-coding verb/stopword lists (which
only fix seen phrasings), this asks the parser LLM which names denote the same
relation, using each predicate's example sentence as context — so it generalizes
to unseen wording.

The LLM proposal is then guarded deterministically (same arity, same predicate
family) before any rename is applied, bounding over-merge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from exact.type1.parser.schemas import ParserResult, _predicate_family
from exact.type1.prompts import get_system_prompt_predicate_reconcile

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient


class PredicateMergeGroup(ParserResult):
    """One cluster of predicate names that denote the same relation."""

    canonical: str
    aliases: list[str]


class PredicateMergeResult(ParserResult):
    """All synonym clusters found across the premise predicate vocabulary."""

    groups: list[PredicateMergeGroup]


class PredicateReconciler:
    """Propose synonym merges via one LLM call, then guard them deterministically."""

    def __init__(self, client: ParserClient, max_tokens: int = 512) -> None:
        self.client = client
        self.max_tokens = max_tokens

    async def reconcile(self, signatures: list[dict[str, object]]) -> dict[str, str]:
        """Return a guarded ``{alias_name: canonical_name}`` rename map.

        Only names that share arity and predicate family with their canonical are
        kept. Returns an empty map when there is nothing safe to merge.
        """
        arity_by_name: dict[str, int] = {}
        for sig in signatures:
            arity_by_name[str(sig["name"])] = int(sig["arity"])  # type: ignore[call-overload]

        # Nothing to do unless at least two predicates share an arity.
        arities = list(arity_by_name.values())
        if len(arity_by_name) < 2 or len(set(arities)) == len(arities):
            # still allow: two same-arity names is the merge case
            if not any(arities.count(a) > 1 for a in set(arities)):
                return {}

        result = await self.client.parse_as(
            [
                {"role": "system", "content": get_system_prompt_predicate_reconcile()},
                {"role": "user", "content": _render_signatures(signatures)},
            ],
            PredicateMergeResult,
            max_tokens=self.max_tokens,
        )

        rename: dict[str, str] = {}
        for group in result.groups:
            canonical = group.canonical
            if canonical not in arity_by_name:
                continue
            canon_arity = arity_by_name[canonical]
            canon_family = _predicate_family(canonical)
            for alias in group.aliases:
                if alias == canonical or alias not in arity_by_name:
                    continue
                if arity_by_name[alias] != canon_arity:
                    continue  # never merge across arity
                alias_family = _predicate_family(alias)
                if (
                    canon_family is not None
                    and alias_family is not None
                    and canon_family != alias_family
                ):
                    continue  # never merge across predicate family
                rename[alias] = canonical
        return rename


def _render_signatures(signatures: list[dict[str, object]]) -> str:
    lines = [
        f"- {sig['name']}/{sig['arity']}  (example: {sig['example']})"
        for sig in signatures
    ]
    return "PREDICATES:\n" + "\n".join(lines)
