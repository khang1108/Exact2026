"""Option interpreter — turns MCQ options into OptionClaim records.

``OParser`` reads each option relative to the question stem and decides whether
it is a complete proposition, a subject-less fragment (recovered from the stem),
raw FOL, a premise reference, or a YNU answer. Deterministic guards catch raw
FOL and premise references before spending an LLM call. No FOL is produced here;
``ClaimParser`` fills that in later for solvable option types.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from exact.type1.parser.schemas import OptionClaim, OptionClaimResult
from exact.type1.prompts import get_system_prompt_option_claim

if TYPE_CHECKING:
    from exact.type1.parser.client import ParserClient

ChatMessage = dict[str, Any]

# Unambiguous logic symbols. Bare predicate applications like "P(x)" are left to
# the LLM, since natural-language options ("requires GPU(s)") can look similar.
_RAW_FOL_RE = re.compile(r"[∀∃→∧∨¬↔]|->|<->")
_PREMISE_REF_RE = re.compile(
    r"^premises?\s+[\d]+(?:\s*(?:,|and)\s*\d+)*\.?$",
    re.IGNORECASE,
)


class OParser:
    """Interpret each MCQ option as a (possibly unsupported) OptionClaim."""

    def __init__(self, client: ParserClient) -> None:
        self.client = client

    async def parse_options(
        self,
        question_stem: str,
        options: dict[str, str],
    ) -> list[OptionClaim]:
        """Interpret every option; deterministic guards skip the LLM where possible."""

        labels = list(options.keys())
        texts = [options[label] for label in labels]

        # Decide which options need the LLM and which are handled deterministically.
        llm_indices: list[int] = []
        llm_messages: list[list[ChatMessage]] = []
        for i, text in enumerate(texts):
            if self._deterministic_claim(labels[i], text) is None:
                llm_indices.append(i)
                llm_messages.append(self._messages(question_stem, text))

        llm_results: list[OptionClaimResult] = (
            await self.client.parse_many_as(llm_messages, OptionClaimResult)
            if llm_messages
            else []
        )
        llm_by_index = dict(zip(llm_indices, llm_results))

        claims: list[OptionClaim] = []
        for i, label in enumerate(labels):
            deterministic = self._deterministic_claim(label, texts[i])
            if deterministic is not None:
                claims.append(deterministic)
            else:
                claims.append(self._compile(label, llm_by_index[i]))
        return claims

    def _deterministic_claim(self, label: str, text: str) -> OptionClaim | None:
        """Catch premise references and raw FOL without an LLM call."""

        stripped = text.strip()
        if _PREMISE_REF_RE.match(stripped):
            indices = tuple(int(n) for n in re.findall(r"\d+", stripped))
            return OptionClaim(
                label=label,
                option_type="premise_reference",
                claim_text=None,
                ynu_value="none",
                premise_indices=indices,
                raw_fol=None,
            )
        if _RAW_FOL_RE.search(stripped):
            return OptionClaim(
                label=label,
                option_type="raw_fol",
                claim_text=None,
                ynu_value="none",
                premise_indices=(),
                raw_fol=stripped,
            )
        return None

    def _compile(self, label: str, result: OptionClaimResult) -> OptionClaim:
        return OptionClaim(
            label=label,
            option_type=result.option_type,
            claim_text=result.claim_text,
            ynu_value=result.ynu_value,
            premise_indices=tuple(result.premise_indices),
            raw_fol=result.raw_fol,
        )

    def _messages(self, question_stem: str, option_text: str) -> list[ChatMessage]:
        return [
            {"role": "system", "content": get_system_prompt_option_claim().strip()},
            {
                "role": "user",
                "content": f'Stem: "{question_stem}"\nOption: "{option_text}"',
            },
        ]
