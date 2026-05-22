"""

"""

from __future__ import annotations

import re

def normalize_type1_answer(answer: str) -> str:
    raw = answer.strip()

    mcq = re.fullmatch(r"[ABCDabcd]", raw)
    if mcq:
        return raw.upper()

    lower = raw.lower()
    if lower in {"yes", "true"}:
        return "Yes"
    if lower in {"no", "false"}:
        return "No"
    if lower in {"uncertain", "unknown", "cannot determine", "not enough information"}:
        return "Uncertain"

    if "uncertain" in lower or "cannot determine" in lower or "not enough" in lower:
        return "Uncertain"
    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"

    match = re.search(r"\banswer\s+is\s+([ABCD])\b", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return raw
