"""Extract multiple-choice options embedded inside a question string.

Many EXACT questions carry their options inline ("Stem\\nA. ...\\nB) ...") with
no separate options payload. Both ``A.`` and ``A)`` label styles occur, so the
options-only data file misses the ``A)`` cases — always parse from the text.
"""

from __future__ import annotations

import re

_OPTION_LINE = re.compile(r"^\s*([A-E])[.)]\s*(.*)$")


def parse_mcq_options(question_text: str) -> tuple[str, dict[str, str]]:
    """Split a question into (stem, {label: option_text}).

    Supports both ``A.`` and ``A)`` label styles. Returns an empty option dict
    when the text contains no option lines (i.e. a polar/open question).
    """

    stem_lines: list[str] = []
    options: dict[str, str] = {}
    for line in question_text.split("\n"):
        match = _OPTION_LINE.match(line)
        if match:
            options[match.group(1).upper()] = match.group(2).strip()
        elif not options:
            stem_lines.append(line)
    return "\n".join(stem_lines).strip(), options
