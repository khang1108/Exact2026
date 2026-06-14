"""Extract multiple-choice options embedded inside a question string.

Many EXACT questions carry their options inline ("Stem\\nA. ...\\nB) ...") with
no separate options payload. Both ``A.`` and ``A)`` label styles occur, and the
sibling options file misses the ``A)`` cases — so always parse from the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_OPTION_LINE = re.compile(r"^\s*([A-E])([.)])\s*(.*)$")


@dataclass(frozen=True)
class MCQExtraction:
    """Options extracted from one question plus extraction diagnostics."""

    stem: str
    options: dict[str, str]
    marker_style: str  # "dot" | "paren" | "mixed" | "none"
    option_count: int
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


def extract_mcq(question_text: str) -> MCQExtraction:
    """Extract options + diagnostics from a question (handles ``A.`` and ``A)``)."""

    stem_lines: list[str] = []
    options: dict[str, str] = {}
    markers: set[str] = set()
    diagnostics: list[str] = []

    for line in question_text.split("\n"):
        match = _OPTION_LINE.match(line)
        if not match:
            if not options:
                stem_lines.append(line)
            continue
        label, marker, text = match.group(1).upper(), match.group(2), match.group(3).strip()
        markers.add(marker)
        if label in options:
            diagnostics.append(f"MCQ_DUPLICATE_LABEL: {label}")
            continue
        if not text:
            diagnostics.append(f"MCQ_OPTION_TEXT_EMPTY: {label}")
        options[label] = text

    if "." in markers and ")" in markers:
        marker_style = "mixed"
        diagnostics.append("MCQ_MIXED_MARKER_FORMAT")
    elif ")" in markers:
        marker_style = "paren"
        diagnostics.append("MCQ_MARKER_A_PAREN")
    elif "." in markers:
        marker_style = "dot"
        diagnostics.append("MCQ_MARKER_A_DOT")
    else:
        marker_style = "none"

    count = len(options)
    if count == 3:
        diagnostics.append("MCQ_THREE_OPTIONS")
    elif count == 4:
        diagnostics.append("MCQ_FOUR_OPTIONS")
    elif count > 0:
        diagnostics.append(f"MCQ_UNSUPPORTED_OPTION_COUNT: {count}")

    return MCQExtraction(
        stem="\n".join(stem_lines).strip() or question_text.strip(),
        options=options,
        marker_style=marker_style,
        option_count=count,
        diagnostics=tuple(diagnostics),
    )


def parse_mcq_options(question_text: str) -> tuple[str, dict[str, str]]:
    """Back-compat thin wrapper: return just (stem, {label: option_text})."""

    extraction = extract_mcq(question_text)
    return extraction.stem, extraction.options
