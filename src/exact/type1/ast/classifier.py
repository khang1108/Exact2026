from __future__ import annotations

import re

# ─────────────────────────────────────────────
# 4. NOUN STOP WORDS  (for _extract_head_noun)
# ─────────────────────────────────────────────

_NOUN_STOP_WORDS = {
    'is','are','was','were','has','have','had','does','do','did',
    'will','would','can','could','should','must','may','might',
    'if','then','and','or','but','not','that','which','who',
    'follows','requires','studies','passes','loves','gives',
    'takes','makes','needs','wants','likes','hates','knows',
}

PRONOUNS = re.compile(
    r'\b(it|its|they|them|their|he|his|him|she|her)\b',
    re.IGNORECASE
)

# Priority 0: quantified+negation beats logical
# Priority 1: logical (if/then beats bare "a")
# Priority 2: quantified (explicit quantifier words)
# Priority 3: remaining logical connectives
DETERMINISTIC_RULES: list[tuple[str, list[str]]] = [
    ("quantified", [                              # P0: "a X does not..."
        r"^(a|an)\s+\w+.*(not|never|does\s+not|is\s+not)",
        r"\b(all|every|each)\b.*\b(not|never|does\s+not)\b",
        r"\bno\s+one\b",
        r"\bnobody\b",
    ]),
    ("logical", [                                 # P1: if/then wins over "a"
        r"\bif\b.+,",
        r"\bif\b.+\bthen\b",
        r"\bwhenever\b",
        r"\bonly\s+if\b",
        r"\bimplies\b",
        r"\btherefore\b",
        r"\bhence\b",
        r"\bif\s+and\s+only\s+if\b",
    ]),
    ("quantified", [                              # P2: explicit quantifiers
        r"^(a|an)\s+\w+",
        r"\b(all|every|each)\b",
        r"\bsome\b",
        r"\bthere\s+exists\b",
        r"\bthere\s+is\s+a\b",
        r"\bfor\s+all\b",
        r"\bfor\s+every\b",
    ]),
    ("logical", [                                 # P3: remaining connectives
        r"\band\b",
        r"\bor\b",
        r"\bbut\b",
        r"\bdoes\s+not\b",
        r"\bis\s+not\b",
        r"\bcannot\b",
        r"\bnever\b",
        r"\bnot\b",
    ]),
]

def fast_classify(sentence: str) -> str:
    """
    Deterministic sentence classifier.
    Returns "quantified" | "logical" | "atomic".
    """
    s = sentence.lower().strip()

    # Compound adjective guard:
    # "has clean and readable code" → "and" connects adjectives, not clauses
    if re.search(r'\b(has|have|is|are)\b', s):
        if not re.search(r'\b(if|then|implies|whenever)\b', s):
            parts = re.split(r'\band\b', s, maxsplit=1)
            if len(parts) == 2:
                right = parts[1].strip()
                # Right side has no subject → adjective compound → atomic
                if not re.search(r'^(x|y|z|the|a|an)\s+\w+\s+\w+', right):
                    if not re.search(r'\b(is|are|has|have|does|will)\b', right):
                        return "atomic"

    for label, patterns in DETERMINISTIC_RULES:
        for pattern in patterns:
            if re.search(pattern, s):
                return label
    return "atomic"