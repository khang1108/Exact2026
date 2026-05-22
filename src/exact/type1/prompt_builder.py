from __future__ import annotations


def build_type1_prompt(premises: list[str], question: str) -> str:
    numbered = "\n".join(f"P{i + 1}: {premise}" for i, premise in enumerate(premises))

    return f"""
You are solving a closed-world educational logic question.

Use ONLY the given premises. Do not use outside knowledge.

Premises:
{numbered}

Question:
{question}

Rules:
- If it is multiple choice, evaluate each option and choose A/B/C/D.
- If it is Yes/No/Uncertain:
  - Answer Yes if the statement follows from the premises.
  - Answer No if the negation follows from the premises.
  - Answer Uncertain if neither follows.
- Cite premise IDs such as P1, P2 in the explanation.
- Keep explanation concise and verifiable.
- Return JSON only.

JSON schema:
{{
  "answer": "...",
  "explanation": "...",
  "fol": null,
  "cot": ["step 1", "step 2"],
  "premises": ["P1", "P2"],
  "confidence": 0.0
}}
""".strip()
