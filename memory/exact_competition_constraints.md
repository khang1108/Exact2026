---
name: exact-competition-constraints
description: Confirmed constraints and answer label spec for EXACT 2026 Type 1 from organizers (BTC)
metadata:
  type: project
---

# EXACT 2026 Type 1 — Confirmed Competition Constraints

## Answer label format (BTC confirmed)

| Question type | Valid answers |
|---|---|
| MCQ | A, B, C, D |
| Yes/No/Uncertain | **Yes**, **No**, **Uncertain** (NOT "Unknown") |
| Open-ended | Short NL statement (scoring method TBD) |

**Critical:** Internal solver uses "Unknown" for undecidable case — must convert to "Uncertain"
before building `PredictionResponse.answer`. This mapping lives in `_to_competition_label()`
in `src/exact/logic/pipeline.py`.

## Question types (BTC confirmed)

Three types will appear in test set:
1. **Multiple Choice (MCQ)** — options A/B/C/D embedded in question text
2. **Yes/No/Uncertain** — question starts with does/is/can/will/would/should/based on
3. **Open-ended** — everything else; answer is a short NL statement

## Input constraints (BTC confirmed)

- Test set provides only **premises-NL** (natural language). No premises-FOL available.
- All pipeline translation must be from NL only.

## vLLM guided JSON

- vLLM server is a **custom build** that supports `guided_json` body parameter.
- `type1_use_guided_json = True` (default in config.py).
- Requires lm-format-enforcer in vLLM build.

## Timing constraint

- Each request must complete within **60 seconds**.
- Soft deadline: 45s (`type1_soft_deadline_s`) — skip repair/fallback if remaining time < threshold.

## Why: open-ended needs special handling

Current pipeline falls through to YNU path for open-ended → tries to prove a claim that
doesn't exist → always returns Uncertain/Unknown. Needs dedicated LLM CoT path.
Need to confirm from BTC how open-ended is scored (exact match? semantic similarity?).
