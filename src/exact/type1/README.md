# Type 1: Logical QA (MCQ / Yes-No-Uncertain)

Inspired by [NL2Logic](https://arxiv.org/pdf/2602.13237). Rather than asking an LLM to produce FOL in one shot, the pipeline decomposes the problem into small, verifiable sub-steps: frame detection → deterministic AST assembly → schema canonicalization → Z3 entailment.

---

## Pipeline Overview

```
NL Premises
    │
    ▼
PremiseFrameParser        (1 vLLM call/premise)
    │  identifies: kind, variable, restrictor_text,
    │  condition_texts, conclusion_texts,
    │  numeric_constraints, temporal_constraints, modality
    ▼
PremiseFrameCompiler      (0 LLM calls — deterministic)
    │  atomics  ──► FOLParser._parse_atomic()
    │  numerics ──► ConstraintParser.parse_numerics()
    │  temporals──► ConstraintParser.parse_temporals()
    │  assembles: ∀x[R(x)].(A(x) ∧ B(x) → C(x))
    ▼
Generic-class repair       (rule: Students arg → ∀x[Student(x)])
    ▼
PremiseSchema              (dedup predicates, arity check, canonicalize)
    │  renames similar predicates to one canonical name
    │  raises diagnostics for drift / lost constraints
    ▼
FOLNode trees  ◄── verified: bool + issues: list[str]   ──►  PremiseSchema
    │                                                            │
    ├── /parser endpoint  (NL → FOL, stop here)                  │
    │                                                            ▼
    │                                          QuestionSideParser (question side)
    │                                            │  QParser     → QuestionFrameResult
    │                                            │  OParser     → OptionParseBundle
    │                                            │  ClaimParser → FOL (schema-canonicalized)
    │                                            │  QueryVerifier → supported / issues
    │                                            ▼
    │                                          QuerySpec  ──► /qparser endpoint (inspect, stop here)
    ▼                                            │
Z3 FOLSolver  ◄──────────────────────────────────┘
    │  polar      → check_ynu()             → "Yes" / "No" / "Uncertain"
    │  mcq        → check_mcq()             → option label "A".."D"
    │              (none-of-above post-processing: 0 ordinary entailed + NoA present → NoA label)
    │  refutation → check_mcq_refutation()  → the single FALSE option
    │  ynu_mapped → check_ynu() + map YNU verdict → option label
    ▼
_normalize_answer()   ("Uncertain" → configured token, default "Unknown")
    ▼
PredictionResponse
```

---

## Folder Structure

```
type1/
  pipeline.py               # run_type1_pipeline() + fol_node_to_dict()
  prompts.py                # all vLLM system prompts

  ast/
    nodes.py                # FOL AST dataclasses
    classifier.py           # sentence-type classifier (atomic/logical/quantified)

  parser/
    client.py               # ParserClient — HTTP pool + semaphore to vLLM
    router.py               # ParserKind routing + request builders
    parser.py               # FOLParser — recursive NL → FOLNode
    frame_parser.py         # PremiseFrameParser + PremiseFrameCompiler + ConstraintParser
    premise_parser.py       # PremiseParser — premise orchestration + repair + verify
    options.py              # extract_mcq() — extract embedded A./A) options with diagnostics
    qparser.py              # QParser — classify question → QuestionFrameResult
    oparser.py              # OParser — role-aware deterministic OptionParser → OptionParseBundle
    claim_parser.py         # ClaimParser — claim texts → canonicalized FOL
    query_verifier.py       # verify() — pre-solve QuerySpec validation
    question_parser.py      # QuestionSideParser — question orchestration
    schemas.py              # frame/query result models, PremiseSchema, QuerySpec, OptionRole

  models/
    schemas.py              # Predicate, domain schema models

  solvers/
    z3_solver.py            # FOLSolver — Z3 entailment / refutation
```

---

## AST Grammar

```
Term       ::= Variable | Constant

FunctionTerm ::= name(Variable*)          -- numeric-valued attribute, e.g. GPA(x)
NumericTerm  ::= float                    -- 2.5, 80.0, 65.0
DateTerm     ::= str                      -- "June1", "May15", "AddDropDeadline"

Atomic     ::= PredicateName(Term+)

Comparison ::= FunctionTerm op NumericTerm     -- GPA(x) >= 2.5
             | FunctionTerm op FunctionTerm    -- Credits(x) >= MinCredits(y)
             | FunctionTerm op DateTerm        -- Submission(x) < June1
             where op ∈ { =, !=, <, <=, >, >= }

Quantified ::= ∀x[Restrictor].Body
             | ∃x.Body

Logical    ::= ¬Formula
               Formula ∧ Formula
             | Formula ∨ Formula
             | Formula → Formula
             | Formula ↔ Formula

Formula    ::= Atomic
             | Comparison
             | Quantified
             | Logical
```

---

## Frame Kinds

`PremiseFrameParser` maps each NL sentence to one of:

| kind | assembled shape |
|---|---|
| `fact` | `A(x) ∧ B(x)` |
| `existential_fact` | `∃x.(A(x) ∧ B(x))` |
| `numeric_fact` | `A(x) ∧ (f(x) op n)` |
| `universal_rule` | `∀x[R(x)].(A(x) → B(x))` |
| `deontic_rule` | `∀x[R(x)].(A(x) → Required/Allowed B(x))` |
| `permission_rule` | `∀x[R(x)].(A(x) → AllowedB(x))` |
| `prohibition_rule` | `∀x[R(x)].(A(x) → NOT CanB(x))` |
| `numeric_rule` | `∀x[R(x)].(f(x) op n → B(x))` |
| `temporal_rule` | `∀x[R(x)].(f(x) < date → B(x))` |
| `equivalence` | `∀x.(A(x) ↔ B(x))` |
| `meta_rule` | full recursive FOLParser fallback |
| `unsupported` | full recursive FOLParser fallback |

---

## Question Side (`QuestionSideParser`)

The question side mirrors the premise side: classify first, compile FOL second, never emit FOL from the classifier. It produces a `QuerySpec` describing *how* to answer the question, then the pipeline routes to the matching solver call.

```
QuestionSideParser
  ├── QParser        question                  → QuestionFrameResult   (1 LLM call)
  ├── OParser        stem + options            → OptionParseBundle     (deterministic-first; LLM only for unresolved fragments)
  ├── ClaimParser    claim texts + schema      → FOLNode[]             (possessive norm + IF_ALL_THEN_ALL + FOLParser + canonicalize)
  └── query_verifier QuerySpec                 → supported / issues    (deterministic)
```

**`question_format`** — `polar` (yes/no), `mcq`, or `open_wh`.

**`solver_mode`** — what to compute and how it routes:

| solver_mode | meaning | routing (v1) |
|---|---|---|
| `entailment` | does a claim follow (default) | `check_ynu` (polar) / `check_mcq` (mcq) |
| `refutation` | "which is false / not true / cannot" | `check_mcq_refutation` |
| `ynu_mapped` | MCQ whose options are Yes/Uncertain/No | `check_ynu` on the stem, map verdict → label |
| `strongest_conclusion` | strongest / most significant | classified; **deferred** → Unknown |
| `fewest_premise` | follows from fewest premises | `check_mcq_fewest_premises` |
| `premise_selection` | which premises support a conclusion | classified; **deferred** → Unknown |
| `unsupported` | open-WH / no decidable claim | → Unknown |

**`can_interpretation`** — disambiguates the word "can":
- `meta_inference` — "which statement **can be inferred**" → logical entailment, "can" dropped from the claim.
- `object_modal` — "**Can** Tuan take the course?" → ability/permission kept inside the claim (`Tuan can take the course`).

### OParser — Role-Aware Deterministic OptionParser (B10)

`OParser.parse_options(stem, options) → OptionParseBundle`

Each MCQ option is classified into one of 9 **`OptionRole`** values in deterministic priority order (first match wins). LLM calls are only made as a fallback for unresolved modal/predicate fragments.

| priority | `OptionRole` | detection rule | is_selectable |
|---|---|---|---|
| 1 | `RAW_FOL` | contains `∀ ∃ → ↔ ¬ ∧ ∨` or `ForAll(` / `Exists(` | ✗ |
| 2 | `PREMISE_REFERENCE` | `^premises?\s+\d+(\s*[,and]\s*\d+)*\.?$` | ✗ |
| 3 | `YNU_ANSWER` | bare polarity token: `yes/no/uncertain/unknown` alone or followed by `[.,]` only | ✗ |
| 4 | `NONE_OF_ABOVE` | `none of the (above\|following)` | ✗ |
| 5 | `SUBJECTLESS_MODAL_FRAGMENT` | starts with a modal (`can/cannot/may/must/should/…`) with no preceding subject | ✓ (after realization) |
| 6 | `PREDICATE_FRAGMENT` | starts with fragment starter (`eligible/needs/only/requires/able/…`) — no subject | ✓ (after realization) |
| 7 | `CONJUNCTIVE_CLAIM` | full claim containing `both … and` | ✓ |
| 8 | `FULL_CLAIM` | default for any complete proposition | ✓ |
| 9 | `UNKNOWN` | empty / label-combo / unresolved fragment | ✗ |

**YNU bare-token rule** — `YNU_ANSWER` matches only when the polarity token is the *entire* option or is immediately followed by punctuation. "No one is qualified." and "No AI models use deep learning." are `FULL_CLAIM`, not `YNU_ANSWER`. `ynu_mapped` solver mode only triggers when ≥ 2 options have role `YNU_ANSWER`.

**Fragment realization** — for `SUBJECTLESS_MODAL_FRAGMENT` and `PREDICATE_FRAGMENT`, `OParser` first tries deterministic subject recovery from the question stem (patterns: `about X`, `for X`, `true (about|of|for) X`, `(can|is|does|will|should) X`). On success, the subject is prepended deterministically (modal fragments: `"{subject} {modal…}"`; predicate fragments: `"{subject} is {frag}"`). Unresolved fragments fall back to one batched LLM `FragmentRealizationResult` call. Still-unresolved → role `UNKNOWN`, `is_selectable=False`.

**Pronoun resolution** — for `FULL_CLAIM` and `CONJUNCTIVE_CLAIM` options that start with `he/she/they/it/his/her`, the pronoun is replaced with the subject recovered from the question stem before the claim is sent to `ClaimParser`. Emits `PRONOUN_RESOLVED` diagnostic tag.

**`OptionParseBundle`** — returned by `OParser`, carried through `QuestionParseBundle`:
- `options: tuple[OptionClaim, ...]`
- `marker_style: str` — `"dot"` / `"paren"` / `"mixed"`
- `option_count: int` — 3 or 4 in the training set
- `role_distribution: dict[str, int]`
- `verified: bool`, `issues: tuple[str, ...]`, `extraction_diagnostics: tuple[str, ...]`

**`OptionClaim` fields:**

| field | meaning |
|---|---|
| `label` | `"A"` … `"D"` |
| `original_text` | raw text from the question (including marker) |
| `normalized_text` | stripped of leading `A. ` / `A) ` markers |
| `role` | `OptionRole` — one of the 9 values above |
| `claim_text` | realized claim sentence (non-null only for selectable roles) |
| `raw_fol` | raw FOL string when `role == RAW_FOL` |
| `premise_indices` | list of referenced premise indices when `role == PREMISE_REFERENCE` |
| `ynu_value` | `"yes"` / `"no"` / `"uncertain"` / `"none"` |
| `is_selectable` | `True` for roles the solver scores; `False` for YNU/NoA/raw_fol/premise_ref/unknown |
| `fol` | `FOLNode` (filled by ClaimParser for selectable options that compiled) |
| `diagnostics` | per-option diagnostic tags |

**None-of-above solver post-processing** — `check_mcq(premises, options, none_of_above_label=None)`:
- Exactly 1 ordinary option entailed → return that label.
- 0 ordinary options entailed **and** `none_of_above_label` present → return the none-of-above label.
- Otherwise → `"Uncertain"`.

**Embedded option extraction** (`options.py / extract_mcq`) — handles both `A.` and `A)` marker styles. The `options_mcq.json` sidecar misses the 13 records (134–146) that use `A)` markers; `extract_mcq` catches all of them. Diagnostics: `MCQ_MARKER_A_DOT`, `MCQ_MARKER_A_PAREN`, `MCQ_MIXED_MARKER_FORMAT`, `MCQ_THREE_OPTIONS`, `MCQ_FOUR_OPTIONS`, `MCQ_DUPLICATE_LABEL`, `MCQ_OPTION_TEXT_EMPTY`.

### ClaimParser — Claim text → canonical FOL

`ClaimParser.parse_claims(claim_texts, schema) → (FOLNodes, renames)`

Before sending to `FOLParser`, each claim text goes through two deterministic pre-processing steps:

1. **Possessive normalization** — `"X's Y"` → `"Y of X"` (e.g. `"John's GPA"` → `"GPA of John"`). Prevents possessive phrases from being collapsed into a phantom entity argument like `JohnsGPA`.

2. **IF_ALL_THEN_ALL detection** — matches `"if all/every X …, then all/every Y …"` and builds a meta-implication deterministically:
   - Splits into antecedent and consequent sub-texts
   - Parses both via `FOLParser` (in the same batch as regular texts)
   - Combines with `LogicalNode(IMPLIES, ∀x.P(x), ∀y.Q(y))`
   
   Without this, the recursive FOLParser produces the wrong nested structure `∀x[P(x)].(Q(x) → ∀y.R(y))`.

After pre-processing, texts are sent to `FOLParser.parse_many` in one batch, then `schema.canonicalize` renames predicates to match the premise vocabulary.

**Semantic-family canonicalization guard** — `PremiseSchema.canonicalize` blocks cross-family predicate renaming. Predicates are classified into three families:

| family | example predicates |
|---|---|
| `requirement` | `Requires`, `Needs`, `Required`, `MustHave`, `Prerequisite` |
| `achievement` | `QualifiesFor`, `EligibleFor`, `Receives`, `Earns`, `Achieves` |
| `action` | `Pass`, `Complete`, `Submit`, `Take`, `Finish` |

A claim predicate in one family is never renamed to a schema predicate in a different family, even if they share the same semantic key. This prevents `Requires(Sophia, X)` from being silently merged with `QualifiesFor(Sophia)`.

---

## Schema Diagnostics

Raised during `PremiseSchema` build and `_verify_bundle()`:

| tag | meaning |
|---|---|
| `ARITY_DRIFT` | same predicate appears as `/1` and `/2` across premises |
| `SCHEMA_SIMILAR_PREDICATES` | two predicates share the same semantic key (non-blocking) |
| `GENERIC_CLASS_USED_AS_CONSTANT` | `Students`, `AIModels` used as a constant — repaired automatically for rule-like frames |
| `ONLY_IF_DIRECTION_CHECK` | "only if/when" detected — implication direction may be reversed |
| `NUMERIC_CONSTRAINT_LOST` | premise has numeric signals but no `ComparisonNode` in the AST |
| `TEMPORAL_CONSTRAINT_LOST` | premise has temporal signals but no `ComparisonNode` in the AST |
| `UNSUPPORTED_MODAL_NOT_NECESSARILY` | epistemic non-entailment; Z3 cannot model it |

Question-side (`query_verifier`, set on `QuerySpec.issues`):

| tag | meaning |
|---|---|
| `QUERY_NO_CLAIM` | polar / ynu_mapped question has no testable claim FOL |
| `QUERY_NO_SOLVABLE_OPTIONS` | MCQ has fewer than two options that compiled to FOL |
| `QUERY_OPTIONS_UNSUPPORTED` | all options are raw FOL or premise references |
| `QUERY_NO_YNU_OPTIONS` | `ynu_mapped` question has no Yes/No/Uncertain options |
| `QUERY_MODE_DEFERRED` | strongest/premise_selection — classified but not solved in v1 |
| `QUERY_OPEN_WH_UNSUPPORTED` | open-WH question with no decidable claim |
| `QUERY_NONE_OF_ABOVE_PRESENT` | informational: a NONE_OF_ABOVE option is present; handled by solver post-processing |

Option-side (set on `OptionClaim.diagnostics`):

| tag | meaning |
|---|---|
| `PRONOUN_RESOLVED` | pronoun subject (he/she/they/it) was replaced with the recovered stem subject |
| `RAW_FOL_UNSUPPORTED` | option contains raw FOL symbols — classified but not compiled |
| `FRAGMENT_SUBJECT_RECOVERED` | deterministic stem-pattern subject recovery succeeded |
| `FRAGMENT_LLM_REPAIR` | LLM `FragmentRealizationResult` call was used for subject recovery |

Proof-connectivity diagnostics (set per claim/option in
`routing_diagnostics.proof_connectivity` and merged into serialized option diagnostics):

| tag | meaning |
|---|---|
| `CLAIM_PREDICATE_NOT_IN_SCHEMA` | claim predicate signature is absent from the premise schema |
| `CLAIM_ARITY_NOT_IN_SCHEMA` | predicate name exists, but not with the claim's arity |
| `CLAIM_CONSTANT_NOT_IN_SCHEMA` | claim constant is absent; a likely alias is reported when found |
| `CLAIM_UNRESOLVED_PRONOUN` | claim text or AST still contains a personal pronoun |
| `CLAIM_SCHEMA_LOW_CONNECTIVITY` | weighted predicate/constant connectivity score is below `0.75` |
| `CLAIM_CANONICALIZATION_BLOCKED` | claim did not compile or a likely semantic match was not canonicalized |

The dashboard contains structured predicate and constant match records plus a
human-readable `report`. When the solver returns `Z3_TRUE_UNCERTAIN`,
`z3_uncertainty_interpretation` is set to `REAL_LOGICAL_UNCERTAINTY` or
`SYMBOL_MISMATCH_LIKELY`; this does not change the answer returned by Z3.
The score averages exact symbols (`1.0`), likely constant aliases (`0.5`),
semantic predicate candidates (`0.4`), arity mismatches (`0.25`), and absent
symbols (`0.0`); an unresolved pronoun halves the final score.

---

## API Endpoints

| method | path | description |
|---|---|---|
| `GET` | `/health` | liveness check |
| `POST` | `/parser` | NL premises → FOL ASTs + verified + issues + renames |
| `POST` | `/qparser` | question (+ options + premises) → `QuerySpec` (classify only, no solving) |
| `POST` | `/predict` | full pipeline → answer |
| `POST` | `/z3` | alias for `/predict` |
| `POST` | `/premises` | lighter version of `/parser` (no verified/issues/renames) |

**`/parser` request / response:**
```json
{ "premises": ["All students must pass at least 5 courses", "Alice is a student"] }
```
```json
{
  "premises": [
    { "id": "premise-1", "original_text": "...", "fol": "∀x[Student(x)].(PassedCourses(x) >= 5.0)", "ast": {...} }
  ],
  "verified": true, "issues": [], "renames": []
}
```

**`/qparser` request / response:** (options may be embedded in `question` as `A.`/`A)` lines)
```json
{ "premises": ["..."], "question": "Which conclusion can be inferred?\nA. ...\nB. ..." }
```
```json
{
  "question_format": "mcq",
  "solver_mode": "entailment",
  "can_interpretation": "meta_inference",
  "main_claim_text": null, "main_claim_fol": null, "negate_claim": false,
  "supported": true, "issues": [],
  "marker_style": "dot",
  "role_distribution": { "FULL_CLAIM": 3, "CONJUNCTIVE_CLAIM": 1 },
  "extraction_diagnostics": ["MCQ_FOUR_OPTIONS"],
  "option_claims": [
    {
      "label": "A", "role": "FULL_CLAIM",
      "normalized_text": "Alice passes the exam.",
      "claim_text": "Alice passes the exam.",
      "fol": "Passes(Alice)",
      "is_selectable": true,
      "ynu_value": "none", "premise_indices": [], "raw_fol": null,
      "diagnostics": []
    }
  ]
}
```

---

## Status

### Done

- [x] `AtomicNode`, `LogicalNode`, `QuantifiedNode` — core FOL AST
- [x] `ComparisonNode` + `FunctionTerm` / `NumericTerm` / `DateTerm` — numeric & temporal comparisons (Issue 6)
- [x] `FOLParser` — recursive atomic / logical / quantified / coreference parsing
- [x] `PremiseFrameParser` — one LLM call classifies the logical skeleton of each premise
- [x] `PremiseFrameCompiler` — deterministic AST assembly from frame fragments
- [x] `ConstraintParser` — `"at least 5 courses"` → `CompletedCourses(x) >= 5.0` (Issue 6)
- [x] Temporal constraint parsing — `"before June 1"` → `Submission(x) < June1` (Issue 6)
- [x] `PremiseSchema` — predicate deduplication, arity checking, semantic canonicalization
- [x] `ARITY_DRIFT` diagnostic with `/0, /1` format (Issue 9)
- [x] Atomic prompt: Rules 5-8 for stable unary/binary predicate preference (Issue 9)
- [x] Generic-class constant repair — `Students` arg → `∀x[Student(x)]` for rule-like frames (Issue 8)
- [x] `NUMERIC_CONSTRAINT_LOST` / `TEMPORAL_CONSTRAINT_LOST` verifier diagnostics (Issue 6)
- [x] Deontic mapping — `must/may/cannot` → `Required*/Allowed*/NOT Can*` predicate prefixes
- [x] `FOLSolver` — Z3 `check_ynu()`, `check_mcq()`, `check_mcq_refutation()`, `check_mcq_fewest_premises()`
- [x] Z3 real-arithmetic for `ComparisonNode` via `Entity→Real` function declarations (Issue 6)
- [x] `run_type1_pipeline()` — wires all stages into one call
- [x] `/parser` API endpoint
- [x] B03 parser quality eval notebook
- [x] **`QuestionSideParser`** — `QParser` + `OParser` + `ClaimParser` + `query_verifier`
- [x] Question classification — `question_format` / `solver_mode` / `can_interpretation` (meta vs object "can")
- [x] Solver routing — entailment / refutation / ynu_mapped / fewest-premise (strongest/premise_selection deferred)
- [x] Configurable uncertain token — `EXACT_TYPE1_UNCERTAIN_TOKEN` (default `"Unknown"`)
- [x] `/qparser` API endpoint + B05 question-parser eval notebook
- [x] **B10 OParser** — 9-role deterministic-first OptionParser (B10)
  - [x] `OptionRole` taxonomy (RAW_FOL / PREMISE_REFERENCE / YNU_ANSWER / NONE_OF_ABOVE / SUBJECTLESS_MODAL_FRAGMENT / PREDICATE_FRAGMENT / CONJUNCTIVE_CLAIM / FULL_CLAIM / UNKNOWN)
  - [x] `OptionParseBundle` with `marker_style`, `role_distribution`, `extraction_diagnostics`
  - [x] YNU bare-token rule (fixes 10× over-counting; "No one…" → FULL_CLAIM)
  - [x] Deterministic fragment realization (stem-pattern subject recovery + LLM fallback)
  - [x] `extract_mcq()` with both `A.` and `A)` marker support + extraction diagnostics
  - [x] None-of-above solver post-processing in `check_mcq(…, none_of_above_label)`
  - [x] B06 eval notebook (offline classifier checks + live structural eval via `/qparser`)
- [x] **Parser bug fixes** (post-B10)
  - [x] `numeric_fact` compiler: numeric constraints now included in output (was silently dropped)
  - [x] `_camel()` helper: multi-word constant args in constraints (`"Professor John"` → `"ProfessorJohn"`)
  - [x] `extract_mcq` deduplication in `_and_nodes` (A∧A removed before AND-folding)
  - [x] Single `extract_mcq` call per request (pipeline + router no longer double-extract)
  - [x] **ClaimParser**: possessive normalization (`"John's GPA"` → `"GPA of John"`)
  - [x] **ClaimParser**: IF_ALL_THEN_ALL meta-implication — deterministic split before FOLParser
  - [x] **PremiseSchema**: semantic-family canonicalization guard (REQUIREMENT / ACHIEVEMENT / ACTION)
  - [x] **OParser**: pronoun resolution for FULL_CLAIM / CONJUNCTIVE_CLAIM options (`PRONOUN_RESOLVED`)
- [x] **Proof-connectivity dashboard**: per-claim symbol matches, diagnostics, and uncertainty interpretation

### Missing / Incomplete

- [ ] **Raw-FOL & premise-reference options** — tagged but not compiled (no FOL-string → AST parser; no premise-ref resolution).
- [ ] **Open-ended question type** — `open_wh` is detected and routed to Unknown; no free-text answering.
- [ ] **SaT sentence segmentation** — multi-sentence premises are not split before parsing; each premise string is sent whole.
- [ ] **Coreference resolution** — `build_coreference_request()` exists in the router but is not called anywhere in the live pipeline.
- [ ] **Rephrasing** — `build_rephrase_request()` exists but is not wired into any pre-processing pass.
- [ ] **Claim-side frame parser (Bug 2)** — `ClaimParser` currently sends all claims directly to the recursive `FOLParser`. It needs a lighter version of the premise-side frame architecture: requirement/modal/possessive attribute patterns need deterministic frame detection before the LLM sees them.
- [ ] **Purpose clause parsing (Bug 4)** — `"X needs to A to B"` / `"X must A in order to B"` patterns lose the purpose clause and the requirement modality. Should produce `RequiresFor(X, A, B)` instead of `Pass(X, A)`. Requires the claim-side frame parser (Bug 2) to be implemented first.
- [ ] **Confidence scoring** — `PredictionResponse.confidence` is always `None`.
- [ ] **Chain-of-thought** — `cot` is a few fixed strings; no step-by-step reasoning trace.
- [ ] **Unknown fallback quality** — when `verified=False` or unsupported, the pipeline returns the Unknown token with no further reasoning. An LLM-based fallback for hard cases is not implemented.
- [ ] **Numeric/temporal constraints in questions** — `ConstraintParser` is only wired for premises, not for question or option texts.
- [ ] **Remaining ranking solver modes** — `strongest_conclusion` and `premise_selection` are classified but return the Unknown token.

---

## Parser Model Service

The parser runs as a dedicated vLLM service (Qwen3-1.7B on ThunderCompute A6000):

```bash
bash scripts/serve_type1_parser.sh
```

Key environment variables:

```dotenv
EXACT_TYPE1_PARSER_BASE_URL=http://127.0.0.1:8001/v1
EXACT_TYPE1_PARSER_MODEL=type1-parser
EXACT_TYPE1_PARSER_API_KEY=exact-parser-token
EXACT_TYPE1_PARSER_CONCURRENCY=32
EXACT_TYPE1_PARSER_SOURCE_MODEL=Qwen/Qwen3-1.7B
EXACT_TYPE1_PARSER_SERVER_GPU_MEMORY_UTILIZATION=0.25
```

The application creates one shared `ParserClient` at startup (via `build_parser_client_from_settings()`) and closes it at shutdown. Do not create a client per request.

```python
from exact.type1.parser import (
    PremiseParser, QuestionSideParser, build_parser_client_from_settings,
)

client = build_parser_client_from_settings()
premise_parser = PremiseParser.from_parser_client(client)
question_parser = QuestionSideParser.from_parser_client(client)

bundle = await premise_parser.parse_premises(["All students must pass.", "Alice is a student."])
print(bundle.verified)          # True / False
for text, tree in zip(bundle.premises, bundle.trees):
    print(text, "→", repr(tree))

q = await question_parser.parse_question(
    "Which conclusion can be inferred?\nA. Alice passes.\nB. Alice fails.",
    options=None,               # extracted from the question body
    schema=bundle.schema,       # share the premise vocabulary
)
print(q.spec.question_format, q.spec.solver_mode, q.spec.supported)
for c in q.spec.option_claims:
    print(c.label, c.role, c.is_selectable, repr(c.fol))

await client.aclose()
```
