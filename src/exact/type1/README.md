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
    │                                            │  OParser     → OptionClaim[]
    │                                            │  ClaimParser → FOL (schema-canonicalized)
    │                                            │  QueryVerifier → supported / issues
    │                                            ▼
    │                                          QuerySpec  ──► /qparser endpoint (inspect, stop here)
    ▼                                            │
Z3 FOLSolver  ◄──────────────────────────────────┘
    │  polar      → check_ynu()             → "Yes" / "No" / "Uncertain"
    │  mcq        → check_mcq()             → option label "A".."D"
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
    options.py              # parse_mcq_options — extract embedded A./A) options
    qparser.py              # QParser — classify question → QuestionFrameResult
    oparser.py              # OParser — MCQ options → OptionClaim[]
    claim_parser.py         # ClaimParser — claim texts → canonicalized FOL
    query_verifier.py       # verify() — pre-solve QuerySpec validation
    question_parser.py      # QuestionSideParser — question orchestration
    schemas.py              # frame/query result models, PremiseSchema, QuerySpec

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
             | Formula ∧ Formula
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
  ├── OParser        stem + options            → OptionClaim[]         (1 LLM call/option)
  ├── ClaimParser    claim texts + schema      → FOLNode[]             (FOLParser + canonicalize)
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
| `fewest_premise` | follows from fewest premises | classified; **deferred** → Unknown |
| `premise_selection` | which premises support a conclusion | classified; **deferred** → Unknown |
| `unsupported` | open-WH / no decidable claim | → Unknown |

**`can_interpretation`** — disambiguates the word "can":
- `meta_inference` — "which statement **can be inferred**" → logical entailment, "can" dropped from the claim.
- `object_modal` — "**Can** Tuan take the course?" → ability/permission kept inside the claim (`Tuan can take the course`).

**`OptionClaim.option_type`** — options are not always self-contained propositions:

| option_type | handling |
|---|---|
| `proposition` | parsed to FOL |
| `fragment` | subject recovered from the stem, then parsed to FOL |
| `raw_fol` | `∀x …` formula — tagged, **not** compiled in v1 |
| `premise_reference` | "Premises 1, 3, 7" → indices — tagged, not compiled in v1 |
| `ynu_answer` | "Yes, all mastered…" → `ynu_value`, drives `ynu_mapped` |

Embedded options (`A.` **and** `A)` lines inside the question body) are extracted by `parse_mcq_options` when no separate `options` payload is supplied.

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
| `QUERY_MODE_DEFERRED` | strongest/fewest/premise_selection — classified but not solved in v1 |
| `QUERY_OPEN_WH_UNSUPPORTED` | open-WH question with no decidable claim |

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
  "option_claims": [
    { "label": "A", "option_type": "proposition", "claim_text": "...", "fol": "...", "ynu_value": "none", "premise_indices": [], "raw_fol": null }
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
- [x] `FOLSolver` — Z3 `check_ynu()`, `check_mcq()`, `check_mcq_refutation()`
- [x] Z3 real-arithmetic for `ComparisonNode` via `Entity→Real` function declarations (Issue 6)
- [x] `run_type1_pipeline()` — wires all stages into one call
- [x] `/parser` API endpoint
- [x] B03 parser quality eval notebook
- [x] **`QuestionSideParser`** — `QParser` + `OParser` + `ClaimParser` + `query_verifier`
- [x] Question classification — `question_format` / `solver_mode` / `can_interpretation` (meta vs object "can")
- [x] MCQ option interpretation — proposition / fragment (subject recovery) / raw_fol / premise_reference / ynu_answer
- [x] Embedded option extraction — `parse_mcq_options` handles `A.` and `A)`
- [x] Solver routing — entailment / refutation / ynu_mapped (strongest/fewest/premise_selection deferred)
- [x] Configurable uncertain token — `EXACT_TYPE1_UNCERTAIN_TOKEN` (default `"Unknown"`)
- [x] `/qparser` API endpoint + B05 question-parser eval notebook

### Missing / Incomplete

- [ ] **Ranking solver modes** — `strongest_conclusion`, `fewest_premise`, `premise_selection` are classified but return the Unknown token (need minimal-subset / ranking logic in Z3).
- [ ] **Raw-FOL & premise-reference options** — tagged but not compiled (no FOL-string → AST parser; no premise-ref resolution).
- [ ] **Open-ended question type** — `open_wh` is detected and routed to Unknown; no free-text answering.
- [ ] **SaT sentence segmentation** — multi-sentence premises are not split before parsing; each premise string is sent whole.
- [ ] **Coreference resolution** — `build_coreference_request()` exists in the router but is not called anywhere in the live pipeline.
- [ ] **Rephrasing** — `build_rephrase_request()` exists but is not wired into any pre-processing pass.
- [ ] **Confidence scoring** — `PredictionResponse.confidence` is always `None`.
- [ ] **Chain-of-thought** — `cot` is a few fixed strings; no step-by-step reasoning trace.
- [ ] **Unknown fallback quality** — when `verified=False` or unsupported, the pipeline returns the Unknown token with no further reasoning. An LLM-based fallback for hard cases is not implemented.
- [ ] **Numeric/temporal constraints in questions** — `ConstraintParser` is only wired for premises, not for question or option texts.

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
    print(c.label, c.option_type, repr(c.fol))

await client.aclose()
```
