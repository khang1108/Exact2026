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
FOLNode trees  ◄── verified: bool + issues: list[str]
    │
    ├── /parser endpoint  (NL → FOL, stop here)
    │
    ▼
Question / option parsing  (FOLParser.parse_many, schema-canonicalized)
    ▼
Z3 FOLSolver
    │  check_ynu()  → "Yes" / "No" / "Uncertain"
    │  check_mcq()  → option label "A" / "B" / "C" / "D"
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
    premise_parser.py       # PremiseParser — orchestration + repair + verify
    schemas.py              # PremiseFrameResult, PremiseSchema, parser result models
    qparser.py              # QuestionParser  ⚠ stub — not implemented

  models/
    schemas.py              # Predicate, domain schema models

  solvers/
    z3_solver.py            # FOLSolver — Z3 entailment
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

---

## API Endpoints

| method | path | description |
|---|---|---|
| `GET` | `/health` | liveness check |
| `POST` | `/parser` | NL premises → FOL ASTs + verified + issues + renames |
| `POST` | `/predict` | full pipeline → answer |
| `POST` | `/z3` | alias for `/predict` |
| `POST` | `/premises` | lighter version of `/parser` (no verified/issues/renames) |

**`/parser` request:**
```json
{ "premises": ["All students must pass at least 5 courses", "Alice is a student"] }
```

**`/parser` response:**
```json
{
  "premises": [
    { "id": "premise-1", "original_text": "...", "fol": "∀x[Student(x)].(PassedCourses(x) >= 5.0)", "ast": {...} }
  ],
  "verified": true,
  "issues": [],
  "renames": []
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
- [x] `FOLSolver` — Z3 `check_ynu()` and `check_mcq()`
- [x] Z3 real-arithmetic for `ComparisonNode` via `Entity→Real` function declarations (Issue 6)
- [x] `run_type1_pipeline()` — wires all stages into one call
- [x] `/parser` API endpoint
- [x] B03 parser quality eval notebook

### Missing / Incomplete

- [ ] **`qparser.py` — QuestionParser** is an empty stub. Questions and MCQ options are currently parsed by `FOLParser.parse_many()` (plain recursive parse, no frame decomposition).
- [ ] **Frame-based question parsing** — questions like *"Is Alice eligible?"* or *"Which students qualify?"* should go through a question-specific frame before hitting the solver.
- [ ] **Open-ended question type** — competition has MCQ, YNU, and open-ended; open-ended is not handled.
- [ ] **SaT sentence segmentation** — multi-sentence premises are not split before parsing; each premise string is sent whole.
- [ ] **Coreference resolution** — `build_coreference_request()` exists in the router but is not called anywhere in the live pipeline.
- [ ] **Rephrasing** — `build_rephrase_request()` exists but is not wired into any pre-processing pass.
- [ ] **Confidence scoring** — `PredictionResponse.confidence` is always `None`.
- [ ] **Chain-of-thought** — `cot` is two fixed strings; no step-by-step reasoning trace.
- [ ] **`"Uncertain"` fallback quality** — when `verified=False`, the pipeline always returns `"Uncertain"` with no further reasoning. A LLM-based fallback for hard / unverified cases is not implemented.
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
from exact.type1.parser import FOLParser, PremiseParser, build_parser_client_from_settings

client = build_parser_client_from_settings()
parser = PremiseParser.from_parser_client(client)

bundle = await parser.parse_premises(["All students must pass.", "Alice is a student."])
print(bundle.verified)          # True / False
print(bundle.verification_issues)
for text, tree in zip(bundle.premises, bundle.trees):
    print(text, "→", repr(tree))

await client.aclose()
```
