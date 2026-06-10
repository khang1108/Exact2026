# Type 1 Logic Framework

This package implements the Type 1 branch for educational yes/no/unknown questions that require explicit logical reasoning. The current design is a small neuro-symbolic framework: an LLM translates natural language into a compact Horn-style intermediate representation, then deterministic symbolic solvers decide the answer and produce a traceable proof.

## Goal

For EXACT Type 1, the system should answer from the given premises only and expose why the answer is correct. The main objective is therefore not just prediction accuracy, but controlled reasoning:

- translate premises and questions into inspectable logical objects;
- derive consequences with deterministic solvers;
- preserve premise provenance for explanation and later S2-style proof checking;
- return `Yes`, `No`, or `Unknown` instead of hallucinating unsupported answers.

## Package layout

```text
src/exact/logic/
├── pipeline.py              # Type 1 orchestration: PredictionRequest → PredictionResponse
├── ir/                      # Core formula IR (Atom, Formula, TranslatedProblem, …)
├── parsing/                 # Deterministic atom/FOL parsers (tests, MCQ helpers)
│   ├── parser.py
│   └── fol_parser.py
├── translation/             # LLM NL→logic autoformalizer
│   ├── llm_translator.py
│   └── prompts.py
├── kb/                      # KnowledgeBase build + premise cache
├── explain/                 # Proof trace → explanation / cot / FOL text
└── README.md

src/exact/symbolic_solvers/
├── base.py                         # Solver protocol
├── forward_chain/solver.py         # Default Horn forward-chain solver with unification
└── z3_prop/                        # Typed formula Z3 entailment backend
```

`logic/` owns representation, translation, KB, explanation, and orchestration.
`symbolic_solvers/` owns execution backends. This separation keeps parser,
solver, and explanation layers replaceable.

## End-to-end algorithm

The default Type 1 path is implemented in `pipeline.py::run_type1_pipeline`.

```text
PredictionRequest
      │
      ▼
collect premises_nl + question
      │
      ▼
translate_with_llm()
      │
      └── validates JSON with Pydantic schemas
      │
      ▼
build_kb_from_parsed_premises()
      │
      └── KnowledgeBase(facts, rules, source_idx, warnings, parser_version)
      │
      ▼
ForwardChainSolver().solve(kb, query.claim)
      │
      ├── derive closure of all reachable atoms
      ├── check claim in closure      -> Yes
      ├── check negated claim closure -> No
      └── otherwise                  -> Unknown
      │
      ▼
explain_result()
      │
      └── explanation + proof steps + cited premise labels
      │
      ▼
PredictionResponse
```

## Core IR

The framework uses a compact Horn-style representation from `ir/`.

### `Atom`

An atomic statement:

```python
Atom(pred="student", args=("sophia",), negated=False)
Atom(pred="eligible_for_certificate", args=("sophia",), negated=True)
```

- `pred`: normalized predicate name.
- `args`: optional tuple of constants or variables.
- `negated`: explicit logical negation.
- `text`: optional original/normalized text for display; it is not part of equality.

Variables are represented as strings beginning with `?`, for example `?x`.

### `Fact`

A directly stated premise:

```python
Fact(atom=Atom("student", ("sophia",)), source_idx=0, text="Sophia is a student.")
```

`source_idx` points back to the original premise, so explanations can cite `P1`, `P2`, etc.

### `Rule`

A Horn rule whose conditions must all hold before deriving the conclusion:

```python
Rule(
    conditions=(
        Atom("student", ("?x",)),
        Atom("passed_assessment", ("?x",)),
    ),
    conclusion=Atom("qualified_for_advanced_courses", ("?x",)),
    source_idx=2,
    text="Students who pass the assessment qualify.",
)
```

This means:

```text
student(?x) AND passed_assessment(?x)
  -> qualified_for_advanced_courses(?x)
```

### `ProofStep`

One proof DAG node:

```python
ProofStep(
    derived=Atom("qualified_for_advanced_courses", ("sophia",)),
    used_premises=(0, 1, 2),
    rule_idx=2,
    parents=(
        Atom("student", ("sophia",)),
        Atom("passed_assessment", ("sophia",)),
    ),
)
```

The solver stores grounded parent atoms, not schematic rule conditions, so later proof rendering can reconstruct the derivation precisely.

## Translation layer

There is one runtime translation path: the LLM semantic parser.

`translation/llm_translator.py` treats the LLM as a parser, not as the final judge. It asks the model to return structured JSON:

```json
{
  "premises": [
    {
      "source_idx": 0,
      "facts": [{"text": "Sophia is a student", "negated": false}],
      "rules": []
    }
  ],
  "query": {
    "claim": {"text": "Sophia qualifies", "negated": false}
  }
}
```

The output is validated by Pydantic models before conversion into `ParsedPremise` and `Query`. If the LLM is missing or returns invalid output, the pipeline logs the error and raises.

### Parser utilities

`parsing/parser.py` is intentionally conservative and is not used as a runtime premise translator. It remains useful for unit tests and for converting already-isolated MCQ option text into solver atoms. It currently supports:

- direct facts: `A.` -> `Fact(A)`;
- simple conditionals: `If A then B.` -> `Rule(A -> B)`;
- conjunctions: `If A and B, then C.` -> `Rule(A AND B -> C)`;
- simple explicit negation markers like `not`, `does not`, `do not`, `did not`.

For competition accuracy, the LLM translator should handle richer educational wording.

## Knowledge base layer

`kb/kb.py` builds a `KnowledgeBase`:

```python
KnowledgeBase(
    raw_premises=(...),
    facts=(...),
    rules=(...),
    premise_hash="...",
    parser_version="llm_translator_v1",
    warnings=(...),
)
```

The KB separates premise parsing from solving. This is useful because the same premise set can be cached or reused with different questions, and parser-version hashing prevents stale logical forms from mixing across parser changes.

## Default solver: forward chaining with unification

The default backend is `ForwardChainSolver` in `symbolic_solvers/forward_chain/solver.py`.

### High-level behavior

1. Seed `known` with all premise facts.
2. Store one `ProofStep` for each known fact.
3. Repeatedly scan rules until no new atom is derived.
4. For every rule, try to match each condition against known atoms.
5. Carry variable bindings across all conditions.
6. Apply the final binding to the conclusion.
7. Add the grounded conclusion to `known` and store a proof step.

### Example

Known facts:

```text
student(sophia)
passed_assessment(sophia)
```

Rule:

```text
student(?x) AND passed_assessment(?x)
  -> qualified_for_advanced_courses(?x)
```

Unification produces:

```python
{"?x": "sophia"}
```

The solver derives:

```text
qualified_for_advanced_courses(sophia)
```

### Why unification matters

Without unification, Python equality compares the argument tuples directly:

```python
("?x",) != ("sophia",)
```

So `student(?x)` would never match `student(sophia)`. The current solver fixes this with:

- `unify(pattern, ground_fact, subst)`;
- `apply_subst(atom, subst)`;
- recursive condition matching across known atoms;
- consistency checks so the same variable must map to the same constant across conditions.

This prevents invalid derivations such as using `student(sophia)` with `passed_assessment(noah)` for a rule requiring the same `?x`.

## Answer semantics

`solve_query(kb, claim)` returns:

- `Yes` if `claim` is derived;
- `No` if `claim.negation()` is derived;
- `Unknown` if neither the claim nor its negation is derivable.

This closed-world answer policy is intentionally conservative: absence of proof is `Unknown`, not `No`.

## Explanation layer

`explain/explain.py` converts `SolveResult` into the API-facing fields:

- `explanation`: short natural-language summary;
- `cot`: symbolic proof trace from `ProofStep.natural_language`;
- `premises`: cited premise labels such as `P1`, `P2`;
- `fol`: readable formalization from `kb_to_fol_like_text()`.

The explanation does not ask the LLM to justify the result. It is generated from the deterministic proof trace.

## Z3 backend

`symbolic_solvers/z3_solver/` is an alternative backend that encodes the current Horn-like IR into Boolean Z3 formulas. It is useful as a future verifier or richer entailment engine, but the current production Type 1 pipeline uses `ForwardChainSolver` because it gives direct proof traces.

Current Z3 scope:

- Boolean atoms;
- facts as asserted symbols;
- Horn rules as implications;
- positive/negative atom exclusion.

Future extensions can add numeric constraints, dates, credits, GPA comparisons, and typed first-order structures through `Theory` in `ir/`.

## Extension points

### Improve translation accuracy

Most Type 1 gains will likely come from better `translation/llm_translator.py` prompts/schema and post-processing. The solver can only prove what the translator represents correctly.

Recommended direction:

- preserve entities as arguments instead of baking them into predicate names;
- normalize equivalent predicates consistently;
- extract variable rules from generic statements;
- keep premise source indices exact.

### Add richer solver support

Forward chaining is ideal for Horn rules. If the dataset includes non-Horn logic, arithmetic, constraints, or quantifier-heavy forms, add a backend under `symbolic_solvers/` while keeping the same `SymbolicSolver.solve(kb, claim)` contract.

### Strengthen proof checking

Because every derived atom has a `ProofStep` with parents and cited premises, an S2 verifier can traverse the proof DAG and independently check each step.

## Current limitations

- MCQ option atom parsing is shallow; premise translation is handled by the LLM.
- The IR supports simple predicate-like atoms, but not full first-order logic syntax.
- Forward chaining handles Horn-style derivations; it does not handle disjunction, contradiction management beyond explicit negated atoms, arithmetic, or temporal constraints.
- Z3 encoding is currently Boolean and does not yet exploit the richer `Theory` fields.

## Tests

Focused regression tests for unification live in:

```text
tests/test_forward_chain_unification.py
```

Run them with:

```bash
./exact/bin/python -m pytest tests/test_forward_chain_unification.py
```
