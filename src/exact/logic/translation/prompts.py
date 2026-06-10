"""Prompt builders for the Type 1 formula-level LLM autoformalizer (LINC path).

All prompts here target the formula IR pipeline (Z3PropSolver). The legacy
Horn/KB prompt builders have been removed.
"""

from __future__ import annotations

from openai.types.chat import ChatCompletionMessageParam


_TYPED_FORMULA_NODE_GUIDE = (
    "Formula node shapes (use 'type' as the op key):\n"
    '  atom:    {"type":"atom","pred":"snake_case","args":[TERM],"negated":false}\n'
    '  not:     {"type":"not","arg":FORMULA}\n'
    '  and/or:  {"type":"and","args":[FORMULA,FORMULA,...]}\n'
    '  implies: {"type":"implies","antecedent":FORMULA,"consequent":FORMULA}\n'
    '  iff:     {"type":"iff","left":FORMULA,"right":FORMULA}\n'
    '  forall:  {"type":"forall","variables":["?x"],"body":FORMULA}\n'
    '  exists:  {"type":"exists","variables":["?x"],"body":FORMULA}\n'
    '  compare: {"type":"compare","op":">=","left":TERM,"right":TERM}\n'
    '  in_set:  {"type":"in_set","member":TERM,"options":[TERM,TERM,...]}\n'
    "Term shapes:\n"
    '  constant or variable: "sophia", "?x"\n'
    '  exact number: {"type":"number","value":"6.0"}\n'
    '  function: {"type":"function","name":"gpa","args":["?x"]}\n'
    '  arithmetic: {"type":"arithmetic","op":"*","args":[TERM,TERM]}\n'
)


def build_premises_only_messages(premises: list[str]) -> list[ChatCompletionMessageParam]:
    """Compact premise-only prompt for KnowledgeBase caching."""

    premise_text = "\n".join(f"{idx}: {premise}" for idx, premise in enumerate(premises))
    schema_hint = (
        '{"predicates":[{"name":"pred","arity":1}],'
        '"premises":[{"source_idx":0,"facts":[{"pred":"pred","args":["item"],"negated":false}],'
        '"rules":[{"conditions":[{"pred":"cond","args":["?x"],"negated":false}],'
        '"conclusion":{"pred":"pred","args":["?x"],"negated":false}}]}]}'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. No extra text. "
                "Translate premises into compact Horn-style predicates for a symbolic solver. "
                "CRITICAL: never include a 'text' field in any atom object. "
                "CRITICAL: args must be an array of strings only, never nested objects."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate these premises into Horn-style IR. Output valid JSON matching: {schema_hint}\n"
                "Rules:\n"
                "- NEVER include 'text' fields in facts, conditions, or conclusions — omit them entirely\n"
                "- NEVER include 'gloss' or 'argument_roles' in predicates — only name and arity\n"
                "- pred and constants: lowercase snake_case; variables: ?x, ?y\n"
                "- args must be an array of strings only, e.g. [\"?x\"], never nested objects\n"
                "- Do not use pred values `and`, `or`, `either`, or `not`; split conjunctions and use negated=true for negation\n"
                "- Generic rules use variables (?x); named facts use constants (sofia)\n"
                "- Standalone assertions go in facts; implications go in rules\n"
                "- Every rule must have at least one condition; never output conditions:[]\n"
                "- CRITICAL: if a premise says 'If A and B then C', the rule must have TWO conditions: "
                "[{pred:A,args:[?x]},{pred:B,args:[?x]}] with conclusion {pred:C,args:[?x]}. "
                "NEVER write an identity rule where condition and conclusion share the same pred.\n"
                "- Split conjunctions into separate condition atoms\n"
                "- If a premise has alternatives with `or`, keep only Horn-compatible direct conditions and do not model disjunction\n"
                "- Preserve source_idx exactly\n\n"
                "Example multi-condition rule — Premise: 'If a student completes courses and passes exams, they graduate':\n"
                '{"source_idx":0,"facts":[],"rules":[{"conditions":[{"pred":"completes_courses","args":["?x"],"negated":false},{"pred":"passes_exams","args":["?x"],"negated":false}],"conclusion":{"pred":"graduates","args":["?x"],"negated":false}}]}\n\n'
                f"Premises:\n{premise_text}"
            ),
        },
    ]


def build_query_only_messages(
    question: str,
    predicate_names: tuple[str, ...] = (),
    entity_constants: tuple[str, ...] = (),
) -> list[ChatCompletionMessageParam]:
    """Compact query-only prompt for YNU/open-ended questions."""

    schema_hint = (
        '{"query":{"claim":{"pred":"predicate_name","args":["entity"],"negated":false}}}'
    )
    predicate_instruction = (
        "CRITICAL — query.claim.pred MUST be EXACTLY one of these allowed names "
        "(no variants, no synonyms, no new inventions):\n"
        f"  {', '.join(predicate_names)}\n"
        "If the question's target concept is not literally in this list, pick the "
        "closest matching predicate from the list above.\n"
        if predicate_names
        else ""
    )
    entity_instruction = (
        "CRITICAL — entity names in args MUST be EXACTLY one of these known constants "
        "(copy verbatim, do NOT translate, rephrase, or alter spelling):\n"
        f"  {', '.join(entity_constants)}\n"
        if entity_constants
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. "
                "Translate the question target into one Horn-style query atom. "
                "Never put JSON objects inside args; args must be strings only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate this question target into valid JSON matching: {schema_hint}\n"
                "Rules:\n"
                "- Do not answer the question\n"
                f"{predicate_instruction}"
                f"{entity_instruction}"
                "- pred and constants: lowercase snake_case; variables: ?x, ?y\n"
                "- Prefer atom shape {\"pred\":\"name\",\"args\":[\"entity\"],\"negated\":false}; omit text fields\n"
                "- args must be an array of strings only, e.g. [\"student\"], never nested objects\n"
                "- Mark negated=true only for explicit negation\n\n"
                f"Question:\n{question}"
            ),
        },
    ]


def build_problem_formula_messages(
    premises: list[str],
    question: str,
    options: list[tuple[str, str]] | None,
) -> list[ChatCompletionMessageParam]:
    """Prompt for one-shot formula-level translation of a Type 1 problem.

    One LLM call covers all premises, the question, and every MCQ option using
    a single shared predicate dictionary.  This prevents the vocabulary drift
    that happens when premises and query are translated separately.
    """

    premise_text = "\n".join(f"{idx}: {premise}" for idx, premise in enumerate(premises))

    # Build the goal-type instruction and options block based on question shape.
    if options is None:
        goal_instruction = (
            "Yes/No/Unknown question: return exactly ONE goals item "
            'with role="query", source_idx=-1, label=null.'
        )
        option_text = ""
    else:
        option_lines = "\n".join(f"{label}. {text}" for label, text in options)
        goal_instruction = (
            "Multiple-choice question: return ONE goals item per option "
            '(role="option", source_idx=-1, label = "A"/"B"/"C"/"D").'
        )
        option_text = f"\nOptions:\n{option_lines}\n"

    # Schema hint shown at the top of the user message — tells the LLM the
    # exact output shape without requiring it to read a full JSON Schema spec.
    schema_hint = (
        '{"predicates":{"predicate_name":1},'
        '"premises":[{"source_idx":0,"role":"premise","text":"...","formula":{...}}],'
        '"goals":[{"source_idx":-1,"role":"query","label":null,"text":"...","formula":{...}}]}'
    )

    # ── Few-shot example 1: ground facts + multi-condition rule + atom query ──
    # Teaches the LLM to represent direct assertions (Sophia completed X) as
    # ground atoms (not implications), and how to chain conditions with "and".
    example_1 = (
        "── EXAMPLE 1: ground facts + multi-condition rule (Yes/No question) ──\n"
        "Premises:\n"
        "0: Students who completed the core curriculum and passed the science assessment qualify for advanced courses.\n"
        "1: Sophia completed the core curriculum.\n"
        "2: Sophia passed the science assessment.\n"
        "Question: Does Sophia qualify for advanced courses?\n"
        "Output:\n"
        '{"predicates":{"completed_core":1,"passed_science":1,"qualifies_advanced":1},'
        '"premises":['
        # Premise 0: multi-condition implies → antecedent is an "and" node
        '{"source_idx":0,"role":"premise","text":"Students who completed the core curriculum and passed the science assessment qualify for advanced courses.",'
        '"formula":{"type":"implies",'
        '"antecedent":{"type":"and","args":['
        '{"type":"atom","pred":"completed_core","args":["?x"],"negated":false},'
        '{"type":"atom","pred":"passed_science","args":["?x"],"negated":false}'
        ']},'
        '"consequent":{"type":"atom","pred":"qualifies_advanced","args":["?x"],"negated":false}}},'
        # Premise 1 & 2: direct assertions → ground atoms (NOT wrapped in implies)
        '{"source_idx":1,"role":"premise","text":"Sophia completed the core curriculum.",'
        '"formula":{"type":"atom","pred":"completed_core","args":["sophia"],"negated":false}},'
        '{"source_idx":2,"role":"premise","text":"Sophia passed the science assessment.",'
        '"formula":{"type":"atom","pred":"passed_science","args":["sophia"],"negated":false}}'
        '],'
        # Goal: atom query with named constant, not variable
        '"goals":[{"source_idx":-1,"role":"query","label":null,"text":"Does Sophia qualify for advanced courses?",'
        '"formula":{"type":"atom","pred":"qualifies_advanced","args":["sophia"],"negated":false}}]}'
        "\n\n"
    )

    # ── Few-shot example 2: implication premises + MCQ options as implications ──
    # Teaches the LLM that MCQ options like "if not P then not Q" must be
    # represented as Implies nodes — NOT collapsed into a single atom.
    example_2 = (
        "── EXAMPLE 2: implication premises + MCQ options (including contrapositive) ──\n"
        "Premises:\n"
        "0: If a Python project is well-tested, it is optimized.\n"
        "Question: Which conclusion follows with the fewest premises?\n"
        "Options:\n"
        "A. If a Python project is not optimized, then it is not well-tested.\n"
        "B. If a Python project is optimized, then it is well-tested.\n"
        "Output:\n"
        '{"predicates":{"well_tested":1,"optimized":1},'
        '"premises":['
        '{"source_idx":0,"role":"premise","text":"If a Python project is well-tested, it is optimized.",'
        '"formula":{"type":"implies",'
        '"antecedent":{"type":"atom","pred":"well_tested","args":["?x"],"negated":false},'
        '"consequent":{"type":"atom","pred":"optimized","args":["?x"],"negated":false}}}'
        '],'
        '"goals":['
        # Option A: contrapositive — NOT optimized → NOT well_tested
        '{"source_idx":-1,"role":"option","label":"A","text":"If a Python project is not optimized, then it is not well-tested.",'
        '"formula":{"type":"implies",'
        '"antecedent":{"type":"atom","pred":"optimized","args":["?x"],"negated":true},'
        '"consequent":{"type":"atom","pred":"well_tested","args":["?x"],"negated":true}}},'
        # Option B: converse — optimized → well_tested (NOT the same as premise)
        '{"source_idx":-1,"role":"option","label":"B","text":"If a Python project is optimized, then it is well-tested.",'
        '"formula":{"type":"implies",'
        '"antecedent":{"type":"atom","pred":"optimized","args":["?x"],"negated":false},'
        '"consequent":{"type":"atom","pred":"well_tested","args":["?x"],"negated":false}}}'
        ']}'
        "\n\n"
    )

    # ── Few-shot example 3: typed numeric terms + explicit quantifiers ─────────
    example_3 = (
        "── EXAMPLE 3: numeric threshold + explicit quantifiers ──\n"
        "Premises:\n"
        "0: Alex has trained for 8 months.\n"
        "1: Every person with at least 6 months of training is an eligible trainer.\n"
        "Question: Is there an eligible trainer?\n"
        "Output:\n"
        '{"predicates":{"eligible_trainer":1},'
        '"premises":['
        '{"source_idx":0,"role":"premise","text":"Alex has trained for 8 months.",'
        '"formula":{"type":"compare","op":"=",'
        '"left":{"type":"function","name":"membership_duration","args":["alex"]},'
        '"right":{"type":"number","value":"8"}}},'
        '{"source_idx":1,"role":"premise","text":"Every person with at least 6 months of training is an eligible trainer.",'
        '"formula":{"type":"forall","variables":["?x"],"body":{"type":"implies",'
        '"antecedent":{"type":"compare","op":">=",'
        '"left":{"type":"function","name":"membership_duration","args":["?x"]},'
        '"right":{"type":"number","value":"6"}},'
        '"consequent":{"type":"atom","pred":"eligible_trainer","args":["?x"],"negated":false}}}}'
        '],'
        '"goals":[{"source_idx":-1,"role":"query","label":null,"text":"Is there an eligible trainer?",'
        '"formula":{"type":"exists","variables":["?x"],"body":'
        '{"type":"atom","pred":"eligible_trainer","args":["?x"],"negated":false}}}]}'
        "\n\n"
    )

    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. No extra text. "
                "Translate every premise and goal into a formula tree. "
                "Preserve logical structure exactly — do NOT answer or simplify."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Output valid JSON matching: {schema_hint}\n\n"
                f"{_TYPED_FORMULA_NODE_GUIDE}\n"
                "CRITICAL RULES:\n"
                "1. Use lowercase snake_case for predicate names and ground constants; "
                "   use ?x / ?y for universally quantified variables.\n"
                "2. Direct assertions ('Sophia completed X') → atom node with named constant (sophia), "
                "   NOT wrapped in implies. NEVER use implies for ground facts.\n"
                "3. 'If A and B then C' → implies(and(atom(A), atom(B)), atom(C)). "
                "   The antecedent is an 'and' node with ≥2 children.\n"
                "4. MCQ options that are implications MUST stay as implies nodes. "
                "   NEVER collapse 'if...then...' into a single atom.\n"
                "   Conversely, an MCQ option without an explicit condition MUST NOT become an implies node. "
                "   Translate a direct claim such as 'Sophia is eligible' as an atom.\n"
                "5. Preserve implication direction exactly: 'If A then B' → antecedent=A, consequent=B.\n"
                "6. Preserve explicit quantification: 'all/every/any' → forall; "
                "'some/there exists/at least one' → exists.\n"
                "7. Negation in a compound formula → Not node wrapping the sub-formula, "
                "   not just negated=true on the atom.\n"
                "8. predicates dict: map each predicate name to its arity (integer).\n"
                "9. Every premise must appear in 'premises' with its exact source_idx.\n"
                "10. Preserve OR as an or node and 'if and only if' as iff. Never drop a branch.\n"
                "11. Numeric properties such as GPA, score, duration, count, and thresholds "
                "must use function + compare nodes, not opaque predicate names.\n"
                f"12. {goal_instruction}\n\n"
                "FEW-SHOT EXAMPLES:\n"
                f"{example_1}"
                f"{example_2}"
                f"{example_3}"
                "Now translate:\n"
                f"Premises:\n{premise_text}\n"
                f"Question:\n{question}\n"
                f"{option_text}"
            ),
        },
    ]


def build_formula_premises_only_messages(
    premises: list[str],
) -> list[ChatCompletionMessageParam]:
    """Prompt for translating ONLY premises into formula-level IR (no goals).

    Produces a compact schema omitting the optional 'role' and 'text' fields to
    reduce output tokens — the caller supplies those from the original NL text.
    This is used by the split-translation cache path: premise formulas are
    translated once and reused across all questions in the same premise group.
    """

    premise_text = "\n".join(f"{idx}: {p}" for idx, p in enumerate(premises))

    schema_hint = (
        '{"predicates":{"pred_name":1},'
        '"premises":[{"source_idx":0,"formula":{...}}]}'
    )

    # Compact few-shot: only shows premises side, no goals. Teaches the LLM to
    # output minimal JSON (no text/role fields) to keep token count low.
    example = (
        "── EXAMPLE: translate premises only ──\n"
        "Premises:\n"
        "0: Students who completed the core curriculum and passed the science assessment qualify for advanced courses.\n"
        "1: Sophia completed the core curriculum.\n"
        "2: Sophia passed the science assessment.\n"
        "Output:\n"
        '{"predicates":{"completed_core":1,"passed_science":1,"qualifies_advanced":1},'
        '"premises":['
        '{"source_idx":0,"formula":{"type":"implies",'
        '"antecedent":{"type":"and","args":['
        '{"type":"atom","pred":"completed_core","args":["?x"],"negated":false},'
        '{"type":"atom","pred":"passed_science","args":["?x"],"negated":false}'
        ']},'
        '"consequent":{"type":"atom","pred":"qualifies_advanced","args":["?x"],"negated":false}}},'
        '{"source_idx":1,"formula":{"type":"atom","pred":"completed_core","args":["sophia"],"negated":false}},'
        '{"source_idx":2,"formula":{"type":"atom","pred":"passed_science","args":["sophia"],"negated":false}}'
        ']}\n\n'
    )

    typed_example = (
        "── EXAMPLE: typed numeric premise ──\n"
        "Premises:\n"
        "0: Alex has trained for 8 months.\n"
        "1: Every person with at least 6 months of training is an eligible trainer.\n"
        "Output:\n"
        '{"predicates":{"eligible_trainer":1},'
        '"premises":['
        '{"source_idx":0,"formula":{"type":"compare","op":"=",'
        '"left":{"type":"function","name":"membership_duration","args":["alex"]},'
        '"right":{"type":"number","value":"8"}}},'
        '{"source_idx":1,"formula":{"type":"forall","variables":["?x"],"body":{"type":"implies",'
        '"antecedent":{"type":"compare","op":">=",'
        '"left":{"type":"function","name":"membership_duration","args":["?x"]},'
        '"right":{"type":"number","value":"6"}},'
        '"consequent":{"type":"atom","pred":"eligible_trainer","args":["?x"],"negated":false}}}}'
        ']}\n\n'
    )

    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. No extra text. "
                "Translate every premise into a formula tree. "
                "Omit 'role' and 'text' fields — output only source_idx and formula."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Output valid JSON matching: {schema_hint}\n\n"
                f"{_TYPED_FORMULA_NODE_GUIDE}\n"
                "CRITICAL RULES:\n"
                "1. lowercase snake_case for predicate names and constants; ?x/?y for variables.\n"
                "2. Direct assertions ('Sophia completed X') → atom with named constant, NOT implies.\n"
                "3. 'If A and B then C' → implies(and(atom(A,?x), atom(B,?x)), atom(C,?x)).\n"
                "4. Preserve explicit forall/exists scope, OR branches, iff, and numeric comparisons.\n"
                "5. Omit 'text' and 'role' fields — only source_idx and formula are required.\n"
                "6. predicates dict: map each predicate name to its integer arity.\n"
                "7. Every premise must appear with its exact source_idx.\n\n"
                f"{example}"
                f"{typed_example}"
                "Now translate:\n"
                f"Premises:\n{premise_text}"
            ),
        },
    ]


def build_formula_goals_messages(
    question: str,
    options: list[tuple[str, str]] | None,
    predicate_dict: dict[str, int],
    entity_constants: tuple[str, ...],
) -> list[ChatCompletionMessageParam]:
    """Prompt for translating ONLY goals (query or MCQ options) into formula IR.

    Receives the predicate dictionary produced by the premises-only call so all
    goal atom predicates come from the same shared vocabulary — preventing the
    drift that would occur if goals were translated without knowledge of the
    premise predicates.
    """

    # Predicate vocabulary constraint — force the LLM to stay within the dict.
    pred_list = ", ".join(
        f"{name}/{arity}" for name, arity in sorted(predicate_dict.items())
    ) or "(none — infer from question)"
    entity_list = ", ".join(entity_constants) or "(none — use ?x for generic)"

    if options is None:
        goal_instruction = (
            "Yes/No/Unknown question: return exactly ONE goal "
            'with role="query", source_idx=-1, label=null.'
        )
        option_text = ""
        schema_hint = (
            '{"goals":[{"source_idx":-1,"role":"query","label":null,"text":"...","formula":{...}}]}'
        )
        # Compact few-shot for query goal
        example = (
            "── EXAMPLE: Yes/No query goal ──\n"
            "Predicate dict: qualifies_advanced/1\n"
            "Entity constants: sophia\n"
            "Question: Does Sophia qualify for advanced courses?\n"
            "Output:\n"
            '{"goals":[{"source_idx":-1,"role":"query","label":null,'
            '"text":"Does Sophia qualify for advanced courses?",'
            '"formula":{"type":"atom","pred":"qualifies_advanced","args":["sophia"],"negated":false}}]}\n\n'
        )
    else:
        option_lines = "\n".join(f"{label}. {text}" for label, text in options)
        option_text = f"\nOptions:\n{option_lines}"
        goal_instruction = (
            "Multiple-choice question: return ONE goal per option "
            '(role="option", label="A"/"B"/"C"/"D").'
        )
        schema_hint = (
            '{"goals":['
            '{"source_idx":-1,"role":"option","label":"A","text":"...","formula":{...}},'
            '{"source_idx":-1,"role":"option","label":"B","text":"...","formula":{...}}'
            ']}'
        )
        # Compact few-shot for MCQ goals (includes implies option to teach formula trees)
        example = (
            "── EXAMPLE: MCQ option goals ──\n"
            "Predicate dict: well_tested/1, optimized/1\n"
            "Entity constants: (none)\n"
            "Question: Which is correct?\n"
            "Options:\n"
            "A. If a project is not optimized, it is not well-tested.\n"
            "B. Optimized projects are well-tested.\n"
            "Output:\n"
            '{"goals":['
            '{"source_idx":-1,"role":"option","label":"A",'
            '"text":"If a project is not optimized, it is not well-tested.",'
            '"formula":{"type":"implies",'
            '"antecedent":{"type":"atom","pred":"optimized","args":["?x"],"negated":true},'
            '"consequent":{"type":"atom","pred":"well_tested","args":["?x"],"negated":true}}},'
            '{"source_idx":-1,"role":"option","label":"B",'
            '"text":"Optimized projects are well-tested.",'
            '"formula":{"type":"implies",'
            '"antecedent":{"type":"atom","pred":"optimized","args":["?x"],"negated":false},'
            '"consequent":{"type":"atom","pred":"well_tested","args":["?x"],"negated":false}}}'
            ']}\n\n'
        )

    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. No extra text. "
                "Translate the question/options into goal formula trees using ONLY the provided predicates."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Output valid JSON matching: {schema_hint}\n\n"
                f"{_TYPED_FORMULA_NODE_GUIDE}\n"
                "CRITICAL RULES:\n"
                f"1. ONLY use predicates from this dict: {pred_list}\n"
                f"2. Known entity constants (named individuals from premises): {entity_list}\n"
                "   - If the question asks about a NAMED INDIVIDUAL (e.g. 'Does David qualify?')\n"
                "     → use that entity as a constant: args=[\"david\"], NOT args=[\"?x\"].\n"
                "   - Use ?x ONLY for universal statements ('all students', 'any project').\n"
                "3. MCQ options that are implications MUST be implies nodes — never collapse to atom.\n"
                "   An option without an explicit condition MUST NOT become an implies node. "
                "   Direct claims such as 'Sophia is eligible' are atom nodes.\n"
                "4. 'If not A then not B' → implies(atom(A,negated=true), atom(B,negated=true)).\n"
                "5. Preserve forall/exists scope, OR branches, iff, and numeric comparisons.\n"
                f"6. {goal_instruction}\n\n"
                f"{example}"
                "Now translate:\n"
                f"Question:\n{question}"
                f"{option_text}"
            ),
        },
    ]


def build_full_translation_messages(
    premises: list[str],
    question: str,
) -> list[ChatCompletionMessageParam]:
    """Build a compact autoformalization prompt for a local <=8B LLM."""

    premise_text = "\n".join(f"{idx}: {premise}" for idx, premise in enumerate(premises))
    schema_hint = (
        '{"predicates":[{"name":"predicate_name","arity":1}],'
        '"premises":[{"source_idx":0,"facts":[{"pred":"predicate_name","args":["item"],"negated":false}],'
        '"rules":[{"conditions":[{"pred":"condition_name","args":["?x"],"negated":false}],'
        '"conclusion":{"pred":"predicate_name","args":["?x"],"negated":false}}]}],'
        '"query":{"claim":{"pred":"predicate_name","args":["sophia"],"negated":false}},'
        '"options":[]}'
    )
    examples = (
        "Example 1:\n"
        "Premise 0: If a student completes assignments, the student passes.\n"
        "Premise 1: Sophia completes assignments.\n"
        "Question: Does Sophia pass?\n"
        "JSON: {\"predicates\":[{\"name\":\"completes_assignments\",\"arity\":1},{\"name\":\"passes\",\"arity\":1}],"
        "\"premises\":[{\"source_idx\":0,\"facts\":[],\"rules\":[{\"conditions\":[{\"pred\":\"completes_assignments\",\"args\":[\"?x\"],\"negated\":false}],"
        "\"conclusion\":{\"pred\":\"passes\",\"args\":[\"?x\"],\"negated\":false}}]},"
        "{\"source_idx\":1,\"facts\":[{\"pred\":\"completes_assignments\",\"args\":[\"sophia\"],\"negated\":false}],\"rules\":[]}],"
        "\"query\":{\"claim\":{\"pred\":\"passes\",\"args\":[\"sophia\"],\"negated\":false}},\"options\":[]}\n\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. Keep it compact and valid. Do not use markdown fences. Do not answer the question. "
                "Translate text into Horn-style predicates for a symbolic solver. "
                "Never put JSON objects inside args; args must be strings only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Task: build one predicate dictionary, then formalize premises, query, and MCQ options.\n"
                "Rules:\n"
                "- Output valid JSON only, matching this shape: "
                f"{schema_hint}\n"
                "- Keep output compact; omit atom text fields.\n"
                "- predicates may contain only name and arity; omit gloss and argument_roles unless needed.\n"
                "- Reuse predicate names from predicates everywhere; never invent variants.\n"
                "- pred and constants must be lowercase snake_case; variables use ?x, ?y.\n"
                "- args must be arrays of strings only, never nested JSON objects.\n"
                "- Do not use pred values `and`, `or`, `either`, or `not`; split conjunctions and use negated=true for negation.\n"
                "- Generic rules use variables; named facts/goals use constants.\n"
                "- Standalone assertions go in facts, not rules.\n"
                "- Every rule must have at least one condition; never output conditions:[].\n"
                "- Split conjunctions into separate condition atoms.\n"
                "- If a premise has alternatives with `or`, keep only Horn-compatible direct conditions and do not model disjunction.\n"
                "- Preserve source_idx exactly.\n"
                "- Do not translate A-D options; always set options to []. The pipeline evaluates MCQ options separately.\n"
                "- Mark negated=true only for explicit negation.\n\n"
                f"{examples}\n"
                f"Premises:\n{premise_text}\n\nQuestion:\n{question}"
            ),
        },
    ]


def build_mcq_options_messages(
    question: str,
    predicate_names: tuple[str, ...],
) -> list[ChatCompletionMessageParam]:
    """Prompt to translate each MCQ option text into a KB atom."""

    schema_hint = (
        '{"options":[{"label":"A","claim":{"pred":"name","args":["?x"],"negated":false}},'
        '{"label":"B","claim":{"pred":"name","args":["?x"],"negated":false}}]}'
    )
    predicate_list = ", ".join(predicate_names) if predicate_names else "(use descriptive snake_case)"
    return [
        {
            "role": "system",
            "content": (
                "You are an autoformalizer for educational logic QA. "
                "Return JSON only. No markdown fences. "
                "Map each MCQ option to ONE atom from the given predicate list."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate each MCQ option into a Horn atom. Output JSON matching: {schema_hint}\n"
                "Rules:\n"
                f"- Available predicates: {predicate_list}\n"
                "- Pick the predicate that best captures the option's core claim\n"
                "- negated=true if the option asserts the predicate does NOT hold\n"
                "- args: '?x' for generic entity, snake_case constant for a named entity\n"
                "- Include all options (A, B, C, D) that appear in the question\n\n"
                f"Question:\n{question}"
            ),
        },
    ]
