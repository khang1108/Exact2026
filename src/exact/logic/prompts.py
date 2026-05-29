"""Prompt builders for the Type 1 LLM autoformalizer."""

from __future__ import annotations

from openai.types.chat import ChatCompletionMessageParam


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
                "Formula node shapes (use 'type' as the op key):\n"
                '  atom:    {"type":"atom","pred":"snake_case","args":["?x"],"negated":false}\n'
                '  not:     {"type":"not","arg":FORMULA}\n'
                '  and:     {"type":"and","args":[FORMULA,FORMULA,...]}\n'
                '  or:      {"type":"or","args":[FORMULA,FORMULA,...]}\n'
                '  implies: {"type":"implies","antecedent":FORMULA,"consequent":FORMULA}\n'
                "            (aliases accepted: lhs/rhs instead of antecedent/consequent)\n\n"
                "CRITICAL RULES:\n"
                "1. Use lowercase snake_case for predicate names and ground constants; "
                "   use ?x / ?y for universally quantified variables.\n"
                "2. Direct assertions ('Sophia completed X') → atom node with named constant (sophia), "
                "   NOT wrapped in implies. NEVER use implies for ground facts.\n"
                "3. 'If A and B then C' → implies(and(atom(A), atom(B)), atom(C)). "
                "   The antecedent is an 'and' node with ≥2 children.\n"
                "4. MCQ options that are implications MUST stay as implies nodes. "
                "   NEVER collapse 'if...then...' into a single atom.\n"
                "5. Preserve implication direction exactly: 'If A then B' → antecedent=A, consequent=B.\n"
                "6. 'All X are Y' (universal) → implies(atom(X,?x), atom(Y,?x)) with variable ?x.\n"
                "7. Negation in a compound formula → Not node wrapping the sub-formula, "
                "   not just negated=true on the atom.\n"
                "8. predicates dict: map each predicate name to its arity (integer).\n"
                "9. Every premise must appear in 'premises' with its exact source_idx.\n"
                f"10. {goal_instruction}\n\n"
                "FEW-SHOT EXAMPLES:\n"
                f"{example_1}"
                f"{example_2}"
                "Now translate:\n"
                f"Premises:\n{premise_text}\n"
                f"Question:\n{question}\n"
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
