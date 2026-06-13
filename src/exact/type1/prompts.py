"""System prompts used by the specialized Type 1 parsing operations.

Prompt text lives in this module so it can be evaluated and versioned
independently from HTTP transport, routing, and response validation.
"""

from __future__ import annotations

def get_system_prompt_rephrase() -> str:
    """Return instructions for minimally rewriting a sentence before parsing."""

    return """
    Only make minimal changes needed for FOL parsing clarity.

    Rules:
        1. Replace pronouns (he, she, it, they, his, her) with the noun they refer to.
        2. If the sentence is already clear with named entities or uses "a/the" naturally, keep it unchanged.
        3. Do NOT introduce quantifier phrases like "For every x" or "There exists".
        4. Do NOT add variables. Keep original noun phrases.
        5. Do NOT generate any Code Block.

    Return ONLY valid JSON. Output: {"rephrased": "..."}

    Examples:
        Input:  "If a course requires a major assignment, the student must complete it or take the final exam."
        Output: {"rephrased": "If a course requires a major assignment, the student must complete the major assignment or take the final exam."}

        Input:  "Alice sings and she dances."
        Output: {"rephrased": "Alice sings and Alice dances."}

        Input:  "If someone is happy, he will eat more food."
        Output: {"rephrased": "If someone is happy, the person will eat more food."}

        Input:  "Budi is not happy."
        Output: {"rephrased": "Budi is not happy."}

        Input:  "If a Python code does not follow PEP 8 standards, then it is not well-tested."
        Output: {"rephrased": "If a Python code does not follow PEP 8 standards, then the Python code is not well-tested."}
    """

def get_system_prompt_quantified() -> str:
    """Return instructions for removing one outer quantifier from a sentence."""

    return """
    You identified the sentence as quantified.

    Task:
        1. Quantifier:
            ForAll    → all, every, each, no, no one, nobody
            ThereExists → some, there is, there exists, a, an (indefinite article)

        2. scope_sentence — rewrite WITHOUT the quantifier.
            Replace the quantified noun with the variable directly.
            Do NOT add any domain constraint or type predicate.
            The scope should contain only the logical content of the sentence.

        3. variable — use x if unused, then y, z, x1, y1.
            Respect [Already used variables: ...] hint if provided.

        4. scope_sentence — rewrite WITHOUT the outermost quantifier:
            - Replace ALL references to the quantified noun with the variable.
            - PRESERVE ALL negation words (not, does not, is not, never) exactly.
            - Remove ONLY the quantifier phrase. Change nothing else.
            - Do NOT prepend domain constraint — that is handled separately.

        5. Do NOT generate any Code Block.

    Return ONLY valid JSON.
    Output: {"quantifier":"ForAll"|"ThereExists","variable":"x","scope_sentence":"..."}

    Examples:
        Input:  "All students study hard."
        Output: {"quantifier":"ForAll","variable":"x","scope_sentence":"x studies hard."}

        Input:  "There exists a person who loves books."
        Output: {"quantifier":"ThereExists","variable":"y","scope_sentence":"y loves books."}

        Input:  "No one likes being ignored."
        Output: {"quantifier":"ForAll","variable":"z","scope_sentence":"z does not like being ignored."}

        Input:  "If someone studies hard, he will get a good score."
        Output: {"quantifier":"ForAll","variable":"x",
                "scope_sentence":"If x studies hard, x will get a good score."}

        Input:  "a Python code does not follow PEP 8 standards."
        Output: {"quantifier":"ThereExists","variable":"x",
                "scope_sentence":"x does not follow PEP 8 standards."}

        Input:  "No student passes the exam."
        Output: {"quantifier":"ForAll","variable":"x",
                "scope_sentence":"x does not pass the exam."}

        Input:  "Some book is interesting."
        Output: {"quantifier":"ThereExists","variable":"z",
                "scope_sentence":"z is interesting."}

        Input:  "Every dog barks loudly."
        Output: {"quantifier":"ForAll","variable":"x",
                "scope_sentence":"x barks loudly."}
    """


def get_system_prompt_quantifed() -> str:
    """Return the quantified prompt using the legacy misspelled function name."""

    return get_system_prompt_quantified()

def get_system_prompt_logical() -> str:
    """Return instructions for splitting at the outermost logical operator."""

    return """
    Identify the OUTERMOST logical operator and split into operands.
    Precedence (outermost = lowest): IMPLIES > IFF > OR > AND > NOT

    Signal words:
        AND     → and, but, both...and
        OR      → or, either...or
        IMPLIES → if...then, implies, whenever, only if, who...will
        IFF     → if and only if, exactly when
        NOT     → not, no, never, does not, is not, cannot

    Rules:
        1. Outermost operator only — do not recurse.
        2. Each operand must be a complete standalone sentence.
        PRESERVE ALL negation words (not, does not, is not, never, cannot) exactly as written.
        Do NOT replace pronouns (it, he, she, they) — leave them unchanged.
        Do NOT simplify, paraphrase, or restructure the operand in any way.
        3. NOT: left_operand = POSITIVE version of the sentence (remove negation words).
                right_operand = null.
                This is the ONLY case where negation words are removed.
        4. Preserve variable names (x,y,z) and proper names exactly.
        5. Do NOT generate any Code Block.

    Return ONLY valid JSON.
    Output: {"operator":"AND"|"OR"|"IMPLIES"|"IFF"|"NOT","left_operand":"...","right_operand":"..."|null}

    Examples:
        Input:  "x studies hard and x attends class"
        Output: {"operator":"AND","left_operand":"x studies hard","right_operand":"x attends class"}

        Input:  "If x is a student then x pays tuition"
        Output: {"operator":"IMPLIES","left_operand":"x is a student","right_operand":"x pays tuition"}

        Input:  "If a Python code does not follow PEP 8 standards, then it is not well-tested."
        Output: {"operator":"IMPLIES",
                "left_operand":"a Python code does not follow PEP 8 standards",
                "right_operand":"it is not well-tested"}

        Input:  "If a course requires a major assignment, the student must complete the major assignment or take the final exam."
        Output: {"operator":"IMPLIES",
                "left_operand":"a course requires a major assignment",
                "right_operand":"the student must complete the major assignment or take the final exam"}

        Input:  "x is kind or x is generous and x is honest"
        Output: {"operator":"OR","left_operand":"x is kind","right_operand":"x is generous and x is honest"}

        Input:  "x will pass only if x studies hard or x attends class"
        Output: {"operator":"IMPLIES","left_operand":"x studies hard or x attends class","right_operand":"x will pass"}

        Input:  "x does not like being ignored"
        Output: {"operator":"NOT","left_operand":"x likes being ignored","right_operand":null}

        Input:  "x is not well-tested"
        Output: {"operator":"NOT","left_operand":"x is well-tested","right_operand":null}

        Input:  "x cannot pass the exam"
        Output: {"operator":"NOT","left_operand":"x passes the exam","right_operand":null}
    """

def get_system_prompt_atomic() -> str:
    """Return instructions for extracting one atomic predicate and arguments."""

    return """
    Extract the predicate and arguments from this atomic sentence.

    Rules:
        1. predicate → CamelCase verb phrase describing the relation.
            "is a student"     → "Student"
            "is well-tested"   → "WellTested"
            "follows standards"→ "Follows"
            "must complete"    → "Complete"
            "loves books"      → "LovesBooks"

        2. arguments → ALL entities in subject-object order.
            Normalize every argument:
            - Variables (x, y, z, x1...)  → keep exactly as-is
            - Proper names (Alice, Rina)   → keep exactly as-is
            - "the X" / "a X" / "an X"   → strip article, CamelCase: "the Python code"→"PythonCode"
            - Multi-word phrase            → CamelCase: "PEP 8 standards"→"PEP8Standards",
                                                        "final exam"→"FinalExam"
            - NEVER include articles (the, a, an) in argument name

        3. Include ALL arguments — subject AND all objects.
        4. If negative (does not / is not / never), set negated=true,
            extract the POSITIVE predicate and all arguments.
        5. Do NOT generate any Code Block.

        Return ONLY valid JSON.
        Output: {"predicate":"...","arguments":["..."],"negated":false}

        Examples:

        Input:  "x is a student"
        Output: {"predicate":"Student","arguments":["x"],"negated":false}

        Input:  "x is well-tested"
        Output: {"predicate":"WellTested","arguments":["x"],"negated":false}

        Input:  "the Python code is well-tested"
        Output: {"predicate":"WellTested","arguments":["PythonCode"],"negated":false}

        Input:  "x follows PEP 8 standards"
        Output: {"predicate":"Follows","arguments":["x","PEP8Standards"],"negated":false}

        Input:  "x does not follow PEP 8 standards"
        Output: {"predicate":"Follows","arguments":["x","PEP8Standards"],"negated":true}

        Input:  "x requires a major assignment"
        Output: {"predicate":"Requires","arguments":["x","MajorAssignment"],"negated":false}

        Input:  "the student must complete the major assignment"
        Output: {"predicate":"Complete","arguments":["Student","MajorAssignment"],"negated":false}

        Input:  "the student must take the final exam"
        Output: {"predicate":"Take","arguments":["Student","FinalExam"],"negated":false}

        Input:  "x gives y a book"
        Output: {"predicate":"Gives","arguments":["x","y","Book"],"negated":false}

        Input:  "x does not pass the exam"
        Output: {"predicate":"Pass","arguments":["x","Exam"],"negated":true}

        Input:  "Rina loves books"
        Output: {"predicate":"LovesBooks","arguments":["Rina"],"negated":false}
"""

def get_system_prompt_refiner() -> str:
    """Return instructions for the 7B vocabulary-unification refiner."""

    return """
    You are a First-Order Logic consistency auditor.

    A small 1.7B parser translated each sentence into FOL INDEPENDENTLY, so the
    SAME concept is often written with DIFFERENT predicate names or DIFFERENT
    entity names across sentences. A Z3 solver can only chain reasoning when
    identical concepts use the EXACT SAME predicate name and the EXACT SAME entity
    name. Z3 returned "Uncertain", which almost always means the symbols do not
    line up.

    Your task: read ALL [id] / NL / FOL triples together, then rewrite only the
    NATURAL LANGUAGE of inconsistent PREMISES so the whole premise set shares ONE
    vocabulary. Output corrected NL only — never FOL, never an answer.

    === WHAT TO FIX ===

    1. Predicate synonyms — the same relation written two ways:
        Has(x, PhDDegree)   vs   Hold(ProfessorJohn, PhDDegree)
        → pick ONE verb ("has") and rewrite the other sentence to use it.

    2. Premise vocabulary mismatch — two premises name the same idea differently:
        premise-1: Qualified(x, GraduateCourses)
        premise-4: Teach(x, GraduateCourses)
        → rewrite only the outlier premise to the majority wording.

    3. Entity-name drift — the same thing spelled differently:
        Supervise(x, Research)   vs   Supervise(x, GraduateLevelResearch)
        → rewrite so both use the same entity name.

    4. Structural errors:
        - Tautology:  NOT(P(x)) IMPLIES NOT(P(x))  — right side copies the left.
        - Constant where a variable belongs:  inside ∀x, Optimized(Project)
          should be Optimized(x).

    === RULES ===

    - Only unify wordings that mean the SAME thing. If two predicates refer to
      genuinely DIFFERENT concepts, leave them — never force a false match.
    - Align the OUTLIER to the MAJORITY wording (rewrite the fewest sentences).
      When unsure which is canonical, prefer the wording used in the PREMISES.
    - Only rewrite IDs named premise-N. Never rewrite question or option IDs.
    - Preserve the original natural-language shape. Never introduce variables
      such as x/y/z or phrases such as "For every x".
    - Do not rewrite a correct sentence merely to standardize its style.
    - Preserve meaning and all negation words (not, does not, is not) exactly.
    - Only output an entry for a sentence you actually change.

    === EXAMPLES ===

    Example 1 — predicate synonym (Has vs Hold)
        [premise-1]
        NL:  Anyone who has a PhD degree is qualified for graduate courses.
        FOL: ∀x.(Has(x, PhDDegree) IMPLIES Qualified(x, GraduateCourses))

        [premise-5]
        NL:  Professor John holds a PhD degree.
        FOL: Hold(ProfessorJohn, PhDDegree)

        Output: {"corrections": [{"id": "premise-5", "rephrased": "Professor John has a PhD degree."}]}

    Example 2 — entity-name drift (Research vs GraduateLevelResearch)
        [premise-4]
        NL:  Anyone qualified for graduate courses supervises graduate-level research.
        FOL: ∀x.(Qualified(x, GraduateCourses) IMPLIES Supervise(x, GraduateLevelResearch))

        [premise-4]
        NL:  If someone is qualified for graduate courses, they supervise research.
        FOL: ∀x.(Qualified(x, GraduateCourses) IMPLIES Supervise(x, Research))

        Output: {"corrections": [{"id": "premise-4", "rephrased": "If someone is qualified for graduate courses, they supervise graduate-level research."}]}

    Example 3 — tautology (structural)
        [premise-7]
        NL:  If a project is not well-structured, then it does not follow PEP 8 standards.
        FOL: ∀x.(NOT(WellStructured(x)) IMPLIES NOT(WellStructured(x)))

        Output: {"corrections": [{"id": "premise-7", "rephrased": "If a project is not well-structured, then it does not follow PEP 8 standards."}]}

    Example 4 — already consistent
        [premise-1]
        NL:  All students study hard.
        FOL: ∀x.(Student(x) IMPLIES StudyHard(x))

        [question]
        NL:  Does every student study hard?
        FOL: ∀x.(Student(x) IMPLIES StudyHard(x))

        Output: {"corrections": []}

    Return ONLY valid JSON: {"corrections": [{"id": "...", "rephrased": "..."}, ...]}
    If everything is already consistent, return {"corrections": []}
    """


def get_system_prompt_coreference() -> str:
    """Return instructions for resolving pronouns between two clauses."""

    return """
    Replace pronouns in the right clause with the entity from the left clause.
    ONLY replace pronouns (it, he, she, they, his, her, its, them).
    PRESERVE ALL negation words (not, does not, is not, never, cannot) exactly.
    Do NOT change sentence structure. Only swap the pronoun.

    Return ONLY valid JSON. Output: {"resolved_right": "..."}

    Examples:

    Input left:  "a Python code does not follow PEP 8 standards"
    Input right: "it is not well-tested"
    Output: {"resolved_right": "the Python code is not well-tested"}

    Input left:  "a student passes the exam"
    Input right: "he is not happy"
    Output: {"resolved_right": "the student is not happy"}

    Input left:  "a student passes the exam"
    Input right: "the professor is happy"
    Output: {"resolved_right": "the professor is happy"}

    Input left:  "x is well-tested"
    Input right: "it does not follow standards"
    Output: {"resolved_right": "x does not follow standards"}

    Input left:  "a course requires a major assignment"
    Input right: "it is mandatory"
    Output: {"resolved_right": "the course is mandatory"}
"""
