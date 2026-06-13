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

        2. variable — use x if unused, then y, z, x1, y1.
            Respect [Already used variables: ...] hint if provided.

        3. restrictor_sentence — the noun class or type that restricts the domain.
            - Write as "x is a [noun]" using the chosen variable.
            - Set to null when the quantifier ranges over ALL entities with no domain
              restriction (e.g. "no one", "everything", "something", "someone").

        4. scope_sentence — rewrite WITHOUT the outer quantifier AND the noun class.
            - Replace ALL references to the quantified noun with the variable.
            - PRESERVE ALL negation words (not, does not, is not, never) exactly.
            - Do NOT include the noun class — it belongs in restrictor_sentence.
            - Remove ONLY the quantifier phrase and its noun class. Change nothing else.

        5. Do NOT generate any Code Block.

    Return ONLY valid JSON.
    Output: {"quantifier":"ForAll"|"ThereExists","variable":"x","restrictor_sentence":"..."|null,"scope_sentence":"..."}

    Examples:
        Input:  "All students study hard."
        Output: {"quantifier":"ForAll","variable":"x","restrictor_sentence":"x is a student","scope_sentence":"x studies hard"}

        Input:  "There exists a person who loves books."
        Output: {"quantifier":"ThereExists","variable":"y","restrictor_sentence":"y is a person","scope_sentence":"y loves books"}

        Input:  "No one likes being ignored."
        Output: {"quantifier":"ForAll","variable":"z","restrictor_sentence":null,"scope_sentence":"z does not like being ignored"}

        Input:  "If someone studies hard, he will get a good score."
        Output: {"quantifier":"ForAll","variable":"x","restrictor_sentence":null,"scope_sentence":"if x studies hard, x will get a good score"}

        Input:  "a Python code does not follow PEP 8 standards."
        Output: {"quantifier":"ThereExists","variable":"x","restrictor_sentence":"x is a Python code","scope_sentence":"x does not follow PEP 8 standards"}

        Input:  "No student passes the exam."
        Output: {"quantifier":"ForAll","variable":"x","restrictor_sentence":"x is a student","scope_sentence":"x does not pass the exam"}

        Input:  "Some book is interesting."
        Output: {"quantifier":"ThereExists","variable":"z","restrictor_sentence":"z is a book","scope_sentence":"z is interesting"}

        Input:  "Every dog barks loudly."
        Output: {"quantifier":"ForAll","variable":"x","restrictor_sentence":"x is a dog","scope_sentence":"x barks loudly"}
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

def get_system_prompt_premise_frame() -> str:
    """Return instructions for decomposing a premise into its logical frame."""

    return """
Decompose a natural-language premise into its logical structure. Do NOT generate FOL.

# TASK
Identify the premise kind and extract its structural components.
All text fragments must reference the entity using the stated variable (e.g. "x is a student").

# PREMISE KINDS
- fact            : ground truth about specific named individuals (no universal quantifier)
- universal_rule  : "All/every/each X that satisfies conditions → conclusions"
- existential_fact: "There exists / some X with properties"
- equivalence     : "X is P if and only if X is Q"
- numeric_fact    : ground fact involving a count or number about a named individual
- numeric_rule    : universal rule with a numeric threshold condition
- deontic_rule    : rule with must/should/required
- permission_rule : rule with can/may/allowed
- prohibition_rule: rule with cannot/prohibited/must not
- temporal_rule   : rule with before/after/until/during/when
- meta_rule       : rule about how other rules interact (too complex to decompose)
- unsupported     : anything that cannot be cleanly decomposed

# OUTPUT FORMAT (return ONLY valid JSON)
{
  "kind": "<kind>",
  "variable": "<single lowercase letter or null for fact>",
  "restrictor_text": "<sentence with variable typing the domain, e.g. 'x is a student', or null>",
  "condition_texts": ["<extra condition 1 with variable>", ...],
  "conclusion_texts": ["<conclusion 1 with variable>", ...],
  "fact_texts": ["<atomic fact about named individual>", ...],
  "numeric_constraints": ["<numeric condition with variable>", ...],
  "temporal_constraints": ["<temporal condition with variable>", ...],
  "modality": "none"|"must"|"can"|"may"|"allowed"|"required"|"prohibited"|"not_necessarily",
  "confidence": 1.0
}

# RULES
1. Every text fragment must be a single simple sentence — no "if/then", no sub-clauses.
2. Use the variable consistently across ALL fragments.
3. restrictor_text = the TYPE or CATEGORY of the subject ("x is a student").
4. condition_texts = extra conditions besides the type that the subject must satisfy.
5. conclusion_texts = what is concluded about the subject.
6. fact_texts = atomic facts about named individuals (for "fact"/"numeric_fact" only).
7. numeric_constraints = numeric thresholds ("x has completed at least 5 courses").
8. temporal_constraints = temporal conditions ("x enrolled before the deadline").
9. Do NOT include quantifier phrases in text fragments.
10. One simple predication per item — do NOT combine multiple facts in one string.

# EXAMPLES

Input: "All students must pass the final exam."
Output: {"kind":"deontic_rule","variable":"x","restrictor_text":"x is a student","condition_texts":[],"conclusion_texts":["x must pass the final exam"],"fact_texts":[],"numeric_constraints":[],"temporal_constraints":[],"modality":"must","confidence":1.0}

Input: "Students with active status who have completed at least 5 courses are eligible for advanced classes."
Output: {"kind":"numeric_rule","variable":"x","restrictor_text":"x is a student","condition_texts":["x has active status"],"conclusion_texts":["x is eligible for advanced classes"],"fact_texts":[],"numeric_constraints":["x has completed at least 5 courses"],"temporal_constraints":[],"modality":"none","confidence":1.0}

Input: "Alice is a graduate student."
Output: {"kind":"fact","variable":null,"restrictor_text":null,"condition_texts":[],"conclusion_texts":[],"fact_texts":["Alice is a graduate student"],"numeric_constraints":[],"temporal_constraints":[],"modality":"none","confidence":1.0}

Input: "Every student who has an active status and has paid tuition can enroll in courses."
Output: {"kind":"permission_rule","variable":"x","restrictor_text":"x is a student","condition_texts":["x has an active status","x has paid tuition"],"conclusion_texts":["x can enroll in courses"],"fact_texts":[],"numeric_constraints":[],"temporal_constraints":[],"modality":"can","confidence":1.0}

Input: "A course is required if and only if it is in the core curriculum."
Output: {"kind":"equivalence","variable":"x","restrictor_text":"x is a course","condition_texts":["x is in the core curriculum"],"conclusion_texts":["x is required"],"fact_texts":[],"numeric_constraints":[],"temporal_constraints":[],"modality":"none","confidence":1.0}

Input: "There exists a student who has completed all required courses."
Output: {"kind":"existential_fact","variable":"x","restrictor_text":"x is a student","condition_texts":["x has completed all required courses"],"conclusion_texts":[],"fact_texts":[],"numeric_constraints":[],"temporal_constraints":[],"modality":"none","confidence":1.0}

Input: "Students who enroll after the add/drop deadline cannot withdraw without penalty."
Output: {"kind":"prohibition_rule","variable":"x","restrictor_text":"x is a student","condition_texts":[],"conclusion_texts":["x cannot withdraw without penalty"],"fact_texts":[],"numeric_constraints":[],"temporal_constraints":["x enrolled after the add/drop deadline"],"modality":"prohibited","confidence":1.0}
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
