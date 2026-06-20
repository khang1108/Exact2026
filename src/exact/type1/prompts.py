"""System prompts used by the specialized Type 1 parsing operations.

Prompt text lives in this module so it can be evaluated and versioned
independently from HTTP transport, routing, and response validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exact.type1.parser.schemas import PremiseSchema

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
        IMPLIES → if...then, implies, whenever, only if, only when, who...will
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
        5. DIRECTION of "only if" / "only when":
                "P only if Q"   → left_operand = P, right_operand = Q  (P → Q)
                "P only when Q" → left_operand = P, right_operand = Q  (P → Q)
                This is the OPPOSITE of "if Q then P" — do NOT swap the operands.
        6. Do NOT generate any Code Block.

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
        Output: {"operator":"IMPLIES","left_operand":"x will pass","right_operand":"x studies hard or x attends class"}

        Input:  "A student is eligible only if the student has advisor approval."
        Output: {"operator":"IMPLIES","left_operand":"A student is eligible","right_operand":"the student has advisor approval"}

        Input:  "A student can register only when the student has paid tuition."
        Output: {"operator":"IMPLIES","left_operand":"A student can register","right_operand":"the student has paid tuition"}

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
            "must complete"    → "RequiredComplete"
            "can register"     → "CanRegister"
            "may enroll"       → "AllowedEnroll"
            "eligible for"     → "EligibleFor"
            "required to take" → "RequiresTake"
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
        5. PREFER UNARY for properties and traits: Safe(x), Reliable(x), CostEffective(x),
            EcoFriendly(x), RequiresTrainingData(x), IsEligible(x).
            Do NOT add a constant argument just to describe the type of x.
        6. PREFER BINARY for true subject-object relations that vary per instance:
            Program(x, ProgramName), ApplyingFor(x, Cycle), SubmittedBy(x, Date).
        7. When a binary argument is always the same constant across all premises,
            fuse it into the predicate name instead:
            Requires(x, TrainingData) → RequiresTrainingData(x)
            CostEffective(x, TransportationSystem) → CostEffective(x)
        8. Do NOT generate any Code Block.

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
        Output: {"predicate":"RequiredComplete","arguments":["Student","MajorAssignment"],"negated":false}

        Input:  "the student must take the final exam"
        Output: {"predicate":"RequiredTake","arguments":["Student","FinalExam"],"negated":false}

        Input:  "x can register"
        Output: {"predicate":"CanRegister","arguments":["x"],"negated":false}

        Input:  "x may enroll in the course"
        Output: {"predicate":"AllowedEnroll","arguments":["x","Course"],"negated":false}

        Input:  "x is eligible for advanced classes"
        Output: {"predicate":"EligibleFor","arguments":["x","AdvancedClasses"],"negated":false}

        Input:  "x is required to take the final exam"
        Output: {"predicate":"RequiresTake","arguments":["x","FinalExam"],"negated":false}

        Input:  "x gives y a book"
        Output: {"predicate":"Gives","arguments":["x","y","Book"],"negated":false}

        Input:  "x does not pass the exam"
        Output: {"predicate":"Pass","arguments":["x","Exam"],"negated":true}

        Input:  "x cannot stay"
        Output: {"predicate":"Stay","arguments":["x"],"negated":true}

        Input:  "x is prohibited from staying"
        Output: {"predicate":"Stay","arguments":["x"],"negated":true}

        Input:  "Rina loves books"
        Output: {"predicate":"LovesBooks","arguments":["Rina"],"negated":false}

        Input:  "x is cost-effective"
        Output: {"predicate":"CostEffective","arguments":["x"],"negated":false}

        Input:  "x requires training data"
        Output: {"predicate":"RequiresTrainingData","arguments":["x"],"negated":false}

        Input:  "AI models are safe"
        Output: {"predicate":"Safe","arguments":["x"],"negated":false}

        Input:  "x is reliable"
        Output: {"predicate":"Reliable","arguments":["x"],"negated":false}
"""

def get_system_prompt_atomic_schema_guided(schema: PremiseSchema) -> str:
    """Return an atomic prompt that injects the premise predicate vocabulary.

    When parsing claim/option sentences, this prompt tells the LLM which
    predicate names already exist in the premise schema. The LLM should reuse
    them instead of inventing new ones, eliminating parser drift.
    """

    vocab_lines: list[str] = []
    for pred in schema.predicates:
        alias_part = f"  (aliases: {', '.join(pred.aliases)})" if pred.aliases else ""
        vocab_lines.append(f"  - {pred.name}/{pred.arity}{alias_part}")

    const_lines: list[str] = []
    for const in schema.constants:
        alias_part = f"  (aliases: {', '.join(const.aliases)})" if const.aliases else ""
        const_lines.append(f"  - {const.name}{alias_part}")

    vocab_block = "\n".join(vocab_lines) if vocab_lines else "  (none)"
    const_block = "\n".join(const_lines) if const_lines else "  (none)"

    return f"""\
    Extract the predicate and arguments from this atomic sentence.

    KNOWN PREDICATES (you MUST use these exact names when they match):
{vocab_block}

    KNOWN CONSTANTS (you MUST use these exact names when they match):
{const_block}

    Rules:
        1. If the sentence's meaning matches a KNOWN PREDICATE above, you MUST
           use that exact predicate name (or one of its aliases). Do NOT invent
           a new name for a concept that already has a predicate.
        2. Only create a new predicate name if no KNOWN PREDICATE fits the meaning.
        3. Similarly, if an argument matches a KNOWN CONSTANT, use that exact name.
        4. predicate → CamelCase verb phrase describing the relation.
        5. arguments → ALL entities in subject-object order.
            Normalize every argument:
            - Variables (x, y, z, x1...)  → keep exactly as-is
            - Proper names (Alice, Rina)   → keep exactly as-is
            - "the X" / "a X" / "an X"   → strip article, CamelCase
            - NEVER include articles (the, a, an) in argument name
        6. Include ALL arguments — subject AND all objects.
        7. If negative (does not / is not / never), set negated=true,
            extract the POSITIVE predicate and all arguments.
        8. PREFER UNARY for properties and traits.
        9. PREFER BINARY for true subject-object relations that vary per instance.
        10. Do NOT generate any Code Block.

    Return ONLY valid JSON.
    Output: {{"predicate":"...","arguments":["..."],"negated":false}}
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
11. "not necessarily P" is not negation. Set modality="not_necessarily" and
    preserve P as a positive conclusion text.

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

Input: "Online courses are not necessarily cost-effective."
Output: {"kind":"deontic_rule","variable":"x","restrictor_text":"x is an online course","condition_texts":[],"conclusion_texts":["x is not necessarily cost-effective"],"fact_texts":[],"numeric_constraints":[],"temporal_constraints":[],"modality":"not_necessarily","confidence":1.0}

Input: "Professor John has published at least 3 academic papers."
Output: {"kind":"numeric_fact","variable":null,"restrictor_text":null,"condition_texts":[],"conclusion_texts":[],"fact_texts":[],"numeric_constraints":["Professor John has published at least 3 academic papers"],"temporal_constraints":[],"modality":"none","confidence":1.0}

Input: "John maintains a GPA of 3.8."
Output: {"kind":"numeric_fact","variable":null,"restrictor_text":null,"condition_texts":[],"conclusion_texts":[],"fact_texts":[],"numeric_constraints":["John has a GPA of 3.8"],"temporal_constraints":[],"modality":"none","confidence":1.0}
"""


def get_system_prompt_numeric_constraint() -> str:
    """Return instructions for extracting one numeric comparison from a sentence."""

    return """
    Extract the numeric comparison expressed in this sentence.

    Rules:
        1. function_name → CamelCase name of the numeric attribute (e.g. GPA, CompletedCourses, CreditHours, Credits).
        2. arguments → ALL entity arguments the function is applied to (variables like x, or constants like Sarah), in order.
        3. operator → the comparison direction:
            "at least N" / "no fewer than N" / "minimum N" → ">="
            "more than N" / "greater than N" / "above N"   → ">"
            "at most N"  / "no more than N"  / "maximum N" → "<="
            "fewer than N" / "less than N" / "below N"     → "<"
            "exactly N"  / "equals N" / "is N" / "of N" / "a … of N" / plain number → "="
            "not equal to N" / "different from N"          → "!="
        4. value → the numeric value as a float (integer or decimal).
        5. Do NOT generate any Code Block.

    Return ONLY valid JSON.
    Output: {"function_name":"...","arguments":["..."],"operator":"="|">="|">"|"<="|"<"|"!=","value":0.0}

    Examples:
        Input:  "x has completed at least 5 courses"
        Output: {"function_name":"CompletedCourses","arguments":["x"],"operator":">=","value":5.0}

        Input:  "x has a GPA below 2.5"
        Output: {"function_name":"GPA","arguments":["x"],"operator":"<","value":2.5}

        Input:  "Sarah has accumulated 80 credits"
        Output: {"function_name":"Credits","arguments":["Sarah"],"operator":"=","value":80.0}

        Input:  "x has more than 60 credit hours"
        Output: {"function_name":"CreditHours","arguments":["x"],"operator":">","value":60.0}

        Input:  "x covers at most 65% of total credits"
        Output: {"function_name":"CreditPercentage","arguments":["x"],"operator":"<=","value":65.0}

        Input:  "x has completed exactly 3 required courses"
        Output: {"function_name":"RequiredCoursesCompleted","arguments":["x"],"operator":"=","value":3.0}

        Input:  "John maintains a GPA of 3.8"
        Output: {"function_name":"GPA","arguments":["John"],"operator":"=","value":3.8}

        Input:  "x has a minimum GPA of 2.5"
        Output: {"function_name":"GPA","arguments":["x"],"operator":">=","value":2.5}
    """


def get_system_prompt_temporal_constraint() -> str:
    """Return instructions for extracting one temporal comparison from a sentence."""

    return """
    Extract the temporal comparison expressed in this sentence.

    Rules:
        1. function_name → CamelCase name of the date attribute (e.g. SubmissionDate, EnrollmentDate, ApplicationDate).
        2. arguments → ALL entity arguments the function is applied to (variables or constants).
        3. operator → the temporal relation:
            "before DATE"                         → "before"
            "by DATE" / "no later than DATE"      → "by"
            "after DATE" / "no earlier than DATE" → "after"
            "on DATE"                             → "on"
            "until DATE"                          → "until"
        4. date_value → normalize the date to a CamelCase identifier (e.g. "June1", "May15", "AddDropDeadline", "SemesterStart").
        5. Do NOT generate any Code Block.

    Return ONLY valid JSON.
    Output: {"function_name":"...","arguments":["..."],"operator":"before"|"after"|"on"|"by"|"until","date_value":"..."}

    Examples:
        Input:  "x enrolled before the add/drop deadline"
        Output: {"function_name":"EnrollmentDate","arguments":["x"],"operator":"before","date_value":"AddDropDeadline"}

        Input:  "x must submit by June 1"
        Output: {"function_name":"SubmissionDate","arguments":["x"],"operator":"by","date_value":"June1"}

        Input:  "Hà submitted her application on May 15"
        Output: {"function_name":"ApplicationDate","arguments":["Ha"],"operator":"on","date_value":"May15"}

        Input:  "x registered after the semester start"
        Output: {"function_name":"RegistrationDate","arguments":["x"],"operator":"after","date_value":"SemesterStart"}

        Input:  "x must complete the form before the deadline"
        Output: {"function_name":"CompletionDate","arguments":["x"],"operator":"before","date_value":"Deadline"}
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


def get_system_prompt_question_frame() -> str:
    """Return instructions for classifying a question before any FOL is produced."""

    return """
Classify a question and decide HOW it should be answered. Do NOT generate FOL.

# TASK
Identify the question_format, the solver_mode, how to read the word "can", and—for
polar (yes/no) questions only—the single declarative claim to test.

# QUESTION FORMAT
- polar   : a yes/no/unknown question ("Does it follow that ...?", "Is X true?", "Can X ...?")
- mcq     : a multiple-choice question (the stem refers to lettered options A/B/C/D)
- open_wh : an open "how/what/why" question with no testable single claim

# SOLVER MODE (what we must compute)
- entailment        : does a statement logically follow from the premises (default)
- refutation        : pick the statement that is FALSE / "not true" / "cannot" follow
- strongest_conclusion : pick the strongest / most significant conclusion
- fewest_premise    : pick the conclusion that follows from the FEWEST premises
- premise_selection : pick which premises support a conclusion
- ynu_mapped        : MCQ whose options are themselves Yes / Uncertain / No answers
- unsupported       : open_wh or anything with no decidable claim

# CAN INTERPRETATION (critical)
- meta_inference : "can" means logical entailment — "which statement CAN be inferred / concluded / follows".
                   Use solver_mode=entailment (or refutation), and DO NOT keep "can" inside the claim.
- object_modal   : "can/may" is a real-world ability/permission about a named subject —
                   "Can Tuan take the course?". Keep the modal inside claim_text ("Tuan can take the course").
- none           : the word "can" is absent or irrelevant.

# CLAIM TEXT (polar only)
For polar questions, claim_text = the statement being tested as a plain declarative sentence,
stripped of "does it follow that" / "according to the premises" / "is it true that".
For mcq and open_wh, claim_text = null (the options carry the claims).
Set negate_claim=true only when the polar question asks whether something is FALSE / NOT true.

# TARGET (the entity the question/options are about)
- target_entity   : the named subject the answer is about — a proper noun like "Sophia",
                    "Professor John", "Student A". null if the question is about a class only.
- target_class    : the class/role of that subject ("student", "professor") or null.
- target_property : the property being asked about ("eligible", "qualifies", "GPA") or null.
These let the option/claim normalizer resolve pronouns ("he", "she") and possessives.

# OUTPUT FORMAT (return ONLY valid JSON)
{
  "question_format": "polar"|"mcq"|"open_wh",
  "solver_mode": "entailment"|"refutation"|"strongest_conclusion"|"fewest_premise"|"premise_selection"|"ynu_mapped"|"unsupported",
  "can_interpretation": "none"|"meta_inference"|"object_modal",
  "claim_text": "<declarative claim or null>",
  "negate_claim": false,
  "target_entity": "<proper-noun subject or null>",
  "target_class": "<class/role or null>",
  "target_property": "<property asked about or null>",
  "confidence": 1.0
}

# EXAMPLES

Input: "Does it follow that if all Python projects are well-structured, then all Python projects are optimized, according to the premises?"
Output: {"question_format":"polar","solver_mode":"entailment","can_interpretation":"none","claim_text":"if all Python projects are well-structured, then all Python projects are optimized","negate_claim":false,"confidence":1.0}

Input: "Which conclusion can be inferred from the premises?"
Output: {"question_format":"mcq","solver_mode":"entailment","can_interpretation":"meta_inference","claim_text":null,"negate_claim":false,"confidence":1.0}

Input: "Which conclusion follows with the fewest premises?"
Output: {"question_format":"mcq","solver_mode":"fewest_premise","can_interpretation":"none","claim_text":null,"negate_claim":false,"confidence":1.0}

Input: "Which of the following statements is NOT true?"
Output: {"question_format":"mcq","solver_mode":"refutation","can_interpretation":"none","claim_text":null,"negate_claim":false,"confidence":1.0}

Input: "Which combination of factors most significantly decreases learning efficiency?"
Output: {"question_format":"mcq","solver_mode":"strongest_conclusion","can_interpretation":"none","claim_text":null,"negate_claim":false,"confidence":1.0}

Input: "Can Tuan take the advanced course?"
Output: {"question_format":"polar","solver_mode":"entailment","can_interpretation":"object_modal","claim_text":"Tuan can take the advanced course","negate_claim":false,"confidence":1.0}

Input: "Is it false that every student passes the final exam?"
Output: {"question_format":"polar","solver_mode":"entailment","can_interpretation":"none","claim_text":"every student passes the final exam","negate_claim":true,"confidence":1.0}

Input: "Do all students earn an A or A+?"
Output: {"question_format":"polar","solver_mode":"entailment","can_interpretation":"none","claim_text":"all students earn an A or A+","negate_claim":false,"confidence":1.0}

Input: "Which premises directly support the conclusion about decreased learning efficiency?"
Output: {"question_format":"mcq","solver_mode":"premise_selection","can_interpretation":"none","claim_text":null,"negate_claim":false,"confidence":1.0}

Input: "Do all students earn an A or A+?\nA. Yes, all mastered the subject.\nB. Uncertain.\nC. No, only some earn A+."
Output: {"question_format":"mcq","solver_mode":"ynu_mapped","can_interpretation":"none","claim_text":"all students earn an A or A+","negate_claim":false,"confidence":1.0}

Input: "How do supervised learning models function?"
Output: {"question_format":"open_wh","solver_mode":"unsupported","can_interpretation":"none","claim_text":null,"negate_claim":false,"confidence":1.0}
"""


def get_system_prompt_fragment_realization() -> str:
    """Return instructions for realizing one subjectless option fragment into a claim.

    Used only as a fallback when deterministic subject recovery fails.
    """

    return """
Turn a subjectless multiple-choice option FRAGMENT into a complete declarative claim by
recovering its subject from the question stem. Do NOT generate FOL.

# TASK
The fragment lacks a subject (e.g. "Can be a research mentor"). Find the subject the
question is about (e.g. "Professor Kim") and return the full sentence.

# RULES
1. PRESERVE every modal word exactly: can, cannot, may, must, should, required, allowed, prohibited.
2. PRESERVE negation exactly: "Cannot teach" must stay "cannot teach", never "can teach".
3. Recover the subject from the stem ("about X", "true for X", "Can X ...", "describes the X").
4. If no subject can be recovered, set subject_found=false and claim_text=null. Do NOT invent one.
5. Return ONE complete declarative sentence.

# OUTPUT FORMAT (return ONLY valid JSON)
{ "claim_text": "<full sentence or null>", "subject_found": true }

# EXAMPLES

Stem: "Which statement is true about Professor Kim?"
Fragment: "Can be a research mentor"
Output: {"claim_text":"Professor Kim can be a research mentor.","subject_found":true}

Stem: "Which option describes the professor?"
Fragment: "Cannot teach graduate courses"
Output: {"claim_text":"The professor cannot teach graduate courses.","subject_found":true}

Stem: "Which of the following is true about Hà?"
Fragment: "Eligible for the internship program"
Output: {"claim_text":"Hà is eligible for the internship program.","subject_found":true}

Stem: "Which conclusion follows from the premises?"
Fragment: "Needs advisor approval"
Output: {"claim_text":null,"subject_found":false}
"""


def get_system_prompt_theory_translate() -> str:
    """Return instructions for translating a whole Type 1 problem to FOL at once.

    The model declares a predicate dictionary once and reuses it across every
    premise, the question, and any options — so the same relation keeps the same
    predicate name (the property the per-premise pipeline could not guarantee).
    """

    return r"""
Translate a logic problem (premises + question + options) into first-order logic.
Output ONE shared predicate dictionary, then one FOL string per premise / claim /
option, ALL using the same predicate names.

# FOL GRAMMAR (emit exactly this syntax)
- Quantifiers:  forall x: BODY      exists x: BODY
- Connectives:  &  (and)   |  (or)   ~  (not)   ->  (implies)   <->  (iff)
- Predicate:    Name(arg1, arg2)         e.g. Medical(x), Needs(Bear, Mouse)
- Comparison:   Func(args) OP number     OP in  >= <= != = > <
- Group with parentheses. Variables are lowercase (x, y). Constants/predicates
  are CamelCase. One formula per premise, no line breaks inside a formula.

# CONSISTENCY RULES (most important)
1. Declare every predicate ONCE in "predicates" with name and arity (gloss
   optional — omit it to stay terse). Keep the dictionary minimal: list each
   predicate a single time, never repeat a name, and stop after the predicates
   the premises/question actually use.
2. Use the SAME predicate name everywhere the SAME relation appears — in premises,
   the claim, and options. Never invent a second name for one relation
   (e.g. do NOT use both PriorityStatus and HasPriorityStatus).
3. Keep distinct relations distinct. Never merge different concepts
   (Medical vs Priority vs Dispatched are different predicates).
4. Numbers/thresholds become comparisons: "weighs under 2 kg" -> Weight(x) < 2.
5. Negation: "is not / does not / never" -> ~ on the predicate.
6. "premises" must have exactly one FOL string per input premise, in order.
7. META-EPISTEMIC premises that only say information is ABSENT or UNKNOWN
   ("No premise states whether X", "It is unknown/unspecified whether X",
   "Nothing indicates whether X") carry NO logical content. Output an EMPTY
   STRING "" for that premise (keep one entry per premise, in order). NEVER
   translate them to ~X or any fact — the solver should stay Uncertain when
   nothing entails the claim.
8. COREFERENCE: use ONE constant for the same real-world entity across the whole
   theory. Parts/aspects of one subject are the SAME constant — e.g. "the Atlas
   case", "the Atlas server", "affected Atlas passwords", "the Atlas forensic
   report" all refer to the Atlas case → use a single constant (Atlas) so the
   rule chain connects. Never split one entity into separate constants.
9. DEONTIC "without": "must not X without Y", "should not X without Y",
   "requires Y before X" mean Y is REQUIRED — translate as the requirement
   predicate (e.g. RequiresReview(p)), NOT as a disjunction like
   (~X | Y). Preserve the requirement so it can be derived.

# QUESTION
- question_format = "polar" for a yes/no/uncertain statement, "mcq" if options are
  given, "open_wh" for an open question with no statement to test.
- polar: set "claim" to the FOL of the POSITIVE statement being asked about; leave
  "options" empty.
- mcq: set "options" to one {label, fol} per option; leave "claim" null.
- open_wh: leave both null.

# OUTPUT (return ONLY valid JSON)
{
  "predicates": [{"name": "...", "arity": 1, "gloss": "..."}, ...],
  "premises": ["<fol>", ...],
  "question_format": "polar"|"mcq"|"open_wh",
  "claim": "<fol or null>",
  "options": [{"label": "A", "fol": "<fol>"}, ...]
}

# EXAMPLE
PREMISES:
1. If a package is medical and weighs under 2 kilograms, then it receives priority delivery status.
2. If a package has priority delivery status and its route is clear, then it can be dispatched.
3. Package P is medical and weighs 1 kilogram and its route is clear.
QUESTION:
Can package P be dispatched?
Output:
{"predicates":[{"name":"Medical","arity":1},{"name":"Weight","arity":1},{"name":"PriorityStatus","arity":1},{"name":"RouteClear","arity":1},{"name":"CanDispatch","arity":1}],"premises":["forall x: (Medical(x) & Weight(x) < 2) -> PriorityStatus(x)","forall x: (PriorityStatus(x) & RouteClear(x)) -> CanDispatch(x)","Medical(P) & Weight(P) = 1 & RouteClear(P)"],"question_format":"polar","claim":"CanDispatch(P)","options":[]}
"""
