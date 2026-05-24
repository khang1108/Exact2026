from exact.logic.ir import Atom, Fact, Rule
from exact.logic.kb import KnowledgeBase
from exact.symbolic_solvers.forward_chain.solver import derive_closure, solve_query


def _kb(facts: tuple[Fact, ...], rules: tuple[Rule, ...]) -> KnowledgeBase:
    return KnowledgeBase(
        raw_premises=(),
        facts=facts,
        rules=rules,
        premise_hash="test",
    )


def test_variable_rule_derives_ground_conclusion() -> None:
    kb = _kb(
        facts=(
            Fact(Atom("student", ("sophia",)), 0, "Sophia is a student."),
            Fact(Atom("passed_assessment", ("sophia",)), 1, "Sophia passed the assessment."),
        ),
        rules=(
            Rule(
                conditions=(Atom("student", ("?x",)), Atom("passed_assessment", ("?x",))),
                conclusion=Atom("qualified_for_advanced_courses", ("?x",)),
                source_idx=2,
                text="Students who pass the assessment qualify.",
            ),
        ),
    )

    known, proofs = derive_closure(kb)
    derived = Atom("qualified_for_advanced_courses", ("sophia",))

    assert derived in known
    assert proofs[derived].parents == (
        Atom("student", ("sophia",)),
        Atom("passed_assessment", ("sophia",)),
    )
    assert proofs[derived].used_premises == (0, 1, 2)


def test_cross_condition_binding_consistency_blocks_mismatch() -> None:
    kb = _kb(
        facts=(
            Fact(Atom("student", ("sophia",)), 0, "Sophia is a student."),
            Fact(Atom("passed_assessment", ("noah",)), 1, "Noah passed the assessment."),
        ),
        rules=(
            Rule(
                conditions=(Atom("student", ("?x",)), Atom("passed_assessment", ("?x",))),
                conclusion=Atom("qualified", ("?x",)),
                source_idx=2,
                text="Students who pass qualify.",
            ),
        ),
    )

    known, _ = derive_closure(kb)

    assert Atom("qualified", ("sophia",)) not in known
    assert Atom("qualified", ("noah",)) not in known


def test_repeated_variables_must_match_same_term() -> None:
    kb = _kb(
        facts=(
            Fact(Atom("same_course_pair", ("math", "math")), 0, "Math matches math."),
            Fact(Atom("same_course_pair", ("math", "science")), 1, "Math and science are paired."),
        ),
        rules=(
            Rule(
                conditions=(Atom("same_course_pair", ("?x", "?x")),),
                conclusion=Atom("self_match", ("?x",)),
                source_idx=2,
                text="Matching pairs are self matches.",
            ),
        ),
    )

    known, _ = derive_closure(kb)

    assert Atom("self_match", ("math",)) in known
    assert Atom("self_match", ("science",)) not in known


def test_solve_query_returns_no_for_derived_negated_claim() -> None:
    kb = _kb(
        facts=(Fact(Atom("missed_required_lab", ("sophia",)), 0, "Sophia missed a lab."),),
        rules=(
            Rule(
                conditions=(Atom("missed_required_lab", ("?x",)),),
                conclusion=Atom("eligible_for_certificate", ("?x",), negated=True),
                source_idx=1,
                text="Students who miss a required lab are not eligible.",
            ),
        ),
    )

    result = solve_query(kb, Atom("eligible_for_certificate", ("sophia",)))

    assert result.label == "No"
    assert result.supporting_premises == (0, 1)
    assert result.proof[-1].derived == Atom("eligible_for_certificate", ("sophia",), negated=True)
