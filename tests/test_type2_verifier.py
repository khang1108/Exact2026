from __future__ import annotations

from exact.type2.schemas import (
    Extraction,
    Formula,
    Quantity,
    Type2QuestionKind,
    Type2SolveResult,
    Verification,
)
from exact.type2.solving.units import Q_
from exact.type2.solving.verifier import (
    POST_SOLVER_VERIFICATION_ERROR,
    verify_type2_result,
)


def test_verify_type2_result_accepts_consistent_numerical_result() -> None:
    result = _result(answer="2", value=Q_(2, "ampere"))

    verified = verify_type2_result(result)

    assert verified.error is None
    assert verified.verification.ok is True


def test_verify_type2_result_rejects_answer_that_does_not_match_value() -> None:
    result = _result(answer="3", value=Q_(2, "ampere"))

    verified = verify_type2_result(result)

    assert verified.error == POST_SOLVER_VERIFICATION_ERROR
    assert verified.verification.ok is False
    assert "does not match computed value" in verified.verification.message


def _result(answer: str, value) -> Type2SolveResult:
    extraction = Extraction(
        kind=Type2QuestionKind.NUMERICAL,
        normalized_question="Given voltage 4 volt and resistance 2 ohm, find current.",
        target="current",
        quantities={
            "voltage": Quantity("voltage", Q_(4, "volt"), "voltage = 4 V"),
            "resistance": Quantity("resistance", Q_(2, "ohm"), "resistance = 2 ohm"),
        },
    )
    formula = Formula(
        id="ohms_law_current",
        domain="electricity",
        subfield="dc_circuits",
        target="current",
        required=("voltage", "resistance"),
        output_unit="ampere",
        expression="I = V / R",
        explanation_template="Ohm's law relates current, voltage, and resistance.",
        keywords=("current", "voltage", "resistance"),
        solve=lambda known: known["voltage"] / known["resistance"],
    )
    return Type2SolveResult(
        answer=answer,
        unit="ampere",
        value=value,
        formula=formula,
        extraction=extraction,
        verification=Verification(True, "Solver accepted result."),
        cot=[],
        premises=["Ohm's law relates current, voltage, and resistance."],
        confidence=0.82,
        error=None,
    )
