from exact.type2.execution_policy import (
    PotMode,
    build_execution_policy,
    validate_fallback_output,
)
from exact.type2.schemas import Type2QuestionKind


def test_execution_policy_accepts_current_question_kinds() -> None:
    numerical = build_execution_policy(Type2QuestionKind.NUMERICAL)
    conceptual = build_execution_policy(Type2QuestionKind.CONCEPTUAL)

    assert numerical.pot_mode == PotMode.NUMERIC_SINGLE
    assert conceptual.pot_mode == PotMode.DISABLED
    assert conceptual.use_conceptual_classifier is True


def test_execution_policy_accepts_detailed_question_kind_strings() -> None:
    multi_value = build_execution_policy("multi_value_numeric")
    directional = build_execution_policy("directional_output")

    assert multi_value.pot_mode == PotMode.NUMERIC_MULTI_JSON
    assert directional.pot_mode == PotMode.DISABLED
    assert directional.use_direction_classifier is True


def test_validate_fallback_output_accepts_current_and_detailed_kinds() -> None:
    assert validate_fallback_output(Type2QuestionKind.NUMERICAL, "12.5 N")
    assert validate_fallback_output("directional_output", "left")
    assert validate_fallback_output(
        "qualitative_conceptual",
        '{"answer": "It increases.", "reason": "The field is stronger."}',
    )


def test_new_solver_contract_modules_import() -> None:
    from exact.type2.solver_contract.normalizer import normalize_contract
    from exact.type2.solving.routing_strategies import run_deterministic_solver

    assert callable(normalize_contract)
    assert callable(run_deterministic_solver)
