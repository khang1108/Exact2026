from __future__ import annotations

import pytest

from exact.config import Settings
from exact.type2.execution_policy import (
    PotMode,
    build_execution_policy,
    validate_fallback_output,
)
from exact.type2.extraction.llm_structured import PotCodeSpec
from exact.type2.fallback.executor import ExecutionResult
from exact.type2.formulas.knowledge import RetrievedFormulaContext
from exact.type2.schemas import Extraction, Type2QuestionKind
from exact.type2.solving import pot_solver


CAPACITOR_VOLTAGE_CODE = """\
import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
W = Q_(0.16, 'J')
C = Q_(20, 'microfarad')
U = (2 * W / C).to('V').magnitude
ans = U
ans_unit = 'V'
"""


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


def test_execute_repairs_omitted_capacitor_voltage_root() -> None:
    spec = PotCodeSpec(
        code=CAPACITOR_VOLTAGE_CODE,
        answer_unit="V",
        formula_ids_used=["voltage_from_capacitor_energy_capacitance"],
    )

    execution = pot_solver._execute_code_spec(spec, timeout_seconds=5.0)

    assert execution.ok
    assert float(execution.ans) == pytest.approx(126.4911064)
    assert execution.ans_unit == "V"


def test_prepare_does_not_double_apply_existing_formula_root() -> None:
    correct_code = CAPACITOR_VOLTAGE_CODE.replace(
        "(2 * W / C).to('V')",
        "((2 * W / C) ** 0.5).to('V')",
    )

    prepared = pot_solver._prepare_generated_code(
        correct_code,
        ["voltage_from_capacitor_energy_capacitance"],
    )

    assert prepared.count("** 0.5") == 1


def test_prepare_does_not_apply_root_to_final_variable_conversion() -> None:
    correct_code = """\
import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
W = Q_(0.16, 'J')
C = Q_(20, 'microfarad')
U = (2 * W / C) ** 0.5
ans = U.to('V').magnitude
ans_unit = 'V'
"""

    prepared = pot_solver._prepare_generated_code(
        correct_code,
        ["voltage_from_capacitor_energy_capacitance"],
    )

    assert prepared.count("** 0.5") == 1


def test_prepare_requires_matching_formula_id_before_repairing_root() -> None:
    prepared = pot_solver._prepare_generated_code(CAPACITOR_VOLTAGE_CODE)

    assert "** 0.5" not in prepared


def test_repair_loop_honors_configured_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = ExecutionResult(
        ok=False,
        ans=None,
        stdout="",
        stderr="dimensionality error",
        error="dimensionality error",
    )
    repair_calls = 0

    monkeypatch.setattr(pot_solver, "_execute_code_spec", lambda *_args, **_kwargs: failed)

    def fake_repair(*_args, **_kwargs) -> PotCodeSpec:
        nonlocal repair_calls
        repair_calls += 1
        return PotCodeSpec(code="ans = None", formula_ids_used=[])

    monkeypatch.setattr(pot_solver, "repair_pot_code", fake_repair)

    extraction = Extraction(
        kind=Type2QuestionKind.NUMERICAL,
        normalized_question="Find the voltage.",
        target="voltage",
        quantities={},
    )
    formula_context = RetrievedFormulaContext(formula_ids=(), context="", summaries=[])
    settings = Settings(_env_file=None, type2_pot_max_retries=1)

    _, _, repair_attempts, _ = pot_solver._execute_with_repair_loop(
        extraction,
        PotCodeSpec(code="ans = None", formula_ids_used=[]),
        formula_context,
        settings,
    )

    assert repair_attempts == 1
    assert repair_calls == 1
