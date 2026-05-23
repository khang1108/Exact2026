from __future__ import annotations

from exact.datasets.schemas import PredictionRequest
from exact.type2.extraction.llm_structured import (
    ExtractionQuantitySpec,
    FinalExplanationSpec,
    PotCodeSpec,
    Type2ExtractionSpec,
)
from exact.type2.fallback.executor import execute_python
from exact.type2.pipeline import run_type2_pipeline
from exact.type2.solving.pot_solver import _normalize_pint_prefixes, _prepare_generated_code


def test_type2_pipeline_solves_with_pot_first_llm_code(monkeypatch) -> None:
    def fake_parse_with_llm(question: str, settings=None):
        return Type2ExtractionSpec(
            kind="numerical",
            target="voltage",
            quantities=[
                ExtractionQuantitySpec(
                    name="current",
                    value=2,
                    unit="A",
                    evidence="current is 2 A",
                ),
                ExtractionQuantitySpec(
                    name="resistance",
                    value=5,
                    unit="ohm",
                    evidence="resistance is 5 ohm",
                ),
            ],
            notes=["test candidate"],
        )

    def fake_generate_pot_code(question: str, explanation: str, formula_context: str = "", settings=None):
        return PotCodeSpec(
            code="""
import pint
ureg = pint.UnitRegistry()
I = 2 * ureg.ampere
R = 5 * ureg.ohm
ans_quantity = (I * R).to("V")
ans = ans_quantity.magnitude
ans_unit = "V"
""",
            explanation="Used Ohm's law U = I * R.",
            answer_unit="V",
            formula_ids_used=["ohm_voltage"],
        )

    def fake_generate_final_explanation(
        question,
        answer,
        unit,
        formula_context,
        code_explanation,
        formula_ids_used,
        settings=None,
    ):
        return FinalExplanationSpec(
            explanation=f"Ohm's law gives {answer} {unit}.",
            premises=["Ohm's law: U = I * R.", "I = 2 A.", "R = 5 ohm."],
            cot=["Used the retrieved Ohm law formula."],
        )

    monkeypatch.setattr("exact.type2.pipeline.parse_with_llm", fake_parse_with_llm)
    monkeypatch.setattr("exact.type2.solving.pot_solver.generate_pot_code", fake_generate_pot_code)
    monkeypatch.setattr(
        "exact.type2.solving.pot_solver.generate_final_explanation",
        fake_generate_final_explanation,
    )

    request = PredictionRequest(
        id="physics_pot",
        question="Calculate the voltage when current is 2 A and resistance is 5 ohm.",
    )

    response = run_type2_pipeline(request)

    assert response.error is None
    assert response.answer == "10"
    assert response.unit == "V"
    assert response.fol is None
    assert response.premises == ["Ohm's law: U = I * R.", "I = 2 A.", "R = 5 ohm."]


def test_pot_solver_normalizes_llm_pint_prefix_objects() -> None:
    code = "Q = 3 * ureg.milli * ureg.coulomb\nC = 100 * ureg.micro * ureg.farad"

    normalized = _normalize_pint_prefixes(code)

    assert "ureg.milli" not in normalized
    assert "ureg.micro" not in normalized
    assert "Q = 3 * 1e-3 * ureg.coulomb" in normalized
    assert "C = 100 * 1e-6 * ureg.farad" in normalized


def test_pot_solver_removes_unused_numpy_imports() -> None:
    code = """
import pint
import numpy as np
ureg = pint.UnitRegistry()
ans = 1
ans_unit = "N"
"""

    prepared = _prepare_generated_code(code)

    assert "numpy" not in prepared
    assert "import pint" in prepared


def test_executor_allows_stringifying_pint_units() -> None:
    result = execute_python(
        """
import pint
ureg = pint.UnitRegistry()
C = 100 * 1e-6 * ureg.farad
ans = C.to("microfarad").magnitude
ans_unit = str(C.to("microfarad").units)
""",
        timeout_seconds=5,
    )

    assert result.ok
    assert abs(result.ans - 100) < 1e-9
    assert result.ans_unit == "microfarad"
