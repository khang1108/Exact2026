from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
from exact.type2.pipeline import run_type2_pipeline


def _response(query_id: str, *, domain: str = "DDT") -> PredictionResponse:
    return PredictionResponse(
        id=query_id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=QuestionType.NUMERICAL,
        answer="6",
        explanation="Solved by test stub.",
        unit="H",
        fol=None,
        cot=[],
        premises=[],
        confidence=0.9,
        error=None,
        routing_diagnostics={
            "domain": domain,
            "family": "SELF_INDUCTANCE",
            "target": "inductance",
            "solver": "ddt_deterministic_solver",
            "fallback_used": False,
        },
    )


def _error_response(query_id: str) -> PredictionResponse:
    return PredictionResponse(
        id=query_id,
        task_type=TaskType.TYPE2_PHYSICS,
        question_type=QuestionType.UNKNOWN,
        answer="",
        explanation="No answer.",
        unit=None,
        fol=None,
        cot=[],
        premises=[],
        confidence=0.0,
        error="test_error",
        routing_diagnostics={"domain": "GENERIC", "fallback_used": False},
    )


def _unexpected_runner(name: str):
    def _runner(*_args, **_kwargs):
        raise AssertionError(f"{name} should not run for a DDT-routed question")

    return _runner


def test_type2_pipeline_routes_to_heuristic_domain_without_cascade(monkeypatch):
    from exact.type2.domains.ddt import pipeline as ddt_pipeline
    from exact.type2.domains.ld import pipeline as ld_pipeline
    from exact.type2.domains.nl_energy import pipeline as nl_energy_pipeline
    from exact.type2.domains import router as domain_router
    from exact.type2.domains.td import pipeline as td_pipeline
    from exact.type2.domains.thcb import pipeline as thcb_pipeline

    monkeypatch.setattr(thcb_pipeline, "try_thcb_pipeline", _unexpected_runner("THCB"))
    monkeypatch.setattr(nl_energy_pipeline, "run_nl_energy_pipeline", _unexpected_runner("NL_ENERGY"))
    monkeypatch.setattr(ld_pipeline, "run_ld_pipeline", _unexpected_runner("LD"))
    monkeypatch.setattr(td_pipeline, "run_td_pipeline", _unexpected_runner("TD"))
    monkeypatch.setattr(domain_router, "route_domain_with_metadata", _unexpected_runner("LLM question-kind route"))
    monkeypatch.setattr(
        ddt_pipeline,
        "run_ddt_pipeline",
        lambda request, _settings=None: (_response(request.id), False),
    )

    result = run_type2_pipeline(
        PredictionRequest(
            query_id="DDT385",
            type="type2",
            question=(
                "Given that the induced electromotive force is 0.3 V, and the current "
                "decreases uniformly from 2 A to 0 A in 0.05 s. Calculate the self-inductance."
            ),
        )
    )

    assert result.answer == "6"
    assert result.routing_diagnostics["type2_domain_route"]["domain"] == "DDT"
    assert result.routing_diagnostics["type2_domain_route"]["attempts"] == ["DDT:accepted"]


def test_type2_pipeline_falls_back_to_generic_after_routed_domain_only(monkeypatch):
    from exact.type2 import pipeline as type2_pipeline
    from exact.type2.domains.ddt import pipeline as ddt_pipeline
    from exact.type2.domains.ld import pipeline as ld_pipeline
    from exact.type2.domains.nl_energy import pipeline as nl_energy_pipeline
    from exact.type2.domains.td import pipeline as td_pipeline
    from exact.type2.domains.thcb import pipeline as thcb_pipeline

    monkeypatch.setattr(thcb_pipeline, "try_thcb_pipeline", _unexpected_runner("THCB"))
    monkeypatch.setattr(nl_energy_pipeline, "run_nl_energy_pipeline", _unexpected_runner("NL_ENERGY"))
    monkeypatch.setattr(ld_pipeline, "run_ld_pipeline", _unexpected_runner("LD"))
    monkeypatch.setattr(td_pipeline, "run_td_pipeline", _unexpected_runner("TD"))
    monkeypatch.setattr(ddt_pipeline, "run_ddt_pipeline", lambda *_args, **_kwargs: (None, True))
    monkeypatch.setattr(type2_pipeline, "run_generic_pipeline", lambda request, _settings=None, domain_hint=None: _response(request.id, domain="GENERIC"))

    result = run_type2_pipeline(
        PredictionRequest(
            query_id="DDT149",
            type="type2",
            question="When the current through the solenoid increases rapidly, what happens to the induced electromotive force?",
        )
    )

    assert result.answer == "6"
    assert result.routing_diagnostics["type2_domain_route"]["domain"] == "GENERIC"
    assert result.routing_diagnostics["type2_domain_route"]["routed_domain"] == "DDT"
    assert result.routing_diagnostics["type2_domain_route"]["attempts"] == ["DDT:fallback_requested"]


def test_type2_pipeline_does_not_run_generic_twice_for_td_failure(monkeypatch):
    from exact.type2 import pipeline as type2_pipeline
    from exact.type2.domains.td import pipeline as td_pipeline

    monkeypatch.setattr(td_pipeline, "run_td_pipeline", lambda request, _settings=None: _error_response(request.id))
    monkeypatch.setattr(type2_pipeline, "run_generic_pipeline", _unexpected_runner("second generic fallback"))

    result = run_type2_pipeline(
        PredictionRequest(
            query_id="TD001",
            type="type2",
            question="A capacitor with capacitance C = 500 pF is charged to voltage U = 300 V. Calculate the stored energy.",
        )
    )

    assert result.error == "test_error"
    assert result.routing_diagnostics["type2_domain_route"]["domain"] == "TD"
    assert result.routing_diagnostics["type2_domain_route"]["attempts"] == ["TD:response_has_error"]
