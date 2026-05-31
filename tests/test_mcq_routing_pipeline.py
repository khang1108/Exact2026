"""Tests for Type 1 question routing and multiple-choice pipeline behavior."""

import asyncio

import httpx
import pytest

from exact.config import Settings
from exact.common.schemas import PredictionRequest, QuestionType, TaskType
from exact.llm_client import OpenAICompatibleJsonClient
from exact.logic.fol_parser import parse_fol
from exact.logic.ir import (
    Atom,
    Compare,
    Exists,
    Fact,
    ForAll,
    FormulaItem,
    Function,
    Iff,
    Implies,
    InSet,
    Not,
    Number,
    Rule,
    TranslatedProblem,
)
from exact.logic.kb import KnowledgeBase
from exact.logic.llm_translator import (
    _formula_item_from_raw,
    _formula_from_raw,
    clear_formula_premise_cache,
    translate_formula_premises_only_with_llm,
)
from exact.logic.pipeline import (
    _mcq_translation_warning,
    decide_mcq_winner,
    evaluate_mcq_options,
    extract_options,
    run_type1_pipeline,
)
from exact.app.router import health_check
from exact.router.task_router import TaskRouter
from exact.scripts.audit_type1_ir_coverage import audit_type1_ir_coverage
from exact.scripts import evaluate_type1_predictions
from exact.symbolic_solvers.z3_prop import Z3PropSolver


class RecordingTranslatorClient:
    """Test double that records the prompt sent to the LLM translator."""

    def __init__(self) -> None:
        self.user_prompt = ""
        self.user_prompts: list[str] = []

    def complete_json_sync(self, messages, temperature: float = 0.0, max_tokens: int = 2048):
        self.user_prompt = str(messages[-1]["content"])
        self.user_prompts.append(self.user_prompt)
        if "Translate each MCQ option" in self.user_prompt:
            return {
                "options": [
                    {"label": "A", "text": "Alpha", "claim": {"text": "Alpha", "pred": "alpha", "args": []}},
                    {"label": "B", "text": "Beta", "claim": {"text": "Beta", "pred": "beta", "args": []}},
                    {"label": "C", "text": "Gamma", "claim": {"text": "Gamma", "pred": "gamma", "args": []}},
                    {"label": "D", "text": "Delta", "claim": {"text": "Delta", "pred": "delta", "args": []}},
                ]
            }
        return {
            "predicates": [{"name": "alpha", "arity": 0, "gloss": "alpha", "argument_roles": []}],
            "premises": [{"source_idx": 0, "facts": [{"text": "Alpha", "pred": "alpha", "args": []}], "rules": []}],
            "query": {"claim": {"text": "Which conclusion follows", "pred": "alpha", "args": []}},
            "options": [],
        }


class CountingTranslatorClient:
    """Test double that counts premise-only and query-only LLM calls."""

    def __init__(self) -> None:
        self.premise_calls = 0
        self.query_calls = 0
        self.query_prompt = ""

    def complete_json_sync(self, messages, temperature: float = 0.0, max_tokens: int = 2048):
        user_prompt = str(messages[-1]["content"])
        if "Translate these premises" in user_prompt:
            self.premise_calls += 1
            return {
                "predicates": [{"name": "alpha", "arity": 0, "gloss": "alpha", "argument_roles": []}],
                "premises": [
                    {
                        "source_idx": 0,
                        "facts": [{"text": "Alpha", "pred": "alpha", "args": []}],
                        "rules": [],
                    }
                ],
            }

        self.query_calls += 1
        self.query_prompt = user_prompt
        return {
            "query": {"claim": {"text": "Alpha", "pred": "alpha", "args": []}},
        }


class SamplingVoteTranslatorClient:
    """Test double that emits controlled premise samples for voting tests."""

    def __init__(
        self,
        premise_preds: list[str | None],
        query_pred: str = "alpha",
        cot_answer: str | None = None,
    ) -> None:
        self.premise_preds = premise_preds
        self.query_pred = query_pred
        self.cot_answer = cot_answer
        self.premise_calls = 0
        self.query_calls = 0
        self.cot_calls = 0

    def complete_json_sync(self, messages, temperature: float = 0.0, max_tokens: int = 2048):
        user_prompt = str(messages[-1]["content"])
        if "Translate these premises" in user_prompt:
            pred = self.premise_preds[self.premise_calls]
            self.premise_calls += 1
            predicates = [
                {"name": pred or self.query_pred, "arity": 0, "gloss": pred or self.query_pred, "argument_roles": []}
            ]
            facts = [] if pred is None else [{"text": pred.title(), "pred": pred, "args": [pred]}]
            return {
                "predicates": predicates,
                "premises": [{"source_idx": 0, "facts": facts, "rules": []}],
            }

        if "symbolic solver returned Unknown" in user_prompt:
            self.cot_calls += 1
            return {
                "answer": self.cot_answer or "Unknown",
                "explanation": "CoT fallback checked the premises.",
                "cot": ["Reviewed the given premises.", "Selected the fallback answer."],
                "confidence": 0.66,
            }

        self.query_calls += 1
        return {
            "query": {"claim": {"text": self.query_pred.title(), "pred": self.query_pred, "args": [self.query_pred]}}
        }


class FormulaTranslatorClient:
    """Test double for the one-shot formula-Z3 path."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0
        self.user_prompt = ""

    def complete_json_sync(self, messages, temperature: float = 0.0, max_tokens: int = 2048):
        self.calls += 1
        self.user_prompt = str(messages[-1]["content"])
        return self.payload


def explicit_llm_settings() -> Settings:
    """Return settings that require the supplied test LLM client."""

    return Settings(
        llm_provider="openai",
        llm_base_url=None,
        type1_use_formula_z3=False,
        type1_translation_samples=3,
    )


def formula_z3_settings() -> Settings:
    return Settings(
        llm_provider="openai",
        llm_base_url=None,
        type1_use_formula_z3=True,
        type1_formula_cache_premises=False,
        type1_enable_legacy_fallback=False,
        type1_enable_cot_fallback=False,
    )


def test_router_detects_mcq_for_logic_question() -> None:
    """Ensure logic questions with A-D labels route as multiple choice."""

    request = PredictionRequest.model_validate(
        {
            "id": "mcq",
            "premises-NL": ["A."],
            "question": "Which conclusion follows?\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
        }
    )

    route = TaskRouter().route(request)

    assert route.task_type == TaskType.TYPE1_LOGIC
    assert route.question_type == QuestionType.MCQ


def test_router_detects_ynu_and_type2() -> None:
    """Ensure yes/no/uncertain and physics requests route correctly."""

    ynu_request = PredictionRequest.model_validate(
        {"id": "ynu", "premises-NL": ["A."], "question": "Does Alpha follow?"}
    )
    type2_request = PredictionRequest.model_validate(
        {"id": "physics", "question": "Find velocity."}
    )

    assert TaskRouter().route(ynu_request).question_type == QuestionType.YES_NO_UNCERTAIN
    assert TaskRouter().route(type2_request).task_type == TaskType.TYPE2_PHYSICS


def test_type1_pipeline_requires_json_llm_client_when_unconfigured() -> None:
    """Type 1 should fail clearly instead of substituting a local parser."""

    request = PredictionRequest.model_validate(
        {"id": "needs_llm", "premises-NL": ["Alpha."], "question": "Does Alpha follow?"}
    )

    with pytest.raises(RuntimeError, match="requires a JSON LLM client"):
        run_type1_pipeline(
            request,
            settings=Settings(llm_provider="openai", llm_base_url=None),
            question_type=QuestionType.YES_NO_UNCERTAIN,
        )


def test_type1_pipeline_uses_explicit_translator_client() -> None:
    """Type 1 should use the supplied JSON translator client for YNU questions."""

    from exact.logic.kb import clear_kb_cache
    clear_kb_cache()

    client = RecordingTranslatorClient()
    request = PredictionRequest.model_validate(
        {"id": "uses_llm", "premises-NL": ["Alpha."], "question": "Does Alpha follow?"}
    )

    response = run_type1_pipeline(
        request,
        translator_client=client,
        settings=explicit_llm_settings(),
        question_type=QuestionType.YES_NO_UNCERTAIN,
    )

    assert client.user_prompt
    assert response.answer == "Yes"


def test_type1_formula_z3_pipeline_uses_one_shot_translation() -> None:
    """Formula-Z3 path should translate the full problem once and let Z3 decide."""

    client = FormulaTranslatorClient(
        {
            "predicates": {"alpha": 0},
            "premises": [
                {
                    "source_idx": 0,
                    "role": "premise",
                    "text": "Alpha.",
                    "formula": {"type": "atom", "pred": "alpha", "args": []},
                }
            ],
            "goals": [
                {
                    "source_idx": -1,
                    "role": "query",
                    "text": "Does Alpha follow?",
                    "formula": {"type": "atom", "pred": "alpha", "args": []},
                }
            ],
        }
    )
    request = PredictionRequest.model_validate(
        {"id": "formula_ynu", "premises-NL": ["Alpha."], "question": "Does Alpha follow?"}
    )

    response = run_type1_pipeline(
        request,
        translator_client=client,
        settings=formula_z3_settings(),
        question_type=QuestionType.YES_NO_UNCERTAIN,
    )

    assert client.calls == 1
    assert "Now translate:" in client.user_prompt
    assert response.answer == "Yes"
    assert response.premises == ["P1: Alpha."]
    assert response.cot is not None
    assert "z3_typed_formula_query_answer: Yes" in response.cot


def test_type1_formula_z3_mcq_handles_compound_option() -> None:
    """MCQ options can be implication formulas instead of collapsed atoms."""

    client = FormulaTranslatorClient(
        {
            "predicates": {"well_tested": 1, "optimized": 1},
            "premises": [
                {
                    "source_idx": 0,
                    "role": "premise",
                    "text": "If code is well-tested, the project is optimized.",
                    "formula": {
                        "type": "implies",
                        "antecedent": {"type": "atom", "pred": "well_tested", "args": ["?x"]},
                        "consequent": {"type": "atom", "pred": "optimized", "args": ["?x"]},
                    },
                }
            ],
            "goals": [
                {
                    "source_idx": -1,
                    "role": "option",
                    "label": "A",
                    "text": "If not optimized then not well-tested.",
                    "formula": {
                        "type": "implies",
                        "antecedent": {
                            "type": "not",
                            "arg": {"type": "atom", "pred": "optimized", "args": ["?x"]},
                        },
                        "consequent": {
                            "type": "not",
                            "arg": {"type": "atom", "pred": "well_tested", "args": ["?x"]},
                        },
                    },
                },
                {
                    "source_idx": -1,
                    "role": "option",
                    "label": "B",
                    "text": "The project is optimized.",
                    "formula": {"type": "atom", "pred": "optimized", "args": ["?x"]},
                },
            ],
        }
    )
    request = PredictionRequest.model_validate(
        {
            "id": "formula_mcq",
            "premises-NL": ["If code is well-tested, the project is optimized."],
            "question": (
                "Which conclusion follows?\n"
                "A. If not optimized then not well-tested.\n"
                "B. The project is optimized."
            ),
        }
    )

    response = run_type1_pipeline(
        request,
        translator_client=client,
        settings=formula_z3_settings(),
        question_type=QuestionType.MCQ,
    )

    assert client.calls == 1
    assert response.answer == "A"
    assert response.cot is not None
    assert "z3_typed_formula_mcq_valid_options: A" in response.cot


def test_ynu_pipeline_reuses_premise_cache_across_questions() -> None:
    """YNU questions with the same premise group should not retranslate premises."""

    from exact.logic.kb import clear_kb_cache
    clear_kb_cache()

    client = CountingTranslatorClient()
    settings = explicit_llm_settings()
    first = PredictionRequest.model_validate(
        {"id": "ynu_cache_1", "premises-NL": ["Alpha."], "question": "Does Alpha follow?"}
    )
    second = PredictionRequest.model_validate(
        {"id": "ynu_cache_2", "premises-NL": ["Alpha."], "question": "Is Alpha true?"}
    )

    run_type1_pipeline(
        first,
        translator_client=client,
        settings=settings,
        question_type=QuestionType.YES_NO_UNCERTAIN,
    )
    run_type1_pipeline(
        second,
        translator_client=client,
        settings=settings,
        question_type=QuestionType.YES_NO_UNCERTAIN,
    )

    assert client.premise_calls == 3
    assert client.query_calls == 6
    assert "query.claim.pred MUST be EXACTLY one of these allowed names" in client.query_prompt
    assert "alpha" in client.query_prompt


def test_ynu_symbolic_consistency_vote_prefers_majority_label() -> None:
    """Three sampled premise translations should vote over solver labels."""

    from exact.logic.kb import clear_kb_cache
    clear_kb_cache()

    client = SamplingVoteTranslatorClient(["alpha", "alpha", None], query_pred="alpha")
    request = PredictionRequest.model_validate(
        {"id": "ynu_vote", "premises-NL": ["Alpha."], "question": "Does Alpha follow?"}
    )

    response = run_type1_pipeline(
        request,
        translator_client=client,
        settings=explicit_llm_settings(),
        question_type=QuestionType.YES_NO_UNCERTAIN,
    )

    assert response.answer == "Yes"
    assert response.confidence == pytest.approx(2 / 3)
    assert response.cot is not None
    assert response.cot[-1] == "symbolic_consistency_vote: Yes=2, Unknown=1"
    assert client.cot_calls == 0


def test_ynu_symbolic_unknown_uses_cot_fallback() -> None:
    """When symbolic voting returns Unknown, the LLM reasoning fallback can recover an answer."""

    from exact.logic.kb import clear_kb_cache
    clear_kb_cache()

    client = SamplingVoteTranslatorClient([None, None, None], query_pred="alpha", cot_answer="Yes")
    request = PredictionRequest.model_validate(
        {"id": "ynu_cot", "premises-NL": ["No translated fact."], "question": "Does Alpha follow?"}
    )

    response = run_type1_pipeline(
        request,
        translator_client=client,
        settings=explicit_llm_settings(),
        question_type=QuestionType.YES_NO_UNCERTAIN,
    )

    assert response.answer == "Yes"
    assert response.confidence == pytest.approx(0.66)
    assert response.cot is not None
    assert response.cot[-1] == "cot_fallback_after_symbolic_unknown: answer=Yes"
    assert client.cot_calls == 1


def test_ynu_symbolic_unknown_can_skip_cot_fallback_for_fast_mode() -> None:
    """Fast competition/eval mode can avoid the extra LLM call after symbolic Unknown."""

    from exact.logic.kb import clear_kb_cache
    clear_kb_cache()

    client = SamplingVoteTranslatorClient([None], query_pred="alpha", cot_answer="Yes")
    request = PredictionRequest.model_validate(
        {"id": "ynu_no_cot", "premises-NL": ["No translated fact."], "question": "Does Alpha follow?"}
    )

    response = run_type1_pipeline(
        request,
        translator_client=client,
        settings=explicit_llm_settings().model_copy(
            update={"type1_translation_samples": 1, "type1_enable_cot_fallback": False}
        ),
        question_type=QuestionType.YES_NO_UNCERTAIN,
    )

    assert response.answer == "Uncertain"
    assert client.cot_calls == 0


def test_mcq_symbolic_consistency_vote_prefers_majority_option() -> None:
    """MCQ sampled premise translations should vote over selected option labels."""

    from exact.logic.kb import clear_kb_cache
    clear_kb_cache()

    client = SamplingVoteTranslatorClient(["alpha", "alpha", "gamma"])
    request = PredictionRequest.model_validate(
        {
            "id": "mcq_vote",
            "premises-NL": ["Alpha."],
            "question": "Which conclusion follows?\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
        }
    )

    response = run_type1_pipeline(
        request,
        translator_client=client,
        settings=explicit_llm_settings(),
        question_type=QuestionType.MCQ,
    )

    assert response.answer == "A"
    assert response.confidence == pytest.approx(2 / 3)
    assert response.cot is not None
    assert response.cot[-1] == "symbolic_consistency_vote: A=2, C=1"


def test_extract_options_preserves_labels_and_text() -> None:
    """Ensure multiline option text is preserved under the correct label."""

    question = (
        "Which conclusion follows?\n"
        "A. Alpha\n"
        "B. Beta spans\nmultiple words\n"
        "C. Gamma\n"
        "D. Delta"
    )

    assert extract_options(question) == [
        ("A", "Alpha"),
        ("B", "Beta spans multiple words"),
        ("C", "Gamma"),
        ("D", "Delta"),
    ]


def test_mcq_pipeline_returns_unique_entailed_letter() -> None:
    """Ensure an entailed MCQ option returns its option letter."""

    request = PredictionRequest.model_validate(
        {
            "id": "mcq_unique",
            "premises-NL": ["Alpha."],
            "question": "Which conclusion follows?\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
        }
    )

    from exact.logic.kb import clear_kb_cache
    clear_kb_cache()

    response = run_type1_pipeline(
        request,
        translator_client=RecordingTranslatorClient(),
        settings=explicit_llm_settings(),
        question_type=QuestionType.MCQ,
    )

    assert response.question_type == QuestionType.MCQ
    assert response.answer == "A"


def test_mcq_pipeline_sends_only_premises_to_llm() -> None:
    """Premise-level cache: the LLM prompt must contain premises but not the question or options."""

    client = RecordingTranslatorClient()
    request = PredictionRequest.model_validate(
        {
            "id": "mcq_prompt",
            "premises-NL": ["Alpha."],
            "question": "Which conclusion follows?\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
        }
    )

    from exact.logic.kb import clear_kb_cache
    clear_kb_cache()

    run_type1_pipeline(
        request,
        translator_client=client,
        settings=explicit_llm_settings(),
        question_type=QuestionType.MCQ,
    )

    premise_prompts = [
        prompt for prompt in client.user_prompts if "Translate these premises" in prompt
    ]
    assert premise_prompts
    assert all("0: Alpha." in prompt for prompt in premise_prompts)
    assert all("Which conclusion follows" not in prompt for prompt in premise_prompts)
    assert all("A. Alpha" not in prompt for prompt in premise_prompts)
    assert all("B. Beta" not in prompt for prompt in premise_prompts)


def test_decide_mcq_winner_prefers_fewest_premises_among_entailed_options() -> None:
    """Ensure fewest-premises MCQ tie-breaking uses proof support size."""

    kb = KnowledgeBase(
        raw_premises=(),
        facts=(
            Fact(Atom("alpha"), 0, "Alpha."),
            Fact(Atom("beta"), 1, "Beta."),
        ),
        rules=(
            Rule(
                conditions=(Atom("beta"),),
                conclusion=Atom("gamma"),
                source_idx=2,
                text="If beta then gamma.",
            ),
        ),
        premise_hash="mcq_tie",
    )
    results = evaluate_mcq_options(
        kb,
        [
            ("A", Atom("gamma")),
            ("B", Atom("alpha")),
            ("C", Atom("missing")),
            ("D", Atom("beta")),
        ],
    )

    assert decide_mcq_winner(results, "Which conclusion follows with the fewest premises?") == "B"


def test_guided_json_400_retries_without_http_retry_budget(monkeypatch) -> None:
    payloads = []
    responses = [
        httpx.Response(
            400,
            request=httpx.Request("POST", "http://local.test/v1/chat/completions"),
        ),
        httpx.Response(
            200,
            request=httpx.Request("POST", "http://local.test/v1/chat/completions"),
            json={"choices": [{"message": {"content": '{"answer":"Yes"}'}, "finish_reason": "stop"}]},
        ),
    ]

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, json, headers):
            payloads.append(json)
            return responses.pop(0)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    client = OpenAICompatibleJsonClient(
        api_key="EMPTY",
        base_url="http://local.test/v1",
        model="test-model",
        max_retries=0,
    )

    result = asyncio.run(
        client.complete_json(
            messages=[{"role": "user", "content": "Return JSON"}],
            json_schema={"$ref": "#/$defs/formula"},
        )
    )

    assert result == {"answer": "Yes"}
    assert len(payloads) == 2
    assert "guided_json" in payloads[0]
    assert "guided_json" not in payloads[1]


def test_formula_premise_cache_rejects_incomplete_translation() -> None:
    class IncompleteFormulaClient:
        def __init__(self):
            self.calls = 0

        def complete_json_sync(self, messages, temperature=0.0, max_tokens=2048):
            self.calls += 1
            return {
                "predicates": {"alpha": 0},
                "premises": [
                    {
                        "source_idx": 0,
                        "role": "premise",
                        "text": "Alpha.",
                        "formula": {"type": "atom", "pred": "alpha", "args": []},
                    }
                ],
            }

    clear_formula_premise_cache()
    client = IncompleteFormulaClient()

    with pytest.raises(RuntimeError, match="exactly one formula for every premise"):
        translate_formula_premises_only_with_llm(
            ["Alpha.", "Beta."],
            llm_client=client,
            settings=Settings(llm_provider="openai", llm_base_url=None),
        )

    assert client.calls == 3


def test_formula_z3_strongest_option_uses_implication_ordering() -> None:
    problem = TranslatedProblem(
        predicates={"a": 1, "b": 1},
        premises=(
            FormulaItem(Atom("a", ("sophia",)), 0, "Sophia has A.", "premise"),
            FormulaItem(
                Implies(Atom("a", ("?x",)), Atom("b", ("?x",))),
                1,
                "A implies B.",
                "premise",
            ),
        ),
        goals=(
            FormulaItem(Atom("a", ("sophia",)), -1, "A", "option", "A"),
            FormulaItem(Atom("b", ("sophia",)), -1, "B", "option", "B"),
        ),
    )

    result = Z3PropSolver().solve_mcq(problem, stem="Choose the strongest statement.")

    assert result.answer == "A"
    assert result.supporting_premises == (0,)


def test_formula_z3_ignores_vacuously_true_mcq_implications_when_supported_options_exist() -> None:
    problem = TranslatedProblem(
        predicates={"a": 1, "b": 1, "missing": 1},
        premises=(
            FormulaItem(Atom("a", ("sophia",)), 0, "Sophia has A.", "premise"),
            FormulaItem(Atom("b", ("sophia",)), 1, "Sophia has B.", "premise"),
        ),
        goals=(
            FormulaItem(Atom("a", ("sophia",)), -1, "Sophia has A.", "option", "A"),
            FormulaItem(
                Implies(Not(Atom("a", ("sophia",))), Atom("missing", ("sophia",))),
                -1,
                "If Sophia does not have A, then Sophia is missing.",
                "option",
                "B",
            ),
            FormulaItem(Atom("b", ("sophia",)), -1, "Sophia has B.", "option", "C"),
        ),
    )

    result = Z3PropSolver().solve_mcq(problem, stem="Choose the strongest statement.")

    assert result.answer == "A"
    assert result.valid_labels == ("A", "C")


def test_formula_z3_strongest_prefers_shorter_nonvacuous_proof() -> None:
    problem = TranslatedProblem(
        predicates={"seed": 1, "intermediate": 1, "downstream": 1},
        premises=(
            FormulaItem(Atom("seed", ("sophia",)), 0, "Sophia has seed.", "premise"),
            FormulaItem(
                Implies(Atom("seed", ("?x",)), Atom("intermediate", ("?x",))),
                1,
                "Seed implies intermediate.",
                "premise",
            ),
            FormulaItem(
                Implies(Atom("intermediate", ("?x",)), Atom("downstream", ("?x",))),
                2,
                "Intermediate implies downstream.",
                "premise",
            ),
        ),
        goals=(
            FormulaItem(Atom("downstream", ("sophia",)), -1, "Downstream.", "option", "A"),
            FormulaItem(Atom("intermediate", ("sophia",)), -1, "Intermediate.", "option", "C"),
        ),
    )

    result = Z3PropSolver().solve_mcq(problem, stem="Choose the strongest conclusion.")

    assert result.answer == "C"


def test_formula_parser_accepts_impl_alias_and_unwrapped_formula_item() -> None:
    item = _formula_item_from_raw(
        {
            "type": "impl",
            "antecedent": {"type": "atom", "pred": "studies", "args": ["?x"]},
            "consequent": {"type": "atom", "pred": "passes", "args": ["?x"]},
        },
        default_role="premise",
        default_source_idx=0,
        default_text="Students pass.",
    )

    assert item.formula == Implies(
        Atom("studies", ("?x",), text="studies(?x)"),
        Atom("passes", ("?x",), text="passes(?x)"),
    )


def test_suspicious_direct_mcq_claim_rewritten_as_implication_triggers_warning() -> None:
    problem = TranslatedProblem(
        predicates={"registered_nurse": 1},
        premises=(),
        goals=(
            FormulaItem(
                Implies(
                    Not(Atom("registered_nurse", ("john",))),
                    Not(Atom("registered_nurse", ("john",))),
                ),
                -1,
                "John is not registered.",
                "option",
                "C",
            ),
        ),
    )

    assert _mcq_translation_warning(problem) == (
        "suspicious MCQ goal formalization for option(s) C; symbolic ranking requires verification"
    )


def test_suspicious_mcq_translation_uses_direct_llm_fallback() -> None:
    class SuspiciousMCQClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json_sync(self, messages, temperature=0.0, max_tokens=2048):
            self.calls += 1
            if "Symbolic reasoning could not prove any option" in str(messages[-1]["content"]):
                return {"answer": "C", "explanation": "Direct option check.", "confidence": 0.55}
            return {
                "predicates": {"registered_nurse": 1},
                "premises": [
                    {
                        "source_idx": 0,
                        "role": "premise",
                        "text": "John is registered.",
                        "formula": {"type": "atom", "pred": "registered_nurse", "args": ["john"]},
                    }
                ],
                "goals": [
                    {
                        "source_idx": -1,
                        "role": "option",
                        "label": "A",
                        "text": "John is registered.",
                        "formula": {"type": "atom", "pred": "registered_nurse", "args": ["john"]},
                    },
                    {
                        "source_idx": -1,
                        "role": "option",
                        "label": "C",
                        "text": "John is not registered.",
                        "formula": {
                            "type": "implies",
                            "antecedent": {
                                "type": "not",
                                "arg": {"type": "atom", "pred": "registered_nurse", "args": ["john"]},
                            },
                            "consequent": {
                                "type": "not",
                                "arg": {"type": "atom", "pred": "registered_nurse", "args": ["john"]},
                            },
                        },
                    },
                ],
            }

    client = SuspiciousMCQClient()
    request = PredictionRequest.model_validate(
        {
            "id": "suspicious_mcq",
            "premises-NL": ["John is registered."],
            "question": "Which statement is correct?\nA. John is registered.\nC. John is not registered.",
        }
    )
    settings = formula_z3_settings().model_copy(update={"type1_enable_cot_fallback": True})

    response = run_type1_pipeline(
        request,
        translator_client=client,
        settings=settings,
        question_type=QuestionType.MCQ,
    )

    assert response.answer == "C"
    assert client.calls == 2
    assert response.error is not None
    assert "suspicious MCQ goal formalization for option(s) C" in response.error


def test_health_check_exposes_type1_build_marker() -> None:
    assert health_check() == {"status": "ok", "build": "type1-typed-smt-v2"}


def test_type1_evaluator_separates_diagnostics_from_failed_predictions() -> None:
    recovered = evaluate_type1_predictions.evaluate_prediction(
        {
            "id": "recovered",
            "answer": "Uncertain",
            "gold_answer": "Unknown",
            "question_type": "yes_no_uncertain",
            "error": "split translation fallback used",
        },
        case_sensitive=False,
    )
    failed = evaluate_type1_predictions.evaluate_prediction(
        {
            "id": "failed",
            "answer": "",
            "gold_answer": "Yes",
            "question_type": "yes_no_uncertain",
            "error": "connection refused",
        },
        case_sensitive=False,
    )

    summary = evaluate_type1_predictions.summarize([recovered, failed])

    assert recovered.status == "correct"
    assert failed.status == "pipeline_error"
    assert summary["pipeline_errors"] == 1
    assert summary["runtime_diagnostics"] == 2


def test_type1_deadline_exhaustion_returns_valid_failsafe_answer() -> None:
    request = PredictionRequest.model_validate(
        {"id": "deadline", "premises-NL": ["Alpha."], "question": "Does Alpha follow?"}
    )
    settings = formula_z3_settings().model_copy(
        update={
            "type1_enable_legacy_fallback": True,
            "type1_soft_deadline_s": 1e-9,
        }
    )

    response = run_type1_pipeline(
        request,
        translator_client=FormulaTranslatorClient({}),
        settings=settings,
        question_type=QuestionType.YES_NO_UNCERTAIN,
    )

    assert response.answer == "Uncertain"
    assert response.confidence == 0.0
    assert response.cot == ["type1_deadline_exhausted: returned low-confidence fail-safe answer"]


def test_released_type1_premises_parse_and_encode_without_errors() -> None:
    audit = audit_type1_ir_coverage()

    assert audit["records"] == 411
    assert audit["premises"] == 4470
    assert audit["errors"] == []


def test_fol_parser_normalizes_bound_variables_and_supports_extended_nodes() -> None:
    formula = parse_fol(
        "ForAll(s, ranking(s) ∈ {Average, Weak, Poor} ↔ "
        "Exists(t, membership_duration(t) ≥ 6))"
    )

    assert formula == ForAll(
        ("?s",),
        Iff(
            InSet(
                Function("ranking", ("?s",)),
                ("Average", "Weak", "Poor"),
            ),
            Exists(
                ("?t",),
                Compare(">=", Function("membership_duration", ("?t",)), Number("6")),
            ),
        ),
    )


def test_typed_formula_json_parser_accepts_quantified_numeric_terms() -> None:
    formula = _formula_from_raw(
        {
            "type": "forall",
            "variables": ["student"],
            "body": {
                "type": "implies",
                "antecedent": {
                    "type": "compare",
                    "op": ">=",
                    "left": {
                        "type": "function",
                        "name": "membership_duration",
                        "args": ["student"],
                    },
                    "right": {"type": "number", "value": "6"},
                },
                "consequent": {
                    "type": "atom",
                    "pred": "eligible_trainer",
                    "args": ["student"],
                },
            },
        }
    )

    assert formula == ForAll(
        ("?student",),
        Implies(
            Compare(">=", Function("membership_duration", ("?student",)), Number("6")),
            Atom("eligible_trainer", ("?student",), text="eligible_trainer(?student)"),
        ),
    )


def test_typed_smt_solver_handles_numeric_membership_and_existential_reasoning() -> None:
    premise_texts = (
        "membership_duration(Alex) = 8",
        "ForAll(x, membership_duration(x) ≥ 6 → eligible_trainer(x))",
        "ranking(Alex) = Average",
        "ForAll(x, ranking(x) ∈ {Average, Weak, Poor} → must_attend_workshop(x))",
    )
    problem = TranslatedProblem(
        predicates={"eligible_trainer": 1, "must_attend_workshop": 1, "verified_trainer": 1},
        premises=(
            *tuple(
                FormulaItem(parse_fol(text), index, text, "premise")
                for index, text in enumerate(premise_texts)
            ),
            FormulaItem(
                ForAll(
                    ("person",),
                    Implies(
                        Atom("eligible_trainer", ("person",)),
                        Atom("verified_trainer", ("person",)),
                    ),
                ),
                len(premise_texts),
                "Every eligible trainer is verified.",
                "premise",
            ),
        ),
        goals=(
            FormulaItem(
                parse_fol(
                    "Exists(x, eligible_trainer(x) ∧ must_attend_workshop(x) "
                    "∧ verified_trainer(x))"
                ),
                -1,
                "At least one eligible trainer must attend the workshop.",
                "query",
            ),
        ),
    )

    result = Z3PropSolver().solve_query(problem)

    assert result.answer == "Yes"
    assert result.theory_status == "sat"
