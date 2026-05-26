"""Tests for Type 1 question routing and multiple-choice pipeline behavior."""

import pytest

from exact.config import Settings
from exact.common.schemas import PredictionRequest, QuestionType, TaskType
from exact.logic.ir import Atom, Fact, Rule
from exact.logic.kb import KnowledgeBase
from exact.logic.pipeline import (
    decide_mcq_winner,
    evaluate_mcq_options,
    extract_options,
    run_type1_pipeline,
)
from exact.router.task_router import TaskRouter


class RecordingTranslatorClient:
    """Test double that records the prompt sent to the LLM translator."""

    def __init__(self) -> None:
        self.user_prompt = ""

    def complete_json_sync(self, messages, temperature: float = 0.0, max_tokens: int = 2048):
        self.user_prompt = str(messages[-1]["content"])
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


def explicit_llm_settings() -> Settings:
    """Return settings that require the supplied test LLM client."""

    return Settings(llm_provider="openai", llm_base_url=None)


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
            settings=Settings(llm_provider="openai", llm_base_url=None, mock_llm=True),
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
    assert "Allowed predicate names from premise translation: alpha" in client.query_prompt


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

    assert "0: Alpha." in client.user_prompt
    assert "Which conclusion follows" not in client.user_prompt
    assert "A. Alpha" not in client.user_prompt
    assert "B. Beta" not in client.user_prompt


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
