"""Tests for Type 1 question routing and multiple-choice pipeline behavior."""

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


def no_llm_settings() -> Settings:
    """Return deterministic settings that cannot trigger LLM translation."""

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

    response = run_type1_pipeline(
        request,
        settings=no_llm_settings(),
        allow_heuristic_fallback=True,
        question_type=QuestionType.MCQ,
    )

    assert response.question_type == QuestionType.MCQ
    assert response.answer == "A"


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
