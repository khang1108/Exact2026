from __future__ import annotations

import asyncio
from typing import Any

from exact.common.schemas import PredictionRequest
from exact.type1.ast import AtomicNode, LogicalNode, QuantifiedNode
from exact.type1.ast.classifier import fast_classify
from exact.type1.models.schemas import Predicate
from exact.type1.parser import FOLParser
from exact.type1.parser.router import ParserRequest
from exact.type1.parser.schemas import AtomicResult, LogicalResult, QuantifiedResult
from exact.type1.pipeline import _clean_question, run_type1_pipeline
from exact.type1.refiner import Type1Refiner
from exact.type1.solvers import FOLSolver


def atom(name: str, *arguments: str) -> AtomicNode:
    return AtomicNode(
        predicate=Predicate(
            name=name,
            arg_sorts=["Entity"] * len(arguments),
            aliases=[],
        ),
        arguments=list(arguments),
    )


class StructuralParserClient:
    async def parse_request(self, request: ParserRequest) -> Any:
        text = request.messages[-1]["content"]
        if request.kind == "quantified":
            if "all students pass" in text.lower():
                return QuantifiedResult(
                    quantifier="ForAll",
                    variable="y",
                    scope_sentence="y passes",
                )
            return QuantifiedResult(
                quantifier="ForAll",
                variable="x",
                scope_sentence="x studies and x passes",
            )
        if request.kind == "logical":
            if "if all students study" in text.lower():
                return LogicalResult(
                    operator="IMPLIES",
                    left_operand="all students study",
                    right_operand="all students pass",
                )
            return LogicalResult(
                operator="AND",
                left_operand="x studies",
                right_operand="x passes",
            )
        return AtomicResult(
            predicate="Passes" if "passes" in text else "Studies",
            arguments=["y" if "y passes" in text else "x"],
            negated=False,
        )


class MappingParser:
    def __init__(self, nodes: dict[str, AtomicNode]) -> None:
        self.nodes = nodes

    async def parse_many(self, sentences) -> list[AtomicNode]:
        return [self.nodes[sentence] for sentence in sentences]


class FakeRefiner:
    def __init__(self, corrections: dict[str, str]) -> None:
        self.corrections = corrections

    async def refine(self, items) -> dict[str, str]:
        return self.corrections


class FakeRefinerClient:
    async def complete_json(self, **kwargs) -> dict:
        return {
            "corrections": [
                {
                    "id": "question",
                    "rephrased": "For every x, does x receive academic distinction?",
                },
                {
                    "id": "premise-1",
                    "rephrased": "For every x, if x studies, then x passes.",
                },
                {
                    "id": "premise-2",
                    "rephrased": "Alice does not study.",
                },
                {
                    "id": "premise-3",
                    "rephrased": "Professor John has a PhD degree.",
                },
            ]
        }


def test_quantified_scope_recurses_when_it_becomes_logical() -> None:
    async def scenario() -> None:
        parser = FOLParser(StructuralParserClient())  # type: ignore[arg-type]
        result = await parser.parse("Every student studies and passes")

        assert isinstance(result, QuantifiedNode)
        assert isinstance(result.body, LogicalNode)
        assert result.body.operator == "AND"

    asyncio.run(scenario())


def test_proposition_level_quantifiers_are_not_lifted() -> None:
    async def scenario() -> None:
        parser = FOLParser(StructuralParserClient())  # type: ignore[arg-type]
        result = await parser.parse("If all students study, then all students pass")

        assert isinstance(result, LogicalNode)
        assert isinstance(result.left, QuantifiedNode)
        assert isinstance(result.right, QuantifiedNode)

    asyncio.run(scenario())


def test_solver_entails_a_valid_rule_chain() -> None:
    solver = FOLSolver()
    rule = QuantifiedNode(
        "FORALL",
        "x",
        LogicalNode("IMPLIES", atom("Studies", "x"), atom("Passes", "x")),
    )

    result = solver.check_ynu([rule, atom("Studies", "Alice")], atom("Passes", "Alice"))
    assert result.answer == "Yes"
    assert result.premises_used == ["premise-1", "premise-2"]


def test_solver_does_not_explode_from_inconsistent_premises() -> None:
    solver = FOLSolver()
    negated = LogicalNode("NOT", atom("Studies", "Alice"))

    assert (
        solver.check_ynu([atom("Studies", "Alice"), negated], atom("Passes", "Alice")).answer
        == "Uncertain"
    )
    assert (
        solver.check_mcq(
        [atom("Studies", "Alice"), negated],
        {"A": atom("Passes", "Alice")},
        ).answer
        == "Uncertain"
    )


def test_classifier_does_not_quantify_object_all() -> None:
    assert fast_classify("John has completed all required courses.") == "atomic"
    assert fast_classify("John attends all classes.") == "atomic"
    assert fast_classify("All students complete required courses.") == "quantified"


def test_parser_injects_bound_variable_into_generic_consequent_subject() -> None:
    parser = FOLParser(StructuralParserClient())  # type: ignore[arg-type]

    assert parser._inject_variable(  # noqa: SLF001
        "the project is optimized",
        "x",
        "a Python code is well-tested",
    ) == "x is optimized"


def test_solver_mcq_prefers_smallest_nontrivial_support() -> None:
    solver = FOLSolver()
    direct = QuantifiedNode(
        "FORALL",
        "x",
        LogicalNode("IMPLIES", atom("Studies", "x"), atom("Passes", "x")),
    )
    second_step = QuantifiedNode(
        "FORALL",
        "x",
        LogicalNode("IMPLIES", atom("Passes", "x"), atom("Graduates", "x")),
    )
    tautology = LogicalNode("IMPLIES", atom("Graduates", "x"), atom("Graduates", "x"))

    result = solver.check_mcq(
        [direct, second_step, atom("Studies", "Alice")],
        {
            "A": atom("Passes", "Alice"),
            "B": atom("Graduates", "Alice"),
            "C": tautology,
        },
    )

    assert result.answer == "A"
    assert result.premises_used == ["premise-1", "premise-3"]


def test_refiner_rejects_goal_variable_and_negation_rewrites() -> None:
    async def scenario() -> None:
        refiner = Type1Refiner(FakeRefinerClient())  # type: ignore[arg-type]
        result = await refiner.refine(
            [
                {"id": "question", "nl": "Does John receive distinction?", "fol": "Receives(John)"},
                {"id": "premise-1", "nl": "If a student studies, they pass.", "fol": "P"},
                {"id": "premise-2", "nl": "Alice studies.", "fol": "Q"},
                {"id": "premise-3", "nl": "Professor John holds a PhD degree.", "fol": "R"},
            ]
        )

        assert result == {"premise-3": "Professor John has a PhD degree."}

    asyncio.run(scenario())


def test_pipeline_rejects_refinement_that_stays_uncertain() -> None:
    async def scenario() -> None:
        parser = MappingParser(
            {
                "Original premise": atom("Original", "Alice"),
                "Bad rewrite": atom("Other", "Alice"),
                "Target": atom("Target", "Alice"),
            }
        )
        payload = PredictionRequest(
            query_id="refinement-guard",
            type="type1",
            query="Target",
            premises=["Original premise"],
        )

        response = await run_type1_pipeline(
            payload,
            parser,  # type: ignore[arg-type]
            FOLSolver(),
            FakeRefiner({"premise-1": "Bad rewrite"}),  # type: ignore[arg-type]
        )

        assert response.answer == "Uncertain"
        assert response.fol is not None and "Original(Alice)" in response.fol
        diagnostics = response.routing_diagnostics or {}
        assert diagnostics["refinement_applied"] is False
        assert diagnostics["refinement_log"][0]["accepted"] is False

    asyncio.run(scenario())


def test_clean_question_removes_meta_reasoning_wrappers() -> None:
    assert _clean_question(
        "Does it follow that if all projects are structured, then all projects are optimized, "
        "according to the premises?"
    ) == "if all projects are structured, then all projects are optimized"
    assert _clean_question(
        "Is the statement true?\nStatement: If a code is optimized, then it is well-tested."
    ) == "If a code is optimized, then it is well-tested."


def test_pipeline_exposes_premises_used() -> None:
    async def scenario() -> None:
        parser = MappingParser(
            {
                "If Alice studies, then Alice passes": QuantifiedNode(
                    "FORALL",
                    "x",
                    LogicalNode("IMPLIES", atom("Studies", "x"), atom("Passes", "x")),
                ),
                "Alice studies": atom("Studies", "Alice"),
                "Alice passes": atom("Passes", "Alice"),
            }
        )
        payload = PredictionRequest(
            query_id="premises-used",
            type="type1",
            query="Alice passes",
            premises=["If Alice studies, then Alice passes", "Alice studies"],
        )
        response = await run_type1_pipeline(payload, parser, FOLSolver())  # type: ignore[arg-type]
        assert response.answer == "Yes"
        assert response.premises_used == ["premise-1", "premise-2"]

    asyncio.run(scenario())
