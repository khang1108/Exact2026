"""Z3 entailment over formula-level Type 1 IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exact.logic.ir import FormulaItem, TranslatedProblem
from exact.symbolic_solvers.z3_prop.encoder import (
    build_theory,
    collect_domain,
    encode_goal,
)

_OPTION_ORDER = ("A", "B", "C", "D")


class InconsistentTheoryError(RuntimeError):
    """Raised when translated premises are UNSAT and need repair."""


@dataclass(frozen=True)
class EncodedOption:
    label: str
    z3: Any
    item: FormulaItem


@dataclass(frozen=True)
class FormulaZ3Result:
    answer: str | None
    mode: str
    theory_status: str
    valid_labels: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    core_sizes: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Z3PropSolver:
    """Finite-domain propositional solver for full formula IR."""

    name: str = "z3_prop_formula"
    timeout_ms: int = 5000

    def solve_query(self, problem: TranslatedProblem) -> FormulaZ3Result:
        constraints, symbol_table = build_theory(problem)
        domain = collect_domain(problem)
        query = _single_goal(problem, role="query")
        if query is None:
            return FormulaZ3Result(
                answer=None,
                mode=self.name,
                theory_status="unknown",
                warnings=("problem has no query goal",),
            )

        phi = encode_goal(query, domain, symbol_table)
        status = theory_status(constraints, self.timeout_ms)
        if status == "unsat":
            return FormulaZ3Result(
                answer=None,
                mode=self.name,
                theory_status=status,
                warnings=("translated premises are inconsistent; trigger repair",),
            )

        try:
            answer = judge_query(constraints, phi, timeout_ms=self.timeout_ms)
        except InconsistentTheoryError as exc:
            return FormulaZ3Result(
                answer=None,
                mode=self.name,
                theory_status="unsat",
                warnings=(str(exc),),
            )
        return FormulaZ3Result(answer=answer, mode=self.name, theory_status=status)

    def solve_mcq(self, problem: TranslatedProblem, stem: str = "") -> FormulaZ3Result:
        constraints, symbol_table = build_theory(problem)
        domain = collect_domain(problem)
        options = tuple(
            EncodedOption(
                label=item.label or "",
                z3=encode_goal(item, domain, symbol_table),
                item=item,
            )
            for item in problem.goals
            if item.role == "option" and item.label
        )
        status = theory_status(constraints, self.timeout_ms)
        if status == "unsat":
            return FormulaZ3Result(
                answer=None,
                mode=self.name,
                theory_status=status,
                warnings=("translated premises are inconsistent; trigger repair",),
            )

        try:
            answer = judge_mcq(constraints, options, stem=stem, timeout_ms=self.timeout_ms)
        except InconsistentTheoryError as exc:
            return FormulaZ3Result(
                answer=None,
                mode=self.name,
                theory_status="unsat",
                warnings=(str(exc),),
            )

        valid = tuple(
            option.label
            for option in sorted(options, key=_option_sort_key)
            if entails(constraints, option.z3, timeout_ms=self.timeout_ms)
        )
        core_sizes = tuple(
            (option.label, unsat_core_size(constraints, option, timeout_ms=self.timeout_ms))
            for option in sorted(options, key=_option_sort_key)
            if option.label in valid
        )
        return FormulaZ3Result(
            answer=answer,
            mode=self.name,
            theory_status=status,
            valid_labels=valid,
            warnings=() if answer else ("no MCQ option was entailed; trigger repair",),
            core_sizes=core_sizes,
        )


def entails(constraints: list[Any], phi: Any, timeout_ms: int = 5000) -> bool:
    """Return True iff T entails phi, i.e. T and not(phi) is UNSAT."""

    from z3 import Not, unsat

    solver = _solver(timeout_ms)
    solver.add(*constraints)
    solver.add(Not(phi))
    return solver.check() == unsat


def judge_query(constraints: list[Any], phi: Any, timeout_ms: int = 5000) -> str:
    """Judge a Yes/No/Unknown query by two UNSAT entailment checks."""

    status = theory_status(constraints, timeout_ms)
    if status == "unsat":
        raise InconsistentTheoryError("translated premises are inconsistent; trigger repair")

    from z3 import Not

    if entails(constraints, phi, timeout_ms=timeout_ms):
        return "Yes"
    if entails(constraints, Not(phi), timeout_ms=timeout_ms):
        return "No"
    return "Unknown"


def judge_mcq(
    constraints: list[Any],
    options: tuple[EncodedOption, ...] | list[EncodedOption],
    stem: str = "",
    timeout_ms: int = 5000,
) -> str | None:
    """Return the selected MCQ label, or None to trigger repair/fallback."""

    status = theory_status(constraints, timeout_ms)
    if status == "unsat":
        raise InconsistentTheoryError("translated premises are inconsistent; trigger repair")

    ordered = sorted(options, key=_option_sort_key)
    valid = [
        option for option in ordered
        if entails(constraints, option.z3, timeout_ms=timeout_ms)
    ]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0].label

    stem_lower = stem.lower()
    if "fewest premises" in stem_lower or "most direct" in stem_lower:
        return min(
            valid,
            key=lambda option: (
                unsat_core_size(constraints, option, timeout_ms=timeout_ms),
                _option_sort_key(option),
            ),
        ).label

    if "strongest" in stem_lower:
        return max(
            valid,
            key=lambda option: (
                count_implied_siblings(option, valid, constraints, timeout_ms=timeout_ms),
                -_option_sort_key(option),
            ),
        ).label

    return valid[0].label


def theory_status(constraints: list[Any], timeout_ms: int = 5000) -> str:
    """Return sat, unsat, or unknown for the translated premise theory."""

    solver = _solver(timeout_ms)
    solver.add(*constraints)
    return str(solver.check())


def unsat_core_size(
    constraints: list[Any],
    option: EncodedOption,
    timeout_ms: int = 5000,
) -> int:
    """Count tracked premise constraints needed to entail an option."""

    from z3 import Bool, Not, unsat

    solver = _solver(timeout_ms)
    solver.set(unsat_core=True)
    for index, constraint in enumerate(constraints):
        solver.assert_and_track(constraint, Bool(f"premise_{index}"))
    solver.add(Not(option.z3))
    if solver.check() != unsat:
        return 10**9
    return len(solver.unsat_core())


def count_implied_siblings(
    option: EncodedOption,
    valid_options: list[EncodedOption],
    constraints: list[Any],
    timeout_ms: int = 5000,
) -> int:
    """Count how many other valid options follow from premises plus option."""

    augmented = [*constraints, option.z3]
    if theory_status(augmented, timeout_ms) != "sat":
        return -1
    return sum(
        1
        for sibling in valid_options
        if sibling.label != option.label
        and entails(augmented, sibling.z3, timeout_ms=timeout_ms)
    )


def _single_goal(problem: TranslatedProblem, role: str) -> FormulaItem | None:
    for item in problem.goals:
        if item.role == role:
            return item
    return None


def _solver(timeout_ms: int):
    from z3 import Solver

    solver = Solver()
    solver.set(timeout=timeout_ms)
    return solver


def _option_sort_key(option: EncodedOption) -> int:
    try:
        return _OPTION_ORDER.index(option.label)
    except ValueError:
        return len(_OPTION_ORDER)
