"""Z3 entailment over formula-level Type 1 IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exact.logic.ir import Implies as FormulaImplies
from exact.logic.ir import FormulaItem, Not as FormulaNot, TranslatedProblem
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
    antecedent_witness: Any | None = None


@dataclass(frozen=True)
class FormulaZ3Result:
    answer: str | None
    mode: str
    theory_status: str
    valid_labels: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    core_sizes: tuple[tuple[str, int], ...] = ()
    supporting_premises: tuple[int, ...] = ()


@dataclass(frozen=True)
class Z3PropSolver:
    """Typed SMT solver for full formula IR with native quantifiers."""

    name: str = "z3_typed_formula"
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
        supporting_premises: tuple[int, ...] = ()
        if answer == "Yes":
            supporting_premises = unsat_core_indices(constraints, phi, timeout_ms=self.timeout_ms)
        elif answer == "No":
            from z3 import Not

            supporting_premises = unsat_core_indices(
                constraints, Not(phi), timeout_ms=self.timeout_ms
            )
        return FormulaZ3Result(
            answer=answer,
            mode=self.name,
            theory_status=status,
            supporting_premises=supporting_premises,
        )

    def solve_mcq(self, problem: TranslatedProblem, stem: str = "") -> FormulaZ3Result:
        constraints, symbol_table = build_theory(problem)
        domain = collect_domain(problem)
        options = tuple(
            EncodedOption(
                label=item.label or "",
                z3=encode_goal(item, domain, symbol_table),
                item=item,
                antecedent_witness=_encode_antecedent_witness(item, domain, symbol_table),
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

        valid_options = _usable_entailed_options(constraints, options, timeout_ms=self.timeout_ms)
        valid = tuple(option.label for option in valid_options)
        core_sizes = tuple(
            (option.label, unsat_core_size(constraints, option, timeout_ms=self.timeout_ms))
            for option in sorted(options, key=_option_sort_key)
            if option.label in valid
        )
        selected = next((option for option in options if option.label == answer), None)
        supporting_premises = (
            unsat_core_indices(constraints, selected.z3, timeout_ms=self.timeout_ms)
            if selected is not None
            else ()
        )
        return FormulaZ3Result(
            answer=answer,
            mode=self.name,
            theory_status=status,
            valid_labels=valid,
            warnings=() if answer else ("no MCQ option was entailed; trigger repair",),
            core_sizes=core_sizes,
            supporting_premises=supporting_premises,
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
    valid = _usable_entailed_options(constraints, ordered, timeout_ms=timeout_ms)
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0].label

    stem_lower = stem.lower()
    if (
        "fewest premises" in stem_lower
        or "most direct" in stem_lower
    ):
        return min(
            valid,
            key=lambda option: (
                unsat_core_size(constraints, option, timeout_ms=timeout_ms),
                _option_sort_key(option),
            ),
        ).label
    if "strongest" in stem_lower:
        # In educational QA, "strongest" asks for the most directly supported
        # usable conclusion. Comparing siblings under the full theory masks the
        # distinction once every candidate has already been proven.
        return min(
            valid,
            key=lambda option: (
                unsat_core_size(constraints, option, timeout_ms=timeout_ms),
                _option_sort_key(option),
            ),
        ).label

    return valid[0].label


def theory_status(constraints: list[Any], timeout_ms: int = 5000) -> str:
    """Return sat, unsat, or unknown for the translated premise theory."""

    solver = _solver(timeout_ms)
    solver.add(*constraints)
    return str(solver.check())


def _usable_entailed_options(
    constraints: list[Any],
    options: tuple[EncodedOption, ...] | list[EncodedOption],
    timeout_ms: int,
) -> list[EncodedOption]:
    entailed = [
        option
        for option in sorted(options, key=_option_sort_key)
        if entails(constraints, option.z3, timeout_ms=timeout_ms)
    ]
    substantive = [
        option
        for option in entailed
        if not _is_vacuous_implication(constraints, option, timeout_ms=timeout_ms)
    ]
    return substantive or entailed


def _is_vacuous_implication(
    constraints: list[Any],
    option: EncodedOption,
    timeout_ms: int,
) -> bool:
    """Return True when an implication is true only because its antecedent is impossible."""

    if option.antecedent_witness is None:
        return False
    from z3 import sat

    solver = _solver(timeout_ms)
    solver.add(*constraints)
    solver.add(option.antecedent_witness)
    return solver.check() != sat


def _encode_antecedent_witness(item: FormulaItem, domain, symbol_table) -> Any | None:
    """Encode ``exists vars. antecedent`` using ``not forall vars. not antecedent``."""

    if not isinstance(item.formula, FormulaImplies):
        return None
    from z3 import Not

    negated_antecedent = FormulaItem(
        formula=FormulaNot(item.formula.antecedent),
        source_idx=item.source_idx,
        text=item.text,
        role=item.role,
        label=item.label,
    )
    return Not(encode_goal(negated_antecedent, domain, symbol_table))


def unsat_core_size(
    constraints: list[Any],
    option: EncodedOption,
    timeout_ms: int = 5000,
) -> int:
    """Count tracked premise constraints needed to entail an option."""

    indices = unsat_core_indices(constraints, option.z3, timeout_ms=timeout_ms)
    if indices or entails(constraints, option.z3, timeout_ms=timeout_ms):
        return len(indices)
    return 10**9


def unsat_core_indices(
    constraints: list[Any],
    phi: Any,
    timeout_ms: int = 5000,
) -> tuple[int, ...]:
    """Return premise indices from one Z3 core that entail ``phi``."""

    from z3 import Bool, Not, unsat

    solver = _solver(timeout_ms)
    solver.set(unsat_core=True)
    markers = []
    for index, constraint in enumerate(constraints):
        marker = Bool(f"premise_{index}")
        markers.append(marker)
        solver.assert_and_track(constraint, marker)
    solver.add(Not(phi))
    if solver.check() != unsat:
        return ()
    core = set(solver.unsat_core())
    return tuple(index for index, marker in enumerate(markers) if marker in core)


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
