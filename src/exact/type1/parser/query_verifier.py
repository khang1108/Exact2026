"""Pre-solve validation of a QuerySpec.

``QueryVerifier`` decides whether a question can be answered by the v1 solver
routing and records why not. It is deterministic and runs after the option
claims have been compiled (so it can see which options carry usable FOL).
"""

from __future__ import annotations

from exact.type1.parser.schemas import OptionClaim

# Solver modes recognized by the classifier but not yet solved in v1.
_DEFERRED_MODES = frozenset(
    {"strongest_conclusion", "fewest_premise", "premise_selection"}
)


def verify(
    question_format: str,
    solver_mode: str,
    main_claim_fol_present: bool,
    option_claims: tuple[OptionClaim, ...],
) -> tuple[bool, list[str]]:
    """Return (supported, issues) for one question before solving."""

    issues: list[str] = []

    if solver_mode == "unsupported" or question_format == "open_wh":
        issues.append("QUERY_OPEN_WH_UNSUPPORTED: no decidable single claim")
        return False, issues

    if solver_mode in _DEFERRED_MODES:
        issues.append(
            f"QUERY_MODE_DEFERRED: solver_mode '{solver_mode}' is classified but "
            f"not solved in v1"
        )
        return False, issues

    if question_format == "polar":
        if not main_claim_fol_present:
            issues.append("QUERY_NO_CLAIM: polar question has no testable claim FOL")
            return False, issues
        return True, issues

    # MCQ (entailment / refutation / ynu_mapped)
    if solver_mode == "ynu_mapped":
        if not _has_ynu_options(option_claims):
            issues.append("QUERY_NO_YNU_OPTIONS: ynu_mapped question has no Yes/No/Uncertain options")
            return False, issues
        if not main_claim_fol_present:
            issues.append("QUERY_NO_CLAIM: ynu_mapped question has no stem claim FOL to test")
            return False, issues
        return True, issues

    solvable = [c for c in option_claims if c.fol is not None]
    if len(solvable) < 2:
        if not option_claims:
            issues.append("QUERY_NO_SOLVABLE_OPTIONS: MCQ has no options")
        elif all(c.option_type in ("raw_fol", "premise_reference") for c in option_claims):
            issues.append(
                "QUERY_OPTIONS_UNSUPPORTED: all options are raw FOL or premise references"
            )
        else:
            issues.append(
                "QUERY_NO_SOLVABLE_OPTIONS: fewer than two options compiled to FOL"
            )
        return False, issues

    return True, issues


def _has_ynu_options(option_claims: tuple[OptionClaim, ...]) -> bool:
    return any(c.ynu_value in ("yes", "no", "uncertain") for c in option_claims)
