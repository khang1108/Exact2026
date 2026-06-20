"""Deterministic diagnostics for claim-to-premise proof connectivity.

Z3 can only connect a claim to premises when both sides use compatible
predicate signatures and constants. This module inspects the canonicalized
claim ASTs before solving so an uncertain result can be separated from an
obvious symbol-vocabulary mismatch.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, TypeVar

from exact.type1.ast.nodes import AtomicNode, ComparisonNode, FOLNode, LogicalNode, QuantifiedNode
from exact.type1.parser.schemas import ConstantSignature, PredicateSignature, PremiseSchema
from exact.type1.parser.schemas import QuestionParseBundle

_LOW_CONNECTIVITY_THRESHOLD = 0.75
_PRONOUN_RE = re.compile(
    r"\b(?:he|her|hers|him|his|it|its|she|their|theirs|them|they)\b",
    re.IGNORECASE,
)
_HARD_MISMATCH_CODES = frozenset(
    {
        "CLAIM_PREDICATE_NOT_IN_SCHEMA",
        "CLAIM_ARITY_NOT_IN_SCHEMA",
        "CLAIM_CONSTANT_NOT_IN_SCHEMA",
        "CLAIM_UNRESOLVED_PRONOUN",
        "CLAIM_SCHEMA_LOW_CONNECTIVITY",
        "CLAIM_CANONICALIZATION_BLOCKED",
    }
)
_T = TypeVar("_T")


def build_proof_connectivity_dashboard(
    q_bundle: QuestionParseBundle,
    schema: PremiseSchema,
) -> dict[str, Any]:
    """Return claim-level symbol connectivity diagnostics for one question."""

    claims: list[dict[str, Any]] = []
    if q_bundle.main_claim_fol is not None or q_bundle.spec.main_claim_text:
        claims.append(
            analyze_claim_connectivity(
                claim_id="question",
                claim_text=q_bundle.spec.main_claim_text,
                fol=q_bundle.main_claim_fol,
                schema=schema,
                expected_to_compile=True,
            )
        )

    for option in q_bundle.spec.option_claims:
        claims.append(
            analyze_claim_connectivity(
                claim_id=option.label,
                claim_text=option.claim_text or option.normalized_text,
                fol=option.fol,
                schema=schema,
                expected_to_compile=option.is_selectable,
            )
        )

    mismatch_claims = [
        claim["claim_id"]
        for claim in claims
        if any(code in _HARD_MISMATCH_CODES for code in claim["diagnostic_codes"])
    ]
    low_connectivity_claims = [
        claim["claim_id"]
        for claim in claims
        if "CLAIM_SCHEMA_LOW_CONNECTIVITY" in claim["diagnostic_codes"]
    ]
    analyzed_scores = [
        claim["proof_connectivity_score"]
        for claim in claims
        if claim["proof_connectivity_score"] is not None
    ]

    dashboard = {
        "claims": claims,
        "claim_count": len(claims),
        "minimum_score": min(analyzed_scores) if analyzed_scores else None,
        "mean_score": (
            round(sum(analyzed_scores) / len(analyzed_scores), 2)
            if analyzed_scores
            else None
        ),
        "claims_with_symbol_mismatch": mismatch_claims,
        "low_connectivity_claims": low_connectivity_claims,
    }
    dashboard["report"] = _format_report(claims)
    return dashboard


def analyze_claim_connectivity(
    *,
    claim_id: str,
    claim_text: str | None,
    fol: FOLNode | None,
    schema: PremiseSchema,
    expected_to_compile: bool = True,
) -> dict[str, Any]:
    """Compare one claim's symbols with the premise-derived schema."""

    unresolved_pronouns = _unresolved_pronouns(claim_text, fol)
    diagnostics: list[str] = []
    if unresolved_pronouns:
        diagnostics.append(
            "CLAIM_UNRESOLVED_PRONOUN: " + ", ".join(unresolved_pronouns)
        )

    if fol is None:
        if expected_to_compile:
            diagnostics.append("CLAIM_CANONICALIZATION_BLOCKED: no claim FOL was produced")
            diagnostics.append(
                "CLAIM_SCHEMA_LOW_CONNECTIVITY: "
                f"score=0.0 threshold={_LOW_CONNECTIVITY_THRESHOLD}"
            )
            score: float | None = 0.0
        else:
            score = None
        return _claim_result(
            claim_id=claim_id,
            claim_text=claim_text,
            predicates=[],
            constants=[],
            unresolved_pronouns=unresolved_pronouns,
            score=score,
            diagnostics=diagnostics,
        )

    predicate_results = [
        _analyze_predicate(name, arity, schema.predicates)
        for name, arity in _collect_predicates(fol)
    ]
    constant_results = [
        _analyze_constant(name, schema.constants)
        for name in _collect_constants(fol)
    ]

    for predicate in predicate_results:
        signature = predicate["signature"]
        if not predicate["found_in_premise_schema"]:
            diagnostics.append(f"CLAIM_PREDICATE_NOT_IN_SCHEMA: {signature}")
        if predicate["arity_mismatch"]:
            available = ", ".join(f"/{arity}" for arity in predicate["available_arities"])
            diagnostics.append(
                f"CLAIM_ARITY_NOT_IN_SCHEMA: {signature}; available arities: {available}"
            )
        if predicate["suspicious_semantic_drift"]:
            diagnostics.append(
                "CLAIM_CANONICALIZATION_BLOCKED: "
                f"{signature} may be semantic drift from {predicate['maybe_schema_match']}"
            )

    for constant in constant_results:
        if not constant["found_in_premise_constants"]:
            detail = f"CLAIM_CONSTANT_NOT_IN_SCHEMA: {constant['name']}"
            if constant["maybe_alias_of"]:
                detail += f"; maybe alias of {constant['maybe_alias_of']}"
            diagnostics.append(detail)

    symbol_scores = [
        result["_score"] for result in [*predicate_results, *constant_results]
    ]
    score = sum(symbol_scores) / len(symbol_scores) if symbol_scores else 0.0
    if unresolved_pronouns:
        score *= 0.5
    score = round(score, 2)
    if score < _LOW_CONNECTIVITY_THRESHOLD:
        diagnostics.append(
            f"CLAIM_SCHEMA_LOW_CONNECTIVITY: score={score} "
            f"threshold={_LOW_CONNECTIVITY_THRESHOLD}"
        )

    for result in [*predicate_results, *constant_results]:
        result.pop("_score", None)
    return _claim_result(
        claim_id=claim_id,
        claim_text=claim_text,
        predicates=predicate_results,
        constants=constant_results,
        unresolved_pronouns=unresolved_pronouns,
        score=score,
        diagnostics=diagnostics,
    )


def interpret_z3_uncertainty(dashboard: dict[str, Any]) -> str:
    """Classify solver uncertainty using the pre-solve connectivity dashboard."""

    if dashboard["claims_with_symbol_mismatch"]:
        return "SYMBOL_MISMATCH_LIKELY"
    return "REAL_LOGICAL_UNCERTAINTY"


def _analyze_predicate(
    name: str,
    arity: int,
    signatures: tuple[PredicateSignature, ...],
) -> dict[str, Any]:
    exact = next(
        (
            signature
            for signature in signatures
            if signature.arity == arity
            and name.casefold()
            in {signature.name.casefold(), *(alias.casefold() for alias in signature.aliases)}
        ),
        None,
    )
    same_name = [
        signature
        for signature in signatures
        if name.casefold()
        in {signature.name.casefold(), *(alias.casefold() for alias in signature.aliases)}
    ]
    candidate, similarity = _closest_name(
        name,
        [
            (signature.name, signature)
            for signature in signatures
            if signature.arity == arity
        ],
    )
    suspicious = exact is None and not same_name and candidate is not None
    if exact is not None:
        symbol_score = 1.0
        match = f"{exact.name}/{exact.arity}"
    elif same_name:
        symbol_score = 0.25
        match = same_name[0].name
    elif candidate is not None:
        symbol_score = 0.4
        match = f"{candidate.name}/{candidate.arity}"
    else:
        symbol_score = 0.0
        match = None
    return {
        "name": name,
        "arity": arity,
        "signature": f"{name}/{arity}",
        "found_in_premise_schema": exact is not None,
        "schema_match": f"{exact.name}/{exact.arity}" if exact is not None else None,
        "available_arities": sorted({signature.arity for signature in same_name}),
        "arity_mismatch": bool(same_name) and exact is None,
        "maybe_schema_match": match if exact is None else None,
        "semantic_similarity": round(similarity, 2) if candidate is not None else None,
        "suspicious_semantic_drift": suspicious,
        "_score": symbol_score,
    }


def _analyze_constant(
    name: str,
    signatures: tuple[ConstantSignature, ...],
) -> dict[str, Any]:
    exact = next(
        (
            signature
            for signature in signatures
            if name.casefold()
            in {signature.name.casefold(), *(alias.casefold() for alias in signature.aliases)}
        ),
        None,
    )
    candidate, similarity = _closest_name(
        name,
        [(signature.name, signature) for signature in signatures],
    )
    return {
        "name": name,
        "found_in_premise_constants": exact is not None,
        "schema_match": exact.name if exact is not None else None,
        "maybe_alias_of": candidate.name if exact is None and candidate is not None else None,
        "semantic_similarity": round(similarity, 2) if candidate is not None else None,
        "_score": 1.0 if exact is not None else (0.5 if candidate is not None else 0.0),
    }


def _closest_name(
    target: str,
    candidates: list[tuple[str, _T]],
) -> tuple[_T | None, float]:
    best: _T | None = None
    best_score = 0.0
    for candidate_name, candidate in candidates:
        score = _name_similarity(target, candidate_name)
        if score > best_score:
            best = candidate
            best_score = score
    if best_score < 0.58:
        return None, best_score
    return best, best_score


def _name_similarity(left: str, right: str) -> float:
    left_compact = _compact_name(left)
    right_compact = _compact_name(right)
    if not left_compact or not right_compact:
        return 0.0
    sequence_score = SequenceMatcher(None, left_compact, right_compact).ratio()
    containment_score = (
        0.75
        if min(len(left_compact), len(right_compact)) >= 4
        and (left_compact in right_compact or right_compact in left_compact)
        else 0.0
    )
    return max(sequence_score, containment_score)


def _compact_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _collect_predicates(node: FOLNode) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []

    def walk(current: FOLNode) -> None:
        if isinstance(current, AtomicNode):
            found.append((current.predicate.name, len(current.arguments)))
        elif isinstance(current, QuantifiedNode):
            walk(current.body)
            if current.restrictor is not None:
                walk(current.restrictor)
        elif isinstance(current, LogicalNode):
            walk(current.left)
            if current.right is not None:
                walk(current.right)

    walk(node)
    return list(dict.fromkeys(found))


def _collect_constants(
    node: FOLNode,
    bound_variables: frozenset[str] = frozenset(),
) -> list[str]:
    if isinstance(node, AtomicNode):
        values = [argument for argument in node.arguments if argument not in bound_variables]
    elif isinstance(node, ComparisonNode):
        values = [argument for argument in node.left.arguments if argument not in bound_variables]
    elif isinstance(node, QuantifiedNode):
        local_variables = bound_variables | {node.variable}
        values = _collect_constants(node.body, local_variables)
        if node.restrictor is not None:
            values.extend(_collect_constants(node.restrictor, local_variables))
    else:
        values = _collect_constants(node.left, bound_variables)
        if node.right is not None:
            values.extend(_collect_constants(node.right, bound_variables))
    return list(dict.fromkeys(values))


def _unresolved_pronouns(claim_text: str | None, fol: FOLNode | None) -> list[str]:
    found = _PRONOUN_RE.findall(claim_text or "")
    if fol is not None:
        found.extend(
            constant for constant in _collect_constants(fol) if _PRONOUN_RE.fullmatch(constant)
        )
    return list(dict.fromkeys(pronoun.casefold() for pronoun in found))


def _claim_result(
    *,
    claim_id: str,
    claim_text: str | None,
    predicates: list[dict[str, Any]],
    constants: list[dict[str, Any]],
    unresolved_pronouns: list[str],
    score: float | None,
    diagnostics: list[str],
) -> dict[str, Any]:
    diagnostic_codes = list(
        dict.fromkeys(item.split(":", 1)[0] for item in diagnostics)
    )
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_predicates": predicates,
        "constants": constants,
        "unresolved_pronouns": unresolved_pronouns,
        "has_suspicious_semantic_drift": any(
            predicate.get("suspicious_semantic_drift", False)
            for predicate in predicates
        ),
        "canonicalization_blocked": (
            "CLAIM_CANONICALIZATION_BLOCKED" in diagnostic_codes
        ),
        "proof_connectivity_score": score,
        "diagnostic_codes": diagnostic_codes,
        "diagnostics": list(dict.fromkeys(diagnostics)),
    }


def _format_report(claims: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for claim in claims:
        heading = (
            "Question claim:"
            if claim["claim_id"] == "question"
            else f"Option {claim['claim_id']}:"
        )
        lines.extend([heading, "  claim predicates:"])
        if claim["claim_predicates"]:
            for predicate in claim["claim_predicates"]:
                found = "yes" if predicate["found_in_premise_schema"] else "no"
                detail = (
                    f"    {predicate['signature']} -> found in premise schema: {found}"
                )
                if predicate["maybe_schema_match"]:
                    detail += f"; maybe match: {predicate['maybe_schema_match']}"
                lines.append(detail)
        else:
            lines.append("    (none)")

        lines.append("  constants:")
        if claim["constants"]:
            for constant in claim["constants"]:
                found = "yes" if constant["found_in_premise_constants"] else "no"
                detail = f"    {constant['name']} -> found: {found}"
                if constant["maybe_alias_of"]:
                    detail += f"; maybe alias of {constant['maybe_alias_of']}"
                lines.append(detail)
        else:
            lines.append("    (none)")

        if claim["unresolved_pronouns"]:
            lines.append(
                "  unresolved_pronouns: " + ", ".join(claim["unresolved_pronouns"])
            )
        lines.append(
            f"  proof_connectivity_score: {claim['proof_connectivity_score']}"
        )
    return "\n".join(lines)
