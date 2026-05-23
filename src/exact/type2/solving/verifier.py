from __future__ import annotations

import math
from dataclasses import replace

from exact.type2.schemas import Type2QuestionKind, Type2SolveResult, Verification


POST_SOLVER_VERIFICATION_ERROR = "type2_result_verification_failed"


class Type2ResultVerifier:
    """Validate a Type 2 solver result before the pipeline accepts it."""

    def verify(self, result: Type2SolveResult) -> Type2SolveResult:
        if result.error is not None:
            return result

        if not result.verification.ok:
            return self._reject(result, result.verification.message)

        if result.extraction.kind == Type2QuestionKind.CONCEPTUAL:
            return self._verify_conceptual(result)

        return self._verify_numerical(result)

    def _verify_conceptual(self, result: Type2SolveResult) -> Type2SolveResult:
        if not result.answer.strip():
            return self._reject(result, "Conceptual result has an empty answer.")
        if not result.premises:
            return self._reject(result, "Conceptual result has no supporting premise.")
        return result

    def _verify_numerical(self, result: Type2SolveResult) -> Type2SolveResult:
        if result.formula is None:
            return self._reject(result, "Numerical result has no formula.")
        if result.value is None:
            return self._reject(result, "Numerical result has no computed value.")
        if not result.answer.strip():
            return self._reject(result, "Numerical result has an empty answer.")
        if result.unit != result.formula.output_unit:
            return self._reject(
                result,
                f"Result unit `{result.unit}` does not match formula output unit "
                f"`{result.formula.output_unit}`.",
            )

        try:
            converted = result.value.to(result.formula.output_unit)
        except Exception as exc:
            return self._reject(result, f"Result value cannot convert to output unit: {exc}")

        magnitude = float(converted.magnitude)
        if not math.isfinite(magnitude):
            return self._reject(result, "Result magnitude is not finite.")

        try:
            answer = float(result.answer)
        except ValueError:
            return self._reject(result, f"Result answer `{result.answer}` is not numeric.")

        if not math.isclose(answer, magnitude, rel_tol=1e-6, abs_tol=1e-9):
            return self._reject(
                result,
                f"Result answer `{result.answer}` does not match computed value `{magnitude}`.",
            )

        return result

    def _reject(self, result: Type2SolveResult, reason: str) -> Type2SolveResult:
        return replace(
            result,
            verification=Verification(False, f"Post-solver verifier failed: {reason}"),
            cot=[
                *result.cot,
                "Post-solver verification rejected the result before fallback.",
            ],
            confidence=0.0,
            error=POST_SOLVER_VERIFICATION_ERROR,
        )


def verify_type2_result(result: Type2SolveResult) -> Type2SolveResult:
    return Type2ResultVerifier().verify(result)
