from __future__ import annotations

from exact.config import Settings, get_settings
from exact.type2.fallback.executor import ExecutionResult, execute_python
from exact.type2.extraction.llm_structured import PotCodeSpec, generate_pot_code, repair_pot_code
from exact.type2.schemas import Extraction, Type2SolveResult, Verification


def run_pot_fallback(
    extraction: Extraction,
    reason: str,
    settings: Settings | None = None,
) -> Type2SolveResult:
    settings = settings or get_settings()
    llm_code = generate_pot_code(
        extraction.normalized_question,
        reason,
        settings=settings,
    )
    if llm_code is None:
        return _unconfigured_result(extraction, reason)

    attempt = _execute_generated_code(llm_code.code, settings.llm_timeout_seconds)
    if attempt.ok:
        return _result_from_execution(extraction, llm_code, attempt)

    repaired = repair_pot_code(
        extraction.normalized_question,
        llm_code.code,
        attempt.error or "execution failed",
        settings=settings,
    )
    if repaired is None:
        return _failed_result(extraction, reason, attempt.error or "execution failed")

    second = _execute_generated_code(repaired.code, settings.llm_timeout_seconds)
    if second.ok:
        return _result_from_execution(extraction, repaired, second)

    return _failed_result(extraction, second.error or "repair failed", second)


def _execute_generated_code(code: str, timeout_seconds: float) -> ExecutionResult:
    return execute_python(code, timeout_seconds=timeout_seconds)


def _result_from_execution(
    extraction: Extraction,
    spec: PotCodeSpec,
    execution: ExecutionResult,
) -> Type2SolveResult:
    answer = _stringify_answer(execution.ans)
    return Type2SolveResult(
        answer=answer,
        unit=spec.answer_unit,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(True, "PoT code executed successfully."),
        cot=[
            "A generated Python program solved the physics question.",
            "The program was executed in a sandbox and produced a result.",
        ],
        premises=[spec.explanation] if spec.explanation else [],
        confidence=0.55,
        error=None,
    )


def _failed_result(
    extraction: Extraction,
    reason: str,
    execution: ExecutionResult | None = None,
) -> Type2SolveResult:
    detail = reason
    if execution and execution.error:
        detail = f"{reason}: {execution.error}"
    return Type2SolveResult(
        answer="",
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(False, detail),
        cot=[
            "Deterministic solving did not produce a verified answer.",
            "PoT code generation/execution was attempted but did not succeed.",
        ],
        premises=[],
        confidence=0.0,
        error="type2_pot_failed",
    )


def _unconfigured_result(extraction: Extraction, reason: str) -> Type2SolveResult:
    return Type2SolveResult(
        answer="",
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(False, f"PoT fallback not configured: {reason}"),
        cot=[
            "Deterministic solving did not produce a verified answer.",
            "No LLM client is configured for PoT code generation.",
        ],
        premises=[],
        confidence=0.0,
        error="type2_pot_not_configured",
    )


def _stringify_answer(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)
