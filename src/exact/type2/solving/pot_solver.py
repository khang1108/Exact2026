from __future__ import annotations

import ast
import re

from exact.config import Settings, get_settings
from exact.type2.extraction.llm_structured import (
    PotCodeSpec,
    generate_final_explanation,
    generate_pot_code,
    repair_pot_code,
)
from exact.type2.fallback.executor import ExecutionResult, execute_python
from exact.type2.formulas.knowledge import RetrievedFormulaContext, canonicalize_formula_ids
from exact.type2.schemas import Extraction, Type2SolveResult, Verification
from exact.type2.solving.solver import solve_extraction
from exact.type2.solving.pot_verifier import verify_pot_execution


POT_SOLVER_NOT_CONFIGURED = "type2_pot_solver_not_configured"
POT_SOLVER_FAILED = "type2_pot_solver_failed"


def solve_with_pot(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
    settings: Settings | None = None,
    generate_explanation: bool = True,
) -> Type2SolveResult:
    settings = settings or get_settings()
    prompt_context = _build_solver_context(extraction, formula_context)
    try:
        code_spec = generate_pot_code(
            extraction.normalized_question,
            "Use the retrieved formulas to solve the problem with Pint.",
            formula_context=prompt_context,
            settings=settings,
        )
    except Exception as exc:
        return _failed_result(extraction, f"LLM code generation returned invalid output: {exc}")
    if code_spec is None:
        return _unconfigured_result(extraction)

    code_spec = _canonicalize_formula_ids(code_spec, formula_context)
    execution = _execute_code_spec(code_spec, settings.llm_timeout_seconds)
    if not execution.ok:
        try:
            repaired = repair_pot_code(
                extraction.normalized_question,
                code_spec.code,
                execution.error or "execution failed",
                settings=settings,
            )
        except Exception as exc:
            return _failed_result(extraction, f"LLM code repair returned invalid output: {exc}")
        if repaired is None:
            fallback = _try_executable_formula_fallback(extraction, formula_context, execution.error)
            if fallback is not None:
                return fallback
            return _failed_result(extraction, execution.error or "execution failed")
        code_spec = repaired
        code_spec = _canonicalize_formula_ids(code_spec, formula_context)
        execution = _execute_code_spec(code_spec, settings.llm_timeout_seconds)

    if not execution.ok:
        fallback = _try_executable_formula_fallback(extraction, formula_context, execution.error)
        if fallback is not None:
            return fallback
        return _failed_result(extraction, execution.error or "execution failed after repair")

    unit = execution.ans_unit or code_spec.answer_unit
    verified = verify_pot_execution(
        execution.ans,
        unit,
        code_spec.formula_ids_used,
        formula_context.formula_ids,
        magnitude_target=extraction.target in {"force", "electric_field"},
    )
    if verified.error is not None:
        fallback = _try_executable_formula_fallback(extraction, formula_context, verified.verification.message)
        if fallback is not None:
            return fallback
        return Type2SolveResult(
            answer="",
            unit=None,
            value=None,
            formula=None,
            extraction=extraction,
            verification=verified.verification,
            cot=["PoT code executed, but verifier rejected the result."],
            premises=[],
            confidence=0.0,
            error=verified.error,
        )

    explanation = _final_explanation(
        extraction,
        verified.answer,
        verified.unit,
        formula_context,
        code_spec,
        settings,
        generate_explanation=generate_explanation,
    )
    return Type2SolveResult(
        answer=verified.answer,
        unit=verified.unit,
        value=verified.value,
        formula=None,
        extraction=extraction,
        verification=verified.verification,
        cot=[
            "Retrieved formula context for the question.",
            "Generated a Pint-based Python program with the LLM.",
            "Executed the program in the sandbox.",
            "Verified the numeric answer, unit, and formula IDs.",
            *explanation.cot,
        ],
        premises=explanation.premises,
        confidence=0.72,
        error=None,
    )


def _execute_code_spec(spec: PotCodeSpec, timeout_seconds: float) -> ExecutionResult:
    code = _prepare_generated_code(spec.code)
    return execute_python(code, timeout_seconds=timeout_seconds)


def _canonicalize_formula_ids(spec: PotCodeSpec, formula_context: RetrievedFormulaContext) -> PotCodeSpec:
    canonical_ids = canonicalize_formula_ids(spec.formula_ids_used, formula_context.summaries)
    if canonical_ids == spec.formula_ids_used:
        return spec
    return spec.model_copy(update={"formula_ids_used": canonical_ids})

def _try_executable_formula_fallback(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
    reason: str | None,
) -> Type2SolveResult | None:
    result = solve_extraction(extraction, preferred_formula_ids=formula_context.formula_ids)
    if result.error is not None:
        return None
    result.cot.insert(0, f"PoT solver failed; executable formula fallback was used. Reason: {reason or 'unknown'}")
    return result
def _strip_code_fence(code: str) -> str:
    text = code.strip()
    match = re.fullmatch(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _prepare_generated_code(code: str) -> str:
    stripped = _strip_code_fence(code)
    normalized = _normalize_pint_prefixes(stripped)
    normalized = _normalize_bare_pint_unit_aliases(normalized)
    normalized = _normalize_pint_constants(normalized)
    normalized = _rewrite_sqrt_calls_for_pint_quantities(normalized)
    return _remove_unused_optional_imports(normalized)


def _normalize_bare_pint_unit_aliases(code: str) -> str:
    """Repair common LLM unit aliases inside Pint expressions."""

    replacements = {
        "m": "ureg.meter",
        "s": "ureg.second",
        "kg": "ureg.kilogram",
        "C": "ureg.coulomb",
        "N": "ureg.newton",
        "J": "ureg.joule",
        "V": "ureg.volt",
        "A": "ureg.ampere",
        "ohm": "ureg.ohm",
    }
    pattern = re.compile(
        r"(?P<op>[*/])\s*(?P<unit>" + "|".join(map(re.escape, replacements)) + r")\b"
        r"(?P<power>\s*\*\*\s*[-+]?\d+)?"
    )

    def replace(match: re.Match[str]) -> str:
        unit = match.group("unit")
        return f"{match.group('op')} {replacements[unit]}{match.group('power') or ''}"

    return pattern.sub(replace, code)


def _normalize_pint_constants(code: str) -> str:
    coulomb_constant = "8.9875517923e9 * ureg.newton * ureg.meter ** 2 / ureg.coulomb ** 2"
    code = code.replace("ureg.constants.Coulomb_constant", f"({coulomb_constant})")
    code = code.replace("ureg.constants.coulomb_constant", f"({coulomb_constant})")
    return code


def _rewrite_sqrt_calls_for_pint_quantities(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    rewritten = _SqrtToPower().visit(tree)
    ast.fix_missing_locations(rewritten)
    try:
        return ast.unparse(rewritten)
    except Exception:
        return code


class _SqrtToPower(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if len(node.args) != 1 or node.keywords:
            return node
        if _is_sqrt_call(node.func):
            return ast.BinOp(
                left=node.args[0],
                op=ast.Pow(),
                right=ast.Constant(value=0.5),
            )
        return node


def _is_sqrt_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Attribute):
        return func.attr == "sqrt" and isinstance(func.value, ast.Name) and func.value.id in {"sympy", "sp"}
    if isinstance(func, ast.Name):
        return func.id == "sqrt"
    return False


def _remove_unused_optional_imports(code: str) -> str:
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped in {"import numpy as np", "import numpy"}:
            continue
        if stripped.startswith("from numpy import "):
            continue
        lines.append(line)
    return "\n".join(lines)


_PINT_PREFIX_SCALE = {
    "tera": "1e12",
    "giga": "1e9",
    "mega": "1e6",
    "kilo": "1e3",
    "centi": "1e-2",
    "milli": "1e-3",
    "micro": "1e-6",
    "nano": "1e-9",
    "pico": "1e-12",
}


def _normalize_pint_prefixes(code: str) -> str:
    """Convert LLM-written Pint prefix objects into numeric scale factors."""

    def replace_prefix(match: re.Match[str]) -> str:
        prefix = match.group(1)
        return f"{_PINT_PREFIX_SCALE[prefix]} * ureg."

    return re.sub(
        r"ureg\.(" + "|".join(_PINT_PREFIX_SCALE) + r")\s*\*\s*ureg\.",
        replace_prefix,
        code,
    )


def _build_solver_context(
    extraction: Extraction,
    formula_context: RetrievedFormulaContext,
) -> str:
    quantities = "\n".join(
        f"- {key}: {quantity.value} ({quantity.evidence})"
        for key, quantity in extraction.quantities.items()
    )
    allowed_ids = "\n- ".join(formula_context.formula_ids) if formula_context.formula_ids else "None"
    return (
        f"Extraction:\n"
        f"- kind: {extraction.kind.value}\n"
        f"- target: {extraction.target}\n"
        f"- quantities:\n{quantities or '- none'}\n\n"
        f"Allowed formula IDs:\n- {allowed_ids}\n\n"
        f"Retrieved formulas:\n{formula_context.context}"
    )


def _final_explanation(
    extraction: Extraction,
    answer: str,
    unit: str | None,
    formula_context: RetrievedFormulaContext,
    code_spec: PotCodeSpec,
    settings: Settings,
    generate_explanation: bool,
):
    from exact.type2.extraction.llm_structured import FinalExplanationSpec

    unit_suffix = f" {unit}" if unit else ""
    if generate_explanation:
        spec = generate_final_explanation(
            extraction.normalized_question,
            answer,
            unit,
            formula_context.context,
            code_spec.explanation or "The generated program computed the verified result.",
            code_spec.formula_ids_used,
            settings=settings,
        )
        if spec is not None:
            return spec

    return FinalExplanationSpec(
        explanation=code_spec.explanation or f"Verified PoT computation gives {answer}{unit_suffix}.",
        premises=[] if not generate_explanation else [formula_context.context] if formula_context.context else [],
        cot=[] if not generate_explanation else ["No LLM final explanation was available; reused the code explanation."],
    )


def _unconfigured_result(extraction: Extraction) -> Type2SolveResult:
    return Type2SolveResult(
        answer="",
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(False, "PoT-first solver requires a configured LLM client."),
        cot=["No LLM client is configured for the PoT-first Type 2 solver."],
        premises=[],
        confidence=0.0,
        error=POT_SOLVER_NOT_CONFIGURED,
    )


def _failed_result(extraction: Extraction, reason: str) -> Type2SolveResult:
    return Type2SolveResult(
        answer="",
        unit=None,
        value=None,
        formula=None,
        extraction=extraction,
        verification=Verification(False, reason),
        cot=["PoT-first solving attempted code generation/execution but failed."],
        premises=[],
        confidence=0.0,
        error=POT_SOLVER_FAILED,
    )
