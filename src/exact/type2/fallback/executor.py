from __future__ import annotations

import ast
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys


ALLOWED_IMPORTS = {"math", "sympy", "pint"}


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    ans: object | None
    stdout: str
    stderr: str
    error: str | None = None


class UnsafeCodeError(ValueError):
    pass


def validate_code(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _validate_import(node)
        elif isinstance(node, ast.Call):
            _validate_call(node)
        elif isinstance(node, ast.Attribute):
            _validate_attribute(node)


def execute_python(code: str, timeout_seconds: float = 5.0) -> ExecutionResult:
    validate_code(code)
    runner = _build_runner_script()
    with tempfile.TemporaryDirectory(prefix="exact_type2_") as tmpdir:
        code_path = Path(tmpdir) / "program.py"
        code_path.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-I", "-c", runner, str(code_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

    if proc.returncode != 0:
        return ExecutionResult(
            ok=False,
            ans=None,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error=(proc.stderr.strip() or proc.stdout.strip() or "execution failed"),
        )

    ans = _parse_ans_marker(proc.stdout)
    if ans is None:
        return ExecutionResult(
            ok=False,
            ans=None,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error="ans not produced",
        )

    return ExecutionResult(ok=True, ans=ans, stdout=proc.stdout, stderr=proc.stderr)


def _validate_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
        names = [alias.name.split(".")[0] for alias in node.names]
    else:
        if node.module is None:
            raise UnsafeCodeError("relative imports are not allowed")
        names = [node.module.split(".")[0]]
    for name in names:
        if name not in ALLOWED_IMPORTS:
            raise UnsafeCodeError(f"import not allowed: {name}")


def _validate_call(node: ast.Call) -> None:
    if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "open", "__import__"}:
        raise UnsafeCodeError(f"call not allowed: {node.func.id}")


def _validate_attribute(node: ast.Attribute) -> None:
    if node.attr.startswith("__"):
        raise UnsafeCodeError("dunder attribute access is not allowed")


def _build_runner_script() -> str:
    return r"""
import builtins
import json
import math
import sys
from pathlib import Path

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "sum": sum,
    "tuple": tuple,
    "__import__": None,
    "int": int,
}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in {"math", "sympy", "pint"}:
        raise ImportError(f"import not allowed: {name}")
    return __import__(name, globals, locals, fromlist, level)

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
glb = {"__builtins__": dict(SAFE_BUILTINS, __import__=_safe_import)}

exec(compile(source, str(path), "exec"), glb, glb)

if "ans" not in glb:
    raise SystemExit("ans not defined")

value = glb["ans"]
if hasattr(value, "magnitude"):
    try:
        value = float(value.magnitude)
    except Exception:
        pass
elif hasattr(value, "item"):
    try:
        value = value.item()
    except Exception:
        pass

print("__ANS__=" + json.dumps(value, default=str))
"""


def _parse_ans_marker(stdout: str) -> object | None:
    marker = "__ANS__="
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            payload = line[len(marker) :]
            return _json_load(payload)
    return None


def _json_load(payload: str) -> object:
    import json

    return json.loads(payload)
