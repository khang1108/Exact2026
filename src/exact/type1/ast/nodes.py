"""First-order logic abstract syntax tree nodes produced by the Type 1 parser."""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from exact.type1.models.schemas import Predicate

__all__ = [
    "AtomicNode",
    "ComparisonNode",
    "DateTerm",
    "FOLNode",
    "FunctionTerm",
    "LogicalNode",
    "NumericTerm",
    "QuantifiedNode",
    "_extract_bound_var",
    "extract_bound_variable",
    "simplify",
]


@dataclass
class AtomicNode:
    """A predicate applied to variables or constants."""

    predicate: Predicate
    arguments: list[str]

    def __repr__(self) -> str:
        return f"{self.predicate.name}({', '.join(self.arguments)})"


@dataclass
class LogicalNode:
    """A logical operation over one or two child formulas."""

    operator: Literal["AND", "OR", "NOT", "IMPLIES", "IFF"]
    left: FOLNode
    right: FOLNode | None = None

    def __repr__(self) -> str:
        if self.operator == "NOT":
            return f"NOT({self.left})"
        return f"({self.left} {self.operator} {self.right})"


@dataclass
class QuantifiedNode:
    """A quantified formula whose body may contain further AST nodes."""

    quantifier: Literal["FORALL", "EXISTS"]
    variable: str
    body: FOLNode
    restrictor: FOLNode | None = None

    @property
    def scope(self) -> FOLNode:
        """Compatibility name for code that refers to a quantifier's scope."""

        return self.body

    def __repr__(self) -> str:
        symbol = "∀" if self.quantifier == "FORALL" else "∃"
        if self.restrictor is not None:
            body_str = str(self.body) if isinstance(self.body, LogicalNode) else f"({self.body})"
            return f"{symbol}{self.variable}[{self.restrictor}].{body_str}"
        if isinstance(self.body, LogicalNode):
            return f"{symbol}{self.variable}.{self.body}"
        return f"{symbol}{self.variable}.({self.body})"


@dataclass
class NumericTerm:
    """A numeric literal used on the right side of a comparison."""

    value: float

    def __repr__(self) -> str:
        return str(self.value)


@dataclass
class DateTerm:
    """A symbolic date or deadline reference (e.g. 'June1', 'AddDropDeadline')."""

    value: str

    def __repr__(self) -> str:
        return self.value


@dataclass
class FunctionTerm:
    """A numeric-valued attribute function applied to entity arguments."""

    name: str
    arguments: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.arguments:
            return f"{self.name}({', '.join(self.arguments)})"
        return self.name


@dataclass
class ComparisonNode:
    """An arithmetic or temporal comparison that yields a boolean formula."""

    operator: Literal["=", ">=", ">", "<=", "<", "!="]
    left: FunctionTerm
    right: NumericTerm | FunctionTerm | DateTerm

    def __repr__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"


FOLNode: TypeAlias = AtomicNode | LogicalNode | QuantifiedNode | ComparisonNode


def _extract_bound_var(node: FOLNode) -> str | None:
    """Return the first quantified variable found while walking ``node``."""

    if isinstance(node, QuantifiedNode):
        return node.variable
    if isinstance(node, LogicalNode):
        left_variable = _extract_bound_var(node.left)
        if left_variable:
            return left_variable
        if node.right is not None:
            return _extract_bound_var(node.right)
    if isinstance(node, ComparisonNode):
        return None
    return None


# Public name used by package consumers; keep the private alias for older code.
extract_bound_variable = _extract_bound_var


def simplify(node: FOLNode) -> FOLNode:
    """Eliminate double negations recursively: NOT(NOT(x)) → x."""
    if isinstance(node, AtomicNode):
        return node
    if isinstance(node, ComparisonNode):
        return node
    if isinstance(node, QuantifiedNode):
        return QuantifiedNode(
            node.quantifier,
            node.variable,
            simplify(node.body),
            simplify(node.restrictor) if node.restrictor is not None else None,
        )
    # LogicalNode
    if node.operator == "NOT":
        inner = simplify(node.left)
        if isinstance(inner, LogicalNode) and inner.operator == "NOT":
            return simplify(inner.left)
        return LogicalNode("NOT", inner)
    return LogicalNode(
        node.operator,
        simplify(node.left),
        simplify(node.right) if node.right is not None else None,
    )
