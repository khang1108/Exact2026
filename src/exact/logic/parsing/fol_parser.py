"""Parser for the mixed FOL notation shipped with the EXACT Type 1 dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass

from exact.logic.ir import (
    And,
    Arithmetic,
    Atom,
    Compare,
    Exists,
    ForAll,
    Formula,
    Function,
    Iff,
    Implies,
    InSet,
    Not,
    Number,
    Or,
    Term,
)

_TOKEN_RE = re.compile(
    r"""
    (?P<SPACE>\s+)
  | (?P<STRING>'[^']*'|"[^"]*")
  | (?P<IMPLIES>->|→|\bimplies\b)
  | (?P<IFF><->|↔)
  | (?P<GE>>=|≥)
  | (?P<LE><=|≤)
  | (?P<NE>!=|≠)
  | (?P<AND>&|∧)
  | (?P<OR>\||∨)
  | (?P<NOT>~|¬|\bnot\b)
  | (?P<QUANT>∀|∃)
  | (?P<IN>∈)
  | (?P<NUMBER>\d+(?:\.\d*)?|\.\d+)
  | (?P<ID>(?:[^\W\d]|_)\w*)
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<LBRACE>\{)
  | (?P<RBRACE>\})
  | (?P<COMMA>,)
  | (?P<PLUS>\+)
  | (?P<MINUS>-)
  | (?P<MUL>\*)
  | (?P<DIV>/)
  | (?P<POW>\^)
  | (?P<EQ>=)
  | (?P<GT>>)
  | (?P<LT><)
    """,
    re.IGNORECASE | re.UNICODE | re.VERBOSE,
)

_COMPARISONS = {
    "EQ": "=",
    "NE": "!=",
    "GT": ">",
    "GE": ">=",
    "LT": "<",
    "LE": "<=",
}
_ARITHMETIC = {
    "PLUS": "+",
    "MINUS": "-",
    "MUL": "*",
    "DIV": "/",
    "POW": "^",
}


class FolParseError(ValueError):
    """Raised when a released FOL expression cannot be parsed."""


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    offset: int


def parse_fol(text: str) -> Formula:
    """Parse one released premise into the typed formula IR."""

    parser = _Parser(_tokenize(text), text)
    formula = parser.parse_formula()
    parser.expect("EOF")
    return formula


def _tokenize(text: str) -> tuple[Token, ...]:
    tokens: list[Token] = []
    offset = 0
    while offset < len(text):
        match = _TOKEN_RE.match(text, offset)
        if match is None:
            raise FolParseError(f"unexpected character {text[offset]!r} at offset {offset}: {text!r}")
        kind = match.lastgroup or ""
        if kind != "SPACE":
            tokens.append(Token(kind, match.group(), offset))
        offset = match.end()
    tokens.append(Token("EOF", "", len(text)))
    return tuple(tokens)


class _Parser:
    def __init__(self, tokens: tuple[Token, ...], source: str):
        self.tokens = tokens
        self.source = source
        self.index = 0
        self.bound_variables: list[str] = []

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def accept(self, *kinds: str) -> Token | None:
        if self.current.kind not in kinds:
            return None
        token = self.current
        self.index += 1
        return token

    def expect(self, kind: str) -> Token:
        token = self.accept(kind)
        if token is None:
            raise FolParseError(
                f"expected {kind}, got {self.current.kind} {self.current.value!r} "
                f"at offset {self.current.offset}: {self.source!r}"
            )
        return token

    def parse_formula(self) -> Formula:
        return self._parse_iff()

    def _parse_iff(self) -> Formula:
        formula = self._parse_implies()
        while self.accept("IFF"):
            formula = Iff(formula, self._parse_implies())
        return formula

    def _parse_implies(self) -> Formula:
        formula = self._parse_or()
        if self.accept("IMPLIES"):
            return Implies(formula, self._parse_implies())
        return formula

    def _parse_or(self) -> Formula:
        formulas = [self._parse_and()]
        while self.accept("OR"):
            formulas.append(self._parse_and())
        return formulas[0] if len(formulas) == 1 else Or(tuple(formulas))

    def _parse_and(self) -> Formula:
        formulas = [self._parse_unary_formula()]
        while self.accept("AND"):
            formulas.append(self._parse_unary_formula())
        return formulas[0] if len(formulas) == 1 else And(tuple(formulas))

    def _parse_unary_formula(self) -> Formula:
        if self.accept("NOT"):
            if self.current.kind in {"LPAREN", "NOT"} or self._is_quantifier():
                return Not(self._parse_unary_formula())
            return Not(self._parse_atomic_formula())
        if self._is_quantifier():
            return self._parse_quantifier()
        if self.accept("LPAREN"):
            formula = self.parse_formula()
            self.expect("RPAREN")
            return formula
        return self._parse_atomic_formula()

    def _is_quantifier(self) -> bool:
        return (
            self.current.kind == "ID"
            and self.current.value.lower() in {"forall", "exists"}
        ) or self.current.value in {"∀", "∃"}

    def _parse_quantifier(self) -> Formula:
        token = self.current
        self.index += 1
        lower = token.value.lower()
        is_forall = lower == "forall" or token.value == "∀"

        if lower in {"forall", "exists"}:
            self.expect("LPAREN")
            variable = self.expect("ID").value
            self.expect("COMMA")
            body = self._parse_bound_body(variable, self.parse_formula)
            self.expect("RPAREN")
        else:
            variable = self.expect("ID").value
            comma_scoped = self.accept("COMMA") is not None
            if comma_scoped:
                body = self._parse_bound_body(variable, self.parse_formula)
            else:
                body = self._parse_bound_body(variable, self._parse_unary_formula)

        constructor = ForAll if is_forall else Exists
        return constructor((f"?{variable.lstrip('?')}",), body)

    def _parse_bound_body(self, variable: str, parse) -> Formula:
        self.bound_variables.append(variable)
        try:
            return parse()
        finally:
            self.bound_variables.pop()

    def _parse_atomic_formula(self) -> Formula:
        if self.current.kind in _COMPARISONS and self.tokens[self.index + 1].kind == "LPAREN":
            op = _COMPARISONS[self.current.kind]
            self.index += 1
            self.expect("LPAREN")
            left = self.parse_term()
            self.expect("COMMA")
            right = self.parse_term()
            self.expect("RPAREN")
            return Compare(op, left, right)

        left = self.parse_term()
        comparison = self.accept(*_COMPARISONS)
        if comparison is not None:
            return Compare(_COMPARISONS[comparison.kind], left, self.parse_term())
        if self.accept("IN"):
            self.expect("LBRACE")
            options = [self.parse_term()]
            while self.accept("COMMA"):
                options.append(self.parse_term())
            self.expect("RBRACE")
            return InSet(left, tuple(options))
        if isinstance(left, Function):
            return Atom(left.name, left.args)
        if isinstance(left, str):
            return Atom(left)
        raise FolParseError(f"term is not a formula at offset {self.current.offset}: {left!r}")

    def parse_term(self) -> Term:
        return self._parse_additive()

    def _parse_additive(self) -> Term:
        term = self._parse_multiplicative()
        while self.current.kind in {"PLUS", "MINUS"}:
            op = _ARITHMETIC[self.current.kind]
            self.index += 1
            term = Arithmetic(op, (term, self._parse_multiplicative()))
        return term

    def _parse_multiplicative(self) -> Term:
        term = self._parse_power()
        while self.current.kind in {"MUL", "DIV"}:
            op = _ARITHMETIC[self.current.kind]
            self.index += 1
            term = Arithmetic(op, (term, self._parse_power()))
        return term

    def _parse_power(self) -> Term:
        term = self._parse_unary_term()
        if self.accept("POW"):
            return Arithmetic("^", (term, self._parse_power()))
        return term

    def _parse_unary_term(self) -> Term:
        if self.accept("MINUS"):
            return Arithmetic("-", (self._parse_unary_term(),))
        if self.current.kind in _COMPARISONS:
            op = _COMPARISONS[self.current.kind]
            self.index += 1
            return Arithmetic(op, (self._parse_unary_term(),))
        return self._parse_primary_term()

    def _parse_primary_term(self) -> Term:
        if token := self.accept("NUMBER"):
            return Number(token.value)
        if token := self.accept("STRING"):
            return token.value[1:-1]
        if token := self.accept("ID"):
            term_name = f"?{token.value}" if token.value in self.bound_variables else token.value
            if not self.accept("LPAREN"):
                return term_name
            args: list[Term] = []
            if not self.accept("RPAREN"):
                args.append(self.parse_term())
                while self.accept("COMMA"):
                    args.append(self.parse_term())
                self.expect("RPAREN")
            return Function(token.value, tuple(args))
        if self.accept("LPAREN"):
            term = self.parse_term()
            self.expect("RPAREN")
            return term
        raise FolParseError(
            f"expected term, got {self.current.kind} {self.current.value!r} "
            f"at offset {self.current.offset}: {self.source!r}"
        )
