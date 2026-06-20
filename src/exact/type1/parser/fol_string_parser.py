"""Recursive-descent parser: ASCII first-order-logic string -> FOLNode AST.

The whole-theory translator asks the LLM to emit each premise/question/option as
one FOL string over a fixed grammar; this turns those strings into the same
``FOLNode`` AST the Z3 solver already consumes. Keeping a single explicit grammar
(rather than per-premise LLM decomposition) is what lets predicate names stay
consistent across a theory.

Grammar (lowest precedence first)::

    formula := 'forall' VAR ':' formula
             | 'exists' VAR ':' formula
             | iff
    iff     := impl ('<->' impl)*
    impl    := disj ('->' disj)?            # right-associative
    disj    := conj ('|' conj)*
    conj    := neg ('&' neg)*
    neg     := ('~'|'!') neg | primary
    primary := '(' formula ')' | comparison | predicate
    predicate  := NAME ('(' termlist ')')?
    comparison := term OP term               # OP: >= <= != = > <
    term    := NAME ('(' termlist ')')? | NUMBER
    termlist:= term (',' term)*

Accepted synonyms: ``and``/``&``/``∧``, ``or``/``|``/``∨``,
``not``/``~``/``!``/``¬``, ``->``/``=>``/``implies``/``→``,
``<->``/``iff``/``↔``, ``forall``/``all``/``∀``, ``exists``/``some``/``∃``.
"""

from __future__ import annotations

import re

from exact.type1.ast.nodes import (
    AtomicNode,
    ComparisonNode,
    FOLNode,
    FunctionTerm,
    LogicalNode,
    NumericTerm,
    QuantifiedNode,
)
from exact.type1.models.schemas import Predicate

__all__ = ["FOLStringParseError", "parse_fol_string"]


class FOLStringParseError(ValueError):
    """Raised when a FOL string does not conform to the grammar."""


_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<arrow_iff>(<->)|(↔)|(\biff\b))
    | (?P<arrow_impl>(->)|(=>)|(→)|(\bimplies\b))
    | (?P<op>>=|<=|!=|=|>|<)
    | (?P<and>&|∧|\band\b)
    | (?P<or>\||∨|\bor\b)
    | (?P<not>~|!|¬|\bnot\b)
    | (?P<forall>∀|\bforall\b|\ball\b)
    | (?P<exists>∃|\bexists\b|\bsome\b)
    | (?P<lparen>\()
    | (?P<rparen>\))
    | (?P<comma>,)
    | (?P<colon>[:.])
    | (?P<number>[+-]?(?:\d+\.\d+|\d+|\.\d+))
    | (?P<name>[A-Za-z_][A-Za-z0-9_']*(?:-[A-Za-z0-9_']+)*)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_COMPARISON_OPS = {">=", "<=", "!=", "=", ">", "<"}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None or match.end() == pos:
            raise FOLStringParseError(f"Unexpected character at {pos}: {text[pos:pos+20]!r}")
        pos = match.end()
        kind = match.lastgroup
        assert kind is not None
        if kind == "ws":
            continue
        tokens.append((kind, match.group()))
    tokens.append(("eof", ""))
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.i = 0

    def _peek(self) -> tuple[str, str]:
        return self.tokens[self.i]

    def _next(self) -> tuple[str, str]:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _expect(self, kind: str) -> tuple[str, str]:
        if self._peek()[0] != kind:
            raise FOLStringParseError(f"Expected {kind}, got {self._peek()!r}")
        return self._next()

    # --- grammar ---------------------------------------------------------
    def parse(self) -> FOLNode:
        node = self._formula()
        if self._peek()[0] != "eof":
            raise FOLStringParseError(f"Trailing tokens: {self._peek()!r}")
        return node

    def _formula(self) -> FOLNode:
        kind = self._peek()[0]
        if kind in ("forall", "exists"):
            self._next()
            var = self._expect("name")[1]
            # optional ':' or '.' separator
            if self._peek()[0] == "colon":
                self._next()
            body = self._formula()
            quant = "FORALL" if kind == "forall" else "EXISTS"
            return QuantifiedNode(quantifier=quant, variable=var, body=body, restrictor=None)
        return self._iff()

    def _iff(self) -> FOLNode:
        left = self._impl()
        while self._peek()[0] == "arrow_iff":
            self._next()
            right = self._impl()
            left = LogicalNode(operator="IFF", left=left, right=right)
        return left

    def _impl(self) -> FOLNode:
        left = self._disj()
        if self._peek()[0] == "arrow_impl":
            self._next()
            right = self._impl()  # right-associative
            return LogicalNode(operator="IMPLIES", left=left, right=right)
        return left

    def _disj(self) -> FOLNode:
        left = self._conj()
        while self._peek()[0] == "or":
            self._next()
            right = self._conj()
            left = LogicalNode(operator="OR", left=left, right=right)
        return left

    def _conj(self) -> FOLNode:
        left = self._neg()
        while self._peek()[0] == "and":
            self._next()
            right = self._neg()
            left = LogicalNode(operator="AND", left=left, right=right)
        return left

    def _neg(self) -> FOLNode:
        if self._peek()[0] == "not":
            self._next()
            return LogicalNode(operator="NOT", left=self._neg(), right=None)
        return self._primary()

    def _primary(self) -> FOLNode:
        kind = self._peek()[0]
        if kind == "lparen":
            self._next()
            node = self._formula()
            self._expect("rparen")
            return node
        # comparison or predicate — both start with a term (name/number)
        return self._comparison_or_predicate()

    def _comparison_or_predicate(self) -> FOLNode:
        # parse a leading term; if followed by a comparison op it's a comparison,
        # otherwise it must be a predicate atom.
        start = self.i
        name_tok = self._peek()
        if name_tok[0] == "number":
            left_term = self._term()
            return self._finish_comparison(left_term)
        if name_tok[0] != "name":
            raise FOLStringParseError(f"Expected atom, got {name_tok!r}")
        name = self._next()[1]
        args: list[str] = []
        func_args: list[str] = []
        if self._peek()[0] == "lparen":
            self._next()
            args, func_args = self._arglist()
            self._expect("rparen")
        if self._peek()[0] == "op":
            # it was the left side of a comparison: rewind to a FunctionTerm
            left_term = FunctionTerm(name=name, arguments=func_args)
            return self._finish_comparison(left_term)
        # plain predicate atom
        del start
        predicate = Predicate(name=name, arg_sorts=["Entity"] * len(args), aliases=[])
        return AtomicNode(predicate=predicate, arguments=args)

    def _finish_comparison(self, left: FunctionTerm | NumericTerm) -> ComparisonNode:
        op = self._expect("op")[1]
        if op not in _COMPARISON_OPS:
            raise FOLStringParseError(f"Bad comparison operator {op!r}")
        right = self._term_value()
        if not isinstance(left, FunctionTerm):
            # numeric on the left — flip so left is always the function term
            if isinstance(right, FunctionTerm):
                left, right, op = right, left, _flip_op(op)
            else:
                raise FOLStringParseError("Comparison needs at least one function term")
        return ComparisonNode(operator=op, left=left, right=right)  # type: ignore[arg-type]

    def _term(self) -> NumericTerm | FunctionTerm:
        return self._term_value()

    def _term_value(self) -> NumericTerm | FunctionTerm:
        tok = self._peek()
        if tok[0] == "number":
            self._next()
            return NumericTerm(value=float(tok[1]))
        if tok[0] == "name":
            name = self._next()[1]
            fargs: list[str] = []
            if self._peek()[0] == "lparen":
                self._next()
                _, fargs = self._arglist()
                self._expect("rparen")
            return FunctionTerm(name=name, arguments=fargs)
        raise FOLStringParseError(f"Expected term, got {tok!r}")

    def _arglist(self) -> tuple[list[str], list[str]]:
        """Parse comma-separated arguments. Returns (atom_args, func_args).

        Arguments are bare identifiers/numbers (no nested predicates) — names are
        kept verbatim for AtomicNode arguments and FunctionTerm arguments alike.
        """
        names: list[str] = []
        if self._peek()[0] == "rparen":
            return names, names
        while True:
            tok = self._peek()
            if tok[0] in ("name", "number"):
                self._next()
                names.append(tok[1])
            else:
                raise FOLStringParseError(f"Bad argument {tok!r}")
            if self._peek()[0] == "comma":
                self._next()
                continue
            break
        return names, names


def _flip_op(op: str) -> str:
    return {">": "<", "<": ">", ">=": "<=", "<=": ">=", "=": "=", "!=": "!="}[op]


def parse_fol_string(text: str) -> FOLNode:
    """Parse one ASCII FOL string into a FOLNode AST.

    Raises FOLStringParseError on any grammar violation so callers can fall back
    or flag the premise rather than feed Z3 a malformed tree.
    """
    if not text or not text.strip():
        raise FOLStringParseError("empty FOL string")
    return _Parser(_tokenize(text)).parse()
