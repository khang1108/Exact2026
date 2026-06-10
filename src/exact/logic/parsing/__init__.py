"""Deterministic text parsers for atoms and released FOL strings."""

from exact.logic.parsing.fol_parser import parse_fol
from exact.logic.parsing.parser import atom_from_text

__all__ = ["atom_from_text", "parse_fol"]
