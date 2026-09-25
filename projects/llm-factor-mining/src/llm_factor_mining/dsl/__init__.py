"""Typed formulaic-factor language: nodes, parser, validator, canonical forms.

All time-series operators look backward only and every window is a positive
integer literal, so an expression cannot reference data after its evaluation
date.  See :mod:`llm_factor_mining.dsl.operators` for the full argument.
"""

from .canonical import (
    NearestReference,
    canonical_string,
    canonicalize,
    complexity,
    jaccard,
    nearest_reference,
    structural_hash,
    structural_similarity,
    subtree_hashes,
)
from .library import LIBRARY, LibraryFactor, library_nodes
from .nodes import Call, Constant, Node, Terminal, to_expression, walk
from .operators import OPERATORS, TERMINALS, ArgKind, OperatorSpec, TerminalSpec, describe_language
from .parser import ParseError, parse, tokenize
from .validate import (
    ERROR_CODES,
    DSLLimits,
    ExpressionStats,
    ValidationIssue,
    ValidationResult,
    expression_stats,
    lookback,
    validate,
)

__all__ = [
    "ERROR_CODES",
    "LIBRARY",
    "OPERATORS",
    "TERMINALS",
    "ArgKind",
    "Call",
    "Constant",
    "DSLLimits",
    "ExpressionStats",
    "LibraryFactor",
    "NearestReference",
    "Node",
    "OperatorSpec",
    "ParseError",
    "Terminal",
    "TerminalSpec",
    "ValidationIssue",
    "ValidationResult",
    "canonical_string",
    "canonicalize",
    "complexity",
    "describe_language",
    "expression_stats",
    "jaccard",
    "library_nodes",
    "lookback",
    "nearest_reference",
    "parse",
    "structural_hash",
    "structural_similarity",
    "subtree_hashes",
    "to_expression",
    "tokenize",
    "validate",
    "walk",
]
