"""Static checks for factor expressions.

Validation is structural and never touches data.  It returns every problem it
finds as a machine-readable :class:`ValidationIssue` so an LLM proposer can be
given precise feedback.  Error codes:

``PARSE``            syntax error (from :mod:`.parser`)
``UNKNOWN_OP``       operator name not in the registry
``UNKNOWN_TERMINAL`` identifier is not a data field
``ARITY``            wrong number of arguments
``TYPE``             argument kind mismatch (e.g. an expression or a non-integer
                     literal where an integer window is required)
``WINDOW_RANGE``     window outside ``[min_window, max_window]``
``CONST_RANGE``      literal outside the operator's or the global bounds
``DEPTH``            tree deeper than ``max_depth``
``SIZE``             more than ``max_nodes`` nodes
``LOOKBACK``         cumulative lookback along some root-to-leaf path exceeds
                     ``max_lookback`` sessions
``CONSTANT_ONLY``    expression contains no data terminal
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .nodes import Call, Constant, Node, Terminal, walk
from .operators import OPERATORS, TERMINALS, ArgKind
from .parser import ParseError, parse


ERROR_CODES: tuple[str, ...] = (
    "PARSE",
    "UNKNOWN_OP",
    "UNKNOWN_TERMINAL",
    "ARITY",
    "TYPE",
    "WINDOW_RANGE",
    "CONST_RANGE",
    "DEPTH",
    "SIZE",
    "LOOKBACK",
    "CONSTANT_ONLY",
)


@dataclass(frozen=True)
class DSLLimits:
    """Search-space bounds.  Defaults admit a 12-month momentum signal."""

    max_depth: int = 10
    max_nodes: int = 48
    max_window: int = 252
    max_lookback: int = 504
    max_abs_literal: float = 1.0e6

    def __post_init__(self) -> None:
        if min(self.max_depth, self.max_nodes, self.max_window, self.max_lookback) < 1:
            raise ValueError("DSL limits must be positive")
        if self.max_window < 2:
            raise ValueError("max_window must be at least 2")
        if not self.max_abs_literal > 0:
            raise ValueError("max_abs_literal must be positive")


@dataclass(frozen=True)
class ValidationIssue:
    """One structured problem; ``path`` is the child-index path from the root."""

    code: str
    message: str
    path: tuple[int, ...] = ()
    position: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
            "position": self.position,
        }


@dataclass(frozen=True)
class ExpressionStats:
    """Size and history requirements of an expression."""

    n_nodes: int
    depth: int
    lookback: int
    n_operators: int
    n_terminals: int
    n_literals: int
    operators: frozenset[str]
    terminals: frozenset[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_nodes": self.n_nodes,
            "depth": self.depth,
            "lookback": self.lookback,
            "n_operators": self.n_operators,
            "n_terminals": self.n_terminals,
            "n_literals": self.n_literals,
            "operators": sorted(self.operators),
            "terminals": sorted(self.terminals),
        }


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :func:`validate`; ``ok`` is true exactly when ``errors`` is empty."""

    ok: bool
    errors: tuple[ValidationIssue, ...] = ()
    node: Node | None = None
    stats: ExpressionStats | None = None
    limits: DSLLimits = field(default_factory=DSLLimits)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.errors)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "stats": None if self.stats is None else self.stats.to_dict(),
        }


def lookback(node: Node) -> int:
    """Maximum rows of history needed along any root-to-leaf path.

    The first ``lookback(node)`` sessions of every symbol are necessarily NaN
    when the evaluator uses full windows only.
    """

    if isinstance(node, Terminal):
        spec = TERMINALS.get(node.name)
        return spec.lookback if spec is not None else 0
    if isinstance(node, Constant):
        return 0
    spec = OPERATORS.get(node.op)
    own = 0
    if spec is not None and spec.window_slot is not None and len(node.args) == spec.arity:
        window = node.args[spec.window_slot]
        if isinstance(window, Constant) and math.isfinite(window.value):
            own = spec.lookback(int(window.value))
    inner = max((lookback(arg) for arg in node.args), default=0)
    return own + inner


def depth(node: Node) -> int:
    """Tree depth; a leaf has depth 1."""

    if isinstance(node, Call):
        return 1 + max((depth(arg) for arg in node.args), default=0)
    return 1


def expression_stats(node: Node) -> ExpressionStats:
    """Compute :class:`ExpressionStats` for any (possibly invalid) tree."""

    n_nodes = n_ops = n_terms = n_lits = 0
    operators: set[str] = set()
    terminals: set[str] = set()
    for _, current in walk(node):
        n_nodes += 1
        if isinstance(current, Call):
            n_ops += 1
            operators.add(current.op)
        elif isinstance(current, Terminal):
            n_terms += 1
            terminals.add(current.name)
        else:
            n_lits += 1
    return ExpressionStats(
        n_nodes=n_nodes,
        depth=depth(node),
        lookback=lookback(node),
        n_operators=n_ops,
        n_terminals=n_terms,
        n_literals=n_lits,
        operators=frozenset(operators),
        terminals=frozenset(terminals),
    )


def validate(expression: str | Node, limits: DSLLimits | None = None) -> ValidationResult:
    """Parse (if needed) and statically check an expression against ``limits``."""

    limits = limits or DSLLimits()
    if isinstance(expression, str):
        try:
            node = parse(expression)
        except ParseError as exc:
            issue = ValidationIssue("PARSE", exc.message, (), exc.position)
            return ValidationResult(False, (issue,), None, None, limits)
    else:
        node = expression

    errors: list[ValidationIssue] = []
    for path, current in walk(node):
        errors.extend(_check_node(current, path, limits))

    stats = expression_stats(node)
    if stats.depth > limits.max_depth:
        errors.append(
            ValidationIssue("DEPTH", f"depth {stats.depth} exceeds max_depth {limits.max_depth}")
        )
    if stats.n_nodes > limits.max_nodes:
        errors.append(
            ValidationIssue("SIZE", f"{stats.n_nodes} nodes exceed max_nodes {limits.max_nodes}")
        )
    if stats.lookback > limits.max_lookback:
        errors.append(
            ValidationIssue(
                "LOOKBACK",
                f"cumulative lookback {stats.lookback} exceeds max_lookback {limits.max_lookback}",
            )
        )
    if stats.n_terminals == 0:
        errors.append(
            ValidationIssue("CONSTANT_ONLY", "expression must reference at least one data field")
        )
    return ValidationResult(not errors, tuple(errors), node, stats, limits)


def _check_node(
    node: Node, path: tuple[int, ...], limits: DSLLimits
) -> list[ValidationIssue]:
    if isinstance(node, Terminal):
        if node.name not in TERMINALS:
            return [
                ValidationIssue(
                    "UNKNOWN_TERMINAL",
                    f"unknown data field {node.name!r}; allowed: {', '.join(TERMINALS)}",
                    path,
                )
            ]
        return []
    if isinstance(node, Constant):
        if not math.isfinite(node.value) or abs(node.value) > limits.max_abs_literal:
            return [
                ValidationIssue(
                    "CONST_RANGE",
                    f"literal {node.value!r} outside +/-{limits.max_abs_literal:g}",
                    path,
                )
            ]
        return []

    spec = OPERATORS.get(node.op)
    if spec is None:
        return [ValidationIssue("UNKNOWN_OP", f"unknown operator {node.op!r}", path)]
    if len(node.args) != spec.arity:
        return [
            ValidationIssue(
                "ARITY",
                f"{node.op} takes {spec.arity} argument(s), got {len(node.args)}",
                path,
            )
        ]

    issues: list[ValidationIssue] = []
    for index, (kind, arg) in enumerate(zip(spec.arg_kinds, node.args)):
        arg_path = (*path, index)
        if kind is ArgKind.SERIES:
            continue
        if not isinstance(arg, Constant):
            wanted = "an integer window literal" if kind is ArgKind.WINDOW else "a numeric literal"
            issues.append(
                ValidationIssue("TYPE", f"{node.op} argument {index + 1} must be {wanted}", arg_path)
            )
            continue
        if kind is ArgKind.WINDOW:
            if not arg.integer:
                issues.append(
                    ValidationIssue(
                        "TYPE",
                        f"{node.op} window must be an integer literal, got {arg.value!r}",
                        arg_path,
                    )
                )
                continue
            window = int(arg.value)
            if window < spec.min_window or window > limits.max_window:
                issues.append(
                    ValidationIssue(
                        "WINDOW_RANGE",
                        f"{node.op} window {window} outside [{spec.min_window}, {limits.max_window}]",
                        arg_path,
                    )
                )
        elif spec.const_bounds is not None:
            low, high = spec.const_bounds
            if not low <= arg.value <= high:
                issues.append(
                    ValidationIssue(
                        "CONST_RANGE",
                        f"{node.op} parameter {arg.value!r} outside [{low:g}, {high:g}]",
                        arg_path,
                    )
                )
    return issues
