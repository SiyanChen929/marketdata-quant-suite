"""Typed random-tree generation and subtree surgery for the factor language.

Only *series* slots are ever replaced or grown, so window and parameter
literals stay literals; every generated tree is additionally checked with
:func:`llm_factor_mining.dsl.validate.validate` by the callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..dsl.nodes import Call, Constant, Node, Terminal
from ..dsl.operators import OPERATORS, TERMINALS, ArgKind


LITERAL_SLOT_OPERATORS = frozenset({"add", "sub", "mul", "div"})


@dataclass(frozen=True)
class GrammarConfig:
    """Sampling distribution over expression trees.

    ``max_depth`` counts levels (a bare terminal has depth 1).  A series slot
    below the root becomes a leaf with probability ``p_leaf``; the second
    argument of ``add/sub/mul/div`` may be a literal with probability
    ``p_literal``.  Windows and power exponents are drawn from fixed menus;
    an operator only draws menu windows it accepts (``delay``/``delta`` take
    ``d >= 1``, rolling operators ``d >= 2``).  The menu includes ``1`` so that
    one-session changes such as ``delta(x, 1)`` are in the baselines' search
    space (every planted benchmark signal must be reachable by every arm;
    see :func:`tree_log_probability`).
    """

    max_depth: int = 4
    p_leaf: float = 0.35
    p_literal: float = 0.15
    windows: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60)
    exponents: tuple[float, ...] = (0.5, 2.0)
    literals: tuple[float, ...] = (0.5, 1.0, 2.0)
    operators: tuple[str, ...] = tuple(OPERATORS)
    terminals: tuple[str, ...] = tuple(TERMINALS)

    def __post_init__(self) -> None:
        if self.max_depth < 2:
            raise ValueError("max_depth must be at least 2")
        if not 0.0 <= self.p_leaf < 1.0 or not 0.0 <= self.p_literal < 1.0:
            raise ValueError("probabilities must lie in [0, 1)")
        unknown = sorted(set(self.operators) - set(OPERATORS)) + sorted(set(self.terminals) - set(TERMINALS))
        if unknown:
            raise ValueError(f"unknown grammar symbols {unknown}")
        if not self.windows or min(self.windows) < 1:
            raise ValueError("windows must be positive integers")


def choose(rng: np.random.Generator, items: tuple | list):
    """Uniform choice that preserves the element type (``rng.choice`` coerces)."""

    return items[int(rng.integers(len(items)))]


def random_window(rng: np.random.Generator, config: GrammarConfig, op: str) -> Constant:
    spec = OPERATORS[op]
    allowed = [window for window in config.windows if window >= spec.min_window]
    return Constant(float(choose(rng, allowed)), True)


def random_parameter(rng: np.random.Generator, config: GrammarConfig, op: str) -> Constant:
    spec = OPERATORS[op]
    low, high = spec.const_bounds or (-np.inf, np.inf)
    allowed = [value for value in config.exponents if low <= value <= high] or [1.0]
    return Constant(float(choose(rng, allowed)), False)


def random_tree(
    rng: np.random.Generator,
    config: GrammarConfig,
    *,
    depth: int = 1,
    max_depth: int | None = None,
    allow_literal: bool = False,
) -> Node:
    """Sample a typed tree whose depth never exceeds ``max_depth``."""

    limit = config.max_depth if max_depth is None else max_depth
    if depth >= limit or (depth > 1 and rng.random() < config.p_leaf):
        if allow_literal and rng.random() < config.p_literal:
            return Constant(float(choose(rng, config.literals)), False)
        return Terminal(choose(rng, config.terminals))
    op = choose(rng, config.operators)
    spec = OPERATORS[op]
    args: list[Node] = []
    for index, kind in enumerate(spec.arg_kinds):
        if kind is ArgKind.SERIES:
            args.append(
                random_tree(
                    rng,
                    config,
                    depth=depth + 1,
                    max_depth=limit,
                    allow_literal=op in LITERAL_SLOT_OPERATORS and index == 1,
                )
            )
        elif kind is ArgKind.WINDOW:
            args.append(random_window(rng, config, op))
        else:
            args.append(random_parameter(rng, config, op))
    return Call(op, tuple(args))


def tree_log_probability(node: Node, config: GrammarConfig, *, max_depth: int | None = None) -> float:
    """Log-probability that :func:`random_tree` returns exactly ``node``.

    ``-inf`` means the tree lies outside the sampler's support (an operator,
    terminal, window, parameter or literal not on the menus, or a tree deeper
    than ``max_depth``).  Argument order is taken as written, so the value is a
    lower bound for the probability of the canonical (order-free) class, and
    the resampling of statically invalid trees (which only raises
    probabilities) is ignored.
    """

    limit = config.max_depth if max_depth is None else int(max_depth)

    def count_log(value: object, items: tuple | list) -> float:
        hits = sum(1 for item in items if item == value)
        return math.log(hits / len(items)) if hits and items else -math.inf

    def visit(current: Node, depth: int, allow_literal: bool) -> float:
        forced_leaf = depth >= limit
        p_leaf = 1.0 if forced_leaf else (config.p_leaf if depth > 1 else 0.0)
        if isinstance(current, (Terminal, Constant)):
            if p_leaf == 0.0:
                return -math.inf
            p_literal = config.p_literal if allow_literal else 0.0
            if isinstance(current, Constant):
                if p_literal == 0.0:
                    return -math.inf
                return math.log(p_leaf) + math.log(p_literal) + count_log(current.value, config.literals)
            if p_literal == 1.0:  # pragma: no cover - excluded by GrammarConfig validation
                return -math.inf
            return math.log(p_leaf) + math.log1p(-p_literal) + count_log(current.name, config.terminals)
        if p_leaf == 1.0 or current.op not in OPERATORS:
            return -math.inf
        spec = OPERATORS[current.op]
        if len(current.args) != spec.arity:
            return -math.inf
        total = math.log1p(-p_leaf) + count_log(current.op, config.operators)
        for index, (kind, arg) in enumerate(zip(spec.arg_kinds, current.args)):
            if total == -math.inf:
                return total
            if kind is ArgKind.SERIES:
                total += visit(arg, depth + 1, current.op in LITERAL_SLOT_OPERATORS and index == 1)
            elif not isinstance(arg, Constant):
                return -math.inf
            elif kind is ArgKind.WINDOW:
                allowed = [float(window) for window in config.windows if window >= spec.min_window]
                total += count_log(arg.value, allowed) if arg.integer else -math.inf
            else:
                low, high = spec.const_bounds or (-np.inf, np.inf)
                allowed = [float(value) for value in config.exponents if low <= value <= high] or [1.0]
                total += count_log(arg.value, allowed)
        return total

    return visit(node, 1, False)


def strip_orientation(node: Node) -> Node:
    """Drop a root ``neg``: the protocol orients every factor by its formation IC sign."""

    while isinstance(node, Call) and node.op == "neg" and len(node.args) == 1:
        node = node.args[0]
    return node


def series_paths(node: Node, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], Node]]:
    """``(path, subtree)`` for the root and every subtree sitting in a series slot."""

    found = [(path, node)]
    if isinstance(node, Call):
        spec = OPERATORS.get(node.op)
        kinds = spec.arg_kinds if spec is not None and len(node.args) == spec.arity else ()
        for index, arg in enumerate(node.args):
            if index < len(kinds) and kinds[index] is ArgKind.SERIES:
                found.extend(series_paths(arg, (*path, index)))
    return found


def replace_at(node: Node, path: tuple[int, ...], replacement: Node) -> Node:
    """Return ``node`` with the subtree at ``path`` replaced."""

    if not path:
        return replacement
    if not isinstance(node, Call):
        raise ValueError("path descends below a leaf")
    head, rest = path[0], path[1:]
    args = list(node.args)
    args[head] = replace_at(args[head], rest, replacement)
    return Call(node.op, tuple(args))


def subtree_at(node: Node, path: tuple[int, ...]) -> Node:
    for index in path:
        if not isinstance(node, Call):
            raise ValueError("path descends below a leaf")
        node = node.args[index]
    return node


def signature_groups() -> dict[tuple[ArgKind, ...], tuple[str, ...]]:
    """Operators grouped by argument signature (for point mutation)."""

    groups: dict[tuple[ArgKind, ...], list[str]] = {}
    for name, spec in OPERATORS.items():
        groups.setdefault(spec.arg_kinds, []).append(name)
    return {kinds: tuple(names) for kinds, names in groups.items()}


__all__ = [
    "GrammarConfig",
    "choose",
    "random_tree",
    "replace_at",
    "series_paths",
    "signature_groups",
    "strip_orientation",
    "subtree_at",
    "tree_log_probability",
]
