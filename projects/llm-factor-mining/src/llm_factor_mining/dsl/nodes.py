"""Immutable abstract-syntax-tree nodes for the factor language."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Terminal:
    """A named data field such as ``close`` or ``returns``."""

    name: str


@dataclass(frozen=True, slots=True)
class Constant:
    """A numeric literal.

    ``integer`` records whether the literal was written without a decimal
    point or exponent; window arguments must be integer literals.
    """

    value: float
    integer: bool = False

    def __post_init__(self) -> None:
        value = float(self.value)
        if value == 0.0:
            value = 0.0  # collapse -0.0 so equal literals hash equally
        object.__setattr__(self, "value", value)
        if self.integer and math.isfinite(value) and not value.is_integer():
            raise ValueError(f"integer literal must be integral, got {value!r}")


@dataclass(frozen=True, slots=True)
class Call:
    """An operator applied to an ordered tuple of argument nodes."""

    op: str
    args: tuple["Node", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))


Node = Terminal | Constant | Call


def children(node: Node) -> tuple[Node, ...]:
    """Return the direct children of ``node``."""

    return node.args if isinstance(node, Call) else ()


def walk(node: Node) -> Iterator[tuple[tuple[int, ...], Node]]:
    """Yield ``(path, node)`` pairs in pre-order; ``path`` indexes child positions."""

    stack: list[tuple[tuple[int, ...], Node]] = [((), node)]
    while stack:
        path, current = stack.pop()
        yield path, current
        kids = children(current)
        for index in range(len(kids) - 1, -1, -1):
            stack.append(((*path, index), kids[index]))


def format_number(constant: Constant) -> str:
    """Render a literal so that the parser reads back the same value and kind."""

    if constant.integer:
        return str(int(constant.value))
    return repr(float(constant.value))


def to_expression(node: Node) -> str:
    """Render ``node`` faithfully in function-call syntax (no reordering)."""

    if isinstance(node, Terminal):
        return node.name
    if isinstance(node, Constant):
        return format_number(node)
    return f"{node.op}({','.join(to_expression(arg) for arg in node.args)})"
