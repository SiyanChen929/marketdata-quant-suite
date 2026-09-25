"""Canonical forms, structural hashes, complexity and structural similarity.

Canonicalization sorts the series arguments of commutative operators
(``add``, ``mul``, ``ts_corr``, ``ts_cov``) and normalizes literal kinds in
non-window slots, so ``close + open``, ``add(open, close)`` and
``open + close`` share one canonical string and hash.  Associativity is *not*
normalized (``(a + b) + c`` and ``a + (b + c)`` hash differently); this is a
deliberate, documented limitation that keeps canonicalization cheap and exact.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib

from .nodes import Call, Constant, Node, Terminal, format_number
from .operators import OPERATORS, ArgKind
from .validate import expression_stats


HASH_NAMESPACE = "llm-factor-mining/dsl-v1"
LITERAL_PLACEHOLDER = "#"


def canonicalize(node: Node) -> Node:
    """Return the canonical tree (commutative arguments sorted, literals normalized)."""

    if not isinstance(node, Call):
        return node
    args = tuple(canonicalize(arg) for arg in node.args)
    spec = OPERATORS.get(node.op)
    if spec is not None and len(args) == spec.arity:
        args = tuple(
            _normalize_literal(arg, kind) for arg, kind in zip(args, spec.arg_kinds)
        )
        if spec.commutative:
            head = sorted(args[:2], key=canonical_string)
            args = (*head, *args[2:])
    return Call(node.op, args)


def canonical_string(node: Node, *, abstract_literals: bool = False) -> str:
    """Canonical function-call rendering; equal strings mean equal canonical trees.

    With ``abstract_literals`` every numeric literal renders as ``#`` so that
    window or scale variants of one idea collapse to the same skeleton.
    """

    if isinstance(node, Terminal):
        return node.name
    if isinstance(node, Constant):
        return LITERAL_PLACEHOLDER if abstract_literals else format_number(node)
    spec = OPERATORS.get(node.op)
    args = node.args
    if spec is not None and len(args) == spec.arity:
        args = tuple(_normalize_literal(arg, kind) for arg, kind in zip(args, spec.arg_kinds))
    rendered = [canonical_string(arg, abstract_literals=abstract_literals) for arg in args]
    if spec is not None and spec.commutative and len(rendered) == spec.arity:
        rendered[:2] = sorted(rendered[:2])
    return f"{node.op}({','.join(rendered)})"


def structural_hash(node: Node, *, abstract_literals: bool = False) -> str:
    """SHA-256 hex digest of the namespaced canonical string (dedup key)."""

    payload = f"{HASH_NAMESPACE}\n{canonical_string(node, abstract_literals=abstract_literals)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def complexity(node: Node) -> float:
    """Parsimony score: operators + data terminals + 0.5 per numeric literal."""

    stats = expression_stats(node)
    return float(stats.n_operators + stats.n_terminals) + 0.5 * stats.n_literals


def subtree_hashes(node: Node, *, abstract_literals: bool = False) -> frozenset[str]:
    """Hashes of every operator or data-terminal subtree (bare literals excluded)."""

    found: set[str] = set()

    def visit(current: Node) -> None:
        if isinstance(current, Constant):
            return
        found.add(structural_hash(current, abstract_literals=abstract_literals))
        if isinstance(current, Call):
            for arg in current.args:
                visit(arg)

    visit(node)
    return frozenset(found)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard index; two empty sets are defined as identical (1.0)."""

    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def structural_similarity(
    left: Node, right: Node, *, abstract_literals: bool = False
) -> float:
    """Jaccard similarity of subtree-hash sets, in [0, 1]."""

    return jaccard(
        subtree_hashes(left, abstract_literals=abstract_literals),
        subtree_hashes(right, abstract_literals=abstract_literals),
    )


@dataclass(frozen=True)
class NearestReference:
    """Most similar reference expression and its similarity score."""

    index: int | None
    similarity: float


def nearest_reference(
    node: Node,
    references: Iterable[Node],
    *,
    abstract_literals: bool = False,
) -> NearestReference:
    """Return the most structurally similar reference (novelty = 1 - similarity)."""

    target = subtree_hashes(node, abstract_literals=abstract_literals)
    best_index: int | None = None
    best = 0.0
    for index, reference in enumerate(references):
        score = jaccard(target, subtree_hashes(reference, abstract_literals=abstract_literals))
        if best_index is None or score > best:
            best_index, best = index, score
    return NearestReference(best_index, best)


def _normalize_literal(node: Node, kind: ArgKind) -> Node:
    if isinstance(node, Constant) and kind is not ArgKind.WINDOW and node.integer:
        return Constant(node.value, False)
    return node
