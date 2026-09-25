"""Baseline A: seeded random sampling from the factor grammar.

The proposer ignores all feedback; it measures what the search protocol finds
when proposals carry no information.  Every returned expression passes static
validation.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from ..dsl.canonical import structural_hash
from ..dsl.nodes import Node, to_expression
from ..dsl.validate import DSLLimits, validate
from .base import Proposal, ProposalContext
from .trees import GrammarConfig, random_tree


def random_valid_tree(
    rng: np.random.Generator,
    config: GrammarConfig,
    limits: DSLLimits,
    *,
    max_attempts: int = 200,
    max_depth: int | None = None,
) -> tuple[Node, int]:
    """Sample until a tree passes validation; returns ``(tree, attempts)``."""

    for attempt in range(1, max_attempts + 1):
        tree = random_tree(rng, config, max_depth=max_depth)
        if validate(tree, limits).ok:
            return tree, attempt
    raise RuntimeError(f"no valid expression in {max_attempts} attempts; check GrammarConfig vs DSLLimits")


class RandomGrammarProposer:
    """Seeded i.i.d. grammar sampler (avoids repeating its own earlier samples)."""

    name = "random"

    def __init__(
        self,
        seed: int = 0,
        *,
        grammar: GrammarConfig | None = None,
        limits: DSLLimits | None = None,
        max_repeat_attempts: int = 50,
    ) -> None:
        self.seed = int(seed)
        self.grammar = grammar or GrammarConfig()
        self.limits = limits or DSLLimits()
        self.max_repeat_attempts = int(max_repeat_attempts)
        self._rng = np.random.default_rng(self.seed)
        self._seen: set[str] = set()

    def propose(self, context: ProposalContext) -> list[Proposal]:
        proposals: list[Proposal] = []
        for _ in range(context.n_requested):
            tree, attempts = random_valid_tree(self._rng, self.grammar, self.limits)
            for _ in range(self.max_repeat_attempts):
                if structural_hash(tree) not in self._seen:
                    break
                tree, more = random_valid_tree(self._rng, self.grammar, self.limits)
                attempts += more
            self._seen.add(structural_hash(tree))
            proposals.append(
                Proposal(
                    expression=to_expression(tree),
                    rationale="uniform random sample from the typed grammar",
                    metadata={"generator": "random_grammar", "attempts": attempts},
                )
            )
        return proposals

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "grammar": asdict(self.grammar),
            "uses_feedback": False,
        }


__all__ = ["RandomGrammarProposer", "random_valid_tree"]
