"""Baseline B: genetic programming over expression trees (Koza, 1992 style).

Fitness of an evaluated expression is ``|formation ICIR| - lambda * complexity``
(the sign is free because the protocol orients every factor by the sign of its
formation IC).  Each round the proposer

1. merges formation feedback into an archive keyed by structural hash;
2. takes the fittest ``population_size`` archive members as the population;
3. breeds children by subtree crossover, subtree mutation or point mutation,
   with parents picked by tournament selection, plus a fraction of random
   immigrants for diversity.

Children are statically valid and never repeat an archived or previously
proposed expression.  All randomness flows from one seeded generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from ..dsl.canonical import structural_hash
from ..dsl.nodes import Call, Constant, Node, Terminal, to_expression, walk
from ..dsl.operators import OPERATORS, ArgKind
from ..dsl.parser import ParseError, parse
from ..dsl.validate import DSLLimits, validate
from .base import EvaluatedFeedback, Proposal, ProposalContext
from .random_grammar import random_valid_tree
from .trees import (
    GrammarConfig,
    choose,
    random_tree,
    replace_at,
    series_paths,
    signature_groups,
)


@dataclass(frozen=True)
class EvolutionConfig:
    """Genetic-programming hyper-parameters (fixed before any experiment)."""

    population_size: int = 30
    tournament_size: int = 3
    p_crossover: float = 0.5
    p_subtree_mutation: float = 0.3
    p_point_mutation: float = 0.2
    immigrant_fraction: float = 0.1
    complexity_penalty: float = 0.002
    mutation_max_depth: int = 3
    max_attempts: int = 40

    def __post_init__(self) -> None:
        total = self.p_crossover + self.p_subtree_mutation + self.p_point_mutation
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError("operator probabilities must sum to 1")
        if self.population_size < 2 or self.tournament_size < 1:
            raise ValueError("population_size >= 2 and tournament_size >= 1 required")
        if not 0.0 <= self.immigrant_fraction <= 1.0:
            raise ValueError("immigrant_fraction must lie in [0, 1]")


class EvolutionaryProposer:
    """Seeded GP baseline driven only by formation-window feedback."""

    name = "evolutionary"

    def __init__(
        self,
        seed: int = 0,
        *,
        config: EvolutionConfig | None = None,
        grammar: GrammarConfig | None = None,
        limits: DSLLimits | None = None,
    ) -> None:
        self.seed = int(seed)
        self.config = config or EvolutionConfig()
        self.grammar = grammar or GrammarConfig()
        self.limits = limits or DSLLimits()
        self._rng = np.random.default_rng(self.seed)
        self._archive: dict[str, tuple[Node, float]] = {}
        self._proposed: set[str] = set()
        self._groups = signature_groups()

    # ---------------------------------------------------------------- fitness

    def fitness(self, feedback: EvaluatedFeedback) -> float:
        icir = feedback.formation_icir
        if not math.isfinite(icir):
            return -math.inf
        return abs(icir) - self.config.complexity_penalty * feedback.complexity

    def _ingest(self, context: ProposalContext) -> None:
        for item in context.evaluated:
            try:
                node = parse(item.expression)
            except ParseError:  # pragma: no cover - feedback is canonical DSL
                continue
            self._archive[structural_hash(node)] = (node, self.fitness(item))

    def population(self) -> list[tuple[str, Node, float]]:
        """Fittest archive members, ties broken by hash for determinism."""

        members = [
            (key, node, score) for key, (node, score) in self._archive.items() if math.isfinite(score)
        ]
        members.sort(key=lambda item: (-item[2], item[0]))
        return members[: self.config.population_size]

    # --------------------------------------------------------------- breeding

    def _tournament(self, population: list[tuple[str, Node, float]]) -> tuple[str, Node, float]:
        size = min(self.config.tournament_size, len(population))
        picks = self._rng.choice(len(population), size=size, replace=False)
        return min((population[int(index)] for index in picks), key=lambda item: (-item[2], item[0]))

    def _crossover(self, left: Node, right: Node) -> Node:
        path, _ = choose(self._rng, series_paths(left))
        _, donor = choose(self._rng, series_paths(right))
        return replace_at(left, path, donor)

    def _subtree_mutation(self, parent: Node) -> Node:
        path, _ = choose(self._rng, series_paths(parent))
        budget = max(2, min(self.config.mutation_max_depth, self.grammar.max_depth))
        return replace_at(parent, path, random_tree(self._rng, self.grammar, max_depth=budget))

    def _point_mutation(self, parent: Node) -> Node:
        sites: list[tuple[tuple[int, ...], Node, str]] = []
        for path, node in walk(parent):
            if isinstance(node, Call) and len(self._groups.get(OPERATORS[node.op].arg_kinds, ())) > 1:
                sites.append((path, node, "operator"))
            elif isinstance(node, Terminal) and len(self.grammar.terminals) > 1:
                sites.append((path, node, "terminal"))
            if isinstance(node, Call):
                spec = OPERATORS[node.op]
                if spec.window_slot is not None:
                    sites.append(((*path, spec.window_slot), node.args[spec.window_slot], f"window:{node.op}"))
        if not sites:
            return parent
        path, node, kind = choose(self._rng, sites)
        if kind == "operator":
            assert isinstance(node, Call)
            options = [name for name in self._groups[OPERATORS[node.op].arg_kinds] if name != node.op]
            return replace_at(parent, path, Call(choose(self._rng, options), node.args))
        if kind == "terminal":
            assert isinstance(node, Terminal)
            options = [name for name in self.grammar.terminals if name != node.name]
            return replace_at(parent, path, Terminal(choose(self._rng, options)))
        op = kind.split(":", 1)[1]
        current = int(node.value) if isinstance(node, Constant) else None
        options = [w for w in self.grammar.windows if w >= OPERATORS[op].min_window and w != current]
        if not options:
            return parent
        return replace_at(parent, path, Constant(float(choose(self._rng, options)), True))

    def _is_new_and_valid(self, node: Node) -> bool:
        key = structural_hash(node)
        return key not in self._archive and key not in self._proposed and validate(node, self.limits).ok

    def _breed(self, population: list[tuple[str, Node, float]]) -> tuple[Node, dict[str, Any]]:
        cfg = self.config
        for _ in range(cfg.max_attempts):
            draw = self._rng.random()
            first = self._tournament(population)
            if draw < cfg.p_crossover:
                second = self._tournament(population)
                child = self._crossover(first[1], second[1])
                meta = {"operation": "crossover", "parents": [first[0], second[0]]}
            elif draw < cfg.p_crossover + cfg.p_subtree_mutation:
                child = self._subtree_mutation(first[1])
                meta = {"operation": "subtree_mutation", "parents": [first[0]]}
            else:
                child = self._point_mutation(first[1])
                meta = {"operation": "point_mutation", "parents": [first[0]]}
            if self._is_new_and_valid(child):
                return child, meta
        return self._immigrant(), {"operation": "immigrant_fallback", "parents": []}

    def _immigrant(self) -> Node:
        for _ in range(self.config.max_attempts):
            tree, _ = random_valid_tree(self._rng, self.grammar, self.limits)
            if self._is_new_and_valid(tree):
                return tree
        tree, _ = random_valid_tree(self._rng, self.grammar, self.limits)
        return tree

    # ------------------------------------------------------------------ API

    def propose(self, context: ProposalContext) -> list[Proposal]:
        self._ingest(context)
        population = self.population()
        n = context.n_requested
        breeding = len(population) >= 2
        n_immigrants = n if not breeding else int(round(self.config.immigrant_fraction * n))
        proposals: list[Proposal] = []
        for index in range(n):
            if index < n_immigrants:
                child = self._immigrant()
                meta: dict[str, Any] = {"operation": "initial" if not breeding else "immigrant", "parents": []}
            else:
                child, meta = self._breed(population)
            self._proposed.add(structural_hash(child))
            proposals.append(
                Proposal(
                    expression=to_expression(child),
                    rationale=f"genetic programming: {meta['operation']}",
                    metadata={"generator": "evolutionary", **meta},
                )
            )
        return proposals

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "evolution": asdict(self.config),
            "grammar": asdict(self.grammar),
            "fitness": "abs(formation ICIR) - complexity_penalty * complexity",
            "uses_feedback": True,
        }


__all__ = ["EvolutionConfig", "EvolutionaryProposer"]
