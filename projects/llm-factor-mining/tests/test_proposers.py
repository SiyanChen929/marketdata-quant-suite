from __future__ import annotations

import numpy as np
import pytest

from llm_factor_mining.dsl import parse, structural_hash, validate
from llm_factor_mining.dsl.nodes import Call, Constant, Terminal
from llm_factor_mining.dsl.validate import DSLLimits, expression_stats
from llm_factor_mining.proposers import (
    EvaluatedFeedback,
    EvolutionConfig,
    EvolutionaryProposer,
    GrammarConfig,
    ProposalContext,
    RandomGrammarProposer,
    random_tree,
)
from llm_factor_mining.proposers.trees import replace_at, series_paths, subtree_at


GRAMMAR = "grammar card"


def context(round_index: int = 0, n: int = 10, **kwargs) -> ProposalContext:
    return ProposalContext(round_index=round_index, n_requested=n, grammar=GRAMMAR, **kwargs)


def feedback(expression: str, icir: float, complexity: float = 3.0) -> EvaluatedFeedback:
    return EvaluatedFeedback(expression, icir / 10, icir, icir * 5, 0.3, complexity)


def test_random_tree_respects_depth_and_types() -> None:
    rng = np.random.default_rng(0)
    config = GrammarConfig(max_depth=4)
    for _ in range(300):
        tree = random_tree(rng, config)
        stats = expression_stats(tree)
        assert stats.depth <= 4
        assert isinstance(tree, Call)  # the root is never a bare leaf


def test_series_paths_and_replace() -> None:
    node = parse("ts_mean(close - open, 5)")
    paths = dict(series_paths(node))
    assert () in paths and (0,) in paths and (0, 0) in paths and (0, 1) in paths
    assert (1,) not in paths  # the window literal is not a series slot
    replaced = replace_at(node, (0, 1), Terminal("low"))
    assert subtree_at(replaced, (0, 1)) == Terminal("low")
    assert subtree_at(replaced, (1,)) == Constant(5.0, True)


def test_random_proposer_valid_deterministic_and_non_repeating() -> None:
    first = RandomGrammarProposer(seed=5)
    batches = [first.propose(context(r, 25)) for r in range(4)]
    expressions = [p.expression for batch in batches for p in batch]
    assert len(expressions) == 100
    assert all(validate(text).ok for text in expressions)
    assert len({structural_hash(parse(text)) for text in expressions}) == 100
    again = RandomGrammarProposer(seed=5)
    assert [p.expression for p in again.propose(context(0, 25))] == [p.expression for p in batches[0]]
    other = RandomGrammarProposer(seed=6)
    assert [p.expression for p in other.propose(context(0, 25))] != [p.expression for p in batches[0]]
    assert first.describe()["uses_feedback"] is False
    assert batches[0][0].metadata["generator"] == "random_grammar"


def test_random_proposer_honours_limits() -> None:
    limits = DSLLimits(max_depth=3, max_window=20, max_lookback=40)
    proposer = RandomGrammarProposer(seed=1, grammar=GrammarConfig(max_depth=3, windows=(2, 5, 10, 20)), limits=limits)
    for item in proposer.propose(context(0, 40)):
        assert validate(item.expression, limits).ok


def test_evolutionary_initial_round_is_random_then_breeds_from_feedback() -> None:
    proposer = EvolutionaryProposer(seed=3)
    initial = proposer.propose(context(0, 12))
    assert {item.metadata["operation"] for item in initial} == {"initial"}
    parents = [
        feedback("ts_mean(returns,5)", 0.5),
        feedback("cs_rank(volume)", 0.4),
        feedback("delta(close,3)", -0.45),
        feedback("ts_std(returns,20)", 0.05),
        feedback("abs(open)", float("nan")),
    ]
    children = proposer.propose(context(1, 30, last_round=tuple(parents)))
    operations = {item.metadata["operation"] for item in children}
    assert operations & {"crossover", "subtree_mutation", "point_mutation"}
    archived = {structural_hash(parse(item.expression)) for item in parents}
    for child in children:
        assert validate(child.expression).ok
        assert structural_hash(parse(child.expression)) not in archived
        for parent in child.metadata.get("parents", []):
            assert parent in archived
    # the NaN-fitness expression never enters the population
    population = proposer.population()
    assert structural_hash(parse("abs(open)")) not in {key for key, _, _ in population}
    assert population[0][0] == structural_hash(parse("ts_mean(returns,5)"))


def test_evolutionary_fitness_uses_abs_icir_and_complexity() -> None:
    proposer = EvolutionaryProposer(seed=0, config=EvolutionConfig(complexity_penalty=0.01))
    assert proposer.fitness(feedback("close", -0.4, 10.0)) == pytest.approx(0.4 - 0.1)
    assert proposer.fitness(feedback("close", float("nan"))) == float("-inf")
    with pytest.raises(ValueError):
        EvolutionConfig(p_crossover=0.9, p_subtree_mutation=0.3, p_point_mutation=0.2)


def test_evolutionary_is_deterministic() -> None:
    parents = (feedback("ts_mean(returns,5)", 0.5), feedback("cs_rank(volume)", 0.4), feedback("delta(close,3)", 0.3))

    def run(seed: int) -> list[str]:
        proposer = EvolutionaryProposer(seed=seed)
        proposer.propose(context(0, 5))
        return [item.expression for item in proposer.propose(context(1, 20, last_round=parents))]

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_tree_log_probability_matches_the_sampler() -> None:
    import math
    from collections import Counter

    from llm_factor_mining.dsl.nodes import to_expression
    from llm_factor_mining.proposers.trees import strip_orientation, tree_log_probability

    config = GrammarConfig()
    rng = np.random.default_rng(11)
    n = 60_000
    counts = Counter(to_expression(random_tree(rng, config)) for _ in range(n))
    for expression in ("abs(close)", "delay(volume,1)", "ts_mean(returns,5)"):
        p = math.exp(tree_log_probability(parse(expression), config))
        assert abs(counts[expression] - n * p) < 4.5 * math.sqrt(n * p), expression
    assert tree_log_probability(parse("close"), config) == -math.inf  # the root is never a leaf
    assert tree_log_probability(parse("ts_mean(close, 7)"), config) == -math.inf  # 7 is not on the menu
    assert tree_log_probability(parse("ts_mean(ts_mean(ts_mean(ts_mean(close, 5), 5), 5), 5)"), config) == -math.inf
    assert strip_orientation(parse("-(-ts_mean(returns, 5))")) == parse("ts_mean(returns, 5)")
    assert 1 in config.windows  # one-session changes (delta(x, 1)) are in the baselines' search space
