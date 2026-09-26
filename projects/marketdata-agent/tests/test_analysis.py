"""Pre-registered statistics: threshold rule, repetition collapse, McNemar, multiplicity, bootstrap, power."""

from __future__ import annotations

import math

import pytest

from marketdata_agent.bench.analysis import (
    all_repetitions_success,
    benjamini_hochberg,
    binomial_tail,
    cluster_bootstrap_ci,
    holm,
    mcnemar_exact,
    min_successes,
    noninferiority_paired_n,
    power_threshold,
    power_threshold_clustered,
    threshold_decision,
)


def test_threshold_rule():
    assert threshold_decision(96, 96, 0.95) == "supported"  # lower bound 0.9615
    assert threshold_decision(95, 96, 0.95) == "inconclusive"  # lower bound 0.943
    assert threshold_decision(50, 100, 0.95) == "not_supported"
    assert threshold_decision(0, 300, 0.02, direction="at_most") == "supported"
    assert threshold_decision(30, 100, 0.02, direction="at_most") == "not_supported"
    assert threshold_decision(0, 0, 0.9) == "inconclusive"
    with pytest.raises(ValueError):
        threshold_decision(1, 2, 0.5, direction="sideways")


def test_all_repetitions_success_collapses_by_item():
    outcomes = [("a", True), ("a", True), ("b", True), ("b", False), ("c", False)]
    assert all_repetitions_success(outcomes) == {"a": True, "b": False, "c": False}


def test_exact_binomial_and_mcnemar_known_values():
    assert binomial_tail(10, 0, 0.3) == 1.0 and binomial_tail(10, 11, 0.3) == 0.0
    assert binomial_tail(3, 3, 0.5) == pytest.approx(0.125)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 * 0.5**6)
    assert mcnemar_exact(6, 0, alternative="greater") == pytest.approx(0.5**6)
    assert mcnemar_exact(5, 5) == 1.0


def test_multiplicity_procedures():
    p = [0.001, 0.02, 0.03, 0.2]
    assert benjamini_hochberg(p, 0.05) == [True, True, True, False]
    assert holm(p, 0.05) == [True, False, False, False]
    assert benjamini_hochberg([], 0.05) == [] and holm([], 0.05) == []


def test_cluster_bootstrap_is_deterministic_and_brackets_the_mean():
    data = {"a": [1, 1, 1], "b": [0, 0], "c": [1, 0]}
    mean, low, high = cluster_bootstrap_ci(data, n_boot=2000, seed=3)
    assert mean == pytest.approx(4 / 7) and low <= mean <= high
    assert cluster_bootstrap_ci(data, n_boot=2000, seed=3) == (mean, low, high)


def test_power_helpers():
    assert min_successes(96, 0.95) == 96 and min_successes(10, 0.99) is None
    assert power_threshold(96, 0.95, 1.0) == 1.0
    assert power_threshold(96, 0.95, 0.99**3) == pytest.approx((0.99**3) ** 96)
    independent = power_threshold(300, 0.9, 0.95)
    assert power_threshold_clustered([30] * 10, 0.9, 0.95, 0.0) == independent
    assert power_threshold_clustered([30] * 10, 0.9, 0.95, 0.2, n_sim=4000) < independent  # clustering costs power
    n = noninferiority_paired_n(0.10, 0.05)
    assert n == math.ceil(0.10 * (1.6448536 + 0.8416212) ** 2 / 0.05**2)


def test_mcnemar_power_known_values():
    from marketdata_agent.bench.analysis import mcnemar_power

    # With no discordance against arm 1, the one-sided exact test needs b >= 5 at alpha 0.05 (0.5**5 < 0.05 < 0.5**4).
    assert mcnemar_power(5, 1.0, 0.0) == pytest.approx(1.0)
    assert mcnemar_power(5, 0.5, 0.0) == pytest.approx(0.5**5)
    assert mcnemar_power(4, 1.0, 0.0) == 0.0
    assert mcnemar_power(100, 0.0, 0.0) == 0.0
    assert mcnemar_power(300, 0.05, 0.0) > mcnemar_power(300, 0.05, 0.01) > mcnemar_power(300, 0.02, 0.01)


def test_the_holm_adjusted_sweep_size_is_the_preregistered_one():
    from marketdata_agent.bench import design
    from marketdata_agent.bench.analysis import noninferiority_paired_n

    needed = noninferiority_paired_n(0.2, 0.05, alpha=0.05 / design.HOLM_COMPARISONS)
    answerable_per_suite = sum(n for (category, _), n in design.EVALUATION_COUNTS.items() if category in {"lookup", "compute", "multi_step"})
    assert needed == 628 and design.SWEEP_SUITES * answerable_per_suite >= needed
