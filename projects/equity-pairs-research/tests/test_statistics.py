from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_pairs import statistics
from equity_pairs.statistics import PairScreenSettings


def test_benjamini_hochberg_adjusts_monotonically_and_preserves_missing_order():
    pvalues = [0.01, 0.04, np.nan, 0.03, 0.002]

    adjusted = statistics.benjamini_hochberg(pvalues)

    np.testing.assert_allclose(
        adjusted[[0, 1, 3, 4]],
        np.array([0.02, 0.04, 0.04, 0.008]),
    )
    assert np.isnan(adjusted[2])
    assert np.isnan(statistics.benjamini_hochberg([np.nan, np.inf])).all()


def _cointegrated_prices(observations: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(20240229)
    log_x = 4.5 + np.cumsum(rng.normal(0.0, 0.01, observations))
    residual = np.zeros(observations)
    shocks = rng.normal(0.0, 0.012, observations)
    for index in range(1, observations):
        residual[index] = 0.70 * residual[index - 1] + shocks[index]
    log_y = 0.30 + 1.20 * log_x + residual
    return pd.DataFrame({"A": np.exp(log_y), "B": np.exp(log_x)})


def _independent_random_walk_prices(observations: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    log_a = 4.5 + np.cumsum(rng.normal(0.0, 0.012, observations))
    log_b = 4.1 + np.cumsum(rng.normal(0.0, 0.014, observations))
    return pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)})


def test_evaluate_pair_distinguishes_seeded_cointegrated_and_independent_walks():
    settings = PairScreenSettings(
        minimum_observations=100,
        minimum_history_ratio=0.95,
        adf_maxlag=1,
        workers=1,
    )
    i1_lookup = {"A": True, "B": True}

    cointegrated = statistics.evaluate_pair(
        "A",
        "B",
        "Information Technology",
        _cointegrated_prices(),
        expected_observations=250,
        settings=settings,
        i1_lookup=i1_lookup,
    )
    independent = statistics.evaluate_pair(
        "A",
        "B",
        "Information Technology",
        _independent_random_walk_prices(),
        expected_observations=250,
        settings=settings,
        i1_lookup=i1_lookup,
    )

    assert cointegrated["screen_status"] == "ok"
    assert cointegrated["dependent"] == "A"
    assert cointegrated["beta"] == pytest.approx(1.20, abs=0.08)
    assert cointegrated["pair_pvalue"] < 0.01
    assert cointegrated["pair_pvalue"] == pytest.approx(
        2.0 * min(cointegrated["eg_pvalue_ab"], cointegrated["eg_pvalue_ba"])
    )
    assert cointegrated["positive_hedge_ratio"] is True
    assert cointegrated["both_look_i1"] is True
    assert independent["screen_status"] == "ok"
    assert independent["pair_pvalue"] > 0.10


def test_evaluate_pair_short_circuits_when_history_is_insufficient(monkeypatch):
    prices = _cointegrated_prices(observations=40)
    settings = PairScreenSettings(minimum_observations=100, workers=1)
    monkeypatch.setattr(
        statistics,
        "_safe_coint",
        lambda *_args, **_kwargs: pytest.fail("cointegration test should not run"),
    )

    result = statistics.evaluate_pair(
        "A", "B", "Sector", prices, expected_observations=100, settings=settings
    )

    assert result["screen_status"] == "insufficient_history"
    assert result["observations"] == 40
    assert result["history_ratio"] == pytest.approx(0.4)
    assert np.isnan(result["pair_pvalue"])


def test_select_top_pairs_prioritizes_strict_candidates_then_deterministic_fallbacks():
    diagnostics = pd.DataFrame(
        [
            {
                "sector": "Tech",
                "pair": "B__C",
                "screen_status": "ok",
                "pair_pvalue": 0.001,
                "half_life_days": 4.0,
                "eligible": False,
            },
            {
                "sector": "Tech",
                "pair": "A__B",
                "screen_status": "ok",
                "pair_pvalue": 0.020,
                "half_life_days": 3.0,
                "eligible": True,
            },
            {
                "sector": "Tech",
                "pair": "A__C",
                "screen_status": "ok",
                "pair_pvalue": 0.002,
                "half_life_days": 5.0,
                "eligible": False,
            },
        ]
    )

    selected = statistics.select_top_pairs(diagnostics, top_n=3)

    assert selected["pair"].tolist() == ["A__B", "B__C", "A__C"]
    assert selected["selection_status"].tolist() == ["strict", "fallback", "fallback"]
    assert selected["selection_rank_in_sector"].tolist() == [1, 2, 3]
