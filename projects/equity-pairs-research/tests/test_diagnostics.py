from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_pairs.diagnostics import (
    equal_weight_market_proxy,
    pair_correlation_diagnostics,
    overlap_correlation_diagnostics,
    portfolio_exposure_diagnostics,
    regression_diagnostics,
    shock_diagnostics,
)


def test_equal_weight_market_proxy_and_regression_beta():
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {"A": [100, 101, 102, 101, 103, 104], "B": [50, 51, 52, 52, 53, 54]},
        index=dates,
    )
    market = equal_weight_market_proxy(prices)
    strategy = 0.2 + 1.5 * market.fillna(0)
    diagnostics = regression_diagnostics(strategy.iloc[1:], market.iloc[1:])

    assert diagnostics["observations"] == 5
    assert diagnostics["market_beta"] == pytest.approx(1.5)
    assert diagnostics["market_correlation"] == pytest.approx(1.0)


def test_shock_and_pair_correlation_diagnostics():
    dates = pd.bdate_range("2024-01-02", periods=5)
    market = pd.Series([-0.03, 0.01, 0.025, -0.005, -0.021], index=dates)
    strategies = pd.DataFrame({"portfolio": [0.01, 0, -0.01, 0.002, 0.02]}, index=dates)
    shocks = shock_diagnostics(strategies, market, shock_threshold=0.02)
    down = shocks.query("regime == 'market_down_at_least_2%' and portfolio == 'portfolio'").iloc[0]
    assert down["observations"] == 2
    assert down["mean_daily_return"] == pytest.approx(0.015)

    pair_returns = pd.DataFrame(
        {"A__B": [0.01, 0, -0.01, 0, 0.02], "A__C": [0.02, 0, -0.02, 0, 0.04]},
        index=dates,
    )
    active = pair_returns.ne(0).astype(int)
    matrix, active_pairs, summary = pair_correlation_diagnostics(
        pair_returns, active, minimum_active_overlap=3
    )
    assert matrix.loc["A__B", "A__C"] == pytest.approx(1.0)
    assert active_pairs.iloc[0]["active_return_correlation"] == pytest.approx(1.0)
    assert summary["active_correlation_count"] == 1

    selected = pd.DataFrame(
        {
            "pair": ["A__B", "A__C"],
            "dependent": ["A", "A"],
            "independent": ["B", "C"],
        }
    )
    detail, overlap_summary = overlap_correlation_diagnostics(
        matrix, active_pairs, selected
    )
    assert bool(detail.iloc[0]["shares_underlying"]) is True
    assert detail.iloc[0]["shared_tickers"] == "A"
    assert overlap_summary.iloc[0]["mean_absolute_active_correlation"] == pytest.approx(1.0)


def test_portfolio_exposure_filters_to_selected_and_aggregates_shared_names():
    date = pd.Timestamp("2024-01-02")
    selected = pd.DataFrame(
        {
            "pair": ["A__B", "A__C"],
            "dependent": ["A", "A"],
            "independent": ["B", "C"],
        }
    )
    signals = pd.DataFrame(
        {
            "date": [date, date, date],
            "pair": ["A__B", "A__C", "X__Y"],
            "weight_dependent": [0.5, -0.5, 0.5],
            "weight_independent": [-0.5, 0.5, -0.5],
        }
    )
    daily, names, summary = portfolio_exposure_diagnostics(signals, selected)

    assert summary.iloc[0]["selected_pairs"] == 2
    assert summary.iloc[0]["mean_sleeve_gross_exposure"] == pytest.approx(1.0)
    assert summary.iloc[0]["mean_netted_gross_exposure"] == pytest.approx(0.5)
    assert names.loc[names["ticker"].eq("A"), "exposure"].iloc[0] == pytest.approx(0.0)
    assert daily.iloc[0]["maximum_absolute_name_exposure"] == pytest.approx(0.25)
