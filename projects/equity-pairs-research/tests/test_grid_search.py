from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from equity_pairs.backtest import backtest_selected_pairs
from equity_pairs.config import StrategyConfig
from equity_pairs.grid_search import (
    GridSearchSpec,
    build_parameter_return_cache,
    build_parameter_sets,
    run_parameter_grid,
    select_best_parameters,
)


def _synthetic_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=360)
    time = np.arange(len(dates), dtype=float)
    market = 100.0 * np.exp(0.0004 * time + 0.018 * np.sin(time / 31.0))
    spreads = {
        "A": 0.055 * np.sin(time / 7.0),
        "C": 0.045 * np.sin(time / 9.0 + 0.7),
        "E": 0.065 * np.sin(time / 6.0 + 1.3),
    }
    prices = pd.DataFrame(
        {
            "B": market,
            "A": market * np.exp(spreads["A"]),
            "D": market * 0.82,
            "C": market * 0.82 * np.exp(spreads["C"]),
            "F": market * 1.18,
            "E": market * 1.18 * np.exp(spreads["E"]),
        },
        index=dates,
    )
    selected = pd.DataFrame(
        [
            {"pair": "A__B", "sector": "Tech", "dependent": "A", "independent": "B", "alpha": 0.0, "beta": 1.0},
            {"pair": "C__D", "sector": "Tech", "dependent": "C", "independent": "D", "alpha": 0.0, "beta": 1.0},
            {"pair": "E__F", "sector": "Industrials", "dependent": "E", "independent": "F", "alpha": 0.0, "beta": 1.0},
        ]
    )
    return selected, prices


def _base_config() -> StrategyConfig:
    values = {
        "zscore_lookback": 20,
        "zscore_min_periods": 10,
        "entry_z": 1.5,
        "exit_z": 0.25,
        "stop_z": 4.0,
        "maximum_holding_days": 20,
        "transaction_cost_bps": 5.0,
        "annual_short_borrow_bps": 30.0,
    }
    if "zscore_method" in {field.name for field in fields(StrategyConfig)}:
        values["zscore_method"] = "sma"
    return StrategyConfig(**values)


def test_parameter_grid_is_deterministic_valid_and_includes_baseline():
    config = _base_config()
    spec = GridSearchSpec(
        zscore_methods=("sma",),
        zscore_lookbacks=(40, 20, 20),
        entry_zs=(1.5, 1.0),
        exit_zs=(0.25, 1.25),
        maximum_holding_days=(30, 20),
    )

    first = build_parameter_sets(spec, config)
    second = build_parameter_sets(spec, config)

    assert first == second
    assert [row.config_id for row in first] == sorted(row.config_id for row in first)
    assert len({row.config_id for row in first}) == len(first)
    assert sum(row.is_baseline for row in first) == 1
    assert all(row.exit_z < row.entry_z < config.stop_z for row in first)
    assert all(row.zscore_min_periods <= row.zscore_lookback for row in first)


def test_grid_search_tunes_before_lockbox_and_reports_both_portfolios():
    selected, prices = _synthetic_universe()
    config = _base_config()
    spec = GridSearchSpec(
        zscore_methods=("sma",),
        zscore_lookbacks=(20, 40),
        entry_zs=(1.0, 1.5),
        exit_zs=(0.25,),
        maximum_holding_days=(20,),
    )
    heldout_start = prices.index[80]
    lockbox_start = prices.index[250]
    heldout_end = prices.index[-1]

    result = run_parameter_grid(
        selected,
        prices,
        heldout_start,
        lockbox_start,
        heldout_end,
        config,
        spec,
    )

    tuning = result.summary.loc[result.summary["split"].eq("tuning")]
    lockbox = result.summary.loc[result.summary["split"].eq("lockbox")]
    assert set(tuning["portfolio"]) == {"equal_weight", "sector_balanced"}
    assert set(lockbox["portfolio"]) == {"equal_weight", "sector_balanced"}
    assert set(lockbox["config_id"]) <= {
        result.best_parameters["config_id"],
        next(row.config_id for row in build_parameter_sets(spec, config) if row.is_baseline),
    }
    assert (pd.to_datetime(tuning["window_end"]) < lockbox_start).all()
    assert (pd.to_datetime(lockbox["window_start"]) >= lockbox_start).all()
    assert result.ranking.iloc[0]["config_id"] == result.best_parameters["config_id"]
    assert result.failures.empty
    assert not result.comparison.empty
    assert {
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "total_trades",
        "median_trades_per_pair",
        "annual_turnover",
        "positive_pair_fraction",
    }.issubset(result.summary.columns)
    assert selected[["alpha", "beta"]].to_dict("list") == {
        "alpha": [0.0, 0.0, 0.0],
        "beta": [1.0, 1.0, 1.0],
    }


def test_selection_never_uses_lockbox_performance():
    summary = pd.DataFrame(
        [
            {"split": "tuning", "portfolio": "sector_balanced", "config_id": "steady", "sharpe": 1.2, "cagr": 0.08, "max_drawdown": -0.05, "annual_turnover": 2.0, "total_trades": 10, "median_trades_per_pair": 5.0},
            {"split": "tuning", "portfolio": "sector_balanced", "config_id": "jackpot", "sharpe": 0.4, "cagr": 0.03, "max_drawdown": -0.02, "annual_turnover": 1.0, "total_trades": 10, "median_trades_per_pair": 5.0},
            {"split": "lockbox", "portfolio": "sector_balanced", "config_id": "steady", "sharpe": -3.0, "cagr": -0.20, "max_drawdown": -0.30, "annual_turnover": 2.0, "total_trades": 4, "median_trades_per_pair": 2.0},
            {"split": "lockbox", "portfolio": "sector_balanced", "config_id": "jackpot", "sharpe": 9.0, "cagr": 0.90, "max_drawdown": -0.01, "annual_turnover": 1.0, "total_trades": 4, "median_trades_per_pair": 2.0},
        ]
    )

    ranking = select_best_parameters(summary)

    assert ranking.iloc[0]["config_id"] == "steady"
    assert set(ranking["config_id"]) == {"steady", "jackpot"}


@pytest.mark.parametrize("method", ["sma", "ewma"])
def test_fast_tuning_matches_canonical_backtester(method):
    selected, prices = _synthetic_universe()
    config = _base_config()
    spec = GridSearchSpec(
        zscore_methods=(method,),
        zscore_lookbacks=(20,),
        entry_zs=(1.5,),
        exit_zs=(0.25,),
        maximum_holding_days=(20,),
    )
    arguments = (
        selected,
        prices,
        prices.index[80],
        prices.index[250],
        prices.index[-1],
        config,
        spec,
    )

    fast = run_parameter_grid(*arguments, fast_tuning=True)
    canonical = run_parameter_grid(*arguments, fast_tuning=False)
    columns = [
        "portfolio",
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
        "total_trades",
        "median_trades_per_pair",
        "positive_pair_fraction",
    ]
    fast_tuning = fast.summary.loc[fast.summary["split"].eq("tuning"), columns].sort_values(
        "portfolio"
    )
    canonical_tuning = canonical.summary.loc[
        canonical.summary["split"].eq("tuning"), columns
    ].sort_values("portfolio")

    assert fast_tuning["portfolio"].tolist() == canonical_tuning["portfolio"].tolist()
    np.testing.assert_allclose(
        fast_tuning.drop(columns="portfolio").to_numpy(dtype=float),
        canonical_tuning.drop(columns="portfolio").to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_ewma_grid_fails_clearly_when_config_does_not_support_it():
    if "zscore_method" in {field.name for field in fields(StrategyConfig)}:
        pytest.skip("StrategyConfig now supports EWMA")
    with pytest.raises(ValueError, match="only the SMA grid"):
        build_parameter_sets(
            GridSearchSpec(
                zscore_methods=("ewma",),
                zscore_lookbacks=(20,),
                entry_zs=(1.5,),
                exit_zs=(0.25,),
                maximum_holding_days=(20,),
            ),
            _base_config(),
        )


def test_parameter_return_cache_matches_canonical_with_non_equal_weights():
    selected, prices = _synthetic_universe()
    config = _base_config()
    spec = GridSearchSpec(
        zscore_methods=("sma",),
        zscore_lookbacks=(20,),
        entry_zs=(1.5,),
        exit_zs=(0.25,),
        maximum_holding_days=(20,),
    )
    weights = pd.Series({"A__B": 0.60, "C__D": 0.25, "E__F": 0.15})
    start = prices.index[100]
    end = prices.index[300]

    cache = build_parameter_return_cache(
        selected,
        prices,
        start,
        end,
        config,
        spec,
        weights,
    )

    assert cache.returns.shape == (201, 1)
    assert cache.turnover.shape == cache.returns.shape
    assert cache.pair_weights.equals(weights.rename_axis("pair").rename("pair_weight"))
    identifier = cache.returns.columns[0]
    _, signals, _, canonical_pair_returns = backtest_selected_pairs(
        selected,
        prices.loc[:end],
        prices.index[0],
        config,
    )
    expected_pair_returns = canonical_pair_returns.loc[start:end, weights.index]
    pd.testing.assert_frame_equal(
        cache.pair_returns[identifier],
        expected_pair_returns.rename_axis(columns="pair"),
    )
    expected_portfolio = expected_pair_returns.dot(weights)
    pd.testing.assert_series_equal(
        cache.returns[identifier],
        expected_portfolio.rename(identifier).rename_axis("date"),
    )

    positions = signals.pivot(index="date", columns="pair", values="position").sort_index()
    previous = positions.shift(1, fill_value=0)
    expected_entries = ((previous.eq(0)) & positions.ne(0)).loc[start:end, weights.index]
    pd.testing.assert_frame_equal(
        cache.positions[identifier],
        positions.loc[start:end, weights.index]
        .astype(np.int8)
        .rename_axis(index="date", columns="pair"),
        check_freq=False,
    )
    pd.testing.assert_frame_equal(
        cache.entries[identifier],
        expected_entries.rename_axis(index="date", columns="pair"),
        check_freq=False,
    )
    pd.testing.assert_series_equal(
        cache.entry_counts.loc[identifier, weights.index],
        expected_entries.sum().rename(identifier),
        check_names=False,
    )
    assert cache.entry_counts.loc[identifier, "total_entries"] == int(
        expected_entries.to_numpy().sum()
    )

    canonical_turnover = signals.pivot(
        index="date", columns="pair", values="turnover"
    ).sort_index()
    expected_turnover = canonical_turnover.loc[start:end, weights.index].dot(weights)
    pd.testing.assert_series_equal(
        cache.turnover[identifier],
        expected_turnover.rename(identifier).rename_axis("date"),
        check_freq=False,
    )


def test_parameter_return_cache_discards_future_prices_but_keeps_warmup():
    selected, prices = _synthetic_universe()
    config = _base_config()
    spec = GridSearchSpec(
        zscore_methods=("sma",),
        zscore_lookbacks=(20, 40),
        entry_zs=(1.5,),
        exit_zs=(0.25,),
        maximum_holding_days=(20,),
    )
    weights = {"A__B": 0.50, "C__D": 0.30, "E__F": 0.20}
    start = prices.index[100]
    end = prices.index[250]
    changed_future = prices.copy()
    changed_future.loc[changed_future.index > end, "A"] *= 100.0
    changed_future.loc[changed_future.index > end, "D"] *= 0.01

    original = build_parameter_return_cache(
        selected, prices, start, end, config, spec, weights
    )
    mutated = build_parameter_return_cache(
        selected, changed_future, start, end, config, spec, weights
    )

    pd.testing.assert_frame_equal(original.returns, mutated.returns)
    pd.testing.assert_frame_equal(original.turnover, mutated.turnover)
    assert original.returns.shape[1] == 2
    assert original.returns.index.min() == start
    assert original.returns.index.max() == end
    # A fully warmed-up cache need not discard the first returned date.
    assert np.isfinite(original.pair_returns[original.returns.columns[0]].iloc[0]).all()


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"A__B": 0.6, "C__D": 0.4}, "align exactly"),
        ({"A__B": 0.6, "C__D": 0.3, "E__F": 0.2}, "sum to one"),
        ({"A__B": 0.6, "C__D": 0.5, "E__F": -0.1}, "cannot be negative"),
    ],
)
def test_parameter_return_cache_rejects_invalid_fixed_weights(weights, message):
    selected, prices = _synthetic_universe()
    with pytest.raises(ValueError, match=message):
        build_parameter_return_cache(
            selected,
            prices,
            prices.index[100],
            prices.index[250],
            _base_config(),
            GridSearchSpec(
                zscore_methods=("sma",),
                zscore_lookbacks=(20,),
                entry_zs=(1.5,),
                exit_zs=(0.25,),
                maximum_holding_days=(20,),
            ),
            weights,
        )
