from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_pairs import backtest
from equity_pairs.backtest import PairModel
from equity_pairs.config import StrategyConfig


def test_target_state_emits_entry_and_all_exit_events():
    index = pd.bdate_range("2024-01-02", periods=11)
    zscore = pd.Series(
        [np.nan, -2.1, -1.0, 0.4, 2.1, 4.1, -2.2, np.nan, -2.2, -1.5, -1.4],
        index=index,
    )
    config = StrategyConfig(maximum_holding_days=2)

    target, events = backtest._target_state(zscore, config)

    assert target.tolist() == [0, 1, 1, 0, -1, 0, 1, 0, 1, 1, 0]
    assert events.tolist() == [
        "flat",
        "enter_long_spread",
        "hold",
        "exit_mean_reversion",
        "enter_short_spread",
        "exit_stop",
        "enter_long_spread",
        "exit_missing_signal",
        "enter_long_spread",
        "hold",
        "exit_max_holding",
    ]


def test_backtest_lags_execution_one_day_and_charges_entry_exit_and_borrow_costs(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 110.0, 121.0, 121.0, 121.0],
            "B": [100.0] * 6,
        },
        index=dates,
    )
    decision_targets = [0, 1, 1, 0, 0, 0]
    decision_events = [
        "flat",
        "enter_long_spread",
        "hold",
        "exit_mean_reversion",
        "flat",
        "flat",
    ]

    def fixed_target_state(zscore, _config):
        return (
            pd.Series(
                decision_targets,
                index=zscore.index,
                dtype=int,
                name="target_position",
            ),
            pd.Series(
                decision_events,
                index=zscore.index,
                dtype="string",
                name="decision_event",
            ),
        )

    monkeypatch.setattr(backtest, "_target_state", fixed_target_state)
    config = StrategyConfig(
        transaction_cost_bps=10.0,
        annual_short_borrow_bps=252.0,
    )
    model = PairModel(
        pair="A__B",
        sector="Tech",
        dependent="A",
        independent="B",
        alpha=0.0,
        beta=1.0,
    )

    result = backtest.run_pair_backtest(model, prices, dates[0], config)
    signals = result.signals

    assert signals["target_position"].tolist() == decision_targets
    assert signals["position"].tolist() == [0, 0, 1, 1, 0, 0]
    assert signals.loc[dates[1], "decision_event"] == "enter_long_spread"
    assert signals.loc[dates[1], "position"] == 0
    assert signals.loc[dates[2], "effective_event"] == "enter_long_spread"
    assert signals.loc[dates[2], "position"] == 1
    assert signals.loc[dates[3], "decision_event"] == "exit_mean_reversion"
    assert signals.loc[dates[3], "position"] == 1
    assert signals.loc[dates[4], "effective_event"] == "exit_mean_reversion"
    assert signals.loc[dates[4], "position"] == 0

    drift_turnover = 1.0 / 21.0
    np.testing.assert_allclose(
        signals["turnover"], [0.0, 0.0, 1.0, drift_turnover, 1.0, 0.0]
    )
    np.testing.assert_allclose(
        signals["transaction_cost"],
        [0.0, 0.0, 0.001, drift_turnover * 0.001, 0.001, 0.0],
    )
    np.testing.assert_allclose(
        signals["borrow_cost"], [0.0, 0.0, 0.00005, 0.00005, 0.0, 0.0]
    )
    np.testing.assert_allclose(signals["gross_return"], [0.0, 0.0, 0.05, 0.05, 0.0, 0.0])
    np.testing.assert_allclose(
        signals["net_return"],
        [0.0, 0.0, 0.04895, 0.05 - drift_turnover * 0.001 - 0.00005, -0.001, 0.0],
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == "long_spread"
    assert trade["entry_date"] == dates[2]
    assert trade["exit_date"] == dates[4]
    assert trade["exit_reason"] == "exit_mean_reversion"
    assert bool(trade["closed"]) is True
    assert trade["transaction_cost_sum"] == pytest.approx(0.002 + drift_turnover * 0.001)
    assert trade["borrow_cost_sum"] == pytest.approx(0.0001)
    assert trade["gross_return_sum"] == pytest.approx(0.10)
    expected_trade_return = (
        (1.0 + 0.04895)
        * (1.0 + 0.05 - drift_turnover * 0.001 - 0.00005)
        * (1.0 - 0.001)
        - 1.0
    )
    assert trade["net_return"] == pytest.approx(expected_trade_return)
    assert result.metrics["trade_count"] == 1
    assert result.metrics["closed_trade_count"] == 1
    assert result.metrics["transaction_cost_sum"] == pytest.approx(
        0.002 + drift_turnover * 0.001
    )
    assert result.metrics["borrow_cost_sum"] == pytest.approx(0.0001)


@pytest.mark.parametrize("beta", [0.0, -1.0, np.nan, np.inf])
def test_backtest_rejects_nonpositive_or_nonfinite_hedge_ratios(beta):
    prices = pd.DataFrame(
        {"A": [100.0, 101.0], "B": [100.0, 99.0]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    model = PairModel("A__B", "Tech", "A", "B", 0.0, beta)

    with pytest.raises(ValueError, match="positive finite hedge ratio"):
        backtest.run_pair_backtest(model, prices, prices.index[0], StrategyConfig())


def test_compute_zscore_supports_sma_and_causal_ewma():
    spread = pd.Series(
        np.linspace(-1.0, 1.0, 30) + 0.1 * np.sin(np.arange(30)),
        index=pd.bdate_range("2024-01-02", periods=30),
    )
    sma = StrategyConfig(zscore_method="sma", zscore_lookback=20, zscore_min_periods=10)
    ewma = StrategyConfig(zscore_method="ewma", zscore_lookback=20, zscore_min_periods=10)

    sma_center, sma_scale, sma_z = backtest.compute_zscore(spread, sma)
    ewma_center, ewma_scale, ewma_z = backtest.compute_zscore(spread, ewma)

    pd.testing.assert_series_equal(
        sma_center,
        spread.rolling(20, min_periods=10).mean(),
    )
    pd.testing.assert_series_equal(
        sma_scale,
        spread.rolling(20, min_periods=10).std(ddof=1),
    )
    assert sma_z.notna().sum() > 0
    assert ewma_z.notna().sum() > 0
    assert not np.allclose(sma_z.dropna(), ewma_z.dropna())

    # Appending a future observation cannot alter any already-computed value.
    extended = pd.concat([spread, pd.Series([25.0], index=[spread.index[-1] + pd.offsets.BDay()])])
    _, _, extended_z = backtest.compute_zscore(extended, ewma)
    pd.testing.assert_series_equal(ewma_z, extended_z.loc[spread.index], check_freq=False)
