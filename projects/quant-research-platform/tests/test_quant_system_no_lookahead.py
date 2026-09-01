from __future__ import annotations

import pandas as pd

from quant_system.backtest.engine import DailyBacktestEngine
from quant_system.config import AppConfig, ExecutionConfig, PortfolioConfig


def test_target_from_close_executes_next_open_not_same_close() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "open": 10, "high": 10, "low": 10, "close": 10, "adj_close": 10, "volume": 1_000_000},
            {"date": "2024-01-02", "symbol": "AAA", "open": 20, "high": 20, "low": 20, "close": 20, "adj_close": 20, "volume": 1_000_000},
            {"date": "2024-01-03", "symbol": "AAA", "open": 30, "high": 30, "low": 30, "close": 30, "adj_close": 30, "volume": 1_000_000},
        ]
    )
    targets = pd.DataFrame([{"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "target_weight": 1.0}])
    config = AppConfig(
        portfolio=PortfolioConfig(initial_capital=1000, target_gross_exposure=1, target_net_exposure=1),
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0, short_borrow_fee_annual=0),
    )
    result = DailyBacktestEngine(prices, config).run(targets)
    trade = result.trades.iloc[0]
    assert trade["date"] == pd.Timestamp("2024-01-02")
    assert trade["price"] == 20
    assert result.equity_curve.loc[result.equity_curve["date"] == pd.Timestamp("2024-01-01"), "equity"].iloc[0] == 1000


def test_signal_date_without_next_bar_does_not_trade() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "open": 10, "high": 10, "low": 10, "close": 10, "adj_close": 10, "volume": 1_000_000},
        ]
    )
    targets = pd.DataFrame([{"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "target_weight": 1.0}])
    result = DailyBacktestEngine(prices, AppConfig()).run(targets)
    assert result.trades.empty


def test_capacity_uses_prior_day_adv_not_execution_day_volume() -> None:
    prices = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "symbol": "AAA",
                "open": 10,
                "high": 10,
                "low": 10,
                "close": 10,
                "adj_close": 10,
                "volume": 100_000_000 if i == 5 else 1_000_000,
            }
            for i in range(6)
        ]
    )
    targets = pd.DataFrame([{"date": pd.Timestamp("2024-01-05"), "symbol": "AAA", "target_weight": 1.0}])
    config = AppConfig(
        portfolio=PortfolioConfig(initial_capital=10_000_000, target_gross_exposure=1, target_net_exposure=1),
        execution=ExecutionConfig(
            slippage_bps=0,
            commission_per_share=0,
            short_borrow_fee_annual=0,
            daily_turnover_cap=10.0,
            max_adv_participation=0.10,
            market_impact_bps_per_1pct_adv=0,
            spread_bps=0,
            min_liquidity_cost_bps=0,
        ),
    )
    result = DailyBacktestEngine(prices, config).run(targets)
    trade = result.trades.iloc[0]
    assert trade["date"] == pd.Timestamp("2024-01-06")
    assert trade["adv20_dollars"] == 10_000_000
    assert trade["quantity"] * trade["price"] == 1_000_000
