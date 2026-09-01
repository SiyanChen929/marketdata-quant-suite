from __future__ import annotations

import pandas as pd

from quant_system.backtest.engine import DailyBacktestEngine
from quant_system.config import AppConfig, ExecutionConfig, PortfolioConfig, RiskConfig


def test_minute_symbol_stop_exits_same_day_and_changes_equity() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    prices = pd.DataFrame(
        [
            {"date": dates[0], "symbol": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1_000_000},
            {"date": dates[1], "symbol": "AAA", "open": 100, "high": 101, "low": 80, "close": 80, "adj_close": 80, "volume": 1_000_000},
        ]
    )
    intraday = pd.DataFrame(
        [
            {"datetime": "2026-01-05 09:35:00", "date": dates[1], "symbol": "AAA", "open": 100, "high": 100, "low": 99, "close": 99, "volume": 1000},
            {"datetime": "2026-01-05 10:00:00", "date": dates[1], "symbol": "AAA", "open": 99, "high": 99, "low": 90, "close": 90, "volume": 2000},
            {"datetime": "2026-01-05 16:00:00", "date": dates[1], "symbol": "AAA", "open": 88, "high": 90, "low": 80, "close": 80, "volume": 3000},
        ]
    )
    config = AppConfig(
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0),
        portfolio=PortfolioConfig(initial_capital=100_000, target_gross_exposure=1.0, target_net_exposure=1.0),
        risk=RiskConfig(
            intraday_enabled=True,
            intraday_symbol_stop_loss_pct=0.08,
            intraday_portfolio_drawdown_limit=0.50,
            intraday_breadth_down_pct=1.10,
        ),
    )
    targets = pd.DataFrame({"date": [dates[0]], "symbol": ["AAA"], "target_weight": [1.0]})
    result = DailyBacktestEngine(prices, config, intraday_prices=intraday).run(targets)
    final_equity = result.equity_curve["equity"].iloc[-1]
    assert final_equity == 90_000
    assert result.equity_curve["intraday_risk_event_count"].iloc[-1] == 1
    intraday_flags = result.trades["intraday_risk_exit"].map(lambda value: bool(value) if pd.notna(value) else False)
    assert bool(intraday_flags.any())
    exit_trade = result.trades[intraday_flags].iloc[-1]
    assert exit_trade["price"] == 90
    assert "same bar close" in exit_trade["reason_for_exit"]


def test_minute_symbol_stop_can_use_next_bar_open_for_conservative_audit() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    prices = pd.DataFrame(
        [
            {"date": dates[0], "symbol": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1_000_000},
            {"date": dates[1], "symbol": "AAA", "open": 100, "high": 101, "low": 80, "close": 80, "adj_close": 80, "volume": 1_000_000},
        ]
    )
    intraday = pd.DataFrame(
        [
            {"datetime": "2026-01-05 09:35:00", "date": dates[1], "symbol": "AAA", "open": 100, "high": 100, "low": 99, "close": 99, "volume": 1000},
            {"datetime": "2026-01-05 10:00:00", "date": dates[1], "symbol": "AAA", "open": 99, "high": 99, "low": 90, "close": 90, "volume": 2000},
            {"datetime": "2026-01-05 16:00:00", "date": dates[1], "symbol": "AAA", "open": 88, "high": 90, "low": 80, "close": 80, "volume": 3000},
        ]
    )
    config = AppConfig(
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0),
        portfolio=PortfolioConfig(initial_capital=100_000, target_gross_exposure=1.0, target_net_exposure=1.0),
        risk=RiskConfig(
            intraday_enabled=True,
            intraday_fill_policy="next_bar_open",
            intraday_symbol_stop_loss_pct=0.08,
            intraday_portfolio_drawdown_limit=0.50,
            intraday_breadth_down_pct=1.10,
        ),
    )
    targets = pd.DataFrame({"date": [dates[0]], "symbol": ["AAA"], "target_weight": [1.0]})
    result = DailyBacktestEngine(prices, config, intraday_prices=intraday).run(targets)
    final_equity = result.equity_curve["equity"].iloc[-1]
    assert final_equity == 88_000
    assert result.equity_curve["intraday_risk_event_count"].iloc[-1] == 1
    intraday_flags = result.trades["intraday_risk_exit"].map(lambda value: bool(value) if pd.notna(value) else False)
    assert bool(intraday_flags.any())
    exit_trade = result.trades[intraday_flags].iloc[-1]
    assert exit_trade["price"] == 88
    assert "next bar open" in exit_trade["reason_for_exit"]


def test_without_minute_data_intraday_engine_is_noop() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    prices = pd.DataFrame(
        [
            {"date": dates[0], "symbol": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1_000_000},
            {"date": dates[1], "symbol": "AAA", "open": 100, "high": 101, "low": 80, "close": 80, "adj_close": 80, "volume": 1_000_000},
        ]
    )
    config = AppConfig(
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0),
        portfolio=PortfolioConfig(initial_capital=100_000, target_gross_exposure=1.0, target_net_exposure=1.0),
        risk=RiskConfig(intraday_enabled=True, intraday_symbol_stop_loss_pct=0.08),
    )
    targets = pd.DataFrame({"date": [dates[0]], "symbol": ["AAA"], "target_weight": [1.0]})
    result = DailyBacktestEngine(prices, config).run(targets)
    assert result.equity_curve["equity"].iloc[-1] == 80_000
