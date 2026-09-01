from __future__ import annotations

import pandas as pd

from quant_system.backtest.engine import DailyBacktestEngine
from quant_system.config import AppConfig, EventsConfig, ExecutionConfig, PortfolioConfig, RiskConfig
from quant_system.events.earnings import align_earnings_to_dates
from quant_system.events.event_risk import score_event_risk
from quant_system.portfolio.risk import apply_event_risk_to_targets


def test_event_risk_blocks_high_risk_targets() -> None:
    targets = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "A", "target_weight": 0.1, "event_risk_score": 80},
            {"date": "2024-01-01", "symbol": "B", "target_weight": 0.1, "event_risk_score": 20},
        ]
    )
    out = apply_event_risk_to_targets(targets, EventsConfig(block_high_event_risk=True, max_event_risk_score_for_normal_strategies=70))
    assert out.loc[out["symbol"] == "A", "target_weight"].iloc[0] == 0.0
    assert out.loc[out["symbol"] == "B", "target_weight"].iloc[0] == 0.1


def test_earnings_event_risk_covers_pre_and_post_event_window() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-06", "2026-05-07", "2026-05-08", "2026-05-11"]),
            "symbol": ["AAOI"] * 4,
        }
    )
    earnings = pd.DataFrame(
        {
            "symbol": ["AAOI"],
            "event_date": [pd.Timestamp("2026-05-07")],
            "known_date": [pd.Timestamp("2026-04-16")],
            "event_risk_score": [70.0],
        }
    )

    scored = score_event_risk(align_earnings_to_dates(prices, earnings))

    by_date = scored.set_index("date")
    assert by_date.loc[pd.Timestamp("2026-05-06"), "days_to_earnings"] == 1
    assert by_date.loc[pd.Timestamp("2026-05-07"), "days_to_earnings"] == 0
    assert by_date.loc[pd.Timestamp("2026-05-08"), "days_to_earnings"] == -1
    assert by_date.loc[pd.Timestamp("2026-05-08"), "event_risk_score"] >= 70
    assert pd.isna(by_date.loc[pd.Timestamp("2026-05-11"), "days_to_earnings"])


def test_atr_stop_exits_next_open() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 5},
            {"date": "2024-01-02", "symbol": "AAA", "open": 100, "high": 101, "low": 80, "close": 80, "adj_close": 80, "volume": 1_000_000, "atr_14": 5},
            {"date": "2024-01-03", "symbol": "AAA", "open": 79, "high": 80, "low": 78, "close": 79, "adj_close": 79, "volume": 1_000_000, "atr_14": 5},
        ]
    )
    targets = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "target_weight": 1.0, "stop_loss_atr": 2.0, "trailing_stop_atr": 10.0},
        ]
    )
    config = AppConfig(
        portfolio=PortfolioConfig(initial_capital=1000, target_gross_exposure=1, target_net_exposure=1),
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0, short_borrow_fee_annual=0),
        risk=RiskConfig(atr_stop_multiple=2.0, trailing_stop_atr_multiple=10.0),
    )
    result = DailyBacktestEngine(prices, config).run(targets)
    assert list(result.trades["date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert result.trades.iloc[1]["reason_for_exit"].startswith("ATR stop loss")


def test_zero_target_closes_small_dust_position() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-02", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-03", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
        ]
    )
    targets = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "target_weight": 0.0002},
            {"date": pd.Timestamp("2024-01-02"), "symbol": "AAA", "target_weight": 0.0, "reason_for_exit": "close dust"},
        ]
    )
    config = AppConfig(
        portfolio=PortfolioConfig(initial_capital=100_000),
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0, short_borrow_fee_annual=0, min_trade_notional=1),
    )
    result = DailyBacktestEngine(prices, config).run(targets)
    assert len(result.trades) == 2
    assert result.trades.iloc[-1]["target_weight"] == 0.0


def test_dust_close_does_not_create_visible_micro_trade() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-02", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-03", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
        ]
    )
    targets = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "target_weight": 0.000005},
            {"date": pd.Timestamp("2024-01-02"), "symbol": "AAA", "target_weight": 0.0},
        ]
    )
    config = AppConfig(
        portfolio=PortfolioConfig(initial_capital=100_000),
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0, short_borrow_fee_annual=0, min_trade_notional=1),
    )
    result = DailyBacktestEngine(prices, config).run(targets)
    assert result.trades.empty
    assert result.positions.empty


def test_sub_dollar_residual_position_is_removed_at_close() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-02", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 0.50, "adj_close": 0.50, "volume": 1_000_000, "atr_14": 1},
        ]
    )
    targets = pd.DataFrame([{"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "target_weight": 0.001}])
    config = AppConfig(
        portfolio=PortfolioConfig(initial_capital=100_000),
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0, short_borrow_fee_annual=0, min_trade_notional=1),
    )
    result = DailyBacktestEngine(prices, config).run(targets)
    assert result.positions.empty


def test_capacity_and_turnover_limits_cap_trade_notional() -> None:
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
                "volume": 1000,
                "atr_14": 1,
            }
            for i in range(6)
        ]
    )
    targets = pd.DataFrame([{"date": pd.Timestamp("2024-01-05"), "symbol": "AAA", "target_weight": 1.0}])
    config = AppConfig(
        portfolio=PortfolioConfig(initial_capital=100_000),
        execution=ExecutionConfig(
            slippage_bps=0,
            commission_per_share=0,
            short_borrow_fee_annual=0,
            min_trade_notional=1,
            max_adv_participation=0.10,
            daily_turnover_cap=0.50,
        ),
    )
    result = DailyBacktestEngine(prices, config).run(targets)
    notional = result.trades.iloc[0]["quantity"] * result.trades.iloc[0]["price"]
    assert notional == 1000
    assert result.trades.iloc[0]["date"] == pd.Timestamp("2024-01-06")
    assert result.trades.iloc[0]["adv20_dollars"] == 10_000
    assert result.trades.iloc[0]["capacity_limited"]


def test_min_holding_days_blocks_ordinary_reduction() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-02", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-03", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000, "atr_14": 1},
        ]
    )
    targets = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "target_weight": 0.5},
            {"date": pd.Timestamp("2024-01-02"), "symbol": "AAA", "target_weight": 0.1},
        ]
    )
    config = AppConfig(
        portfolio=PortfolioConfig(initial_capital=100_000),
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0, short_borrow_fee_annual=0, min_trade_notional=1),
        risk=RiskConfig(min_holding_days=5),
    )
    result = DailyBacktestEngine(prices, config).run(targets)
    assert len(result.trades) == 1


def test_long_short_pnl_are_split_by_position_direction() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "LONG", "open": 10, "high": 10, "low": 10, "close": 10, "adj_close": 10, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-01", "symbol": "SHRT", "open": 10, "high": 10, "low": 10, "close": 10, "adj_close": 10, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-02", "symbol": "LONG", "open": 10, "high": 12, "low": 10, "close": 12, "adj_close": 12, "volume": 1_000_000, "atr_14": 1},
            {"date": "2024-01-02", "symbol": "SHRT", "open": 10, "high": 9, "low": 8, "close": 8, "adj_close": 8, "volume": 1_000_000, "atr_14": 1},
        ]
    )
    targets = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "symbol": "LONG", "target_weight": 0.5},
            {"date": pd.Timestamp("2024-01-01"), "symbol": "SHRT", "target_weight": -0.5},
        ]
    )
    config = AppConfig(
        portfolio=PortfolioConfig(initial_capital=1000),
        execution=ExecutionConfig(slippage_bps=0, commission_per_share=0, short_borrow_fee_annual=0),
    )
    result = DailyBacktestEngine(prices, config).run(targets)
    row = result.equity_curve[result.equity_curve["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert row["long_pnl"] == 100
    assert row["short_pnl"] == 100


def test_drawdown_risk_bucket_resets_yearly() -> None:
    engine = DailyBacktestEngine(pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]), AppConfig())
    assert engine._risk_bucket(pd.Timestamp("2025-12-31")) == "2025"
    assert engine._risk_bucket(pd.Timestamp("2026-01-02")) == "2026"
