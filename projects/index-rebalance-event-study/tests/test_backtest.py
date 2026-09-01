from __future__ import annotations

import pandas as pd
import pytest

from index_rebalance_event_study.backtest import (
    _intraday_window_return,
    _minute_to_daily_price_scale,
    aggregate_date_portfolio,
    build_daily_event_ledger,
    performance_metrics,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "announcement_date": ["2035-01-08", "2035-01-08"],
            "effective_close_date": ["2035-01-12", "2035-01-12"],
            "segment": ["primary", "primary"],
            "action": ["add", "delete"],
            "security_name": ["Northstar Synthetic Labs", "Blue Mesa Fictional Industries"],
            "symbol": ["ZZZADD", "ZZZDEL"],
        }
    )


def _daily_bars() -> pd.DataFrame:
    rows = []
    for symbol, entry, effective_open, effective_close, next_open in (
        ("ZZZADD", 100.0, 101.0, 104.0, 103.0),
        ("ZZZDEL", 50.0, 49.0, 47.0, 48.0),
    ):
        for date, open_, close in (
            ("2035-01-09", entry, entry * 1.001),
            ("2035-01-12", effective_open, effective_close),
            ("2035-01-15", next_open, next_open * 1.001),
        ):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_,
                    "high": max(open_, close) * 1.01,
                    "low": min(open_, close) * 0.99,
                    "close": close,
                    "volume": 10_000,
                    "source": "synthetic_test",
                    "finality": "confirmed",
                }
            )
    return pd.DataFrame(rows)


def test_daily_ledger_is_causal_and_directional() -> None:
    ledger, failures = build_daily_event_ledger(_events(), _daily_bars())
    assert failures.empty
    add = ledger.loc[ledger["symbol"].eq("ZZZADD")].iloc[0]
    deletion = ledger.loc[ledger["symbol"].eq("ZZZDEL")].iloc[0]
    assert add["entry_date"] == pd.Timestamp("2035-01-09")
    assert add["announcement_to_effective_return_gross"] == pytest.approx(0.04)
    assert deletion["announcement_to_effective_return_gross"] == pytest.approx(0.06)
    assert ledger["marketdata_finality"].eq("confirmed").all()


def test_missing_effective_session_fails_closed() -> None:
    bars = _daily_bars().loc[lambda frame: ~frame["symbol"].eq("ZZZDEL")]
    ledger, failures = build_daily_event_ledger(_events(), bars)
    assert len(ledger) == 1
    assert failures.loc[0, "reason"] == "missing_confirmed_effective_session"


def test_date_portfolio_balances_add_and_delete_legs() -> None:
    ledger, _ = build_daily_event_ledger(_events(), _daily_bars())
    portfolio = aggregate_date_portfolio(
        ledger,
        "announcement_to_effective_return_gross",
        cost_bps=10.0,
    )
    assert portfolio.loc[0, "gross_return"] == pytest.approx(0.05)
    assert portfolio.loc[0, "net_return"] == pytest.approx(0.049)


def test_single_leg_date_keeps_other_half_in_cash() -> None:
    trades = pd.DataFrame(
        {
            "effective_close_date": pd.to_datetime(["2035-01-12"] * 2),
            "action": ["delete", "delete"],
            "return": [0.02, 0.04],
        }
    )
    portfolio = aggregate_date_portfolio(trades, "return", cost_bps=10.0)
    assert portfolio.loc[0, "gross_exposure"] == 0.5
    assert portfolio.loc[0, "net_return"] == pytest.approx(0.0145)


def test_price_scale_reconciles_large_split_basis() -> None:
    minute = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2035-01-12 09:30", "2035-01-12 15:59"]),
            "open": [106.0, 114.0],
            "high": [106.0, 116.0],
            "low": [105.0, 113.0],
            "close": [105.5, 113.5],
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2035-01-12"]),
            "open": [53.0],
            "high": [58.0],
            "low": [52.5],
            "close": [56.75],
        }
    )
    assert _minute_to_daily_price_scale(minute, daily, pd.Timestamp("2035-01-12")) == 0.5


def test_intraday_window_decomposes_without_overlap() -> None:
    minute = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2035-01-12 15:49:00", "2035-01-12 15:50:00", "2035-01-12 15:59:00"]
            ),
            "open": [100.0, 102.0, 104.0],
            "high": [102.0, 104.0, 106.0],
            "low": [99.0, 101.0, 103.0],
            "close": [101.0, 103.0, 105.0],
            "volume": [1_000, 2_000, 7_000],
        }
    )
    result = _intraday_window_return(minute, pd.Timestamp("2035-01-12"))
    assert result is not None
    assert result["underlying_1549_to_1550_return"] == pytest.approx(0.02)
    assert result["underlying_1550_to_last_minute_return"] == pytest.approx(105 / 102 - 1)


def test_performance_metrics_use_dates_as_observations() -> None:
    portfolio = pd.DataFrame(
        {
            "effective_close_date": pd.to_datetime(["2035-01-12", "2035-04-13"]),
            "gross_return": [0.02, -0.01],
            "net_return": [0.019, -0.011],
            "gross_exposure": [1.0, 1.0],
        }
    )
    metrics = performance_metrics(portfolio)
    assert metrics["event_dates"] == 2
    assert metrics["break_even_cost_bps"] == pytest.approx(50.0)
