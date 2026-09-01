from __future__ import annotations

import pandas as pd

from quant_system.config import UniverseConfig
from quant_system.data.quality import reconcile_daily_ohlcv_with_intraday
from quant_system.universe.filters import apply_price_quality_adjustments, build_asof_tradable_universe, build_tradable_universe


def test_stale_price_symbol_is_not_tradable() -> None:
    prices = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "symbol": "STALE",
                "open": 10,
                "high": 10,
                "low": 10,
                "close": 10,
                "adj_close": 10,
                "volume": 1_000_000,
            }
            for i in range(40)
        ]
    )
    out = build_tradable_universe(
        prices,
        UniverseConfig(
            min_history_days=1,
            min_avg_dollar_volume_20d=1,
            min_avg_volume_20d=1,
            max_stale_price_days=10,
        ),
    )
    row = out.iloc[0]
    assert not row["tradable"]
    assert "stale_price" in row["filter_reason"]


def test_unhandled_corporate_action_jump_is_not_tradable() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "JUMP", "open": 10, "high": 10, "low": 10, "close": 10, "adj_close": 10, "volume": 1_000_000},
            {"date": "2024-01-02", "symbol": "JUMP", "open": 100, "high": 100, "low": 100, "close": 100, "adj_close": 100, "volume": 1_000_000},
        ]
    )
    out = build_tradable_universe(
        prices,
        UniverseConfig(
            min_history_days=1,
            min_avg_dollar_volume_20d=1,
            min_avg_volume_20d=1,
            max_unhandled_corporate_action_return=0.8,
        ),
    )
    row = out.iloc[0]
    assert not row["tradable"]
    assert "possible_unhandled_corporate_action" in row["filter_reason"]


def test_corporate_action_history_is_quarantined_not_dropped_when_clean_segment_is_long_enough() -> None:
    rows = [
        {"date": "2024-01-01", "symbol": "MOMO", "open": 10, "high": 10, "low": 10, "close": 10, "adj_close": 10, "volume": 1_000_000},
        {"date": "2024-01-02", "symbol": "MOMO", "open": 25, "high": 25, "low": 25, "close": 25, "adj_close": 25, "volume": 1_000_000},
    ]
    rows.extend(
        {
            "date": pd.Timestamp("2024-01-03") + pd.Timedelta(days=i),
            "symbol": "MOMO",
            "open": 25 + i * 0.1,
            "high": 25 + i * 0.1,
            "low": 25 + i * 0.1,
            "close": 25 + i * 0.1,
            "adj_close": 25 + i * 0.1,
            "volume": 1_000_000,
        }
        for i in range(70)
    )
    config = UniverseConfig(
        min_history_days=252,
        min_quality_adjusted_history_days=60,
        min_avg_dollar_volume_20d=1,
        min_avg_volume_20d=1,
    )
    adjusted = apply_price_quality_adjustments(pd.DataFrame(rows), config)
    out = build_tradable_universe(adjusted, config)
    row = out.iloc[0]
    assert row["tradable"]
    assert row["limited_history_flag"]
    assert row["quality_adjusted"]
    assert row["filter_reason"] == "quality_adjusted_history"


def test_asof_universe_uses_prior_day_liquidity_not_current_day_spike() -> None:
    rows = []
    for i in range(8):
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "symbol": "SPIKE",
                "open": 10,
                "high": 10,
                "low": 10,
                "close": 10,
                "adj_close": 10,
                "volume": 10_000_000 if i == 5 else 1_000_000,
            }
        )
    panel = build_asof_tradable_universe(
        pd.DataFrame(rows),
        UniverseConfig(min_history_days=1, min_avg_dollar_volume_20d=1, min_avg_volume_20d=1),
    )
    spike_day = panel.loc[panel["date"].eq(pd.Timestamp("2024-01-06"))].iloc[0]
    assert spike_day["avg_volume_20d"] == 1_000_000
    assert spike_day["avg_dollar_volume_20d"] == 10_000_000


def test_intraday_reconciliation_repairs_stale_daily_close() -> None:
    daily = pd.DataFrame(
        [
            {"date": "2026-05-12", "symbol": "MU", "open": 764.5, "high": 782.7, "low": 706.6, "close": 747.0, "adj_close": 747.0, "volume": 1_000},
            {"date": "2026-05-12", "symbol": "NVDA", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "adj_close": 101.0, "volume": 1_000},
        ]
    )
    intraday = pd.DataFrame(
        [
            {"date": "2026-05-12", "datetime": "2026-05-12 19:50:00", "symbol": "MU", "open": 761, "high": 767, "low": 758, "close": 764.2, "volume": 100},
            {"date": "2026-05-12", "datetime": "2026-05-12 19:55:00", "symbol": "MU", "open": 764, "high": 767, "low": 763, "close": 765.8, "volume": 100},
            {"date": "2026-05-12", "datetime": "2026-05-12 19:55:00", "symbol": "NVDA", "open": 101, "high": 102, "low": 100, "close": 101.2, "volume": 100},
        ]
    )

    repaired, report = reconcile_daily_ohlcv_with_intraday(daily, intraday, close_mismatch_threshold=0.01)

    mu = repaired[repaired["symbol"].eq("MU")].iloc[0]
    nvda = repaired[repaired["symbol"].eq("NVDA")].iloc[0]
    assert mu["close"] == 765.8
    assert mu["adj_close"] == 765.8
    assert mu["source"] == "intraday-derived"
    assert mu["finality"] == "provisional"
    assert nvda["close"] == 101.0
    assert report["symbol"].tolist() == ["MU"]
