from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from quant_marketdata import MarketDataStore

from quant_system.backtest.costs import commission, daily_short_borrow_cost
from quant_system.backtest.metrics import equity_metrics, max_drawdown
from quant_system.backtest.reports import build_data_provenance
from quant_system.config import AppConfig


def test_max_drawdown() -> None:
    equity = pd.Series([100, 120, 90, 110])
    assert round(max_drawdown(equity), 4) == -0.25


def test_equity_metrics_include_drawdown_and_turnover() -> None:
    curve = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "equity": [100, 110, 105, 120],
            "gross_exposure": [0, 1, 1, 1],
            "net_exposure": [0, 0.5, 0.5, 0.5],
            "turnover": [0, 1, 0.1, 0.1],
        }
    )
    metrics = equity_metrics(curve)
    assert round(metrics["total_return"], 6) == 0.2
    assert round(metrics["trailing_one_year_return"], 6) == 0.2
    assert metrics["max_drawdown"] < 0
    assert metrics["average_turnover"] > 0


def test_equity_metrics_trailing_one_year_return() -> None:
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2025-01-01", "2025-06-01", "2026-01-01"]),
            "equity": [100, 150, 180, 300],
        }
    )
    metrics = equity_metrics(curve)
    assert round(metrics["trailing_one_year_return"], 6) == 1.0


def test_cost_models() -> None:
    assert commission(100, 0.005) == 0.5
    assert round(daily_short_borrow_cost(10_000, 0.025), 4) == round(10_000 * 0.025 / 252, 4)


def test_marketdata_provenance_snapshots_confirmed_manifest(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QUANT_DATA_HOME", str(tmp_path))
    bars = pd.DataFrame(
        {
            "date": ["2024-01-03"],
            "symbol": ["AAA"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.5],
            "volume": [1000],
            "source": ["marketdata.app"],
            "finality": ["confirmed"],
        }
    )
    MarketDataStore().write_bars(bars, finality="confirmed", resolution="D")

    provenance = build_data_provenance(
        AppConfig(data_provider="marketdata"),
        SimpleNamespace(prices=bars),
    )

    assert provenance["sources"] == ["marketdata.app"]
    assert provenance["finalities"] == ["confirmed"]
    assert len(provenance["data_manifest_sha256"]) == 64
    assert provenance["manifest_snapshot"]["resolutions"] == ["D"]


def test_marketdata_provenance_includes_exact_reference_cache_snapshots() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2024-01-03"],
            "symbol": ["AAA"],
            "close": [10.5],
            "source": ["marketdata.app"],
            "finality": ["confirmed"],
        }
    )
    snapshot = {
        "schema_version": 1,
        "cache_key": "marketdata_earnings_AAA_2024-01-01_2024-12-31",
        "request": {"provider": "marketdata.app", "dataset": "stock_earnings", "symbol": "AAA"},
        "request_sha256": "a" * 64,
        "fetched_at": "2026-09-01T12:00:00+00:00",
        "rows": 4,
        "columns": ["symbol", "reported_eps"],
        "csv_sha256": "b" * 64,
    }
    bars.attrs["reference_data_snapshots"] = [snapshot]

    provenance = build_data_provenance(AppConfig(data_provider="csv"), SimpleNamespace(prices=bars))

    assert provenance["reference_data_snapshots"] == [snapshot]
    assert len(provenance["reference_data_manifest_sha256"]) == 64
