from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from quant_system import cli


def _row(symbol: str, date: str, close: float = 10.0) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": date,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "adj_close": close,
        "volume": 1000,
        "source": "marketdata.app",
        "finality": "confirmed",
    }


def test_incremental_market_update_starts_after_each_symbols_latest_date(tmp_path, monkeypatch):
    report = tmp_path / "report"
    calls: list[tuple[str, str, str]] = []

    class FakeStore:
        root = tmp_path / "external-store"

        def read_bars(self, **kwargs):
            assert kwargs["finality"] == "confirmed"
            return pd.DataFrame([_row("AAA", "2024-01-02")]).drop(columns="adj_close")

    class FakeProvider:
        def __init__(self):
            self.store = FakeStore()

        def get_ohlcv(self, symbol, start, end, timeframe):
            calls.append((symbol, start, end))
            return pd.DataFrame([_row(symbol, end, 11.0)])

    monkeypatch.setattr(cli, "MarketDataAPIProvider", FakeProvider)
    config = SimpleNamespace(
        data_provider="marketdata",
        data_path=str(tmp_path / "unused"),
        start_date="2024-01-01",
        timeframe="1d",
        universe=SimpleNamespace(custom_symbols=["AAA", "BBB"], symbols=[]),
    )

    cli.cmd_update_market_data(config, through="2024-01-05", jobs=2, report_out=str(report))

    assert set(calls) == {("AAA", "2024-01-03", "2024-01-05"), ("BBB", "2024-01-01", "2024-01-05")}
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    assert summary["rows_fetched"] == 2
    assert summary["latest_market_date"] == "2024-01-05"


def test_latest_completed_us_equity_date_excludes_in_progress_session():
    before_close = pd.Timestamp("2026-08-21 11:06:00", tz="America/New_York")
    after_close_buffer = pd.Timestamp("2026-08-21 16:16:00", tz="America/New_York")

    assert cli._latest_completed_us_equity_date(before_close) == pd.Timestamp("2026-08-20")
    assert cli._latest_completed_us_equity_date(after_close_buffer) == pd.Timestamp("2026-08-21")


def test_merge_downloaded_ohlcv_removes_incomplete_session(tmp_path):
    prices = tmp_path / "prices.csv"
    pd.DataFrame(
        [
            _row("AAA", "2026-08-20", 10.0),
            _row("AAA", "2026-08-21", 11.0),
        ]
    ).to_csv(prices, index=False)

    merged = cli._merge_downloaded_ohlcv(
        prices,
        pd.DataFrame(),
        maximum_date=pd.Timestamp("2026-08-20"),
    )

    assert merged["date"].tolist() == ["2026-08-20"]
