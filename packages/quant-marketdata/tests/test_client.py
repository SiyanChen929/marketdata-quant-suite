from __future__ import annotations

from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quant_marketdata import (
    CANONICAL_COLUMNS,
    CredentialError,
    DataContractError,
    DataUnavailableError,
    MarketDataClient,
    MarketDataStore,
    OPTION_CHAIN_COLUMNS,
)

from conftest import FakeResponse, FakeSession, candle_payload


AFTER_CLOSE = lambda: datetime(2024, 1, 3, 16, 30, tzinfo=ZoneInfo("America/New_York"))


def test_daily_fetch_uses_explicit_adjustments_and_safe_exact_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETDATA_TOKEN", "test-token-never-persist")
    store = MarketDataStore(tmp_path)
    session = FakeSession([FakeResponse(candle_payload("2024-01-02", "2024-01-03"))])
    client = MarketDataClient(store=store, session=session, now=AFTER_CLOSE)

    bars = client.get_daily_bars("aapl", "2024-01-02", "2024-01-03")

    assert tuple(bars.columns) == CANONICAL_COLUMNS
    assert bars["symbol"].tolist() == ["AAPL", "AAPL"]
    assert bars["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert bars["finality"].eq("confirmed").all()
    call = session.calls[0]
    assert call["url"].endswith("/stocks/candles/D/AAPL/")
    assert call["params"]["adjustsplits"] == "true"
    assert call["params"]["adjustdividends"] == "true"
    assert call["headers"]["Authorization"] == "Bearer test-token-never-persist"

    cache_manifests = list(
        (tmp_path / "raw" / "marketdata" / "confirmed" / "request-cache").rglob("*.json")
    )
    assert len(cache_manifests) == 1
    manifest_text = cache_manifests[0].read_text(encoding="utf-8")
    assert "test-token-never-persist" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["request"]["adjustsplits"] == "true"
    assert manifest["request"]["adjustdividends"] == "true"

    monkeypatch.delenv("MARKETDATA_TOKEN")
    cached = client.get_daily_bars("AAPL", "2024-01-02", "2024-01-03")
    pd.testing.assert_frame_equal(bars, cached)
    assert len(session.calls) == 1


def test_cache_miss_requires_environment_token(tmp_path, monkeypatch):
    monkeypatch.delenv("MARKETDATA_TOKEN", raising=False)
    client = MarketDataClient(
        store=MarketDataStore(tmp_path),
        session=FakeSession([]),
        now=AFTER_CLOSE,
    )
    with pytest.raises(CredentialError, match="MARKETDATA_TOKEN"):
        client.get_daily_bars("AAPL", "2024-01-02", "2024-01-03")


def test_confirmed_today_rejected_before_conservative_cutoff(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETDATA_TOKEN", "offline-test-token")
    before_close = lambda: datetime(
        2024, 1, 3, 15, 59, tzinfo=ZoneInfo("America/New_York")
    )
    session = FakeSession([FakeResponse(candle_payload("2024-01-03"))])
    client = MarketDataClient(
        store=MarketDataStore(tmp_path),
        session=session,
        now=before_close,
    )

    with pytest.raises(DataUnavailableError, match="16:15 ET"):
        client.get_daily_bars("AAPL", "2024-01-03", "2024-01-03")
    assert not (tmp_path / "lake" / "confirmed").exists()


def test_provisional_today_uses_separate_staging_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETDATA_TOKEN", "offline-test-token")
    before_close = lambda: datetime(
        2024, 1, 3, 15, 59, tzinfo=ZoneInfo("America/New_York")
    )
    client = MarketDataClient(
        store=MarketDataStore(tmp_path),
        session=FakeSession([FakeResponse(candle_payload("2024-01-03"))]),
        now=before_close,
    )

    bars = client.get_daily_bars(
        "AAPL",
        "2024-01-03",
        "2024-01-03",
        finality="provisional",
    )
    assert bars["finality"].eq("provisional").all()
    assert (
        tmp_path
        / "staging"
        / "provisional"
        / "bars"
        / "resolution=D"
        / "year=2024"
        / "bars.parquet"
    ).is_file()
    assert not (tmp_path / "lake" / "confirmed").exists()


def test_generic_stock_bars_preserve_intraday_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETDATA_TOKEN", "offline-test-token")
    timestamps = [
        int(pd.Timestamp("2024-01-03 15:35:00Z").timestamp()),
        int(pd.Timestamp("2024-01-03 15:40:00Z").timestamp()),
    ]
    payload = {
        "s": "ok",
        "t": timestamps,
        "o": [100.0, 100.5],
        "h": [101.0, 101.5],
        "l": [99.0, 100.0],
        "c": [100.5, 101.0],
        "v": [1000, 1200],
    }
    session = FakeSession([FakeResponse(payload)])
    client = MarketDataClient(
        store=MarketDataStore(tmp_path),
        session=session,
        now=AFTER_CLOSE,
    )

    bars = client.get_stock_bars(
        "AAPL",
        "2024-01-03",
        "2024-01-03",
        resolution="5min",
        finality="provisional",
    )
    assert bars["date"].tolist() == [
        pd.Timestamp("2024-01-03 10:35:00"),
        pd.Timestamp("2024-01-03 10:40:00"),
    ]
    restored = client.store.read_bars(
        "AAPL",
        "2024-01-03",
        "2024-01-03",
        finality="provisional",
        resolution="5",
    )
    pd.testing.assert_frame_equal(bars, restored)
    assert session.calls[0]["url"].endswith("/stocks/candles/5/AAPL/")


def test_bulk_daily_deduplicates_symbols_and_returns_canonical_order(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETDATA_TOKEN", "offline-test-token")
    session = FakeSession(
        [
            FakeResponse(candle_payload("2024-01-02")),
            FakeResponse(candle_payload("2024-01-02")),
        ]
    )
    client = MarketDataClient(
        store=MarketDataStore(tmp_path),
        session=session,
        now=AFTER_CLOSE,
    )

    bars = client.get_bulk_daily_bars(
        ["msft", "AAPL", "MSFT"],
        "2024-01-02",
        "2024-01-02",
    )
    assert tuple(bars.columns) == CANONICAL_COLUMNS
    assert bars["symbol"].tolist() == ["AAPL", "MSFT"]
    assert len(session.calls) == 2


def test_no_data_response_is_cached_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETDATA_TOKEN", "offline-test-token")
    session = FakeSession([FakeResponse({"s": "no_data"})])
    client = MarketDataClient(
        store=MarketDataStore(tmp_path),
        session=session,
        now=AFTER_CLOSE,
    )
    first = client.get_daily_bars("NONE", "2024-01-02", "2024-01-03")
    monkeypatch.delenv("MARKETDATA_TOKEN")
    second = client.get_daily_bars("NONE", "2024-01-02", "2024-01-03")
    assert first.empty and second.empty
    assert len(session.calls) == 1


def test_option_chain_uses_shared_transport_provenance_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETDATA_TOKEN", "option-token-never-persist")
    expiration = int(pd.Timestamp("2024-02-16 05:00:00Z").timestamp())
    updated = int(pd.Timestamp("2024-01-03 15:45:00Z").timestamp())
    payload = {
        "s": "ok",
        "optionSymbol": ["AAPL240216C00150000"],
        "underlying": ["AAPL"],
        "expiration": [expiration],
        "side": ["call"],
        "strike": [150.0],
        "dte": [44],
        "bid": [9.8],
        "ask": [10.2],
        "mid": [10.0],
        "last": [10.1],
        "volume": [120],
        "openInterest": [2500],
        "iv": [0.25],
        "delta": [0.6],
        "gamma": [0.03],
        "theta": [-0.02],
        "vega": [0.12],
        "updated": [updated],
    }
    session = FakeSession([FakeResponse(payload)])
    client = MarketDataClient(
        store=MarketDataStore(tmp_path),
        session=session,
        now=AFTER_CLOSE,
    )

    chain = client.get_option_chain(
        "aapl",
        date="2024-01-03",
        side="call",
        min_open_interest=100,
    )
    assert tuple(chain.columns) == OPTION_CHAIN_COLUMNS
    assert chain.loc[0, "source"] == "marketdata.app"
    assert chain.loc[0, "bid_ask_spread"] == pytest.approx(0.4)
    call = session.calls[0]
    assert call["url"].endswith("/options/chain/AAPL/")
    assert call["params"]["side"] == "call"
    assert call["params"]["minOpenInterest"] == 100

    manifests = list(
        (tmp_path / "raw" / "marketdata" / "options" / "request-cache").rglob("*.json")
    )
    assert len(manifests) == 1
    assert "option-token-never-persist" not in manifests[0].read_text(encoding="utf-8")
    monkeypatch.delenv("MARKETDATA_TOKEN")
    cached = client.get_option_chain(
        "AAPL",
        date="2024-01-03",
        side="call",
        min_open_interest=100,
    )
    assert cached.loc[0, "option_symbol"] == "AAPL240216C00150000"
    assert len(session.calls) == 1


def test_option_chain_rejects_unknown_filters_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKETDATA_TOKEN", "offline-test-token")
    client = MarketDataClient(
        store=MarketDataStore(tmp_path),
        session=FakeSession([]),
        now=AFTER_CLOSE,
    )
    with pytest.raises(DataContractError, match="unsupported option-chain filters"):
        client.get_option_chain("AAPL", unsupported=True)
