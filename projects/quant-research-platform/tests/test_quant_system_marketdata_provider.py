from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest
from quant_marketdata import MarketDataStore

from quant_system.config import AppConfig
from quant_system.data.intraday_provider import MarketDataIntradayProvider
from quant_system.data.marketdata_provider import MarketDataAPIProvider
from quant_system.data.marketdata_store_provider import MarketDataStoreProvider
from quant_system.research import provider_for


@dataclass
class DummyResponse:
    payload: dict
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self.payload


@dataclass
class DummySession:
    get_func: object

    def get(self, url, **kwargs):
        return self.get_func(url, **kwargs)


def test_marketdata_provider_fetches_candles_with_bearer_header(monkeypatch, tmp_path) -> None:
    payload = {
        "s": "ok",
        "t": [1777608000, 1777694400],
        "o": [278.86, 280.0],
        "h": [287.22, 282.0],
        "l": [278.37, 279.0],
        "c": [280.14, 281.0],
        "v": [79915442, 70000000],
    }
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append((url, params, headers, timeout))
        assert url.startswith("https://api.marketdata.app/v1/")
        assert url.rstrip("/").rsplit("/", 1)[-1] == "AAPL"
        assert "/D/" in url
        assert params["from"] == "2026-05-01"
        assert params["to"] == "2026-05-05"
        assert params["adjustsplits"] == "true"
        assert headers["Accept"] == "application/json"
        assert headers["Authorization"] == "Bearer TOKEN"
        assert timeout == 30.0
        return DummyResponse(payload)

    monkeypatch.setenv("MARKETDATA_TOKEN", "TOKEN")
    provider = MarketDataAPIProvider(cache_dir=tmp_path, session=DummySession(fake_get))
    frame = provider.get_ohlcv("AAPL", "2026-05-01", "2026-05-05")
    assert len(frame) == 2
    assert frame["symbol"].tolist() == ["AAPL", "AAPL"]
    assert frame["close"].tolist() == [280.14, 281.0]
    assert frame["adj_close"].tolist() == [280.14, 281.0]
    assert frame["source"].eq("marketdata.app").all()
    assert frame["finality"].eq("confirmed").all()
    assert str(frame.loc[0, "date"].date()) == "2026-05-01"
    assert len(calls) == 1


def test_marketdata_provider_reads_marketdata_token_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MARKETDATA_TOKEN", "ENV_TOKEN")

    def fake_get(url, params, headers, timeout):
        assert headers["Authorization"] == "Bearer ENV_TOKEN"
        return DummyResponse({"s": "no_data"})

    provider = MarketDataAPIProvider(cache_dir=tmp_path, session=DummySession(fake_get))
    frame = provider.get_ohlcv("AAPL", "2026-05-01", "2026-05-05")
    assert frame.empty


def test_marketdata_provider_uses_cache(monkeypatch, tmp_path) -> None:
    count = {"requests": 0}

    def fake_get(url, params, headers, timeout):
        count["requests"] += 1
        return DummyResponse(
            {
                "s": "ok",
                "t": [1777608000],
                "o": [100],
                "h": [101],
                "l": [99],
                "c": [100.5],
                "v": [1000],
            }
        )

    monkeypatch.setenv("MARKETDATA_TOKEN", "TOKEN")
    provider = MarketDataAPIProvider(cache_dir=tmp_path, session=DummySession(fake_get))
    first = provider.get_ohlcv("MSFT", "2026-05-01", "2026-05-05")
    second = provider.get_ohlcv("MSFT", "2026-05-01", "2026-05-05")
    assert count["requests"] == 1
    pd.testing.assert_frame_equal(first, second)


def test_marketdata_provider_reuses_completed_session_cache_for_open_ended_requests(monkeypatch, tmp_path) -> None:
    count = {"requests": 0}

    def fake_get(url, params, headers, timeout):
        count["requests"] += 1
        return DummyResponse(
            {
                    "s": "ok",
                    "t": [1777608000 + count["requests"] * 86400],
                    "o": [100],
                    "h": [105],
                "l": [99],
                "c": [100.0 + count["requests"]],
                "v": [1000],
            }
        )

    monkeypatch.setenv("MARKETDATA_TOKEN", "TOKEN")
    provider = MarketDataAPIProvider(cache_dir=tmp_path, session=DummySession(fake_get))
    first = provider.get_ohlcv("MSFT", "2026-05-01", None)
    second = provider.get_ohlcv("MSFT", "2026-05-01", None)
    assert count["requests"] == 1
    pd.testing.assert_frame_equal(first, second)


def test_marketdata_provider_raises_clear_api_error(tmp_path) -> None:
    provider = MarketDataAPIProvider(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="MarketData.app error"):
        provider._payload_to_ohlcv("BAD", {"s": "error", "errmsg": "bad symbol"})


def test_marketdata_provider_requires_token(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MARKETDATA_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    provider = MarketDataAPIProvider(cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="MARKETDATA_TOKEN"):
        provider.get_ohlcv("AAPL", "2026-05-01", "2026-05-05")


def test_marketdata_provider_rejects_constructor_credentials(tmp_path) -> None:
    with pytest.raises(ValueError, match="MARKETDATA_TOKEN"):
        MarketDataAPIProvider(api_key="TOKEN", cache_dir=tmp_path)


def test_intraday_provider_uses_shared_provisional_store(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append((url, params))
        assert url.rstrip("/").endswith("AAPL")
        assert "/5/" in url
        assert params["adjustdividends"] == "true"
        return DummyResponse(
            {
                "s": "ok",
                "t": [1704292200, 1704292500],
                "o": [100.0, 100.5],
                "h": [101.0, 101.2],
                "l": [99.8, 100.2],
                "c": [100.5, 101.0],
                "v": [1000, 1200],
            }
        )

    monkeypatch.setenv("MARKETDATA_TOKEN", "TOKEN")
    provider = MarketDataIntradayProvider(cache_dir=tmp_path, session=DummySession(fake_get))
    frame = provider.get_intraday("AAPL", "2024-01-03", "2024-01-03", "5min")

    assert len(calls) == 1
    assert frame["source"].eq("marketdata.app").all()
    assert frame["finality"].eq("provisional").all()
    stored = provider.read_stored_intraday(["AAPL"], "2024-01-03", "2024-01-03")
    assert len(stored) == 2
    assert stored["datetime"].nunique() == 2


def test_intraday_provider_rejects_constructor_credentials(tmp_path) -> None:
    with pytest.raises(ValueError, match="MARKETDATA_TOKEN"):
        MarketDataIntradayProvider(api_key="TOKEN", cache_dir=tmp_path)


def test_research_provider_reads_confirmed_store_without_network(monkeypatch, tmp_path) -> None:
    store = MarketDataStore(tmp_path)
    store.write_bars(
        pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "symbol": ["AAPL"],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000],
                "source": ["marketdata.app"],
                "finality": ["confirmed"],
            }
        ),
        finality="confirmed",
        resolution="D",
    )
    monkeypatch.delenv("MARKETDATA_TOKEN", raising=False)
    provider = MarketDataStoreProvider(store)

    frame = provider.get_bulk_ohlcv(["AAPL"], "2024-01-01", "2024-01-31")

    assert frame.loc[0, "adj_close"] == 101.0
    assert frame.loc[0, "finality"] == "confirmed"


def test_marketdata_research_selector_uses_store_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QUANT_DATA_HOME", str(tmp_path))
    selected = provider_for(AppConfig(data_provider="marketdata"))
    assert isinstance(selected, MarketDataStoreProvider)
