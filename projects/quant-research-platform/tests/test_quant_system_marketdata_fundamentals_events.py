from __future__ import annotations

import pandas as pd

from quant_system.data.marketdata_earnings import MarketDataEarningsClient, earnings_payload_to_frame
from quant_system.config import AppConfig, EventsConfig
from quant_system.events.provider import MarketDataEventDataProvider, normalize_marketdata_events
from quant_system.fundamentals.provider import MarketDataFundamentalDataProvider, normalize_marketdata_fundamentals
from quant_system.research import load_events


def test_marketdata_earnings_payload_normalizes_column_arrays() -> None:
    payload = {
        "s": "ok",
        "symbol": ["AAA", "AAA"],
        "fiscalYear": [2024, 2024],
        "fiscalQuarter": [1, 2],
        "date": [1711843200, 1719705600],
        "reportDate": [1714521600, 1722384000],
        "reportTime": ["after market close", "before market open"],
        "reportedEPS": [1.0, 1.2],
        "estimatedEPS": [0.9, 1.0],
        "surpriseEPS": [0.1, 0.2],
        "surpriseEPSpct": [11.1, 20.0],
        "updated": [1714521600, 1722384000],
    }
    frame = earnings_payload_to_frame("AAA", payload)
    assert list(frame["symbol"]) == ["AAA", "AAA"]
    assert list(frame["reported_eps"]) == [1.0, 1.2]
    assert frame["report_date"].iloc[0] == pd.Timestamp("2024-04-30")


def test_marketdata_fundamental_provider_maps_earnings_to_factors(monkeypatch, tmp_path) -> None:
    earnings = pd.DataFrame(
        {
            "symbol": ["AAA"] * 5,
            "report_date": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"]),
            "reported_eps": [1.0, 1.1, 1.2, 1.3, 1.5],
            "estimated_eps": [0.9, 1.0, 1.1, 1.2, 1.4],
            "surprise_eps_pct": [10, 10, 10, 10, 7],
        }
    )
    provider = MarketDataFundamentalDataProvider(cache_dir=tmp_path)
    monkeypatch.setattr(provider.earnings_client, "get_earnings", lambda symbol, start, end: earnings)
    out = provider.get_fundamentals("AAA", "2023-01-01", None)
    assert "eps_yoy_growth" in out.columns
    assert out["eps_yoy_growth"].iloc[-1] == 0.5
    assert out["guidance_up_or_down"].iloc[-1] == 7


def test_marketdata_event_provider_uses_report_date_as_known_date(monkeypatch, tmp_path) -> None:
    earnings = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2024-05-01"]),
            "report_time": ["after market close"],
            "reported_eps": [1.0],
            "estimated_eps": [0.9],
            "surprise_eps": [0.1],
            "surprise_eps_pct": [11.1],
        }
    )
    provider = MarketDataEventDataProvider(cache_dir=tmp_path)
    monkeypatch.setattr(provider.earnings_client, "get_earnings", lambda symbol, start, end: earnings)
    out = provider.get_earnings_calendar(["AAA"], "2024-01-01", None)
    assert out["event_date"].iloc[0] == pd.Timestamp("2024-05-01")
    assert out["known_date"].iloc[0] == pd.Timestamp("2024-05-01")
    assert out["event_type"].iloc[0] == "earnings"


def test_marketdata_earnings_client_uses_fast_failure_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MARKETDATA_TOKEN", "test-token")
    monkeypatch.delenv("MARKETDATA_EARNINGS_RETRIES", raising=False)
    monkeypatch.delenv("MARKETDATA_EARNINGS_TIMEOUT", raising=False)
    client = MarketDataEarningsClient(cache_dir=tmp_path)
    assert client.retries == 1
    assert client.timeout_seconds == 8


def test_marketdata_or_csv_events_overlay_csv_rows(tmp_path) -> None:
    events_path = tmp_path / "events.csv"
    events_path.write_text(
        "symbol,event_date,known_date,event_type,event_risk_score,expected_move\n"
        "AAA,2026-05-07,2026-04-16,earnings,50,0.15\n",
        encoding="utf-8",
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-06", "2026-05-07", "2026-05-08"]),
            "symbol": ["AAA"] * 3,
        }
    )
    marketdata = pd.DataFrame(
        {
            "symbol": ["BBB"],
            "report_date": [pd.Timestamp("2026-05-07")],
            "reported_eps": [1.0],
            "estimated_eps": [1.0],
            "surprise_eps": [0.0],
            "surprise_eps_pct": [0.0],
        }
    )
    config = AppConfig(
        start_date="2026-01-01",
        events_path=str(events_path),
        events=EventsConfig(enabled=True, provider="marketdata_or_csv"),
    )

    out = load_events(config, ["AAA", "BBB"], prices, [], marketdata_earnings=marketdata)
    aaa = out[out["symbol"] == "AAA"].set_index("date")

    assert aaa.loc[pd.Timestamp("2026-05-07"), "event_risk_score"] >= 70
    assert aaa.loc[pd.Timestamp("2026-05-08"), "days_to_earnings"] == -1


def test_marketdata_event_provider_major_events_empty_until_endpoint_exists(tmp_path) -> None:
    provider = MarketDataEventDataProvider(cache_dir=tmp_path)
    out = provider.get_major_events(["AAA"], "2024-01-01", None)
    assert out.empty


def _manifested(frame: pd.DataFrame, cache_key: str) -> pd.DataFrame:
    frame.attrs["cache_manifest"] = {
        "cache_key": cache_key,
        "request_sha256": f"request-{cache_key}",
        "csv_sha256": f"csv-{cache_key}",
    }
    return frame


def test_runtime_normalizers_preserve_consumed_cache_manifests() -> None:
    events = _manifested(
        pd.DataFrame({"eventDate": ["2026-01-02"], "eventType": ["earnings"]}),
        "events-AAA",
    )
    fundamentals = _manifested(
        pd.DataFrame({"knownDate": ["2026-01-02"], "epsYoYGrowth": [0.2]}),
        "fundamentals-AAA",
    )

    normalized_events = normalize_marketdata_events("AAA", events)
    normalized_fundamentals = normalize_marketdata_fundamentals("AAA", fundamentals)

    assert normalized_events.attrs["reference_cache_manifests"][0]["cache_key"] == "events-AAA"
    assert normalized_fundamentals.attrs["reference_cache_manifests"][0]["cache_key"] == "fundamentals-AAA"


def test_empty_earnings_responses_remain_in_bulk_provenance(monkeypatch, tmp_path) -> None:
    client = MarketDataEarningsClient(cache_dir=tmp_path)
    empty = _manifested(pd.DataFrame(), "earnings-no-data-AAA")
    monkeypatch.setattr(client, "get_earnings", lambda *_args: empty)

    out = client.get_bulk_earnings(["AAA"], "2026-01-01", "2026-01-31")

    assert out.empty
    assert out.attrs["reference_cache_manifests"][0]["cache_key"] == "earnings-no-data-AAA"


def test_event_provider_keeps_empty_runtime_snapshot_with_nonempty_earnings(monkeypatch, tmp_path) -> None:
    provider = MarketDataEventDataProvider(cache_dir=tmp_path)
    empty_runtime = _manifested(pd.DataFrame(), "events-no-data-AAA")
    earnings = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2026-01-15"]),
            "surprise_eps_pct": [0.0],
        }
    )
    monkeypatch.setattr(provider.runtime_client, "get_events", lambda *_args: empty_runtime)
    monkeypatch.setattr(provider.earnings_client, "get_bulk_earnings", lambda *_args: earnings)

    out = provider.get_earnings_calendar(["AAA"], "2026-01-01", "2026-01-31")

    assert not out.empty
    assert out.attrs["reference_cache_manifests"][0]["cache_key"] == "events-no-data-AAA"
