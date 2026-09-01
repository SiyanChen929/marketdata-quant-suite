from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import requests

from equity_pairs import data


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", content=b""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = content

    def json(self):
        return self._payload


class SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingMarketDataClient:
    def __init__(self, bars):
        self.bars = bars
        self.calls = []

    def get_bulk_daily_bars(self, symbols, start, end, finality="confirmed", refresh=False):
        self.calls.append(
            {
                "symbols": symbols,
                "start": start,
                "end": end,
                "finality": finality,
                "refresh": refresh,
            }
        )
        return self.bars.copy()


def _bars(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
            "finality",
        ],
    )


def test_normalize_market_symbols_is_ordered_and_vendor_neutral():
    assert data.normalize_market_symbol(" brk.b ") == "BRK.B"
    assert data.normalize_market_symbols(["brk.b", "AAPL", "BRK.B"]) == (
        "BRK.B",
        "AAPL",
    )
    with pytest.raises(ValueError):
        data.normalize_market_symbol("bad symbol")
    with pytest.raises(ValueError):
        data.normalize_market_symbol(pd.NA)


def test_fetch_sp500_constituents_normalizes_schema(monkeypatch):
    source = pd.DataFrame(
        {
            "Symbol": ["BF.B", "AAPL"],
            "Security": ["Brown-Forman", "Apple"],
            "GICS Sector": ["Consumer Staples", "Information Technology"],
            "GICS Sub-Industry": ["Distillers", "Technology Hardware"],
            "CIK": [14693, 320193],
            "Unneeded": [1, 2],
        }
    )
    session = SequenceSession([FakeResponse(text="<html />")])
    monkeypatch.setattr(data.pd, "read_html", lambda _: [source])

    result = data.fetch_sp500_constituents(session=session, max_retries=0)

    assert list(result.columns) == [
        "ticker",
        "source_ticker",
        "security",
        "sector",
        "subindustry",
        "cik",
    ]
    assert result["ticker"].tolist() == ["BF.B", "AAPL"]
    assert result["cik"].tolist() == ["0000014693", "0000320193"]
    assert session.calls[0][0] == data.SP500_WIKIPEDIA_URL


def test_http_get_retries_without_leaking_params():
    secret = "a" * 32
    session = SequenceSession([FakeResponse(status_code=429), FakeResponse(status_code=503)])

    with pytest.raises(data.DataDownloadError) as error:
        data._safe_get(
            session,
            "https://example.test/data",
            params={"api_key": secret},
            timeout=1.0,
            max_retries=1,
            backoff_factor=0,
            sleep_func=lambda _: None,
            request_name="test service",
        )

    assert len(session.calls) == 2
    assert secret not in str(error.value)


def test_http_get_retries_request_exception_then_succeeds():
    session = SequenceSession(
        [requests.ConnectionError("offline"), FakeResponse(payload={"ok": True})]
    )

    response = data._safe_get(
        session,
        "https://example.test/data",
        params=None,
        timeout=1.0,
        max_retries=1,
        backoff_factor=0,
        sleep_func=lambda _: None,
        request_name="test service",
    )

    assert response.json() == {"ok": True}


def test_confirmed_close_uses_shared_bulk_contract_and_exclusive_end():
    client = RecordingMarketDataClient(
        _bars(
            [
                ("2020-01-01", "AAPL", 10, 12, 9, 11, 100, "marketdata", "confirmed"),
                ("2020-01-01", "BRK.B", 20, 22, 19, 21, 200, "marketdata", "confirmed"),
                ("2020-01-02", "AAPL", 11, 13, 10, 12, 110, "marketdata", "confirmed"),
                ("2020-01-03", "AAPL", 12, 14, 11, 13, 120, "marketdata", "confirmed"),
            ]
        )
    )

    result = data.download_confirmed_close(
        ["aapl", "BRK.B", "AAPL"],
        "2020-01-01",
        "2020-01-03",
        client=client,
        refresh=True,
    )

    assert client.calls == [
        {
            "symbols": ["AAPL", "BRK.B"],
            "start": "2020-01-01",
            "end": "2020-01-03",
            "finality": "confirmed",
            "refresh": True,
        }
    ]
    assert result.index.tolist() == pd.to_datetime(["2020-01-01", "2020-01-02"]).tolist()
    assert result.columns.tolist() == ["AAPL", "BRK.B"]
    assert result.loc[pd.Timestamp("2020-01-01"), "BRK.B"] == 21
    assert result.attrs["source"] == "marketdata"
    assert result.attrs["finality"] == "confirmed"


def test_confirmed_close_rejects_nonconfirmed_rows():
    client = RecordingMarketDataClient(
        _bars(
            [
                ("2020-01-01", "AAPL", 10, 12, 9, 11, 100, "marketdata", "provisional"),
            ]
        )
    )

    with pytest.raises(data.DataDownloadError, match="non-confirmed"):
        data.download_confirmed_close(["AAPL"], "2020-01-01", "2020-01-02", client=client)


def test_confirmed_close_rejects_incomplete_schema():
    client = RecordingMarketDataClient(
        pd.DataFrame(
            {"date": ["2020-01-01"], "symbol": ["AAPL"], "finality": ["confirmed"]}
        )
    )

    with pytest.raises(data.DataDownloadError, match="missing required columns"):
        data.download_confirmed_close(["AAPL"], "2020-01-01", "2020-01-02", client=client)


def test_snapshot_round_trip_and_checksum_guard(tmp_path):
    frame = pd.DataFrame(
        {"series": [1.0, np.nan]}, index=pd.to_datetime(["2021-01-01", "2021-01-02"])
    )
    path = tmp_path / "snapshot.csv"
    data.save_timeseries_snapshot(frame, path, metadata={"source": "fixture"})

    loaded, metadata = data.load_timeseries_snapshot(path)

    pd.testing.assert_frame_equal(loaded, frame.rename_axis("date"))
    assert metadata["source"] == "fixture"
    path.write_text(path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    with pytest.raises(data.DataDownloadError, match="checksum"):
        data.load_timeseries_snapshot(path)


@pytest.mark.parametrize(
    "metadata",
    [{"api_key": "secret"}, {"nested": {"accessToken": "secret"}}, {"credentials": ["secret"]}],
)
def test_snapshot_metadata_rejects_secrets(tmp_path, metadata):
    frame = pd.DataFrame({"value": [1]}, index=pd.to_datetime(["2021-01-01"]))
    with pytest.raises(ValueError, match="credentials"):
        data.save_timeseries_snapshot(frame, tmp_path / "unsafe.csv", metadata=metadata)


@pytest.mark.parametrize("value", [None, "", "short", "A" * 32, "a" * 31 + "!"])
def test_fred_key_is_env_only_and_validated(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FRED_API_KEY", raising=False)
    else:
        monkeypatch.setenv("FRED_API_KEY", value)
    with pytest.raises(data.InvalidFREDApiKey) as error:
        data._fred_api_key_from_env()
    assert not value or value not in str(error.value)


def test_fred_download_retries_parses_missing_and_uses_exclusive_end(monkeypatch):
    key = "a" * 32
    monkeypatch.setenv("FRED_API_KEY", key)
    session = SequenceSession(
        [
            FakeResponse(status_code=503),
            FakeResponse(
                payload={
                    "observations": [
                        {"date": "2020-01-01", "value": "1.5"},
                        {"date": "2020-01-02", "value": "."},
                        {"date": "2020-01-03", "value": "9.9"},
                    ]
                }
            ),
        ]
    )

    result = data.download_fred_observations(
        {"fed_funds": "DFF"},
        "2020-01-01",
        "2020-01-03",
        session=session,
        max_retries=1,
        backoff_factor=0,
        sleep_func=lambda _: None,
    )

    assert result.index.tolist() == pd.to_datetime(["2020-01-01", "2020-01-02"]).tolist()
    assert result.loc[pd.Timestamp("2020-01-01"), "fed_funds"] == 1.5
    assert np.isnan(result.loc[pd.Timestamp("2020-01-02"), "fed_funds"])
    params = session.calls[-1][1]["params"]
    assert params["observation_end"] == "2020-01-02"
    assert params["api_key"] == key


def test_fred_snapshot_omits_key_and_can_load_offline(monkeypatch, tmp_path):
    key = "b" * 32
    monkeypatch.setenv("FRED_API_KEY", key)
    cache = tmp_path / "macro.csv"
    session = SequenceSession(
        [FakeResponse(payload={"observations": [{"date": "2021-01-01", "value": "2"}]})]
    )
    first = data.download_fred_series(
        {"ten_year": "DGS10"},
        "2021-01-01",
        "2021-01-02",
        session=session,
        cache_path=cache,
        max_retries=0,
    )
    monkeypatch.delenv("FRED_API_KEY")
    second = data.download_fred_series(
        {"ten_year": "DGS10"},
        "2021-01-01",
        "2021-01-02",
        session=SequenceSession([AssertionError("network should not be called")]),
        cache_path=cache,
    )

    pd.testing.assert_frame_equal(first, second)
    sidecar = data.snapshot_metadata_path(cache).read_text(encoding="utf-8")
    metadata = json.loads(sidecar)
    assert metadata["source"] == "fred_current_vintage"
    assert key not in sidecar
    assert "api_key" not in sidecar.lower()
