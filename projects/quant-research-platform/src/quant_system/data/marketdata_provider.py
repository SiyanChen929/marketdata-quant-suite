"""MarketData.app provider adapter.

Secrets are read from ``MARKETDATA_TOKEN`` only and sent with header-based
authentication by the suite's shared MarketData client.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from quant_marketdata import MarketDataClient, MarketDataStore

from quant_system.data.provider import MarketDataProvider
from quant_system.utils.validation import validate_confirmed_marketdata


class MarketDataAPIProvider(MarketDataProvider):
    """MarketData.app historical candles provider."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache",
        retries: int = 3,
        pause_seconds: float = 0.25,
        adjust_splits: bool = True,
        session: Any | None = None,
    ) -> None:
        if api_key is not None:
            raise ValueError("Pass MarketData credentials through MARKETDATA_TOKEN, not constructor arguments")
        self.base_url = base_url or os.getenv("MARKETDATA_BASE_URL", "https://api.marketdata.app/v1")
        self.retries = retries
        self.pause_seconds = pause_seconds
        self.adjust_splits = adjust_splits
        store = MarketDataStore() if str(cache_dir) == "data/cache" else MarketDataStore(root=cache_dir)
        self.store = store
        self.client = MarketDataClient(
            store=store,
            session=session,
            base_url=self.base_url,
            retries=retries,
            backoff_seconds=pause_seconds,
        )

    def get_ohlcv(self, symbol: str, start: str, end: str | None, timeframe: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV from MarketData.app historical candles."""

        resolution = self._resolution(timeframe)
        if resolution != "D":
            raise ValueError("MarketDataAPIProvider handles daily bars; use MarketDataIntradayProvider for intraday resolutions")
        end_value = end or _latest_completed_us_equity_date().date().isoformat()
        frame = self.client.get_daily_bars(
            symbol,
            start,
            end_value,
            finality="confirmed",
        )
        compatible = frame.loc[:, ["symbol", "date", "open", "high", "low", "close", "volume", "source", "finality"]].copy()
        compatible["adj_close"] = compatible["close"]
        return validate_confirmed_marketdata(compatible)

    @staticmethod
    def _payload_to_ohlcv(symbol: str, payload: Any) -> pd.DataFrame:
        """Map MarketData.app or common OHLCV JSON shapes to the canonical schema.

        MarketData.app returns column-oriented arrays with ``s/o/h/l/c/v/t``.
        The fallback branch supports row-oriented payloads for local mocks and
        future adapter variants.
        """

        if isinstance(payload, dict):
            status = payload.get("s")
            if status == "no_data":
                return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])
            if status == "error":
                raise ValueError(f"MarketData.app error for {symbol}: {payload.get('errmsg', payload)}")
            if {"o", "h", "l", "c", "v", "t"}.issubset(payload):
                return MarketDataAPIProvider._marketdata_column_payload_to_ohlcv(symbol, payload)
        records = payload
        if isinstance(payload, dict):
            for key in ("data", "bars", "prices", "results", "candles"):
                if key in payload:
                    records = payload[key]
                    break
        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])
        aliases = {
            "date": ["date", "datetime", "timestamp", "time", "t"],
            "open": ["open", "o"],
            "high": ["high", "h"],
            "low": ["low", "l"],
            "close": ["close", "c"],
            "adj_close": ["adj_close", "adjusted_close", "adjClose", "ac", "close", "c"],
            "volume": ["volume", "v"],
        }
        out = pd.DataFrame()
        out["symbol"] = frame.get("symbol", symbol)
        for canonical, candidates in aliases.items():
            source = next((candidate for candidate in candidates if candidate in frame.columns), None)
            if source is None:
                raise ValueError(f"MarketData payload missing {canonical}; available columns={list(frame.columns)}")
            out[canonical] = _normalize_date(frame[source]) if canonical == "date" else frame[source]
        return out

    @staticmethod
    def _marketdata_column_payload_to_ohlcv(symbol: str, payload: dict[str, Any]) -> pd.DataFrame:
        """Normalize MarketData.app column-oriented candles payload."""

        times = payload.get("t", [])
        out = pd.DataFrame(
            {
                "symbol": symbol.upper(),
                "date": _normalize_date(pd.Series(times)),
                "open": pd.to_numeric(pd.Series(payload.get("o", [])), errors="coerce"),
                "high": pd.to_numeric(pd.Series(payload.get("h", [])), errors="coerce"),
                "low": pd.to_numeric(pd.Series(payload.get("l", [])), errors="coerce"),
                "close": pd.to_numeric(pd.Series(payload.get("c", [])), errors="coerce"),
                "adj_close": pd.to_numeric(pd.Series(payload.get("c", [])), errors="coerce"),
                "volume": pd.to_numeric(pd.Series(payload.get("v", [])), errors="coerce"),
            }
        )
        return out

    @staticmethod
    def _resolution(timeframe: str) -> str:
        """Map framework timeframe names to MarketData.app resolution names."""

        normalized = timeframe.lower()
        mapping = {
            "1d": "D",
            "d": "D",
            "daily": "D",
            "1w": "W",
            "w": "W",
            "weekly": "W",
            "1m": "M",
            "m": "M",
            "monthly": "M",
            "1min": "1",
            "1minute": "1",
            "5min": "5",
            "15min": "15",
            "30min": "30",
            "1h": "H",
            "h": "H",
        }
        return mapping.get(normalized, timeframe)

    def get_universe(self, name: str) -> list[str]:
        """Fetch a provider universe.

        MarketData.app supports symbol-specific endpoints; broad historical
        universe membership should come from your security master or a dedicated
        point-in-time constituents dataset.
        """

        raise NotImplementedError("MarketData.app universe endpoint is not wired; provide custom_symbols or a CSV security master.")


def _normalize_date(values: pd.Series) -> pd.Series:
    """Normalize MarketData timestamps or ISO dates to session dates."""

    if values.empty:
        return pd.Series(dtype="datetime64[ns]")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        unit = "ms" if numeric.abs().max() > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return pd.to_datetime(values, format="mixed", errors="coerce").dt.normalize()


def _latest_completed_us_equity_date(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return a conservative latest date for an open-ended daily request."""

    current = now if now is not None else pd.Timestamp.now(tz="America/New_York")
    if current.tzinfo is None:
        current = current.tz_localize("America/New_York")
    else:
        current = current.tz_convert("America/New_York")
    session_date = current.tz_localize(None).normalize()
    close_buffer = current.normalize() + pd.Timedelta(hours=16, minutes=15)
    return session_date if current >= close_buffer else session_date - pd.Timedelta(days=1)
