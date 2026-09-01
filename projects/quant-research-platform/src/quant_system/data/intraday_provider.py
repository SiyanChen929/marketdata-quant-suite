"""Intraday OHLCV providers for execution-risk simulation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from quant_marketdata import DataUnavailableError, MarketDataClient, MarketDataStore

from quant_system.data.marketdata_provider import MarketDataAPIProvider
from quant_system.utils.validation import require_columns


INTRADAY_COLUMNS = [
    "symbol",
    "datetime",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "finality",
]


def validate_intraday_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate intraday OHLCV without normalizing away timestamps."""

    require_columns(frame, ["symbol", "datetime", "open", "high", "low", "close", "volume"], "Intraday OHLCV")
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["datetime"] = pd.to_datetime(out["datetime"], format="mixed", errors="coerce")
    out["date"] = pd.to_datetime(out.get("date", out["datetime"]), format="mixed", errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "source" not in out:
        out["source"] = "csv"
    if "finality" not in out:
        out["finality"] = "provisional"
    out["source"] = out["source"].astype(str)
    out["finality"] = out["finality"].astype(str)
    out = out.dropna(subset=["symbol", "datetime", "date", "open", "high", "low", "close"])
    return out[INTRADAY_COLUMNS].sort_values(["date", "datetime", "symbol"]).reset_index(drop=True)


class CSVIntradayDataProvider:
    """CSV fallback for minute or 5-minute bars."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._full_cache: pd.DataFrame | None = None

    def get_bulk_intraday(self, symbols: list[str], start: str, end: str | None, timeframe: str = "5min") -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=INTRADAY_COLUMNS)
        wanted = {symbol.upper() for symbol in symbols}
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize() if end else None
        data = self._load_full_file()
        if data.empty:
            return pd.DataFrame(columns=INTRADAY_COLUMNS)
        mask = data["symbol"].isin(wanted) & data["date"].ge(start_ts)
        if end_ts is not None:
            mask &= data["date"].le(end_ts)
        return data.loc[mask].copy().reset_index(drop=True)

    def _load_full_file(self) -> pd.DataFrame:
        """Load the full intraday CSV through a validated binary mirror."""

        if self._full_cache is not None:
            return self._full_cache
        cache_path = self._binary_cache_path()
        if cache_path.exists():
            try:
                self._full_cache = pd.read_pickle(cache_path)
                return self._full_cache
            except Exception:
                cache_path.unlink(missing_ok=True)
        raw = pd.read_csv(self.path)
        self._full_cache = validate_intraday_ohlcv(raw)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._full_cache.to_pickle(cache_path)
        return self._full_cache

    def _binary_cache_path(self) -> Path:
        stat = self.path.stat()
        payload = f"{self.path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return self.path.parent / f"{self.path.stem}_{digest}.pkl"


class MarketDataIntradayProvider:
    """MarketData.app intraday candles provider preserving timestamps."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache",
        retries: int = 2,
        pause_seconds: float = 0.25,
        timeout_seconds: float = 30.0,
        chunk_days: int = 30,
        session: Any | None = None,
    ) -> None:
        if api_key is not None:
            raise ValueError("Pass MarketData credentials through MARKETDATA_TOKEN, not constructor arguments")
        store = MarketDataStore() if str(cache_dir) == "data/cache" else MarketDataStore(root=cache_dir)
        self.store = store
        self.client = MarketDataClient(
            store=store,
            session=session,
            base_url=base_url or "https://api.marketdata.app/v1",
            retries=retries,
            backoff_seconds=pause_seconds,
            timeout=timeout_seconds,
        )
        self.chunk_days = chunk_days

    def get_bulk_intraday(self, symbols: list[str], start: str, end: str | None, timeframe: str = "5min") -> pd.DataFrame:
        frames = []
        for symbol in symbols:
            try:
                frame = self.get_intraday(symbol, start, end, timeframe)
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INTRADAY_COLUMNS)

    def get_intraday(self, symbol: str, start: str, end: str | None, timeframe: str = "5min") -> pd.DataFrame:
        end_value = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end_value).normalize()
        if (end_ts - start_ts).days > self.chunk_days:
            return self._get_intraday_chunked(symbol, start_ts, end_ts, timeframe)
        return self._get_intraday_single(symbol, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"), timeframe)

    def _get_intraday_chunked(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + pd.Timedelta(days=self.chunk_days), end)
            try:
                frame = self._get_intraday_single(symbol, cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), timeframe)
            except DataUnavailableError:
                frame = pd.DataFrame()
            if not frame.empty:
                frames.append(frame)
            cursor = chunk_end + pd.Timedelta(days=1)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INTRADAY_COLUMNS)
        if not out.empty:
            out = validate_intraday_ohlcv(out)
        return out

    def _get_intraday_single(self, symbol: str, start: str, end: str | None, timeframe: str = "5min") -> pd.DataFrame:
        resolution = MarketDataAPIProvider._resolution(timeframe)
        symbol = symbol.upper()
        bars = self.client.get_stock_bars(
            symbol,
            start,
            end or pd.Timestamp.today().strftime("%Y-%m-%d"),
            resolution=resolution,
            finality="provisional",
        )
        if bars.empty:
            return pd.DataFrame(columns=INTRADAY_COLUMNS)
        frame = bars.rename(columns={"date": "datetime"}).copy()
        frame["date"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
        return validate_intraday_ohlcv(frame)

    def read_stored_intraday(
        self,
        symbols: list[str],
        start: str,
        end: str | None,
        timeframe: str = "5min",
    ) -> pd.DataFrame:
        """Read provisional intraday bars already present in the shared lake."""

        bars = self.store.read_bars(
            symbols=symbols,
            start=start,
            end=end,
            finality="provisional",
            resolution=MarketDataAPIProvider._resolution(timeframe),
        )
        if bars.empty:
            return pd.DataFrame(columns=INTRADAY_COLUMNS)
        frame = bars.rename(columns={"date": "datetime"}).copy()
        frame["date"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
        return validate_intraday_ohlcv(frame)

    @staticmethod
    def _payload_to_intraday(symbol: str, payload: Any) -> pd.DataFrame:
        if isinstance(payload, dict):
            status = payload.get("s")
            if status == "no_data":
                return pd.DataFrame(columns=INTRADAY_COLUMNS)
            if status == "error":
                raise ValueError(f"MarketData.app intraday error for {symbol}: {payload.get('errmsg', payload)}")
            if {"o", "h", "l", "c", "v", "t"}.issubset(payload):
                return pd.DataFrame(
                    {
                        "symbol": symbol.upper(),
                        "datetime": _parse_marketdata_time(pd.Series(payload.get("t", []))),
                        "open": pd.to_numeric(pd.Series(payload.get("o", [])), errors="coerce"),
                        "high": pd.to_numeric(pd.Series(payload.get("h", [])), errors="coerce"),
                        "low": pd.to_numeric(pd.Series(payload.get("l", [])), errors="coerce"),
                        "close": pd.to_numeric(pd.Series(payload.get("c", [])), errors="coerce"),
                        "volume": pd.to_numeric(pd.Series(payload.get("v", [])), errors="coerce"),
                        "source": "marketdata.app",
                        "finality": "provisional",
                    }
                )
            for key in ("data", "bars", "prices", "results", "candles"):
                if key in payload:
                    payload = payload[key]
                    break
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame(columns=INTRADAY_COLUMNS)
        rename = {
            "timestamp": "datetime",
            "time": "datetime",
            "t": "datetime",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        frame = frame.rename(columns=rename)
        if "symbol" not in frame:
            frame["symbol"] = symbol.upper()
        if "datetime" in frame:
            frame["datetime"] = _parse_marketdata_time(frame["datetime"])
        return frame


def _parse_marketdata_time(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.8:
        unit = "ms" if numeric.dropna().median() > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce")
    return pd.to_datetime(values, format="mixed", errors="coerce")
