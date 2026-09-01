"""Confirmed daily-bar adapter for the suite's shared MarketData package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from quant_marketdata import (
    CANONICAL_COLUMNS as SHARED_CANONICAL_COLUMNS,
    MarketDataClient,
    MarketDataStore,
)


CANONICAL_COLUMNS = tuple(SHARED_CANONICAL_COLUMNS)


class CanonicalBarError(ValueError):
    """Raised when shared market data violate the canonical daily schema."""


class BulkDailyClient(Protocol):
    """Structural type used to keep adapter tests offline."""

    def get_bulk_daily_bars(
        self,
        symbols: Sequence[str],
        start: str,
        end: str,
        finality: str = "confirmed",
        refresh: bool = False,
    ) -> pd.DataFrame: ...


def external_data_home(value: str | Path | None = None) -> Path:
    """Return the required external cache root."""

    configured = str(value) if value is not None else os.getenv("QUANT_DATA_HOME", "")
    if not configured.strip():
        raise RuntimeError("Set QUANT_DATA_HOME to an external shared-data directory")
    return Path(configured).expanduser().resolve()


def validate_confirmed_daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate the shared confirmed daily-bar contract."""

    missing = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
    if missing:
        raise CanonicalBarError(f"Missing canonical columns: {', '.join(missing)}")
    bars = frame.loc[:, CANONICAL_COLUMNS].copy()
    try:
        bars["date"] = pd.to_datetime(bars["date"], errors="raise").dt.normalize()
    except (TypeError, ValueError) as exc:
        raise CanonicalBarError("Invalid date column") from exc
    bars["symbol"] = bars["symbol"].astype(str).str.upper().str.strip()
    bars["source"] = bars["source"].astype(str).str.strip()
    bars["finality"] = bars["finality"].astype(str).str.lower().str.strip()
    if bars["symbol"].eq("").any() or bars["source"].eq("").any():
        raise CanonicalBarError("symbol and source must be non-empty")
    if not bars["finality"].eq("confirmed").all():
        raise CanonicalBarError("Formal event studies accept confirmed bars only")
    price_columns = ["open", "high", "low", "close"]
    for column in (*price_columns, "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    numeric = bars[[*price_columns, "volume"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise CanonicalBarError("OHLCV values must be finite")
    if (bars[price_columns] <= 0).any().any() or (bars["volume"] < 0).any():
        raise CanonicalBarError("Prices must be positive and volume non-negative")
    if (
        bars["high"].lt(bars[["open", "close"]].max(axis=1)).any()
        or bars["low"].gt(bars[["open", "close"]].min(axis=1)).any()
        or bars["high"].lt(bars["low"]).any()
    ):
        raise CanonicalBarError("Incoherent OHLC row")
    if bars.duplicated(["date", "symbol"]).any():
        raise CanonicalBarError("Duplicate (date, symbol) rows")
    return bars.sort_values(["date", "symbol"]).reset_index(drop=True)


@dataclass
class ConfirmedDailyBars:
    """Thin adapter that fixes the shared client to confirmed daily data."""

    client: BulkDailyClient

    @classmethod
    def from_environment(
        cls,
        *,
        data_home: str | Path | None = None,
        session: object | None = None,
    ) -> "ConfirmedDailyBars":
        """Build the shared store/client without reading any repo-local token file."""

        store = MarketDataStore(root=external_data_home(data_home))
        client = MarketDataClient(store=store, session=session)
        return cls(client=client)

    def get_bulk_daily_bars(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Retrieve canonical daily bars with immutable confirmed finality."""

        normalized_symbols = sorted({str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()})
        if not normalized_symbols:
            raise ValueError("At least one symbol is required")
        start_date = pd.Timestamp(start).normalize()
        end_date = pd.Timestamp(end).normalize()
        if start_date > end_date:
            raise ValueError("start must not follow end")
        raw = self.client.get_bulk_daily_bars(
            symbols=normalized_symbols,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            finality="confirmed",
            refresh=refresh,
        )
        bars = validate_confirmed_daily_bars(raw)
        if not bars.empty:
            outside = bars["date"].lt(start_date) | bars["date"].gt(end_date)
            if outside.any():
                raise CanonicalBarError("Shared client returned rows outside the requested range")
            unexpected = sorted(set(bars["symbol"]) - set(normalized_symbols))
            if unexpected:
                raise CanonicalBarError(f"Unexpected symbols: {', '.join(unexpected)}")
        return bars


def event_daily_bars(
    events: pd.DataFrame,
    adapter: ConfirmedDailyBars,
    *,
    lookback_days: int = 7,
    forward_days: int = 3,
    benchmark_symbols: Sequence[str] = (),
    refresh: bool = False,
) -> pd.DataFrame:
    """Request one confirmed daily panel covering a normalized event table."""

    required = {"announcement_date", "effective_close_date", "symbol"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Missing event columns: {', '.join(missing)}")
    symbols = sorted(set(events["symbol"].astype(str).str.upper()) | set(benchmark_symbols))
    start = pd.to_datetime(events["announcement_date"]).min() - pd.Timedelta(days=lookback_days)
    end = pd.to_datetime(events["effective_close_date"]).max() + pd.Timedelta(days=forward_days)
    return adapter.get_bulk_daily_bars(symbols, start, end, refresh=refresh)
