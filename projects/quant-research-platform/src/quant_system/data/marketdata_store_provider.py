"""Read-only research adapter for the canonical external MarketData lake."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from quant_marketdata import MarketDataStore

from quant_system.data.provider import MarketDataProvider
from quant_system.utils.validation import validate_confirmed_marketdata


class MarketDataStoreProvider(MarketDataProvider):
    """Serve formal daily research from confirmed, already-ingested bars."""

    def __init__(self, store: MarketDataStore | None = None) -> None:
        self.store = store or MarketDataStore()

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str | None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        return self.get_bulk_ohlcv([symbol], start, end, timeframe)

    def get_bulk_ohlcv(
        self,
        symbols: list[str],
        start: str,
        end: str | None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        if str(timeframe).strip().lower() not in {"1d", "d", "daily"}:
            raise ValueError("Formal store provider currently serves daily resolution only")
        bars = self.store.read_bars(
            symbols=[str(symbol).upper() for symbol in symbols],
            start=start,
            end=end,
            finality="confirmed",
            resolution="D",
        )
        snapshot_attrs = dict(bars.attrs)
        compatible = bars.copy()
        compatible["adj_close"] = compatible.get("close")
        validated = validate_confirmed_marketdata(compatible)
        validated.attrs.update(snapshot_attrs)
        return validated

    def get_universe(self, name: str) -> list[str]:
        raise NotImplementedError(
            "Use a point-in-time security master or explicit custom_symbols; "
            "the price lake is not a universe-membership source"
        )
