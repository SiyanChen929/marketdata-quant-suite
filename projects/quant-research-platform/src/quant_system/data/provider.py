"""Market data provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    """Abstract market data adapter.

    All implementations return symbol, date, open, high, low, close,
    adj_close, and volume. Production MarketData adapters also retain source
    and finality lineage.
    """

    @abstractmethod
    def get_ohlcv(self, symbol: str, start: str, end: str | None, timeframe: str = "1d") -> pd.DataFrame:
        """Return historical OHLCV for one symbol."""

    def get_bulk_ohlcv(self, symbols: list[str], start: str, end: str | None, timeframe: str = "1d") -> pd.DataFrame:
        """Return historical OHLCV for many symbols."""

        frames = [self.get_ohlcv(symbol, start, end, timeframe) for symbol in symbols]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """Return corporate actions.

        TODO: implement split/dividend point-in-time actions in concrete paid-data adapters.
        """

        return pd.DataFrame(columns=["symbol", "date", "action_type", "value"])

    def get_universe(self, name: str) -> list[str]:
        """Return a named universe."""

        raise NotImplementedError(f"Universe {name!r} is not implemented by this provider")
