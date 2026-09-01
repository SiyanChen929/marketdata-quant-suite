"""Market data provider backed by the local Parquet lake."""

from __future__ import annotations

import pandas as pd

from quant_system.data.lake import QuantSystemLake
from quant_system.data.provider import MarketDataProvider
from quant_system.utils.validation import require_columns


class LakeMarketDataProvider(MarketDataProvider):
    """Read daily OHLCV from the shared confirmed QuantSystem lake."""

    def __init__(self, lake: QuantSystemLake | None = None) -> None:
        self.lake = lake or QuantSystemLake()

    def get_ohlcv(self, symbol: str, start: str, end: str | None, timeframe: str = "1d") -> pd.DataFrame:
        if timeframe != "1d":
            raise ValueError("LakeMarketDataProvider daily path supports timeframe='1d'")
        return self._confirmed(self.lake.read_daily([symbol], start, end))

    def get_bulk_ohlcv(self, symbols: list[str], start: str, end: str | None, timeframe: str = "1d") -> pd.DataFrame:
        if timeframe != "1d":
            raise ValueError("LakeMarketDataProvider daily path supports timeframe='1d'")
        return self._confirmed(self.lake.read_daily(symbols, start, end))

    @staticmethod
    def _confirmed(frame: pd.DataFrame) -> pd.DataFrame:
        require_columns(frame, ["source", "finality"], "Daily research lake")
        if not frame.empty and not frame["finality"].astype(str).str.lower().eq("confirmed").all():
            raise ValueError("daily research lake returned provisional rows")
        return frame

    def get_universe(self, name: str) -> list[str]:
        return self.lake.daily_symbols()
