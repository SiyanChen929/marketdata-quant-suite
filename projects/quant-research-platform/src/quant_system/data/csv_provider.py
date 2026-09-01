"""CSV market data provider."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_system.data.provider import MarketDataProvider
from quant_system.utils.validation import validate_ohlcv


class CSVMarketDataProvider(MarketDataProvider):
    """Load OHLCV data from a single CSV or a directory of per-symbol CSVs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cache: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache
        if self.path.is_dir():
            frames = [pd.read_csv(file) for file in sorted(self.path.glob("*.csv"))]
            raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            raw = pd.read_csv(self.path)
        self._cache = validate_ohlcv(raw)
        return self._cache

    def get_ohlcv(self, symbol: str, start: str, end: str | None, timeframe: str = "1d") -> pd.DataFrame:
        """Return one symbol from CSV data."""

        if timeframe != "1d":
            raise ValueError("CSV provider MVP supports daily timeframe only")
        data = self._load()
        mask = (data["symbol"] == symbol.upper()) & (data["date"] >= pd.Timestamp(start))
        if end:
            mask &= data["date"] <= pd.Timestamp(end)
        return data.loc[mask].copy().reset_index(drop=True)

    def get_bulk_ohlcv(self, symbols: list[str], start: str, end: str | None, timeframe: str = "1d") -> pd.DataFrame:
        """Return many symbols from CSV data."""

        data = self._load()
        wanted = {symbol.upper() for symbol in symbols}
        mask = data["symbol"].isin(wanted) & (data["date"] >= pd.Timestamp(start))
        if end:
            mask &= data["date"] <= pd.Timestamp(end)
        return data.loc[mask].copy().reset_index(drop=True)

    def get_universe(self, name: str) -> list[str]:
        """Return all symbols present in the CSV for simple local research."""

        return sorted(self._load()["symbol"].unique().tolist())
