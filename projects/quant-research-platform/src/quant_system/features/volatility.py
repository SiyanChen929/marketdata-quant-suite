"""Volatility features."""

from __future__ import annotations

import pandas as pd


def realized_volatility(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualized rolling realized volatility per symbol."""

    returns = prices.sort_values(["symbol", "date"]).groupby("symbol")["adj_close"].pct_change()
    return returns.groupby(prices.sort_values(["symbol", "date"])["symbol"]).transform(lambda s: s.rolling(window, min_periods=5).std() * (252**0.5))
