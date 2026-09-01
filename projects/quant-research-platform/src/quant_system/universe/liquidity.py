"""Liquidity calculations."""

from __future__ import annotations

import pandas as pd


def add_liquidity_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add rolling volume and dollar-volume features known by next open."""

    out = prices.sort_values(["symbol", "date"]).copy()
    out["dollar_volume"] = out["close"] * out["volume"]
    grouped = out.groupby("symbol", group_keys=False)
    out["avg_volume_20d"] = grouped["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    out["avg_dollar_volume_20d"] = grouped["dollar_volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    return out
