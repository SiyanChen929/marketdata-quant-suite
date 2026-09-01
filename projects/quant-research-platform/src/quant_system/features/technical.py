"""Technical indicators used by strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time technical features per symbol.

    Features are computed using data available at the close of each row. The
    backtest engine shifts signals to next-session execution.
    """

    out = prices.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby("symbol", group_keys=False)
    out["return_1d"] = grouped["adj_close"].pct_change()
    for window in (5, 10, 20, 50, 63, 100, 126, 200, 252):
        out[f"ma_{window}"] = grouped["adj_close"].transform(lambda s: s.rolling(window, min_periods=max(5, window // 3)).mean())
        out[f"ret_{window}d"] = grouped["adj_close"].pct_change(window)
        out[f"vol_{window}d"] = grouped["return_1d"].transform(lambda s: s.rolling(window, min_periods=max(5, window // 3)).std())
        out[f"high_{window}"] = grouped["adj_close"].transform(lambda s: s.rolling(window, min_periods=max(5, window // 3)).max())
        out[f"low_{window}"] = grouped["adj_close"].transform(lambda s: s.rolling(window, min_periods=max(5, window // 3)).min())
        out[f"high_{window}_prior"] = grouped["adj_close"].transform(lambda s: s.shift(1).rolling(window, min_periods=max(5, window // 3)).max())
        out[f"low_{window}_prior"] = grouped["adj_close"].transform(lambda s: s.shift(1).rolling(window, min_periods=max(5, window // 3)).min())
    out["high_252"] = grouped["adj_close"].transform(lambda s: s.rolling(252, min_periods=60).max())
    out["low_252"] = grouped["adj_close"].transform(lambda s: s.rolling(252, min_periods=60).min())
    out["high_252_prior"] = grouped["adj_close"].transform(lambda s: s.shift(1).rolling(252, min_periods=60).max())
    out["low_252_prior"] = grouped["adj_close"].transform(lambda s: s.shift(1).rolling(252, min_periods=60).min())
    out["distance_to_high_252"] = out["adj_close"] / out["high_252"] - 1.0
    out["distance_to_prior_high_252"] = out["adj_close"] / out["high_252_prior"] - 1.0
    out["volume_ma_20"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    out["volume_expansion"] = out["volume"] / out["volume_ma_20"].replace(0, np.nan)
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - grouped["close"].shift(1)).abs()
    low_close = (out["low"] - grouped["close"].shift(1)).abs()
    out["true_range"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr_14"] = grouped["true_range"].transform(lambda s: s.rolling(14, min_periods=5).mean())
    out["rsi_14"] = grouped["adj_close"].transform(_rsi)
    out["bb_mid_20"] = out["ma_20"]
    out["bb_std_20"] = grouped["adj_close"].transform(lambda s: s.rolling(20, min_periods=10).std())
    out["bb_upper_20"] = out["bb_mid_20"] + 2.0 * out["bb_std_20"]
    out["bb_lower_20"] = out["bb_mid_20"] - 2.0 * out["bb_std_20"]
    out["zscore_20"] = (out["adj_close"] - out["bb_mid_20"]) / out["bb_std_20"].replace(0, np.nan)
    out["adx_14"] = grouped.apply(_adx).reset_index(level=0, drop=True).sort_index()
    return out


def cross_sectional_rank(frame: pd.DataFrame, column: str, ascending: bool = True) -> pd.Series:
    """Daily percentile rank for a feature."""

    return frame.groupby("date")[column].rank(pct=True, ascending=ascending)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    plus_dm = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=5).mean()
    plus_di = 100 * plus_dm.rolling(window, min_periods=5).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window, min_periods=5).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window, min_periods=5).mean()
