"""Theme activation and maturity protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd


def confidence_multiplier(theme_active_days: int) -> float:
    """New theme protocol confidence multiplier."""

    return float(min(1.0, np.sqrt(max(theme_active_days, 0) / 252.0)))


def suggested_allocation_multiplier(theme_active_days: int) -> float:
    """Position cap multiplier based on theme maturity."""

    if theme_active_days < 60:
        return 0.3
    if theme_active_days < 120:
        return 0.5
    if theme_active_days < 252:
        return 0.75
    return 1.0


def compute_theme_activation(theme_prices: pd.DataFrame, benchmark_prices: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    """Compute a theme activation score from component-level prices."""

    if theme_prices.empty or benchmark_prices.empty:
        return pd.DataFrame(columns=["date", "theme_score", "theme_active", "theme_active_days", "confidence_multiplier", "suggested_allocation_multiplier"])
    data = theme_prices.sort_values(["symbol", "date"]).copy()
    bench = benchmark_prices[["date", "adj_close"]].rename(columns={"adj_close": "benchmark_close"}).sort_values("date")
    data = data.merge(bench, on="date", how="left")
    grouped = data.groupby("symbol")
    data["ret_21d"] = grouped["adj_close"].pct_change(21)
    data["ret_63d"] = grouped["adj_close"].pct_change(63)
    data["bench_ret_21d"] = data["benchmark_close"].pct_change(21)
    data["bench_ret_63d"] = data["benchmark_close"].pct_change(63)
    data["ma_20"] = grouped["adj_close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    data["high_252"] = grouped["adj_close"].transform(lambda s: s.rolling(252, min_periods=60).max())
    data["volume_z"] = grouped["volume"].transform(lambda s: (s - s.rolling(63, min_periods=20).mean()) / s.rolling(63, min_periods=20).std())
    daily = data.groupby("date").agg(
        rel_21=("ret_21d", lambda s: (s - data.loc[s.index, "bench_ret_21d"]).mean()),
        rel_63=("ret_63d", lambda s: (s - data.loc[s.index, "bench_ret_63d"]).mean()),
        pct_above_20d=("adj_close", lambda s: (s > data.loc[s.index, "ma_20"]).mean()),
        pct_near_high=("adj_close", lambda s: (s / data.loc[s.index, "high_252"] > 0.90).mean()),
        volume_zscore=("volume_z", "mean"),
    ).reset_index()
    daily["theme_score"] = (
        30 * (daily["rel_21"] > 0).astype(float)
        + 30 * (daily["rel_63"] > 0).astype(float)
        + 15 * daily["pct_above_20d"].fillna(0.0)
        + 15 * daily["pct_near_high"].fillna(0.0)
        + 10 * (daily["volume_zscore"].fillna(0.0) > 0).astype(float)
    )
    daily["theme_active"] = (daily["theme_score"] > threshold) & (daily["pct_above_20d"] >= 0.5)
    active_days = []
    count = 0
    for active in daily["theme_active"]:
        count = count + 1 if active else 0
        active_days.append(count)
    daily["theme_active_days"] = active_days
    daily["confidence_multiplier"] = daily["theme_active_days"].apply(confidence_multiplier)
    daily["suggested_allocation_multiplier"] = daily["theme_active_days"].apply(suggested_allocation_multiplier)
    return daily
