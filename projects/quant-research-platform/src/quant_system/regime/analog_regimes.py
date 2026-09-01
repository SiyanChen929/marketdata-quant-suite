"""Analog high-momentum regime detection."""

from __future__ import annotations

import pandas as pd


def identify_analog_regimes(features: pd.DataFrame) -> pd.DataFrame:
    """Identify historical high-momentum regimes independent of theme names."""

    if features.empty or "rs_63d" not in features.columns:
        return pd.DataFrame(columns=["date", "analog_regime"])
    daily = features.groupby("date").agg(
        rs_63d_p90=("rs_63d", lambda s: s.quantile(0.9)),
        breadth_20d=("adj_close", lambda s: 0.0),
        volume_expansion=("volume_expansion", "mean"),
    ).reset_index()
    threshold = daily["rs_63d_p90"].quantile(0.9)
    daily["analog_regime"] = (daily["rs_63d_p90"] >= threshold) & (daily["volume_expansion"].fillna(1.0) > 1.0)
    return daily


def identify_analog_regimes_pit(
    features: pd.DataFrame,
    lookback_days: int = 504,
    quantile: float = 0.80,
) -> pd.DataFrame:
    """Identify high-momentum analog regimes with a point-in-time threshold."""

    if features.empty or "rs_63d" not in features.columns:
        return pd.DataFrame(columns=["date", "analog_regime", "analog_score"])
    data = features.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    daily = data.groupby("date").agg(
        rs_63d_p90=("rs_63d", lambda s: s.quantile(0.9)),
        volume_expansion=("volume_expansion", "mean"),
    ).reset_index()
    min_periods = min(126, max(20, int(lookback_days) // 4))
    daily["rs_threshold"] = (
        daily["rs_63d_p90"]
        .rolling(int(lookback_days), min_periods=min_periods)
        .quantile(float(quantile))
        .shift(1)
    )
    daily["volume_ok"] = daily["volume_expansion"].fillna(1.0) > 1.0
    daily["analog_regime"] = (daily["rs_63d_p90"] >= daily["rs_threshold"]) & daily["volume_ok"]
    daily["analog_score"] = 50.0
    daily.loc[daily["analog_regime"], "analog_score"] = 100.0
    daily.loc[daily["rs_63d_p90"] < daily["rs_threshold"], "analog_score"] = 25.0
    return daily[["date", "analog_regime", "analog_score", "rs_63d_p90", "rs_threshold", "volume_expansion"]]
