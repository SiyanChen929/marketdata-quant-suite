"""Thematic basket support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_thematic_baskets(path: str | Path) -> dict[str, list[str]]:
    """Load theme baskets from YAML."""

    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    baskets = raw.get("themes", raw)
    return {str(name): [str(symbol).upper() for symbol in symbols] for name, symbols in baskets.items()}


def score_theme_membership(symbols: list[str], baskets: dict[str, list[str]]) -> pd.DataFrame:
    """Assign a simple neutral theme score for basket members."""

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        themes = [name for name, members in baskets.items() if symbol in members]
        rows.append({"symbol": symbol, "theme_memberships": ",".join(themes), "theme_score": 60.0 if themes else 50.0})
    return pd.DataFrame(rows)


def dynamic_theme_scores(prices: pd.DataFrame, benchmark: str = "SPY", threshold: float = 60.0) -> pd.DataFrame:
    """Compute point-in-time symbol theme/momentum scores.

    This is intentionally date-local and uses only data available at each close.
    It gives the theme layer real variation while keeping it interpretable:
    relative 21d/63d strength, breadth above the 20d average, proximity to highs,
    and volume expansion.
    """

    if prices.empty:
        return pd.DataFrame(columns=["date", "symbol", "theme_score"])
    data = prices.sort_values(["symbol", "date"]).copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    bench = data[data["symbol"] == benchmark.upper()][["date", "adj_close"]].rename(columns={"adj_close": "benchmark_close"}).sort_values("date")
    bench["benchmark_ret_21d"] = bench["benchmark_close"].pct_change(21)
    bench["benchmark_ret_63d"] = bench["benchmark_close"].pct_change(63)
    data = data.merge(bench, on="date", how="left")
    grouped = data.groupby("symbol", group_keys=False)
    if "rs_21d" not in data:
        data["rs_21d"] = grouped["adj_close"].pct_change(21) - data["benchmark_ret_21d"]
    if "rs_63d" not in data:
        data["rs_63d"] = grouped["adj_close"].pct_change(63) - data["benchmark_ret_63d"]
    if "ma_20" not in data:
        data["ma_20"] = grouped["adj_close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    if "high_252" not in data:
        data["high_252"] = grouped["adj_close"].transform(lambda s: s.rolling(252, min_periods=60).max())
    if "volume_expansion" not in data:
        vol_ma = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        data["volume_expansion"] = data["volume"] / vol_ma.replace(0, pd.NA)
    by_date = data.groupby("date")
    rel21_rank = by_date["rs_21d"].rank(pct=True).fillna(0.5)
    rel63_rank = by_date["rs_63d"].rank(pct=True).fillna(0.5)
    volume_rank = by_date["volume_expansion"].rank(pct=True).fillna(0.5)
    above_20 = (data["adj_close"] > data["ma_20"]).astype(float).where(data["ma_20"].notna(), 0.5)
    near_high = (data["adj_close"] / data["high_252"].replace(0, pd.NA) > 0.90).astype(float).where(data["high_252"].notna(), 0.5)
    score = (
        30.0 * rel21_rank
        + 30.0 * rel63_rank
        + 15.0 * above_20
        + 15.0 * near_high
        + 10.0 * volume_rank
    ).clip(0, 100)
    active = (
        score.ge(float(threshold))
        & data["rs_21d"].fillna(0).gt(0)
        & data["rs_63d"].fillna(0).gt(0)
        & above_20.ge(0.5)
    )
    out = data[["date", "symbol"]].copy()
    out["theme_score"] = score
    out["theme_active"] = active
    out["theme_reason"] = (
        "rel21_rank="
        + rel21_rank.round(2).astype(str)
        + "; rel63_rank="
        + rel63_rank.round(2).astype(str)
        + "; above20="
        + above_20.round(0).astype(int).astype(str)
        + "; near_high="
        + near_high.round(0).astype(int).astype(str)
        + "; volume_rank="
        + volume_rank.round(2).astype(str)
    )
    return out
