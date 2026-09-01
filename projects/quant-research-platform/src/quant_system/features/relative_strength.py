"""Relative strength features."""

from __future__ import annotations

import pandas as pd


def add_relative_strength(prices: pd.DataFrame, benchmark: str = "SPY", windows: tuple[int, ...] = (21, 63)) -> pd.DataFrame:
    """Add returns relative to a benchmark for each date."""

    source_attrs = dict(prices.attrs)
    out = prices.copy()
    bench = out[out["symbol"] == benchmark.upper()][["date", "adj_close"]].rename(columns={"adj_close": "benchmark_adj_close"})
    out = out.merge(bench, on="date", how="left")
    for window in windows:
        symbol_ret = out.sort_values(["symbol", "date"]).groupby("symbol")["adj_close"].pct_change(window)
        bench_ret = out.sort_values(["symbol", "date"]).groupby("symbol")["benchmark_adj_close"].pct_change(window)
        out[f"rs_{window}d"] = symbol_ret - bench_ret
    out = out.drop(columns=["benchmark_adj_close"])
    out.attrs.update(source_attrs)
    return out
