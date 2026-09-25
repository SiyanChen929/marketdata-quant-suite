"""Shared test helpers: deterministic synthetic bars, small markets and fixtures.

This module is deliberately *not* named ``conftest``: the suite's other test
packages import helpers from their own ``conftest`` module, and a second
module of that name would shadow theirs when several suites are collected in
one pytest process.  Test modules import what they need from here, including
the ``synthetic_bars`` and ``panel`` fixtures.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from llm_factor_mining.data import Panel, panel_from_bars


SYNTHETIC_SOURCE = "synthetic"


def symbols(n: int) -> list[str]:
    return [f"S{index:03d}" for index in range(n)]


def calendar(n_dates: int, start: str = "2031-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n_dates, name="date")


def bars_from_close(
    close: pd.DataFrame,
    *,
    seed: int = 0,
    finality: str = "confirmed",
    source: str = SYNTHETIC_SOURCE,
) -> pd.DataFrame:
    """OHLC-consistent canonical long bars around a given close matrix.

    NaN closes mean "no bar" and produce no row.
    """

    rng = np.random.default_rng(seed)
    values = close.to_numpy(dtype="float64")
    previous = np.vstack([values[:1], values[:-1]])
    previous = np.where(np.isnan(previous), values, previous)
    open_ = previous * (1.0 + rng.normal(0.0, 0.002, values.shape))
    high = np.maximum(open_, values) * (1.0 + np.abs(rng.normal(0.0, 0.004, values.shape)))
    low = np.minimum(open_, values) * (1.0 - np.abs(rng.normal(0.0, 0.004, values.shape)))
    volume = np.round(rng.lognormal(13.0, 0.4, values.shape))
    n_dates, n_symbols = values.shape
    frame = pd.DataFrame(
        {
            "date": np.repeat(close.index.to_numpy(), n_symbols),
            "symbol": np.tile(np.asarray(close.columns, dtype=object), n_dates),
            "open": open_.ravel(),
            "high": high.ravel(),
            "low": low.ravel(),
            "close": values.ravel(),
            "volume": volume.ravel(),
        }
    )
    frame = frame.loc[np.isfinite(frame["close"])].reset_index(drop=True)
    frame["source"] = source
    frame["finality"] = finality
    return frame


def random_walk_close(
    n_dates: int = 80,
    n_symbols: int = 12,
    *,
    seed: int = 7,
    vol: float = 0.02,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, vol, size=(n_dates, n_symbols))
    returns[0] = 0.0
    levels = 50.0 + 100.0 * rng.random(n_symbols)
    close = levels * np.cumprod(1.0 + returns, axis=0)
    return pd.DataFrame(close, index=calendar(n_dates), columns=pd.Index(symbols(n_symbols), name="symbol"))


def make_bars(
    n_dates: int = 80,
    n_symbols: int = 12,
    *,
    seed: int = 7,
    finality: str = "confirmed",
) -> pd.DataFrame:
    return bars_from_close(random_walk_close(n_dates, n_symbols, seed=seed), seed=seed + 1, finality=finality)


def perturb_after(
    bars: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    seed: int = 99,
    scale: float = 0.2,
) -> pd.DataFrame:
    """Randomly change every bar strictly after ``cutoff``.

    Each row gets a common price factor plus an extra close-only factor (so
    scale-invariant quantities such as ``(close - open) / open`` also change);
    high/low are then widened to keep the OHLC ordering valid.
    """

    rng = np.random.default_rng(seed)
    out = bars.copy()
    later = (pd.to_datetime(out["date"]) > pd.Timestamp(cutoff)).to_numpy()
    n = int(later.sum())
    common = np.exp(rng.normal(0.0, scale, n))
    close_only = np.exp(rng.normal(0.0, scale / 2.0, n))
    open_ = out.loc[later, "open"].to_numpy() * common
    close = out.loc[later, "close"].to_numpy() * common * close_only
    high = np.maximum.reduce([out.loc[later, "high"].to_numpy() * common, open_, close])
    low = np.minimum.reduce([out.loc[later, "low"].to_numpy() * common, open_, close])
    out.loc[later, "open"] = open_
    out.loc[later, "high"] = high
    out.loc[later, "low"] = low
    out.loc[later, "close"] = close
    out.loc[later, "volume"] = np.round(
        out.loc[later, "volume"].to_numpy() * np.exp(rng.normal(0.0, scale, n))
    )
    return out


def panel_from_close(close: pd.DataFrame, *, seed: int = 0) -> Panel:
    return panel_from_bars(bars_from_close(close, seed=seed))


def tiny_bars(rows: Sequence[tuple[str, str, float, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    frame["source"] = SYNTHETIC_SOURCE
    frame["finality"] = "confirmed"
    return frame


@pytest.fixture()
def synthetic_bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture()
def panel(synthetic_bars: pd.DataFrame) -> Panel:
    return panel_from_bars(synthetic_bars)


# --------------------------------------------------------------------------
# stage-2 helpers: small planted-alpha markets and splits (cached, read-only)
# --------------------------------------------------------------------------

@lru_cache(maxsize=8)
def small_market(snr: float = 0.3, seed: int = 0, n_symbols: int = 24, n_dates: int = 260):
    """A small synthetic market with the default planted signals (do not mutate)."""

    from llm_factor_mining.benchmark.synthetic import SyntheticMarketConfig, simulate_market

    return simulate_market(SyntheticMarketConfig(n_symbols=n_symbols, n_dates=n_dates, snr=snr, seed=seed))


def small_splits(panel: Panel):
    from llm_factor_mining.protocol.splits import chronological_splits

    return chronological_splits(panel.dates, fractions=(0.6, 0.2, 0.2), lag=1, max_horizon=1)


def tiny_search_config(**overrides):
    from llm_factor_mining.evaluate.metrics import MetricConfig
    from llm_factor_mining.search import SearchConfig

    values = {
        "budget": 24,
        "batch_size": 8,
        "metric": MetricConfig(min_names=10),
        "record_timestamps": False,
        "top_k": 3,
        "min_ic_dates": 30,
    }
    values.update(overrides)
    return SearchConfig(**values)
