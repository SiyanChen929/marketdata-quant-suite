"""Independent ground truth computed directly with numpy on the raw bar frame.

This module does not import the tool handlers, :mod:`marketdata_agent.analytics`,
the clock or the as-of view. It receives the full canonical panel, including rows
after the as-of date, and applies every cutoff itself. Agreement between these
functions and the tools is a test (``tests/test_reference.py``), not an assumption,
so the benchmark's ground truth does not validate the tools against themselves.
Where practical a different formulation is used: drawdown from suffix minima
instead of running maxima, and correlation from explicit sums instead of
``corrcoef``.

Conventions (identical in meaning to the tools):

* return: close of the first session on or after ``start`` to the close of the
  last session on or before ``end``;
* volatility: sample standard deviation (n - 1) of the last ``window`` daily log
  returns ending at the last close on or before ``end``, times sqrt(252);
* correlation: Pearson correlation of the last ``window`` log returns over dates
  on which both symbols have a close;
* maximum drawdown: the most negative ``close[j] / close[i] - 1`` over ``i <= j``;
* universe at ``as_of``: symbols with at least one bar dated on or before ``as_of``;
* "latest": the last session on or before ``as_of`` across all symbols.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import math

import numpy as np
import pandas as pd


ANNUALIZATION = 252.0


def _day(value: date | str) -> np.datetime64:
    return np.datetime64(pd.Timestamp(value).date().isoformat(), "D")


@dataclass(frozen=True)
class RawPanel:
    """Numpy arrays extracted once from a canonical long-form frame."""

    dates: np.ndarray  # datetime64[D]
    symbols: np.ndarray  # object
    closes: np.ndarray  # float64

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "RawPanel":
        dates = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[D]")
        return cls(dates, frame["symbol"].astype(str).to_numpy(dtype=object), frame["close"].to_numpy(dtype="float64"))

    def series(self, symbol: str, start: date | str | None = None, end: date | str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Dates and closes for ``symbol`` in ``[start, end]``, sorted by date."""

        mask = self.symbols == symbol
        if start is not None:
            mask &= self.dates >= _day(start)
        if end is not None:
            mask &= self.dates <= _day(end)
        order = np.argsort(self.dates[mask], kind="stable")
        return self.dates[mask][order], self.closes[mask][order]

    def universe(self, as_of: date | str) -> list[str]:
        return sorted(set(self.symbols[self.dates <= _day(as_of)].tolist()))

    def latest_session(self, as_of: date | str) -> str:
        eligible = self.dates[self.dates <= _day(as_of)]
        if eligible.size == 0:
            raise ValueError(f"no session on or before {as_of}")
        return str(eligible.max())

    def sessions(self, start: date | str | None = None, end: date | str | None = None) -> list[str]:
        mask = np.ones(self.dates.shape, dtype=bool)
        if start is not None:
            mask &= self.dates >= _day(start)
        if end is not None:
            mask &= self.dates <= _day(end)
        return [str(d) for d in np.unique(self.dates[mask])]


def close_on(panel: RawPanel, symbol: str, day: date | str) -> float:
    dates, closes = panel.series(symbol, day, day)
    if closes.size != 1:
        raise ValueError(f"{symbol} has {closes.size} bars on {day}")
    return float(closes[0])


def last_close(panel: RawPanel, symbol: str, as_of: date | str) -> tuple[str, float]:
    dates, closes = panel.series(symbol, None, as_of)
    if closes.size == 0:
        raise ValueError(f"{symbol} has no bar on or before {as_of}")
    return str(dates[-1]), float(closes[-1])


def period_return(panel: RawPanel, symbol: str, start: date | str, end: date | str) -> float:
    _, closes = panel.series(symbol, start, end)
    if closes.size < 2:
        raise ValueError(f"{symbol} needs two closes in {start}..{end}")
    return float(closes[-1] / closes[0] - 1.0)


def _log_returns(closes: np.ndarray) -> np.ndarray:
    return np.log(closes[1:]) - np.log(closes[:-1])


def realized_volatility(panel: RawPanel, symbol: str, window: int, end: date | str) -> float:
    _, closes = panel.series(symbol, None, end)
    if closes.size < window + 1:
        raise ValueError(f"{symbol} needs {window + 1} closes up to {end}")
    returns = _log_returns(closes[-(window + 1) :])
    mean = returns.sum() / returns.size
    variance = ((returns - mean) ** 2).sum() / (returns.size - 1)
    return float(math.sqrt(variance) * math.sqrt(ANNUALIZATION))


def correlation(panel: RawPanel, a: str, b: str, window: int, end: date | str) -> float:
    dates_a, closes_a = panel.series(a, None, end)
    dates_b, closes_b = panel.series(b, None, end)
    common, index_a, index_b = np.intersect1d(dates_a, dates_b, assume_unique=True, return_indices=True)
    if common.size < window + 1:
        raise ValueError(f"{a}/{b} share {common.size} closes up to {end}")
    x = _log_returns(closes_a[index_a][-(window + 1) :])
    y = _log_returns(closes_b[index_b][-(window + 1) :])
    dx, dy = x - x.mean(), y - y.mean()
    return float((dx * dy).sum() / math.sqrt((dx * dx).sum() * (dy * dy).sum()))


def max_drawdown(panel: RawPanel, symbol: str, start: date | str, end: date | str) -> float:
    """``min_{i<=j} close[j]/close[i] - 1`` via suffix minima (0 when never below a prior close)."""

    _, closes = panel.series(symbol, start, end)
    if closes.size < 2:
        raise ValueError(f"{symbol} needs two closes in {start}..{end}")
    suffix_min = np.minimum.accumulate(closes[::-1])[::-1]
    return float(min(0.0, float(np.min(suffix_min / closes - 1.0))))


def rank_by_return(panel: RawPanel, symbols: Sequence[str], start: date | str, end: date | str) -> tuple[str, ...]:
    """Symbols ordered by period return, highest first (ties broken by symbol)."""

    scored = [(period_return(panel, s, start, end), s) for s in symbols]
    return tuple(s for _, s in sorted(scored, key=lambda item: (-item[0], item[1])))


def top_by(values: dict[str, float], direction: str) -> str:
    """The symbol with the largest (``max``) or smallest (``min``) value; ties broken by symbol."""

    sign = -1.0 if direction == "max" else 1.0
    return sorted(values, key=lambda s: (sign * values[s], s))[0]
