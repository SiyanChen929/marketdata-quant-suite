"""Pure return, volatility, correlation, and drawdown arithmetic used by the tools.

Conventions (also stated in the tool descriptions):

* period return: close of the first session in ``[start, end]`` to the close of
  the last session in ``[start, end]``;
* realized volatility: sample standard deviation (``ddof=1``) of ``window``
  consecutive daily log returns, annualized by ``sqrt(252)``;
* correlation: Pearson correlation of daily log returns on dates where both
  series have a confirmed close;
* maximum drawdown: ``min_t close_t / max_{s<=t} close_s - 1`` (a number <= 0).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .errors import InsufficientDataError


TRADING_DAYS_PER_YEAR = 252


def _as_prices(closes: Sequence[float] | np.ndarray | pd.Series, minimum: int) -> np.ndarray:
    values = np.asarray(closes, dtype="float64")
    if values.ndim != 1:
        raise ValueError("closes must be one-dimensional")
    if len(values) < minimum:
        raise InsufficientDataError(f"need at least {minimum} closes, got {len(values)}")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise InsufficientDataError("closes must be finite and strictly positive")
    return values


def simple_return(closes: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Return ``last / first - 1``."""

    values = _as_prices(closes, 2)
    return float(values[-1] / values[0] - 1.0)


def log_returns(closes: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Return consecutive daily log returns."""

    values = _as_prices(closes, 2)
    return np.diff(np.log(values))


def realized_volatility(
    closes: Sequence[float] | np.ndarray | pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized sample volatility of the log returns implied by ``closes``."""

    returns = log_returns(_as_prices(closes, 3))
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def return_correlation(
    closes_a: Sequence[float] | np.ndarray | pd.Series,
    closes_b: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Pearson correlation of the log returns of two date-aligned close series."""

    a = log_returns(_as_prices(closes_a, 3))
    b = log_returns(_as_prices(closes_b, 3))
    if len(a) != len(b):
        raise ValueError("close series must be date-aligned and equally long")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        raise InsufficientDataError("correlation is undefined for a constant return series")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class Drawdown:
    """Maximum drawdown with positional indices into the input closes."""

    max_drawdown: float
    peak_index: int
    trough_index: int
    recovery_index: int | None


def max_drawdown(closes: Sequence[float] | np.ndarray | pd.Series) -> Drawdown:
    """Largest peak-to-trough decline, the peak that preceded it, and the recovery point.

    The peak is the first session at which the running maximum in force at the
    trough was reached; recovery is the first later session whose close is at
    least that peak close (``None`` if it never recovers in the sample).
    """

    values = _as_prices(closes, 2)
    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1.0
    trough = int(np.argmin(drawdowns))
    depth = float(drawdowns[trough])
    if depth == 0.0:
        return Drawdown(0.0, 0, 0, None)
    peak_value = running_max[trough]
    peak = int(np.flatnonzero(values[: trough + 1] == peak_value)[0])
    later = np.flatnonzero(values[trough + 1 :] >= peak_value)
    recovery = int(trough + 1 + later[0]) if later.size else None
    return Drawdown(depth, peak, trough, recovery)
