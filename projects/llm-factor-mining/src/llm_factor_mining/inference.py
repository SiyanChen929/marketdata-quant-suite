"""Multiple-testing corrections and selection-adjusted performance statistics.

Conventions:

* p-values that are missing (``NaN``) are treated as ``1.0``: a hypothesis
  that could not be tested still counts toward the family size;
* ``m`` may exceed the number of supplied p-values.  The unsupplied
  hypotheses are treated as having ``p = 1`` (they can never be rejected but
  enlarge the family), which is how a search counts invalid or failed trials;
* Sharpe-type statistics are per period (not annualized).

References: Benjamini & Hochberg (1995); Benjamini & Yekutieli (2001);
Holm's step-down procedure; Bailey & Lopez de Prado (2014) for the deflated
Sharpe ratio.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy import stats


EULER_MASCHERONI = 0.5772156649015329
Alternative = Literal["two-sided", "greater", "less"]
Method = Literal["bh", "by", "holm"]


@dataclass(frozen=True)
class MultipleTestResult:
    """Outcome of a multiple-testing procedure, aligned with the input order."""

    method: str
    alpha: float
    m: int
    rejected: np.ndarray
    adjusted: np.ndarray

    @property
    def n_rejected(self) -> int:
        return int(self.rejected.sum())

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "m": self.m,
            "n_tested": int(self.rejected.size),
            "n_rejected": self.n_rejected,
            "rejected": [bool(value) for value in self.rejected],
            "adjusted": [float(value) for value in self.adjusted],
        }


def _prepare(pvalues: Sequence[float] | np.ndarray, m: int | None, alpha: float) -> tuple[np.ndarray, int]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    p = np.asarray(pvalues, dtype="float64").ravel().copy()
    p[np.isnan(p)] = 1.0
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    family = p.size if m is None else int(m)
    if family < p.size:
        raise ValueError(f"family size m={family} is smaller than the {p.size} supplied p-values")
    return p, family


def _step_up(p: np.ndarray, alpha: float, m: int, scale: float, method: str) -> MultipleTestResult:
    n = p.size
    if n == 0:
        return MultipleTestResult(method, alpha, m, np.zeros(0, dtype=bool), np.zeros(0))
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype="float64")
    passes = ranked <= alpha * ranks / (m * scale)
    k = int(np.flatnonzero(passes)[-1]) + 1 if passes.any() else 0
    adjusted_sorted = np.minimum.accumulate((m * scale * ranked / ranks)[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    rejected = np.zeros(n, dtype=bool)
    rejected[order[:k]] = True
    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted
    return MultipleTestResult(method, alpha, m, rejected, adjusted)


def benjamini_hochberg(
    pvalues: Sequence[float] | np.ndarray, alpha: float = 0.05, *, m: int | None = None
) -> MultipleTestResult:
    """Benjamini-Hochberg (1995) step-up FDR control (independence / PRDS)."""

    p, family = _prepare(pvalues, m, alpha)
    return _step_up(p, alpha, family, 1.0, "bh")


def benjamini_yekutieli(
    pvalues: Sequence[float] | np.ndarray, alpha: float = 0.05, *, m: int | None = None
) -> MultipleTestResult:
    """Benjamini-Yekutieli (2001) FDR control under arbitrary dependence."""

    p, family = _prepare(pvalues, m, alpha)
    harmonic = float(np.sum(1.0 / np.arange(1, family + 1))) if family else 1.0
    return _step_up(p, alpha, family, harmonic, "by")


def holm(
    pvalues: Sequence[float] | np.ndarray, alpha: float = 0.05, *, m: int | None = None
) -> MultipleTestResult:
    """Holm step-down family-wise error control."""

    p, family = _prepare(pvalues, m, alpha)
    n = p.size
    if n == 0:
        return MultipleTestResult("holm", alpha, family, np.zeros(0, dtype=bool), np.zeros(0))
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    multipliers = family - np.arange(n, dtype="float64")
    adjusted_sorted = np.minimum(np.maximum.accumulate(multipliers * ranked), 1.0)
    fails = ranked > alpha / multipliers
    k = int(np.flatnonzero(fails)[0]) if fails.any() else n
    rejected = np.zeros(n, dtype=bool)
    rejected[order[:k]] = True
    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted
    return MultipleTestResult("holm", alpha, family, rejected, adjusted)


def multiple_test(
    pvalues: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    *,
    method: Method = "bh",
    m: int | None = None,
) -> MultipleTestResult:
    """Dispatch to :func:`benjamini_hochberg`, :func:`benjamini_yekutieli` or :func:`holm`."""

    procedures = {"bh": benjamini_hochberg, "by": benjamini_yekutieli, "holm": holm}
    if method not in procedures:
        raise ValueError(f"unknown multiple-testing method {method!r}; use one of {sorted(procedures)}")
    return procedures[method](pvalues, alpha, m=m)


def t_to_p(t_stat: float, *, df: float | None = None, alternative: Alternative = "two-sided") -> float:
    """p-value of a t-statistic (normal reference when ``df`` is ``None``).

    HAC (Newey-West) t-statistics are conventionally referred to the normal
    distribution.  Non-finite statistics return ``NaN``.
    """

    if t_stat is None or not math.isfinite(float(t_stat)):
        return float("nan")
    t = float(t_stat)
    dist = stats.norm if df is None else stats.t(df)
    if alternative == "two-sided":
        return float(min(1.0, 2.0 * dist.sf(abs(t))))
    if alternative == "greater":
        return float(dist.sf(t))
    if alternative == "less":
        return float(dist.cdf(t))
    raise ValueError(f"unknown alternative {alternative!r}")


# --------------------------------------------------------------------------
# deflated Sharpe ratio (Bailey & Lopez de Prado, 2014)
# --------------------------------------------------------------------------


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true per-period Sharpe ratio exceeds ``benchmark``.

    ``Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2))`` with
    sample skewness ``g3`` and (non-excess) kurtosis ``g4``, as used by Bailey &
    Lopez de Prado (2014).  Returns ``NaN`` when undefined.
    """

    values = (sharpe, benchmark, skew, kurtosis)
    if n_obs < 2 or not all(math.isfinite(float(value)) for value in values):
        return float("nan")
    variance_term = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance_term <= 0.0:
        return float("nan")
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1.0) / math.sqrt(variance_term)
    return float(stats.norm.cdf(z))


def expected_maximum_sharpe(n_trials: int, sharpe_variance: float, *, mean: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratios under the null of no skill.

    ``E[max] ~ mean + sqrt(V) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))``
    (Bailey & Lopez de Prado, 2014), with the Euler-Mascheroni constant
    ``gamma``.  With a single trial there is no selection and the result is
    ``mean``.
    """

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if not math.isfinite(sharpe_variance) or sharpe_variance < 0.0:
        raise ValueError("sharpe_variance must be a finite non-negative number")
    if n_trials == 1:
        return float(mean)
    sd = math.sqrt(sharpe_variance)
    inv = stats.norm.ppf
    gamma = EULER_MASCHERONI
    return float(
        mean
        + sd * ((1.0 - gamma) * inv(1.0 - 1.0 / n_trials) + gamma * inv(1.0 - 1.0 / (n_trials * math.e)))
    )


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    *,
    n_trials: int,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe ratio: PSR against the expected maximum null Sharpe.

    ``sharpe_variance`` is the cross-trial variance of the (per-period) Sharpe
    ratios of all trials; ``n_trials`` the number of independent trials.
    """

    threshold = expected_maximum_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(
        sharpe, n_obs, benchmark=threshold, skew=skew, kurtosis=kurtosis
    )


def sample_moments(values: Sequence[float] | np.ndarray) -> tuple[float, float, float, int]:
    """Per-period ``(sharpe, skew, kurtosis, n)`` of a return-like series (NaNs dropped).

    Kurtosis is the non-excess (Pearson) value; a normal series gives 3.
    """

    x = np.asarray(values, dtype="float64").ravel()
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        nan = float("nan")
        return nan, nan, nan, n
    sd = float(x.std(ddof=1))
    sharpe = float(x.mean() / sd) if sd > 0 else float("nan")
    if sd > 0:
        skew = float(stats.skew(x, bias=True))
        kurt = float(stats.kurtosis(x, fisher=False, bias=True))
    else:
        skew = kurt = float("nan")
    return sharpe, skew, kurt, n


__all__ = [
    "EULER_MASCHERONI",
    "MultipleTestResult",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "holm",
    "multiple_test",
    "probabilistic_sharpe_ratio",
    "sample_moments",
    "t_to_p",
]
