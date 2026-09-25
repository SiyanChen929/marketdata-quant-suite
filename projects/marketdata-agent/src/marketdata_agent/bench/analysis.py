"""Pre-registered statistics for the model study (standard library and numpy only).

* :func:`threshold_decision`: the plan's Wilson threshold rule, over distinct items;
* :func:`all_repetitions_success`: collapse repeated episodes to one success per
  distinct item (success only if every repetition succeeds);
* :func:`mcnemar_exact`: exact McNemar test for paired binary outcomes (H2c, H4);
* :func:`benjamini_hochberg` and :func:`holm`: multiplicity control;
* :func:`cluster_bootstrap_ci`: percentile bootstrap that resamples clusters;
* power helpers: :func:`min_successes`, :func:`power_threshold` (independent
  items), :func:`power_threshold_clustered` (beta-binomial clusters with a given
  intra-cluster correlation, by simulation) and :func:`noninferiority_paired_n`.

None of these has been applied to model results yet; they are unit-tested on
known values (``tests/test_analysis.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Literal

import numpy as np

from .scoring import Z_95, wilson_interval


Decision = Literal["supported", "not_supported", "inconclusive"]


def threshold_decision(successes: int, n: int, threshold: float, *, direction: str = "at_least") -> Decision:
    """Apply the Wilson 95% threshold rule.

    ``at_least`` (a rate must reach ``threshold``): supported when the lower
    bound is at or above it, not supported when the upper bound is below it.
    ``at_most`` (a rate must stay under ``threshold``, e.g. over-refusal):
    supported when the upper bound is at or below it, not supported when the
    lower bound is above it. Otherwise inconclusive.
    """

    low, high = wilson_interval(successes, n)
    if low is None or high is None:
        return "inconclusive"
    if direction == "at_least":
        if low >= threshold:
            return "supported"
        return "not_supported" if high < threshold else "inconclusive"
    if direction == "at_most":
        if high <= threshold:
            return "supported"
        return "not_supported" if low > threshold else "inconclusive"
    raise ValueError("direction must be 'at_least' or 'at_most'")


def all_repetitions_success(outcomes: Iterable[tuple[str, bool]]) -> dict[str, bool]:
    """``{item_key: True}`` iff every recorded repetition of the item succeeded."""

    collapsed: dict[str, bool] = {}
    for key, success in outcomes:
        collapsed[key] = collapsed.get(key, True) and bool(success)
    return collapsed


def _binomial_pmf(n: int, k: int, p: float) -> float:
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    log = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)
    return math.exp(log)


def binomial_tail(n: int, k: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``."""

    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(_binomial_pmf(n, j, p) for j in range(k, n + 1)))


def mcnemar_exact(b: int, c: int, *, alternative: str = "two-sided") -> float:
    """Exact McNemar p-value from the discordant counts ``b`` (only arm 1 succeeds) and ``c`` (only arm 2).

    ``alternative="greater"`` tests whether ``b`` exceeds ``c``.
    """

    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    if alternative == "greater":
        return binomial_tail(n, b, 0.5)
    if alternative != "two-sided":
        raise ValueError("alternative must be 'two-sided' or 'greater'")
    k = min(b, c)
    return min(1.0, 2.0 * sum(_binomial_pmf(n, j, 0.5) for j in range(0, k + 1)))


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.05) -> list[bool]:
    """Rejections under the Benjamini-Hochberg step-up procedure at false discovery rate ``q``."""

    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    cutoff = -1
    for rank, index in enumerate(order, start=1):
        if pvalues[index] <= q * rank / m:
            cutoff = rank
    rejected = [False] * m
    for rank, index in enumerate(order, start=1):
        rejected[index] = rank <= cutoff
    return rejected


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Rejections under the Holm step-down procedure (family-wise error ``alpha``)."""

    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    rejected = [False] * m
    for rank, index in enumerate(order):
        if pvalues[index] > alpha / (m - rank):
            break
        rejected[index] = True
    return rejected


def cluster_bootstrap_ci(
    values_by_cluster: Mapping[str, Sequence[float]],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean of all values and a percentile interval from resampling whole clusters with replacement."""

    clusters = [np.asarray(v, dtype="float64") for v in values_by_cluster.values() if len(v)]
    if not clusters:
        raise ValueError("no values")
    sums = np.array([c.sum() for c in clusters])
    sizes = np.array([c.size for c in clusters], dtype="float64")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    tail = (1.0 - level) / 2.0
    return float(sums.sum() / sizes.sum()), float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def min_successes(n: int, threshold: float) -> int | None:
    """Smallest ``k`` whose Wilson lower bound is at or above ``threshold`` (``None`` if even ``n`` fails)."""

    for k in range(n + 1):
        low, _ = wilson_interval(k, n)
        if low is not None and low >= threshold:
            return k
    return None


def power_threshold(n: int, threshold: float, q: float) -> float:
    """``P(supported)`` for ``n`` independent items that each succeed with probability ``q``."""

    k = min_successes(n, threshold)
    return 0.0 if k is None else binomial_tail(n, k, q)


def max_failures_at_most(n: int, threshold: float) -> int | None:
    """Largest ``k`` whose Wilson upper bound is at or below ``threshold`` (``None`` if even 0 fails)."""

    best = None
    for k in range(n + 1):
        _, high = wilson_interval(k, n)
        if high is not None and high <= threshold:
            best = k
        else:
            break
    return best


def power_at_most(n: int, threshold: float, f: float) -> float:
    """``P(supported)`` for an ``at_most`` hypothesis when each of ``n`` items fails with probability ``f``."""

    k = max_failures_at_most(n, threshold)
    if k is None:
        return 0.0
    return 1.0 - binomial_tail(n, k + 1, f)


def power_threshold_clustered(
    cluster_sizes: Sequence[int],
    threshold: float,
    q: float,
    icc: float,
    *,
    n_sim: int = 20_000,
    seed: int = 0,
) -> float:
    """``P(supported)`` when items share a cluster-level success probability.

    Each cluster draws ``p ~ Beta`` with mean ``q`` and intra-cluster
    correlation ``icc`` (``alpha + beta = (1 - icc) / icc``); its items then
    succeed independently with probability ``p``. ``icc = 0`` reduces to
    :func:`power_threshold`.
    """

    sizes = np.asarray([int(s) for s in cluster_sizes if int(s) > 0])
    n = int(sizes.sum())
    k = min_successes(n, threshold)
    if k is None:
        return 0.0
    if icc <= 0.0:
        return power_threshold(n, threshold, q)
    if q >= 1.0:
        return 1.0
    strength = (1.0 - icc) / icc
    rng = np.random.default_rng(seed)
    p = rng.beta(q * strength, (1.0 - q) * strength, size=(n_sim, sizes.size))
    successes = rng.binomial(np.broadcast_to(sizes, p.shape), p).sum(axis=1)
    return float(np.mean(successes >= k))


def noninferiority_paired_n(discordance: float, margin: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items needed for a one-sided paired non-inferiority test when the true difference is zero.

    Normal approximation: the paired difference of success rates has variance
    ``discordance / n``, so ``n = discordance * (z_{1-alpha} + z_{power})^2 / margin^2``.
    """

    if not 0.0 < discordance <= 1.0 or margin <= 0.0:
        raise ValueError("discordance must be in (0, 1] and margin positive")
    z_alpha = _z(1.0 - alpha)
    z_beta = _z(power)
    return int(math.ceil(discordance * (z_alpha + z_beta) ** 2 / margin**2))


def _z(p: float) -> float:
    """Standard-normal quantile by bisection on ``erf`` (accurate to about 1e-10)."""

    low, high = -10.0, 10.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


__all__ = [
    "Z_95",
    "all_repetitions_success",
    "benjamini_hochberg",
    "binomial_tail",
    "cluster_bootstrap_ci",
    "holm",
    "max_failures_at_most",
    "mcnemar_exact",
    "min_successes",
    "noninferiority_paired_n",
    "power_at_most",
    "power_threshold",
    "power_threshold_clustered",
    "threshold_decision",
]
