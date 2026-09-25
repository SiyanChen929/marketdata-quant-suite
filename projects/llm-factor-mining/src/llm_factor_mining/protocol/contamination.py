"""Look-ahead / memorization diagnostics for LLM-proposed factors.

Tools:

* :func:`split_ic_at_cutoff` compares out-of-sample IC before and after a
  language model's knowledge cutoff.  A factor whose edge exists only before
  the cutoff is consistent with memorized, period-specific knowledge rather
  than a persistent effect.  The cutoff date must come from the model
  provider's documentation; this module never assumes one.
* :func:`behavioural_novelty` (primary rediscovery measure) scores a factor by
  its *values*: ``1 - max |mean per-date rank correlation|`` against every
  entry of the reference catalog on the same (formation) data.  A re-spelling
  of a classic factor (``-ts_sum(returns, 5)`` for 5-day reversal) scores
  near 0 however different its syntax.
* :func:`library_novelty` (secondary, syntactic) measures structural distance
  from the catalog (maximum subtree Jaccard similarity).  It cannot tell a
  re-spelled classic factor from new structure and is reported only as a
  description of syntax.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from ..dsl.canonical import nearest_reference
from ..dsl.library import LIBRARY, library_nodes
from ..dsl.nodes import Node
from ..dsl.parser import parse
from ..dsl.validate import DSLLimits
from ..evaluate.engine import Evaluator
from ..evaluate.metrics import ICSummary, default_nw_lags, ic_summary, newey_west_se, rank_ic


@dataclass(frozen=True)
class CutoffComparison:
    """IC statistics on either side of a knowledge cutoff.

    ``pre`` uses signal dates whose forward return *ends* strictly before the
    cutoff session; ``post`` uses signal dates on or after it.  Signal dates
    whose return straddles the cutoff are dropped (``n_straddling``).
    ``diff_z`` is ``(mean_post - mean_pre) / sqrt(se_pre^2 + se_post^2)`` with
    Newey-West standard errors.
    """

    cutoff: str
    pre: ICSummary
    post: ICSummary
    n_straddling: int
    diff_mean: float
    diff_z: float

    def to_dict(self) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        return {
            "cutoff": self.cutoff,
            "pre": {key: clean(value) for key, value in asdict(self.pre).items()},
            "post": {key: clean(value) for key, value in asdict(self.post).items()},
            "n_straddling": self.n_straddling,
            "diff_mean": clean(self.diff_mean),
            "diff_z": clean(self.diff_z),
        }


def split_ic_at_cutoff(
    ic: pd.Series,
    cutoff: str | pd.Timestamp,
    calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    lag: int = 1,
    horizon: int = 1,
    nw_lags: int | None = None,
) -> CutoffComparison:
    """Split a per-date IC series at ``cutoff`` (see :class:`CutoffComparison`)."""

    if lag < 1 or horizon < 1:
        raise ValueError("lag and horizon must be at least 1")
    dates = pd.DatetimeIndex(calendar)
    cut = pd.Timestamp(cutoff)
    series = ic.dropna()
    positions = dates.get_indexer(series.index)
    if (positions < 0).any():
        raise ValueError("IC dates must belong to the calendar")
    cut_position = int(dates.searchsorted(cut, side="left"))
    end_positions = positions + lag + horizon
    pre_mask = end_positions < cut_position
    post_mask = positions >= cut_position
    pre = series[pre_mask]
    post = series[post_mask]
    pre_summary = ic_summary(pre, horizon=horizon, nw_lags=nw_lags)
    post_summary = ic_summary(post, horizon=horizon, nw_lags=nw_lags)
    se_pre = _se(pre, horizon, nw_lags)
    se_post = _se(post, horizon, nw_lags)
    diff = post_summary.mean - pre_summary.mean
    denominator = math.sqrt(se_pre**2 + se_post**2) if math.isfinite(se_pre) and math.isfinite(se_post) else float("nan")
    diff_z = diff / denominator if math.isfinite(diff) and denominator > 0 else float("nan")
    return CutoffComparison(
        cutoff=str(cut.date()),
        pre=pre_summary,
        post=post_summary,
        n_straddling=int((~pre_mask & ~post_mask).sum()),
        diff_mean=float(diff),
        diff_z=float(diff_z),
    )


def _se(values: pd.Series, horizon: int, nw_lags: int | None) -> float:
    lags = default_nw_lags(len(values), horizon) if nw_lags is None else nw_lags
    return newey_west_se(values.to_numpy(), lags)


@dataclass(frozen=True)
class Novelty:
    """Structural distance of one expression from a reference catalog.

    ``similarity`` is the maximum Jaccard similarity of subtree-hash sets
    against any reference (``abstract_similarity`` does the same with every
    numeric literal abstracted, so window variants count as the same idea);
    novelty is ``1 - similarity``.
    """

    expression: str
    nearest: str | None
    similarity: float
    abstract_nearest: str | None
    abstract_similarity: float

    @property
    def novelty(self) -> float:
        return 1.0 - self.similarity

    @property
    def abstract_novelty(self) -> float:
        return 1.0 - self.abstract_similarity

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "novelty": self.novelty,
            "abstract_novelty": self.abstract_novelty,
        }


def library_novelty(
    expression: str | Node,
    references: Iterable[tuple[str, Node]] | None = None,
) -> Novelty:
    """Novelty of ``expression`` against ``references`` (default: the classic library)."""

    node = parse(expression) if isinstance(expression, str) else expression
    if references is None:
        pairs = list(zip((factor.name for factor in LIBRARY), library_nodes()))
    else:
        pairs = list(references)
    names = [name for name, _ in pairs]
    nodes = [ref for _, ref in pairs]
    concrete = nearest_reference(node, nodes)
    abstract = nearest_reference(node, nodes, abstract_literals=True)
    text = expression if isinstance(expression, str) else str(expression)
    return Novelty(
        expression=text,
        nearest=None if concrete.index is None else names[concrete.index],
        similarity=float(concrete.similarity),
        abstract_nearest=None if abstract.index is None else names[abstract.index],
        abstract_similarity=float(abstract.similarity),
    )


@dataclass(frozen=True)
class BehaviouralNovelty:
    """Value-based distance of one factor from a reference catalog.

    ``correlations`` holds the mean per-date Spearman correlation with each
    reference on the scored dates; ``novelty = 1 - max |correlation|``
    (``NaN`` when no correlation is defined).
    """

    nearest: str | None
    max_abs_corr: float
    correlations: Mapping[str, float]

    @property
    def novelty(self) -> float:
        return 1.0 - self.max_abs_corr if math.isfinite(self.max_abs_corr) else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "nearest": self.nearest,
            "max_abs_corr": None if not math.isfinite(self.max_abs_corr) else self.max_abs_corr,
            "novelty": None if not math.isfinite(self.novelty) else self.novelty,
        }


def library_signals(evaluator: Evaluator) -> dict[str, pd.DataFrame]:
    """Values of every reference-catalog entry on ``evaluator``'s panel."""

    return {factor.name: evaluator.evaluate(factor.node) for factor in LIBRARY}


def library_signals_for(panel: Any, *, limits: DSLLimits | None = None) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: evaluate the catalog on ``panel``."""

    return library_signals(Evaluator(panel, limits=limits))


def behavioural_novelty(
    values: pd.DataFrame,
    references: Mapping[str, pd.DataFrame],
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    min_names: int = 10,
) -> BehaviouralNovelty:
    """``1 - max_ref |mean_t rank_corr(values_t, ref_t)|`` over ``dates``."""

    chosen = pd.DatetimeIndex(dates)
    left = values.loc[chosen]
    correlations: dict[str, float] = {}
    for name, reference in references.items():
        series = rank_ic(left, reference.loc[chosen], min_names=min_names).dropna()
        correlations[name] = float(series.mean()) if len(series) else float("nan")
    finite = {name: value for name, value in correlations.items() if math.isfinite(value)}
    if not finite:
        return BehaviouralNovelty(None, float("nan"), correlations)
    nearest = max(sorted(finite), key=lambda name: abs(finite[name]))
    return BehaviouralNovelty(nearest, abs(finite[nearest]), correlations)


def mean_novelty(expressions: Iterable[str | Node], *, abstract: bool = False) -> float:
    """Average (concrete or abstract) novelty; ``NaN`` for an empty set."""

    scores = [
        library_novelty(expression).abstract_novelty if abstract else library_novelty(expression).novelty
        for expression in expressions
    ]
    return float(np.mean(scores)) if scores else float("nan")


__all__ = [
    "BehaviouralNovelty",
    "CutoffComparison",
    "Novelty",
    "behavioural_novelty",
    "library_signals",
    "library_signals_for",
    "library_novelty",
    "mean_novelty",
    "split_ic_at_cutoff",
]
