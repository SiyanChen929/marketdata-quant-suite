"""Leakage-aware factor metrics.

Timing convention (the only one used anywhere in this package): a signal
observed at the close of session ``t`` is traded at the close of session
``t + lag`` (``lag >= 1``) and earns the close-to-close return from
``t + lag`` to ``t + lag + h``::

    fwd_h[t] = close[t + lag + h] / close[t + lag] - 1

Every metric series is indexed by the *signal* date ``t``.

Windows (formation / validation / test) are evaluated without recomputation
leakage: the factor is computed once on the full panel (it is causal), and
metrics are then restricted to the window's signal dates whose forward-return
end date ``t + lag + h`` still lies inside the window.  Equivalently, the last
``lag + h`` sessions of every window are embargoed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# configuration and windows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricConfig:
    """Evaluation settings; ``cost_bps`` is charged per unit of one-way turnover."""

    horizon: int = 1
    lag: int = 1
    n_quantiles: int = 5
    min_names: int = 10
    cost_bps: float = 10.0
    nw_lags: int | None = None
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        _check_timing(self.horizon, self.lag)
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be at least 2")
        if self.min_names < 3:
            raise ValueError("min_names must be at least 3")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.periods_per_year < 1:
            raise ValueError("periods_per_year must be positive")
        if self.nw_lags is not None and self.nw_lags < self.horizon - 1:
            raise ValueError("nw_lags must be at least horizon - 1 for overlapping returns")

    @property
    def effective_min_names(self) -> int:
        return max(self.min_names, self.n_quantiles)


@dataclass(frozen=True)
class EvaluationWindow:
    """Inclusive calendar bounds of one evaluation period (``None`` = open)."""

    name: str = "full"
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        start = None if self.start is None else pd.Timestamp(self.start)
        end = None if self.end is None else pd.Timestamp(self.end)
        if start is not None and end is not None and start > end:
            raise ValueError(f"window {self.name!r} starts after it ends")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


def _check_timing(horizon: int, lag: int) -> None:
    if int(lag) != lag or lag < 1:
        raise ValueError(
            "lag must be an integer >= 1: a signal observed at close t cannot trade before t+1"
        )
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be an integer >= 1")


def forward_returns(close: pd.DataFrame, horizon: int = 1, lag: int = 1) -> pd.DataFrame:
    """Return ``close[t+lag+h] / close[t+lag] - 1`` indexed by signal date ``t``."""

    _check_timing(horizon, lag)
    entry = close.shift(-lag)
    exit_ = close.shift(-(lag + horizon))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (exit_ / entry - 1.0).where(entry > 0)
    return result


def embargoed_signal_dates(
    calendar: pd.DatetimeIndex,
    window: EvaluationWindow | None = None,
    *,
    lag: int,
    horizon: int,
) -> pd.DatetimeIndex:
    """Signal dates in ``window`` whose forward return ends inside the window.

    Session positions refer to ``calendar`` (the panel's trading dates).  A
    signal at position ``p`` is kept only if ``p + lag + horizon`` is at or
    before the last calendar position inside the window.
    """

    _check_timing(horizon, lag)
    dates = pd.DatetimeIndex(calendar)
    window = window or EvaluationWindow()
    inside = np.ones(len(dates), dtype=bool)
    if window.start is not None:
        inside &= dates >= window.start
    if window.end is not None:
        inside &= dates <= window.end
    positions = np.flatnonzero(inside)
    if positions.size == 0:
        return dates[:0]
    last = int(positions[-1])
    keep = positions[positions + lag + horizon <= last]
    return dates[keep]


def restrict(data: pd.DataFrame | pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame | pd.Series:
    """Rows of ``data`` on ``dates`` (missing dates are dropped, not filled)."""

    return data.loc[data.index.intersection(dates)]


# --------------------------------------------------------------------------
# information coefficient
# --------------------------------------------------------------------------


def _row_mask(frame: pd.DataFrame, keep: pd.Series) -> pd.DataFrame:
    factor = np.where(keep.to_numpy(dtype=bool), 1.0, np.nan)
    return frame.mul(factor, axis=0)


def rank_ic(signal: pd.DataFrame, forward: pd.DataFrame, *, min_names: int = 10) -> pd.Series:
    """Per-date Spearman rank correlation between signal and forward return.

    Only names with both values present are used; dates with fewer than
    ``min_names`` such names, or with no rank variation, are NaN.
    """

    both = signal.notna() & forward.notna()
    x = signal.where(both).rank(axis=1, method="average")
    y = forward.where(both).rank(axis=1, method="average")
    x = x.sub(x.mean(axis=1), axis=0)
    y = y.sub(y.mean(axis=1), axis=0)
    numerator = (x * y).sum(axis=1)
    denominator = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = numerator / denominator
    ic = ic.where((both.sum(axis=1) >= min_names) & (denominator > 0))
    ic.name = "rank_ic"
    return ic


def default_nw_lags(n_obs: int, horizon: int = 1) -> int:
    """``max(h - 1, floor(4 (T/100)^(2/9)))``: overlap-aware Newey-West bandwidth."""

    automatic = int(math.floor(4.0 * (max(n_obs, 0) / 100.0) ** (2.0 / 9.0))) if n_obs > 0 else 0
    return max(int(horizon) - 1, automatic)


def newey_west_se(values: np.ndarray | pd.Series | list[float], lags: int) -> float:
    """Newey-West (1987) HAC standard error of the sample mean, Bartlett kernel.

    Non-finite observations are dropped and the remaining ones treated as
    consecutive.  ``lags = 0`` gives ``sqrt(mean((x - xbar)^2) / T)``.
    """

    if lags < 0:
        raise ValueError("lags must be non-negative")
    x = np.asarray(values, dtype="float64").ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan")
    u = x - x.mean()
    long_run = float(u @ u) / n
    for lag in range(1, min(int(lags), n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run += 2.0 * weight * float(u[lag:] @ u[:-lag]) / n
    return math.sqrt(max(long_run, 0.0) / n)


@dataclass(frozen=True)
class ICSummary:
    n: int
    mean: float
    std: float
    icir: float
    t_nw: float
    t_iid: float
    hit_rate: float
    nw_lags: int


def ic_summary(ic: pd.Series, *, horizon: int = 1, nw_lags: int | None = None) -> ICSummary:
    """Mean, dispersion, ICIR, HAC t-statistic and hit rate of an IC series."""

    values = ic.dropna().to_numpy(dtype="float64")
    n = int(values.size)
    lags = default_nw_lags(n, horizon) if nw_lags is None else int(nw_lags)
    if lags < horizon - 1:
        raise ValueError("nw_lags must be at least horizon - 1")
    if n == 0:
        nan = float("nan")
        return ICSummary(0, nan, nan, nan, nan, nan, nan, lags)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    se = newey_west_se(values, lags)
    return ICSummary(
        n=n,
        mean=mean,
        std=std,
        icir=_ratio(mean, std),
        t_nw=_ratio(mean, se),
        t_iid=_ratio(mean, std / math.sqrt(n)) if n > 1 else float("nan"),
        hit_rate=float((values > 0).mean()),
        nw_lags=lags,
    )


# --------------------------------------------------------------------------
# quantiles, turnover and the cost-adjusted long-short portfolio
# --------------------------------------------------------------------------


def quantile_buckets(signal: pd.DataFrame, n_quantiles: int = 5, *, min_names: int = 10) -> pd.DataFrame:
    """Per-date bucket labels 1..n_quantiles (1 = lowest signal); NaN if excluded.

    Bucket ``q`` holds average-tie ranks in ``((q-1) n/Q, q n/Q]``; dates with
    fewer than ``max(min_names, Q)`` valid names are excluded.
    """

    counts = signal.notna().sum(axis=1)
    ranks = signal.rank(axis=1, method="average")
    buckets = np.ceil(ranks.mul(float(n_quantiles)).div(counts.replace(0, np.nan), axis=0))
    buckets = buckets.clip(lower=1, upper=n_quantiles)
    return _row_mask(buckets, counts >= max(min_names, n_quantiles))


def quantile_returns(buckets: pd.DataFrame, forward: pd.DataFrame, n_quantiles: int) -> pd.DataFrame:
    """Equal-weighted mean forward return of each bucket, per date."""

    columns = {}
    for quantile in range(1, n_quantiles + 1):
        columns[quantile] = forward.where(buckets.eq(quantile)).mean(axis=1)
    frame = pd.DataFrame(columns, index=forward.index)
    frame.columns.name = "quantile"
    return frame


def top_quantile_turnover(buckets: pd.DataFrame, n_quantiles: int) -> pd.Series:
    """Share of today's top-bucket names that were not in yesterday's top bucket."""

    top = buckets.eq(n_quantiles)
    previous = top.shift(1, fill_value=False)
    n_top = top.sum(axis=1)
    overlap = (top & previous).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        turnover = 1.0 - overlap / n_top
    turnover = turnover.where((n_top > 0) & (previous.sum(axis=1) > 0))
    turnover.name = "top_quantile_turnover"
    return turnover.astype("float64")


def long_short_weights(signal: pd.DataFrame, n_quantiles: int = 5, *, min_names: int = 10) -> pd.DataFrame:
    """Formation weights: +1 spread over the top bucket, -1 over the bottom bucket.

    Formation uses signal availability only (never future-return
    availability).  Dates without both legs are flat.
    """

    buckets = quantile_buckets(signal, n_quantiles, min_names=min_names)
    top = buckets.eq(n_quantiles).astype("float64")
    bottom = buckets.eq(1).astype("float64")
    n_top = top.sum(axis=1)
    n_bottom = bottom.sum(axis=1)
    weights = top.div(n_top.replace(0, np.nan), axis=0) - bottom.div(n_bottom.replace(0, np.nan), axis=0)
    active = (n_top > 0) & (n_bottom > 0)
    return _row_mask(weights, active).fillna(0.0)


def holding_weights(formation: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Overlapping-portfolio weights: average of the last ``horizon`` formations."""

    return formation.rolling(window=int(horizon), min_periods=1).sum() / float(horizon)


def long_short_backtest(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    config: MetricConfig | None = None,
) -> pd.DataFrame:
    """Daily cost-adjusted returns of the overlapping top-minus-bottom portfolio.

    Weights held on signal date ``t`` earn ``close[t+lag+1] / close[t+lag] - 1``;
    trades implied by ``|w_t - w_{t-1}|`` execute at ``close[t+lag]`` and cost
    ``cost_bps`` per unit traded.  Weight drift between rebalances is ignored.
    A held name with no next-session return contributes zero and is reported
    in ``missing_return_weight``.
    """

    config = config or MetricConfig()
    formation = long_short_weights(
        signal, config.n_quantiles, min_names=config.effective_min_names
    )
    weights = holding_weights(formation, config.horizon)
    daily = forward_returns(close, 1, config.lag)
    gross = (weights * daily.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    if len(weights):
        turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    cost = turnover * (config.cost_bps * 1.0e-4)
    missing = (weights.abs() * daily.isna()).sum(axis=1)
    return pd.DataFrame(
        {
            "gross": gross,
            "turnover": turnover,
            "cost": cost,
            "net": gross - cost,
            "gross_exposure": weights.abs().sum(axis=1),
            "missing_return_weight": missing,
        },
        index=signal.index,
    )


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorReport:
    """Headline metrics of one factor over one embargoed evaluation window.

    ``ic_*`` statistics use the per-date rank IC at horizon ``h`` (ICIR is
    per period, not annualized; ``ic_tstat_nw`` uses ``nw_lags`` Bartlett
    lags).  ``quantile_mean_returns`` and ``long_short_*`` use equal-weighted
    buckets of names with both a signal and an ``h``-session forward return.
    ``portfolio_turnover`` and ``ls_*`` describe the daily overlapping
    top-minus-bottom portfolio of :func:`long_short_backtest` (gross 2,
    ``cost_bps`` per unit one-way turnover, annualized with
    ``periods_per_year``).
    """

    expression: str | None
    expression_hash: str | None
    window: str
    window_start: str | None
    window_end: str | None
    first_signal_date: str | None
    last_signal_date: str | None
    horizon: int
    lag: int
    n_quantiles: int
    min_names: int
    cost_bps: float
    n_signal_dates: int
    n_ic_dates: int
    ic_mean: float
    ic_std: float
    icir: float
    ic_tstat_nw: float
    ic_tstat_iid: float
    ic_hit_rate: float
    nw_lags: int
    coverage: float
    quantile_mean_returns: tuple[float, ...]
    quantile_monotonicity: float
    long_short_mean: float
    long_short_tstat_nw: float
    top_quantile_turnover: float
    portfolio_turnover: float
    ls_gross_ann_return: float
    ls_net_ann_return: float
    ls_net_ann_vol: float
    ls_gross_sharpe: float
    ls_net_sharpe: float
    panel_sha256: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary; NaN and infinities become ``None``."""

        return _json_clean(asdict(self))


@dataclass(frozen=True)
class FactorEvaluation:
    """Report plus the underlying per-date series (window-restricted)."""

    report: FactorReport
    ic: pd.Series
    long_short_spread: pd.Series
    quantile_returns: pd.DataFrame
    portfolio: pd.DataFrame


@dataclass(frozen=True)
class _FullPanelSeries:
    """Per-date series computed once on the full panel, restricted per window."""

    calendar: pd.DatetimeIndex
    ic: pd.Series
    quantile_returns: pd.DataFrame
    top_turnover: pd.Series
    portfolio: pd.DataFrame
    labelled: pd.Series  # names with a forward return, per date
    covered: pd.Series  # names with both a signal and a forward return, per date


def _prepare(signal: pd.DataFrame, close: pd.DataFrame, config: MetricConfig) -> _FullPanelSeries:
    if not signal.index.equals(close.index) or not signal.columns.equals(close.columns):
        raise ValueError("signal and close must share the same dates and symbols")
    q, min_names = config.n_quantiles, config.effective_min_names
    forward = forward_returns(close, config.horizon, config.lag)
    joint_buckets = quantile_buckets(signal.where(forward.notna()), q, min_names=min_names)
    formation_buckets = quantile_buckets(signal, q, min_names=min_names)
    return _FullPanelSeries(
        calendar=pd.DatetimeIndex(close.index),
        ic=rank_ic(signal, forward, min_names=min_names),
        quantile_returns=quantile_returns(joint_buckets, forward, q),
        top_turnover=top_quantile_turnover(formation_buckets, q),
        portfolio=long_short_backtest(signal, close, config),
        labelled=forward.notna().sum(axis=1),
        covered=(signal.notna() & forward.notna()).sum(axis=1),
    )


def _window_evaluation(
    full: _FullPanelSeries,
    config: MetricConfig,
    window: EvaluationWindow,
    *,
    expression: str | None,
    expression_hash: str | None,
    panel_sha256: str | None,
) -> FactorEvaluation:
    h, lag, q = config.horizon, config.lag, config.n_quantiles
    dates = embargoed_signal_dates(full.calendar, window, lag=lag, horizon=h)

    ic = restrict(full.ic, dates).dropna()
    summary = ic_summary(ic, horizon=h, nw_lags=config.nw_lags)

    by_quantile = restrict(full.quantile_returns, dates)
    spread = (by_quantile[q] - by_quantile[1]).dropna()
    spread.name = "long_short_spread"
    spread_lags = default_nw_lags(len(spread), h) if config.nw_lags is None else config.nw_lags
    quantile_means = tuple(float(by_quantile[col].mean()) for col in by_quantile.columns)

    top_turn = restrict(full.top_turnover, dates)
    portfolio = restrict(full.portfolio, dates)
    labelled = int(restrict(full.labelled, dates).sum())
    covered = int(restrict(full.covered, dates).sum())

    ppy = config.periods_per_year
    net, gross = portfolio["net"], portfolio["gross"]
    spread_t = (
        _ratio(float(spread.mean()), newey_west_se(spread.to_numpy(), spread_lags))
        if len(spread) > 1
        else float("nan")
    )
    report = FactorReport(
        expression=expression,
        expression_hash=expression_hash,
        window=window.name,
        window_start=_date_text(window.start),
        window_end=_date_text(window.end),
        first_signal_date=_date_text(dates[0]) if len(dates) else None,
        last_signal_date=_date_text(dates[-1]) if len(dates) else None,
        horizon=h,
        lag=lag,
        n_quantiles=q,
        min_names=config.effective_min_names,
        cost_bps=float(config.cost_bps),
        n_signal_dates=int(len(dates)),
        n_ic_dates=summary.n,
        ic_mean=summary.mean,
        ic_std=summary.std,
        icir=summary.icir,
        ic_tstat_nw=summary.t_nw,
        ic_tstat_iid=summary.t_iid,
        ic_hit_rate=summary.hit_rate,
        nw_lags=summary.nw_lags,
        coverage=covered / labelled if labelled else float("nan"),
        quantile_mean_returns=quantile_means,
        quantile_monotonicity=_monotonicity(quantile_means),
        long_short_mean=float(spread.mean()) if len(spread) else float("nan"),
        long_short_tstat_nw=spread_t,
        top_quantile_turnover=float(top_turn.mean()) if top_turn.notna().any() else float("nan"),
        portfolio_turnover=float(portfolio["turnover"].mean()) if len(portfolio) else float("nan"),
        ls_gross_ann_return=float(gross.mean() * ppy) if len(gross) else float("nan"),
        ls_net_ann_return=float(net.mean() * ppy) if len(net) else float("nan"),
        ls_net_ann_vol=float(net.std(ddof=1) * math.sqrt(ppy)) if len(net) > 1 else float("nan"),
        ls_gross_sharpe=_sharpe(gross, ppy),
        ls_net_sharpe=_sharpe(net, ppy),
        panel_sha256=panel_sha256,
    )
    return FactorEvaluation(report, ic, spread, by_quantile, portfolio)


def evaluate_signal(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    *,
    config: MetricConfig | None = None,
    window: EvaluationWindow | None = None,
    expression: str | None = None,
    expression_hash: str | None = None,
    panel_sha256: str | None = None,
) -> FactorEvaluation:
    """Evaluate a full-panel causal ``signal`` on one embargoed window."""

    config = config or MetricConfig()
    return _window_evaluation(
        _prepare(signal, close, config),
        config,
        window or EvaluationWindow(),
        expression=expression,
        expression_hash=expression_hash,
        panel_sha256=panel_sha256,
    )


def evaluate_windows(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    windows: Iterable[EvaluationWindow],
    *,
    config: MetricConfig | None = None,
    expression: str | None = None,
    expression_hash: str | None = None,
    panel_sha256: str | None = None,
) -> dict[str, FactorEvaluation]:
    """Evaluate one causal signal on several windows (names must be unique).

    Full-panel series are computed once; each window only restricts them.
    """

    config = config or MetricConfig()
    windows = list(windows)
    names = [window.name for window in windows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate window names: {duplicates}")
    full = _prepare(signal, close, config)
    return {
        window.name: _window_evaluation(
            full,
            config,
            window,
            expression=expression,
            expression_hash=expression_hash,
            panel_sha256=panel_sha256,
        )
        for window in windows
    }


def _ratio(numerator: float, denominator: float) -> float:
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or denominator <= 0:
        return float("nan")
    return numerator / denominator


def _sharpe(returns: pd.Series, periods_per_year: int) -> float:
    if len(returns) < 2:
        return float("nan")
    return _ratio(float(returns.mean()), float(returns.std(ddof=1))) * math.sqrt(periods_per_year)


def _monotonicity(values: tuple[float, ...]) -> float:
    """Spearman correlation between bucket index and bucket mean return."""

    array = np.asarray(values, dtype="float64")
    if array.size < 2 or not np.isfinite(array).all():
        return float("nan")
    ranks = pd.Series(array).rank().to_numpy()
    index = np.arange(1, array.size + 1, dtype="float64")
    if np.std(ranks) == 0:
        return float("nan")
    return float(np.corrcoef(index, ranks)[0, 1])


def _date_text(value: pd.Timestamp | None) -> str | None:
    return None if value is None else str(pd.Timestamp(value).date())


def _json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        return _json_clean(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    return value
