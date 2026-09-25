"""Vectorized, causal evaluation of factor ASTs on a :class:`Panel`.

Semantics shared by every operator:

* rows are trading sessions in panel order; columns are symbols;
* rolling operators use ``min_periods = window`` (no partial windows), so a
  window touching a missing observation is NaN;
* cross-sectional operators read one date at a time;
* non-finite intermediate values (``inf``) become NaN;
* the final factor is masked to ``(date, symbol)`` cells that have a
  confirmed bar.

Results are memoized by the structural hash of the canonical subtree, so
shared subexpressions (``returns``, ``ts_std(returns, 20)``...) are computed
once per :class:`Evaluator`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import Panel
from ..dsl.canonical import canonicalize, structural_hash
from ..dsl.nodes import Call, Constant, Node, Terminal
from ..dsl.operators import OPERATORS, TERMINALS, ArgKind
from ..dsl.validate import DSLLimits, validate


DIV_EPS = 1.0e-12
DEGENERATE_REL_EPS = 1.0e-10


class ExpressionError(ValueError):
    """An expression failed validation and cannot be evaluated."""

    def __init__(self, message: str, codes: tuple[str, ...]) -> None:
        super().__init__(message)
        self.codes = codes


@dataclass(frozen=True)
class CacheInfo:
    hits: int
    misses: int
    size: int


# --------------------------------------------------------------------------
# numerical kernels (pure functions of aligned DataFrames)
# --------------------------------------------------------------------------


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype="float64", copy=True)
    values[~np.isfinite(values)] = np.nan
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def _rolling(frame: pd.DataFrame, window: int) -> pd.core.window.rolling.Rolling:
    return frame.rolling(window=window, min_periods=window)


def _shifted(values: np.ndarray, lag: int) -> np.ndarray:
    """Rows shifted down by ``lag`` (older values), NaN-padded at the top."""

    out = np.full_like(values, np.nan)
    if lag == 0:
        out[:] = values
    elif lag < values.shape[0]:
        out[lag:] = values[:-lag]
    return out


def _degenerate(std: pd.DataFrame, mean: pd.DataFrame) -> pd.DataFrame:
    scale = mean.abs().where(mean.abs() > 1.0, 1.0)
    return std <= DEGENERATE_REL_EPS * scale


def op_div(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x / y).where(y.abs() >= DIV_EPS)


def op_log(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x) * np.log1p(x.abs())


def op_sqrt(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x) * np.sqrt(x.abs())


def op_power(x: pd.DataFrame, exponent: float) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.sign(x) * x.abs() ** exponent


def op_delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(d)


def op_delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x - x.shift(d)


def op_ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _rolling(x, d).mean()


def op_ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _rolling(x, d).std(ddof=1)


def op_ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _rolling(x, d).sum()


def op_ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _rolling(x, d).min()


def op_ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _rolling(x, d).max()


def op_ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Average-tie rank of today's value among the last ``d`` values, divided by ``d``."""

    return _rolling(x, d).rank(method="average", ascending=True, pct=True)


def op_ts_zscore(x: pd.DataFrame, d: int) -> pd.DataFrame:
    rolling = _rolling(x, d)
    mean = rolling.mean()
    std = rolling.std(ddof=1)
    return ((x - mean) / std).where(~_degenerate(std, mean))


def _ts_arg_extreme(x: pd.DataFrame, d: int, *, maximum: bool) -> pd.DataFrame:
    values = x.to_numpy(dtype="float64")
    best = _shifted(values, 0)
    arg = np.zeros_like(values)
    invalid = np.isnan(best)
    for lag in range(1, d):
        older = _shifted(values, lag)
        invalid |= np.isnan(older)
        # strict comparison keeps the most recent position on ties; NaN compares False
        better = older > best if maximum else older < best
        best = np.where(better, older, best)
        arg = np.where(better, float(lag), arg)
    arg[invalid] = np.nan
    return pd.DataFrame(arg, index=x.index, columns=x.columns)


def op_ts_argmax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _ts_arg_extreme(x, d, maximum=True)


def op_ts_argmin(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _ts_arg_extreme(x, d, maximum=False)


def op_decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Weighted mean with weight ``d - k`` on the value ``k`` sessions ago."""

    values = x.to_numpy(dtype="float64")
    total = np.zeros_like(values)
    for lag in range(d):
        total += (d - lag) * _shifted(values, lag)
    return pd.DataFrame(total / (d * (d + 1) / 2.0), index=x.index, columns=x.columns)


def _joint(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    both = x.notna() & y.notna()
    return x.where(both), y.where(both)


def op_ts_cov(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    xj, yj = _joint(x, y)
    mean_xy = _rolling(xj * yj, d).mean()
    mean_x = _rolling(xj, d).mean()
    mean_y = _rolling(yj, d).mean()
    return (mean_xy - mean_x * mean_y) * (d / (d - 1.0))


def op_ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    xj, yj = _joint(x, y)
    cov = op_ts_cov(xj, yj, d)
    rx, ry = _rolling(xj, d), _rolling(yj, d)
    std_x, std_y = rx.std(ddof=1), ry.std(ddof=1)
    degenerate = _degenerate(std_x, rx.mean()) | _degenerate(std_y, ry.mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / (std_x * std_y)
    return corr.clip(-1.0, 1.0).where(~degenerate)


def op_cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, method="average", pct=True)


def op_cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    mean = x.mean(axis=1)
    std = x.std(axis=1, ddof=1)
    scale = mean.abs().where(mean.abs() > 1.0, 1.0)
    std = std.where(std > DEGENERATE_REL_EPS * scale)
    return x.sub(mean, axis=0).div(std, axis=0)


def op_cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0)


KERNELS: dict[str, Callable[..., pd.DataFrame]] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": op_div,
    "neg": lambda x: -x,
    "abs": lambda x: x.abs(),
    "sign": lambda x: np.sign(x),
    "log": op_log,
    "sqrt": op_sqrt,
    "power": op_power,
    "delay": op_delay,
    "delta": op_delta,
    "ts_mean": op_ts_mean,
    "ts_std": op_ts_std,
    "ts_sum": op_ts_sum,
    "ts_min": op_ts_min,
    "ts_max": op_ts_max,
    "ts_rank": op_ts_rank,
    "ts_zscore": op_ts_zscore,
    "ts_argmax": op_ts_argmax,
    "ts_argmin": op_ts_argmin,
    "decay_linear": op_decay_linear,
    "ts_corr": op_ts_corr,
    "ts_cov": op_ts_cov,
    "cs_rank": op_cs_rank,
    "cs_zscore": op_cs_zscore,
    "cs_demean": op_cs_demean,
}


def terminal_frame(panel: Panel, name: str) -> pd.DataFrame:
    """Materialize one data terminal from the panel."""

    if name in ("open", "high", "low", "close", "volume"):
        return panel.field(name).astype("float64")
    close = panel.close
    if name == "returns":
        return op_div(close, close.shift(1)) - 1.0
    if name == "vwap_proxy":
        return (panel.high + panel.low + close) / 3.0
    if name == "dollar_volume":
        return close * panel.volume
    raise KeyError(f"unknown terminal {name!r}")


# --------------------------------------------------------------------------
# evaluator
# --------------------------------------------------------------------------


class Evaluator:
    """Evaluate expressions on one panel with an LRU memo keyed by structural hash."""

    def __init__(
        self,
        panel: Panel,
        *,
        limits: DSLLimits | None = None,
        max_cache_entries: int = 64,
        mask_unavailable: bool = True,
    ) -> None:
        if max_cache_entries < 0:
            raise ValueError("max_cache_entries must be non-negative")
        self.panel = panel
        self.limits = limits or DSLLimits()
        self.max_cache_entries = int(max_cache_entries)
        self.mask_unavailable = bool(mask_unavailable)
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def cache_info(self) -> CacheInfo:
        return CacheInfo(self._hits, self._misses, len(self._cache))

    def clear_cache(self) -> None:
        self._cache.clear()

    def evaluate(self, expression: str | Node) -> pd.DataFrame:
        """Validate and evaluate; returns a new date x symbol float64 frame.

        Syntax and static errors both raise :class:`ExpressionError` whose
        ``codes`` are the validator's error codes (``PARSE`` for syntax).
        """

        result = validate(expression, self.limits)
        if not result.ok or result.node is None:
            detail = "; ".join(f"{issue.code}: {issue.message}" for issue in result.errors)
            raise ExpressionError(f"invalid expression: {detail}", result.codes)
        values = self._evaluate(canonicalize(result.node))
        if self.mask_unavailable:
            values = values.where(self.panel.available)
        return values.copy()

    def _evaluate(self, node: Node) -> pd.DataFrame:
        key = structural_hash(node)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            self._cache.move_to_end(key)
            return cached
        self._misses += 1
        values = _finite(self._compute(node))
        if self.max_cache_entries:
            self._cache[key] = values
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)
        return values

    def _compute(self, node: Node) -> pd.DataFrame:
        close = self.panel.close
        if isinstance(node, Terminal):
            if node.name not in TERMINALS:  # pragma: no cover - rejected by validate
                raise KeyError(node.name)
            return terminal_frame(self.panel, node.name)
        if isinstance(node, Constant):
            return pd.DataFrame(node.value, index=close.index, columns=close.columns, dtype="float64")
        assert isinstance(node, Call)
        spec = OPERATORS[node.op]
        args: list[object] = []
        for kind, arg in zip(spec.arg_kinds, node.args):
            if kind is ArgKind.SERIES:
                args.append(self._evaluate(arg))
            else:
                assert isinstance(arg, Constant)
                args.append(int(arg.value) if kind is ArgKind.WINDOW else float(arg.value))
        return KERNELS[node.op](*args)


def evaluate_expression(
    panel: Panel,
    expression: str | Node,
    *,
    limits: DSLLimits | None = None,
) -> pd.DataFrame:
    """One-shot convenience wrapper around :class:`Evaluator`."""

    return Evaluator(panel, limits=limits).evaluate(expression)
