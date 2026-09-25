"""Synthetic markets with planted formulaic alphas, and recovery metrics.

Data-generating process (all draws from one seeded generator)::

    r[t, i] = beta_i * m[t] + sigma_i * eps[t, i]
              + s * sum_k w_k * z_k[t - lag - 1, i]
    z_k[t]  = cs_zscore(g_k(bars up to t))            (a DSL expression)
    s       = snr * mean_i(sigma_i),   sum_k w_k^2 = 1

so a signal observed at the close of ``t`` predicts the close-to-close return
earned by a position entered at the close of ``t + lag`` - exactly what the
evaluator scores under the suite's execution-lag rule.  ``snr`` is the ratio of
the cross-sectional dispersion of the planted return component to the average
idiosyncratic volatility.  Volume follows a log-AR(1) with a common component,
independent of returns; open/high/low are noise around the close path.

Because ``z_k`` depends on prices that themselves contain the planted
component, the path is solved as a fixed point: starting from the alpha-free
path, prices are rebuilt and every ``z_k`` re-evaluated *with the package's own
DSL evaluator* until the returns change by less than ``tol``.  The map is a
contraction for small ``s`` (and exactly causal), so convergence is fast; the
achieved residual is reported.  Bars are emitted in the suite's canonical long
schema with ``source="synthetic"`` and ``finality="confirmed"`` and pass
through :func:`llm_factor_mining.data.panel_from_bars` validation.

This is synthetic data; nothing produced here is evidence about real markets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd

from ..data import PANEL_FIELDS, Panel, panel_content_hash, panel_from_bars
from ..dsl.canonical import structural_similarity
from ..dsl.nodes import Call, Node
from ..dsl.parser import parse
from ..dsl.validate import validate
from ..evaluate.engine import Evaluator
from ..jsonutil import json_safe


SYNTHETIC_SOURCE = "synthetic"
PLANTED_FAMILIES = ("textbook", "library_family", "drawn")


@dataclass(frozen=True)
class PlantedFactor:
    """One ground-truth expression and its (pre-normalization) weight."""

    name: str
    expression: str
    weight: float = 1.0
    family: str = "textbook"

    def __post_init__(self) -> None:
        result = validate(self.expression)
        if not result.ok:
            raise ValueError(f"planted expression {self.expression!r} is invalid: {result.codes}")
        if self.family not in PLANTED_FAMILIES:
            raise ValueError(f"planted family must be one of {PLANTED_FAMILIES}")

    @property
    def node(self) -> Node:
        return parse(self.expression)


# Families: "textbook" - classic factors any literature-aware proposer knows;
# "library_family" - a documented idea (the negative correlation of volume
# changes with returns behind Kakushadze (2016) Alpha#2, whose formula is in
# the reference library as ``volume_change_vs_intraday_return``); "drawn" -
# drawn at random from the typed grammar by the pre-registered procedure of
# :mod:`llm_factor_mining.benchmark.draw`.
FIXED_PLANTED: tuple[PlantedFactor, ...] = (
    PlantedFactor("reversal_5", "-ts_mean(returns, 5)", 1.0, "textbook"),
    PlantedFactor("abnormal_volume_20", "volume / ts_mean(volume, 20)", 1.0, "textbook"),
    PlantedFactor(
        "volume_return_corr_10", "-ts_corr(returns, delta(log(volume), 1), 10)", 1.0, "library_family"
    ),
)
# Candidate k = 40 of the pre-registered draw (results/planted_signal_draw.json):
# the first of 41 candidates meeting criteria C1-C4 of benchmark.draw.
DRAWN_PLANTED = PlantedFactor("drawn_40", "delta(ts_cov(ts_argmin(low,20),cs_demean(open),5),1)", 1.0, "drawn")
DEFAULT_PLANTED: tuple[PlantedFactor, ...] = (*FIXED_PLANTED, DRAWN_PLANTED)


@dataclass(frozen=True)
class SyntheticMarketConfig:
    """Parameters of the synthetic market (dates are fictional business days)."""

    n_symbols: int = 60
    n_dates: int = 750
    snr: float = 0.10
    seed: int = 0
    planted: tuple[PlantedFactor, ...] = DEFAULT_PLANTED
    lag: int = 1
    start: str = "2031-01-02"
    market_mu: float = 0.0003
    market_vol: float = 0.010
    beta_range: tuple[float, float] = (0.5, 1.5)
    idio_vol_range: tuple[float, float] = (0.012, 0.030)
    log_volume_mean: float = 13.0
    log_volume_dispersion: float = 0.6
    volume_persistence: float = 0.7
    volume_noise: float = 0.35
    volume_common_share: float = 0.3
    max_abs_return: float = 0.5
    max_iterations: int = 200
    tol: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.n_symbols < 10 or self.n_dates < 60:
            raise ValueError("need at least 10 symbols and 60 dates")
        if self.snr < 0:
            raise ValueError("snr must be non-negative")
        if self.lag < 1:
            raise ValueError("lag must be at least 1")
        if not self.planted:
            raise ValueError("at least one planted factor is required")
        names = [factor.name for factor in self.planted]
        if len(names) != len(set(names)):
            raise ValueError("planted factor names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class SyntheticMarket:
    """Generated bars, the validated panel and the ground-truth planted signals."""

    config: SyntheticMarketConfig
    bars: pd.DataFrame
    panel: Panel
    planted_signals: Mapping[str, pd.DataFrame]
    alpha_scale: float
    iterations: int
    residual: float
    realized_snr: float
    weights: Mapping[str, float] = field(default_factory=dict)

    def diagnostics(self) -> dict[str, Any]:
        return json_safe(
            {
                "alpha_scale": self.alpha_scale,
                "iterations": self.iterations,
                "residual": self.residual,
                "realized_snr": self.realized_snr,
                "weights": dict(self.weights),
                "content_sha256": self.panel.content_sha256,
            }
        )


def _fast_panel(
    dates: pd.DatetimeIndex, symbols: pd.Index, arrays: Mapping[str, np.ndarray]
) -> Panel:
    fields = {name: pd.DataFrame(arrays[name], index=dates, columns=symbols) for name in PANEL_FIELDS}
    return Panel(**fields, sources=(SYNTHETIC_SOURCE,), content_sha256=panel_content_hash(fields, (SYNTHETIC_SOURCE,)))


def _shift_down(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    out[lag:] = values[:-lag]
    return out


def simulate_market(config: SyntheticMarketConfig | None = None) -> SyntheticMarket:
    """Generate one synthetic market (see the module docstring)."""

    cfg = config or SyntheticMarketConfig()
    rng = np.random.default_rng(cfg.seed)
    t_count, n = cfg.n_dates, cfg.n_symbols
    dates = pd.bdate_range(cfg.start, periods=t_count, name="date")
    symbols = pd.Index([f"SYN{index:03d}" for index in range(n)], name="symbol")

    beta = rng.uniform(*cfg.beta_range, size=n)
    idio = rng.uniform(*cfg.idio_vol_range, size=n)
    market = rng.normal(cfg.market_mu, cfg.market_vol, size=t_count)
    eps = rng.standard_normal((t_count, n))
    start_price = np.exp(rng.normal(math.log(50.0), 0.5, size=n))

    level = rng.normal(cfg.log_volume_mean, cfg.log_volume_dispersion, size=n)
    phi = cfg.volume_persistence
    common = rng.standard_normal(t_count)
    own = rng.standard_normal((t_count, n))
    innovations = cfg.volume_noise * (
        math.sqrt(cfg.volume_common_share) * common[:, None] + math.sqrt(1.0 - cfg.volume_common_share) * own
    )
    state = np.zeros((t_count, n))
    state[0] = innovations[0] / math.sqrt(1.0 - phi**2)
    for t in range(1, t_count):
        state[t] = phi * state[t - 1] + innovations[t]
    volume = np.maximum(np.round(np.exp(level + state)), 1.0)

    gap = rng.normal(0.0, 0.25, size=(t_count, n)) * idio
    up = np.minimum(np.abs(rng.normal(0.0, 0.5, size=(t_count, n))) * idio, 0.2)
    down = np.minimum(np.abs(rng.normal(0.0, 0.5, size=(t_count, n))) * idio, 0.2)

    base = beta * market[:, None] + idio * eps
    base[0] = 0.0
    raw_weights = np.array([factor.weight for factor in cfg.planted], dtype="float64")
    norm = float(np.sqrt(np.sum(raw_weights**2)))
    if norm == 0:
        raise ValueError("planted weights cannot all be zero")
    weights = raw_weights / norm
    scale = cfg.snr * float(idio.mean())
    delay = cfg.lag + 1
    nodes = [Call("cs_zscore", (parse(factor.expression),)) for factor in cfg.planted]

    def build(returns: np.ndarray) -> dict[str, np.ndarray]:
        close = start_price * np.cumprod(1.0 + returns, axis=0)
        previous = np.vstack([close[:1], close[:-1]])
        open_ = previous * np.exp(gap)
        high = np.maximum(open_, close) * (1.0 + up)
        low = np.minimum(open_, close) * (1.0 - down)
        return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}

    def planted_values(arrays: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        engine = Evaluator(_fast_panel(dates, symbols, arrays), max_cache_entries=32)
        return [engine.evaluate(node).to_numpy() for node in nodes]

    returns = base.copy()
    residual = math.inf
    iterations = 0
    alpha = np.zeros_like(base)
    for iterations in range(1, cfg.max_iterations + 1):
        signals = planted_values(build(returns))
        alpha = np.zeros_like(base)
        for weight, values in zip(weights, signals):
            alpha += weight * np.nan_to_num(_shift_down(values, delay), nan=0.0)
        alpha *= scale
        updated = np.clip(base + alpha, -cfg.max_abs_return, cfg.max_abs_return)
        updated[0] = 0.0
        residual = float(np.max(np.abs(updated - returns)))
        returns = updated
        if residual <= cfg.tol:
            break
    else:
        raise RuntimeError(
            f"planted-alpha fixed point did not converge in {cfg.max_iterations} iterations "
            f"(residual {residual:.3g}); lower snr or raise max_iterations"
        )

    arrays = build(returns)
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), n),
            "symbol": np.tile(symbols.to_numpy(dtype=object), t_count),
            **{name: arrays[name].ravel() for name in PANEL_FIELDS},
        }
    )
    frame["source"] = SYNTHETIC_SOURCE
    frame["finality"] = "confirmed"
    panel = panel_from_bars(
        frame,
        metadata={
            "generator": "llm_factor_mining.benchmark.synthetic",
            "seed": cfg.seed,
            "snr": cfg.snr,
            "planted": [factor.name for factor in cfg.planted],
        },
    )
    engine = Evaluator(panel, max_cache_entries=32)
    planted = {factor.name: engine.evaluate(node) for factor, node in zip(cfg.planted, nodes)}
    noise = idio * eps
    dispersion_alpha = np.nanstd(alpha[delay + 1 :], axis=1)
    dispersion_noise = np.nanstd(noise[delay + 1 :], axis=1)
    active = dispersion_alpha > 0
    realized = float(np.mean(dispersion_alpha[active] / dispersion_noise[active])) if active.any() else 0.0
    return SyntheticMarket(
        config=cfg,
        bars=frame,
        panel=panel,
        planted_signals=planted,
        alpha_scale=scale,
        iterations=iterations,
        residual=residual,
        realized_snr=realized,
        weights={factor.name: float(weight) for factor, weight in zip(cfg.planted, weights)},
    )


# --------------------------------------------------------------------------
# recovery metrics
# --------------------------------------------------------------------------


def factor_value_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    dates: pd.DatetimeIndex | Sequence[pd.Timestamp],
    *,
    min_names: int = 10,
    stride: int = 1,
) -> float:
    """Mean per-date Spearman correlation of two factor-value panels on ``dates``."""

    from ..search import mean_rank_correlation  # local import avoids a cycle

    chosen = pd.DatetimeIndex(dates)[:: max(int(stride), 1)]
    return mean_rank_correlation(left.loc[chosen], right.loc[chosen], min_names=min_names)


def recovery_matrix(
    candidates: Mapping[str, pd.DataFrame],
    planted: Mapping[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    *,
    min_names: int = 10,
    stride: int = 1,
) -> dict[str, dict[str, float]]:
    """``{planted_name: {candidate_key: rho}}`` of mean cross-sectional rank correlations."""

    return {
        name: {
            key: factor_value_correlation(values, target, dates, min_names=min_names, stride=stride)
            for key, values in candidates.items()
        }
        for name, target in planted.items()
    }


def best_matches(matrix: Mapping[str, Mapping[str, float]], *, signed: bool = False) -> dict[str, dict[str, Any]]:
    """Per planted factor: the best-matching candidate and its correlation.

    ``signed=False`` ranks by ``|rho|`` (unoriented trials: the protocol may
    still flip their sign).  ``signed=True`` ranks by ``rho`` itself: for
    *oriented* (selected) factors only a positive correlation means the factor
    bets with the planted signal.
    """

    out: dict[str, dict[str, Any]] = {}
    for name, row in matrix.items():
        finite = {key: value for key, value in row.items() if math.isfinite(value)}
        if not finite:
            out[name] = {"best": None, "abs_rho": 0.0, "rho": None}
            continue
        score = (lambda item: finite[item]) if signed else (lambda item: abs(finite[item]))
        key = max(sorted(finite), key=score)
        out[name] = {"best": key, "abs_rho": abs(finite[key]), "rho": finite[key]}
    return out


def structural_match(expression: str, planted: Sequence[PlantedFactor]) -> dict[str, float]:
    """Structural (subtree Jaccard) similarity of ``expression`` to each planted factor."""

    node = parse(expression)
    return {factor.name: structural_similarity(node, parse(factor.expression)) for factor in planted}


__all__ = [
    "DEFAULT_PLANTED",
    "DRAWN_PLANTED",
    "FIXED_PLANTED",
    "PLANTED_FAMILIES",
    "PlantedFactor",
    "SYNTHETIC_SOURCE",
    "SyntheticMarket",
    "SyntheticMarketConfig",
    "best_matches",
    "factor_value_correlation",
    "recovery_matrix",
    "simulate_market",
    "structural_match",
]
