"""Nested, purged walk-forward validation for trading strategies.

The module is deliberately strategy-agnostic.  It can consume either a cached
date-by-configuration return matrix or an evaluator callback with the signature
``(parameter, train_index, evaluation_index) -> returns``.  Hyperparameters are
ranked only on chronological inner validation folds.  The selected parameter is
then frozen and evaluated once on the corresponding outer test fold.

``purge_size`` removes observations immediately before every evaluation window.
``embargo_size`` creates a holdout interval after a test window and those
observations are excluded from later training folds.  Both are expressed in
observations, not calendar days, and every exclusion is exposed in the audit
tables returned by :func:`run_nested_walk_forward`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


ReturnEvaluator = Callable[[Any, pd.Index, pd.Index], pd.Series | Sequence[float] | np.ndarray]
ScoreFunction = Callable[[pd.Series], float]


@dataclass(frozen=True)
class NestedWalkForwardSpec:
    """Controls the chronological outer and inner walk-forward loops.

    ``outer_train_size`` is the initial history available to the first outer
    fold.  With ``expanding=True`` later folds retain all eligible history;
    otherwise each uses at most ``outer_train_size`` eligible observations.
    Inner folds always expand within each outer training sample because they
    simulate the information set available as that training sample evolved.
    """

    outer_train_size: int
    outer_test_size: int
    inner_train_size: int
    inner_validation_size: int
    outer_step_size: int | None = None
    inner_step_size: int | None = None
    purge_size: int = 0
    embargo_size: int = 0
    expanding: bool = True
    maximum_outer_folds: int | None = None
    maximum_inner_folds: int | None = None
    annualization: int = 252
    selection_metric: str = "sharpe"
    minimum_evaluation_observations: int = 2
    bootstrap_samples: int = 1_000
    bootstrap_block_size: int = 10
    confidence_level: float = 0.95
    random_seed: int = 7

    def __post_init__(self) -> None:
        positive = {
            "outer_train_size": self.outer_train_size,
            "outer_test_size": self.outer_test_size,
            "inner_train_size": self.inner_train_size,
            "inner_validation_size": self.inner_validation_size,
            "annualization": self.annualization,
            "minimum_evaluation_observations": self.minimum_evaluation_observations,
            "bootstrap_block_size": self.bootstrap_block_size,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.purge_size < 0 or self.embargo_size < 0:
            raise ValueError("purge_size and embargo_size cannot be negative")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples cannot be negative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        for name, value in (
            ("maximum_outer_folds", self.maximum_outer_folds),
            ("maximum_inner_folds", self.maximum_inner_folds),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.resolved_outer_step < self.outer_test_size + self.embargo_size:
            raise ValueError(
                "outer_step_size must be at least outer_test_size + embargo_size "
                "so outer test windows do not overlap"
            )
        if self.resolved_inner_step < self.inner_validation_size + self.embargo_size:
            raise ValueError(
                "inner_step_size must be at least inner_validation_size + embargo_size "
                "so inner validation windows do not overlap"
            )
        supported = {
            "mean_return",
            "annualized_mean_return",
            "cagr",
            "sharpe",
            "sortino",
            "calmar",
        }
        if self.selection_metric not in supported:
            raise ValueError(
                f"Unsupported selection_metric {self.selection_metric!r}; "
                f"choose one of {sorted(supported)}"
            )

    @property
    def resolved_outer_step(self) -> int:
        return (
            int(self.outer_step_size)
            if self.outer_step_size is not None
            else int(self.outer_test_size + self.embargo_size)
        )

    @property
    def resolved_inner_step(self) -> int:
        return (
            int(self.inner_step_size)
            if self.inner_step_size is not None
            else int(self.inner_validation_size + self.embargo_size)
        )


@dataclass(frozen=True)
class PurgedFold:
    """Indices and exclusions for one chronological train/evaluation split."""

    fold_id: int
    train_index: pd.Index
    test_index: pd.Index
    purge_index: pd.Index
    embargo_index: pd.Index
    prior_embargo_excluded: pd.Index


@dataclass
class NestedWalkForwardResult:
    """Performance output and sufficient tables to audit every decision."""

    spec: NestedWalkForwardSpec
    oos_returns: pd.DataFrame
    performance: pd.DataFrame
    fold_audit: pd.DataFrame
    inner_evaluations: pd.DataFrame
    selection_audit: pd.DataFrame
    outer_evaluations: pd.DataFrame
    bootstrap_confidence_intervals: pd.DataFrame
    data_snooping_diagnostics: pd.DataFrame
    pbo_fold_audit: pd.DataFrame


def _validated_index(index: pd.Index | Sequence[Any]) -> pd.Index:
    result = pd.Index(index)
    if result.empty:
        raise ValueError("index cannot be empty")
    if result.has_duplicates:
        raise ValueError("index must be unique")
    if not result.is_monotonic_increasing:
        raise ValueError("index must be sorted in increasing chronological order")
    return result


def build_purged_walk_forward_folds(
    index: pd.Index | Sequence[Any],
    *,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
    purge_size: int = 0,
    embargo_size: int = 0,
    expanding: bool = True,
    maximum_folds: int | None = None,
) -> list[PurgedFold]:
    """Construct deterministic chronological folds with explicit exclusions.

    Test windows never overlap.  By default a new test starts after the prior
    test and its embargo.  Embargoed observations are conservatively ineligible
    for all subsequent training folds, which makes the exclusion directly
    inspectable and prevents accidental reuse by custom evaluators.
    """

    values = _validated_index(index)
    if initial_train_size <= 0 or test_size <= 0:
        raise ValueError("initial_train_size and test_size must be positive")
    if purge_size < 0 or embargo_size < 0:
        raise ValueError("purge_size and embargo_size cannot be negative")
    resolved_step = test_size + embargo_size if step_size is None else int(step_size)
    if resolved_step < test_size + embargo_size:
        raise ValueError("step_size must be at least test_size + embargo_size")
    if maximum_folds is not None and maximum_folds <= 0:
        raise ValueError("maximum_folds must be positive when provided")

    first_test_start = int(initial_train_size + purge_size)
    prior_embargo_positions: set[int] = set()
    folds: list[PurgedFold] = []
    fold_id = 0
    test_start = first_test_start
    while test_start + test_size <= len(values):
        raw_train_end = test_start - purge_size
        raw_train_start = 0 if expanding else max(0, raw_train_end - initial_train_size)
        raw_positions = np.arange(raw_train_start, raw_train_end, dtype=int)
        excluded_positions = np.array(
            sorted(set(raw_positions.tolist()) & prior_embargo_positions), dtype=int
        )
        if excluded_positions.size:
            keep = ~np.isin(raw_positions, excluded_positions)
            train_positions = raw_positions[keep]
        else:
            train_positions = raw_positions
        purge_positions = np.arange(raw_train_end, test_start, dtype=int)
        test_positions = np.arange(test_start, test_start + test_size, dtype=int)
        embargo_stop = min(len(values), test_start + test_size + embargo_size)
        embargo_positions = np.arange(test_start + test_size, embargo_stop, dtype=int)

        if train_positions.size:
            folds.append(
                PurgedFold(
                    fold_id=fold_id,
                    train_index=values.take(train_positions),
                    test_index=values.take(test_positions),
                    purge_index=values.take(purge_positions),
                    embargo_index=values.take(embargo_positions),
                    prior_embargo_excluded=values.take(excluded_positions),
                )
            )
            fold_id += 1
        prior_embargo_positions.update(embargo_positions.tolist())
        if maximum_folds is not None and len(folds) >= maximum_folds:
            break
        test_start += resolved_step
    return folds


def _first(index: pd.Index) -> Any:
    return index[0] if len(index) else pd.NaT


def _last(index: pd.Index) -> Any:
    return index[-1] if len(index) else pd.NaT


def folds_to_audit_table(
    folds: Sequence[PurgedFold],
    *,
    layer: str,
    parent_outer_fold: int | None = None,
) -> pd.DataFrame:
    """Convert fold definitions to a human- and machine-readable audit table."""

    rows: list[dict[str, Any]] = []
    for fold in folds:
        overlap = fold.train_index.intersection(fold.test_index)
        rows.append(
            {
                "layer": layer,
                "evaluation_role": "lockbox" if layer == "outer" else "inner_validation",
                "parent_outer_fold": parent_outer_fold,
                "fold_id": fold.fold_id,
                "train_start": _first(fold.train_index),
                "train_end": _last(fold.train_index),
                "evaluation_start": _first(fold.test_index),
                "evaluation_end": _last(fold.test_index),
                "train_observations": int(len(fold.train_index)),
                "evaluation_observations": int(len(fold.test_index)),
                "purged_observations": int(len(fold.purge_index)),
                "purge_start": _first(fold.purge_index),
                "purge_end": _last(fold.purge_index),
                "embargoed_after_observations": int(len(fold.embargo_index)),
                "embargo_start": _first(fold.embargo_index),
                "embargo_end": _last(fold.embargo_index),
                "prior_embargo_excluded_from_train": int(
                    len(fold.prior_embargo_excluded)
                ),
                "train_evaluation_overlap": int(len(overlap)),
                "chronological": bool(
                    len(fold.train_index)
                    and len(fold.test_index)
                    and _last(fold.train_index) < _first(fold.test_index)
                ),
            }
        )
    return pd.DataFrame(rows)


def _numeric_returns(values: pd.Series | Sequence[float] | np.ndarray) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def performance_statistics(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    annualization: int = 252,
) -> dict[str, float | int]:
    """Calculate compact return statistics without treating missing data as zero."""

    values = _numeric_returns(returns).dropna()
    observations = int(len(values))
    if not observations:
        return {
            "observations": 0,
            "mean_return": math.nan,
            "annualized_mean_return": math.nan,
            "total_return": math.nan,
            "cagr": math.nan,
            "annual_volatility": math.nan,
            "sharpe": math.nan,
            "sortino": math.nan,
            "max_drawdown": math.nan,
            "calmar": math.nan,
        }
    mean_return = float(values.mean())
    volatility = float(values.std(ddof=1)) if observations > 1 else math.nan
    annual_volatility = volatility * math.sqrt(annualization)
    sharpe = (
        mean_return / volatility * math.sqrt(annualization)
        if np.isfinite(volatility) and volatility > 0
        else math.nan
    )
    downside = values.loc[values < 0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside.to_numpy(dtype=float)))))
        * math.sqrt(annualization)
        if len(downside)
        else 0.0
    )
    sortino = (
        mean_return * annualization / downside_deviation
        if downside_deviation > 0
        else math.nan
    )
    equity = (1.0 + values).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = (
        float(equity.iloc[-1] ** (annualization / observations) - 1.0)
        if equity.iloc[-1] > 0
        else math.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = (
        cagr / abs(max_drawdown)
        if np.isfinite(cagr) and max_drawdown < 0
        else math.nan
    )
    return {
        "observations": observations,
        "mean_return": mean_return,
        "annualized_mean_return": mean_return * annualization,
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def _circular_block_indices(
    observations: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = int(math.ceil(observations / block_size))
    starts = rng.integers(0, observations, size=blocks)
    offsets = np.arange(block_size, dtype=int)
    sampled = (starts[:, None] + offsets[None, :]) % observations
    return sampled.ravel()[:observations]


def bootstrap_performance_confidence_intervals(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    samples: int = 1_000,
    block_size: int = 10,
    confidence_level: float = 0.95,
    annualization: int = 252,
    seed: int = 7,
) -> pd.DataFrame:
    """Percentile CIs from a deterministic circular moving-block bootstrap."""

    if samples < 0:
        raise ValueError("samples cannot be negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    values = _numeric_returns(returns).dropna().to_numpy(dtype=float)
    statistics = (
        "annualized_mean_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
    )
    estimates = performance_statistics(values, annualization=annualization)
    draws: dict[str, list[float]] = {name: [] for name in statistics}
    if samples and len(values):
        rng = np.random.default_rng(seed)
        effective_block_size = min(block_size, len(values))
        for _ in range(samples):
            locations = _circular_block_indices(len(values), effective_block_size, rng)
            metrics = performance_statistics(values[locations], annualization=annualization)
            for name in statistics:
                draws[name].append(float(metrics[name]))
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, Any]] = []
    for name in statistics:
        finite = np.asarray(draws[name], dtype=float)
        finite = finite[np.isfinite(finite)]
        rows.append(
            {
                "statistic": name,
                "estimate": float(estimates[name]),
                "lower": float(np.quantile(finite, alpha)) if len(finite) else math.nan,
                "upper": float(np.quantile(finite, 1.0 - alpha)) if len(finite) else math.nan,
                "confidence_level": float(confidence_level),
                "bootstrap_samples_requested": int(samples),
                "valid_bootstrap_samples": int(len(finite)),
                "block_size": int(min(block_size, len(values))) if len(values) else 0,
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def reality_check(
    candidate_returns: pd.DataFrame,
    *,
    benchmark_returns: pd.Series | float = 0.0,
    samples: int = 1_000,
    block_size: int = 10,
    annualization: int = 252,
    seed: int = 7,
) -> dict[str, Any]:
    """A White-reality-check-style test for the best candidate's mean return.

    Candidate excess returns are centered column-wise to impose the joint null,
    then resampled in common circular blocks.  Common blocks retain temporal
    dependence and cross-candidate correlation.  This is a transparent
    reality-check-style diagnostic, not a claim of an exact finite-sample test.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if candidate_returns.empty or candidate_returns.shape[1] < 1:
        raise ValueError("candidate_returns must contain at least one candidate")
    numeric = candidate_returns.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if np.isscalar(benchmark_returns):
        excess = numeric - float(benchmark_returns)
    else:
        benchmark = pd.to_numeric(benchmark_returns, errors="coerce")
        excess = numeric.sub(benchmark, axis=0)
    excess = excess.dropna(axis=0, how="any")
    if len(excess) < 2:
        raise ValueError("reality_check requires at least two complete observations")
    means = excess.mean(axis=0)
    best_candidate = str(means.idxmax())
    observed_best = float(means.max() * annualization)
    centered = excess.sub(means, axis=1).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    effective_block_size = min(block_size, len(centered))
    null_maxima = np.empty(samples, dtype=float)
    for location in range(samples):
        indices = _circular_block_indices(len(centered), effective_block_size, rng)
        null_maxima[location] = float(centered[indices].mean(axis=0).max() * annualization)
    pvalue = float((1 + np.sum(null_maxima >= observed_best)) / (samples + 1))
    return {
        "method": "centered_joint_circular_block_reality_check_style",
        "observations": int(len(excess)),
        "candidate_count": int(excess.shape[1]),
        "best_candidate": best_candidate,
        "observed_best_annualized_mean_return": observed_best,
        "null_pvalue": pvalue,
        "null_maximum_mean": float(np.mean(null_maxima)),
        "null_maximum_q95": float(np.quantile(null_maxima, 0.95)),
        "bootstrap_samples": int(samples),
        "block_size": int(effective_block_size),
        "seed": int(seed),
    }


def _parameter_payload(parameter: Any) -> str:
    if is_dataclass(parameter):
        value: Any = asdict(parameter)
    elif isinstance(parameter, Mapping):
        value = dict(parameter)
    elif hasattr(parameter, "__dict__"):
        value = {
            str(key): item
            for key, item in vars(parameter).items()
            if not str(key).startswith("_")
        }
    elif isinstance(parameter, (str, int, float, bool)) or parameter is None:
        value = parameter
    else:
        value = repr(parameter)
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _parameter_fingerprint(parameter: Any) -> str:
    return hashlib.sha256(_parameter_payload(parameter).encode("utf-8")).hexdigest()[:16]


def _evaluate_returns(
    evaluator: ReturnEvaluator,
    parameter: Any,
    train_index: pd.Index,
    evaluation_index: pd.Index,
) -> pd.Series:
    raw = evaluator(parameter, train_index.copy(), evaluation_index.copy())
    if isinstance(raw, pd.Series):
        if raw.index.has_duplicates:
            raise ValueError("evaluator returned duplicate dates")
        result = pd.to_numeric(raw, errors="coerce").reindex(evaluation_index)
    else:
        array = np.asarray(raw, dtype=float).reshape(-1)
        if len(array) != len(evaluation_index):
            raise ValueError(
                "array-like evaluator output must have the same length as evaluation_index"
            )
        result = pd.Series(array, index=evaluation_index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)


def _metric_score(
    returns: pd.Series,
    metric: str,
    annualization: int,
    score_function: ScoreFunction | None,
) -> float:
    if score_function is not None:
        value = float(score_function(returns.dropna()))
    else:
        value = float(performance_statistics(returns, annualization=annualization)[metric])
    return value if np.isfinite(value) else -math.inf


def _inner_folds_for_outer(
    outer_fold: PurgedFold,
    spec: NestedWalkForwardSpec,
) -> list[PurgedFold]:
    return build_purged_walk_forward_folds(
        outer_fold.train_index,
        initial_train_size=spec.inner_train_size,
        test_size=spec.inner_validation_size,
        step_size=spec.resolved_inner_step,
        purge_size=spec.purge_size,
        embargo_size=spec.embargo_size,
        expanding=True,
        maximum_folds=spec.maximum_inner_folds,
    )


def pbo_style_diagnostic(
    diagnostic_returns: pd.DataFrame,
    outer_folds: Sequence[PurgedFold],
    selected_by_fold: Mapping[int, str],
    *,
    metric: str,
    annualization: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank inner winners ex post on outer folds (PBO-style, not formal CSCV)."""

    matrix = diagnostic_returns.apply(pd.to_numeric, errors="coerce")
    rows: list[dict[str, Any]] = []
    for fold in outer_folds:
        scores: dict[str, float] = {}
        for candidate_id in matrix.columns:
            values = matrix[str(candidate_id)].reindex(fold.test_index)
            scores[str(candidate_id)] = _metric_score(values, metric, annualization, None)
        ranking = sorted(scores, key=lambda candidate: (-scores[candidate], candidate))
        selected = selected_by_fold[fold.fold_id]
        if selected not in ranking:
            raise ValueError(f"diagnostic_returns is missing selected candidate {selected!r}")
        rank = ranking.index(selected) + 1
        count = len(ranking)
        percentile = (count - rank + 0.5) / count
        rows.append(
            {
                "outer_fold": fold.fold_id,
                "selected_candidate": selected,
                "selected_outer_score": scores[selected],
                "selected_outer_rank": rank,
                "candidate_count": count,
                "selected_outer_percentile": percentile,
                "below_outer_median": bool(percentile < 0.5),
            }
        )
    audit = pd.DataFrame(rows)
    valid = audit["selected_outer_percentile"].dropna() if not audit.empty else pd.Series(dtype=float)
    summary = {
        "method": "chronological_outer_rank_pbo_style",
        "outer_folds": int(len(valid)),
        "pbo_style_probability": float((valid < 0.5).mean()) if len(valid) else math.nan,
        "mean_selected_outer_percentile": float(valid.mean()) if len(valid) else math.nan,
        "note": "Outer-rank degradation diagnostic; not combinatorially symmetric PBO.",
    }
    return audit, summary


def run_nested_walk_forward(
    index: pd.Index | Sequence[Any],
    candidates: Mapping[str, Any],
    evaluator: ReturnEvaluator,
    spec: NestedWalkForwardSpec,
    *,
    score_function: ScoreFunction | None = None,
    diagnostic_returns: pd.DataFrame | None = None,
) -> NestedWalkForwardResult:
    """Select on inner folds and evaluate one frozen candidate per outer fold.

    ``diagnostic_returns`` is optional.  When supplied, it is used only to rank
    every candidate ex post on outer folds for the explicitly labeled PBO-style
    diagnostic; it never affects hyperparameter selection or reported OOS
    returns.  The callback itself is invoked on outer data only for the selected
    candidate.
    """

    chronology = _validated_index(index)
    if not candidates:
        raise ValueError("candidates cannot be empty")
    normalized_candidates = {str(key): value for key, value in candidates.items()}
    if len(normalized_candidates) != len(candidates):
        raise ValueError("candidate identifiers must be unique after string conversion")
    candidate_ids = sorted(normalized_candidates)
    if diagnostic_returns is not None:
        diagnostics = diagnostic_returns.copy()
        diagnostics.columns = diagnostics.columns.map(str)
        missing = sorted(set(candidate_ids) - set(diagnostics.columns))
        if missing:
            raise ValueError(f"diagnostic_returns is missing candidates: {missing}")
        diagnostics = diagnostics.reindex(chronology)
    else:
        diagnostics = None

    outer_folds = build_purged_walk_forward_folds(
        chronology,
        initial_train_size=spec.outer_train_size,
        test_size=spec.outer_test_size,
        step_size=spec.resolved_outer_step,
        purge_size=spec.purge_size,
        embargo_size=spec.embargo_size,
        expanding=spec.expanding,
        maximum_folds=spec.maximum_outer_folds,
    )
    if not outer_folds:
        raise ValueError("the specification produces no complete outer folds")

    fold_audits = [folds_to_audit_table(outer_folds, layer="outer")]
    inner_evaluation_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    outer_evaluation_rows: list[dict[str, Any]] = []
    oos_pieces: list[pd.DataFrame] = []
    reality_rows: list[dict[str, Any]] = []
    selected_by_fold: dict[int, str] = {}

    for outer_fold in outer_folds:
        inner_folds = _inner_folds_for_outer(outer_fold, spec)
        if not inner_folds:
            raise ValueError(
                f"outer fold {outer_fold.fold_id} has no complete inner folds; "
                "reduce inner_train_size, purge_size, or inner_validation_size"
            )
        fold_audits.append(
            folds_to_audit_table(
                inner_folds,
                layer="inner",
                parent_outer_fold=outer_fold.fold_id,
            )
        )
        candidate_pieces: dict[str, list[pd.Series]] = {
            candidate_id: [] for candidate_id in candidate_ids
        }
        candidate_fold_scores: dict[str, list[float]] = {
            candidate_id: [] for candidate_id in candidate_ids
        }
        for inner_fold in inner_folds:
            for candidate_id in candidate_ids:
                parameter = normalized_candidates[candidate_id]
                returns = _evaluate_returns(
                    evaluator,
                    parameter,
                    inner_fold.train_index,
                    inner_fold.test_index,
                )
                valid_observations = int(returns.notna().sum())
                score = (
                    _metric_score(
                        returns,
                        spec.selection_metric,
                        spec.annualization,
                        score_function,
                    )
                    if valid_observations >= spec.minimum_evaluation_observations
                    else -math.inf
                )
                candidate_pieces[candidate_id].append(returns)
                candidate_fold_scores[candidate_id].append(score)
                inner_evaluation_rows.append(
                    {
                        "outer_fold": outer_fold.fold_id,
                        "inner_fold": inner_fold.fold_id,
                        "candidate_id": candidate_id,
                        "evaluation_start": _first(inner_fold.test_index),
                        "evaluation_end": _last(inner_fold.test_index),
                        "observations": valid_observations,
                        "selection_metric": spec.selection_metric,
                        "fold_score": score if np.isfinite(score) else math.nan,
                    }
                )

        aggregate_scores: dict[str, float] = {}
        for candidate_id in candidate_ids:
            pooled = pd.concat(candidate_pieces[candidate_id]).sort_index()
            if pooled.index.has_duplicates:
                raise RuntimeError("inner validation windows unexpectedly overlap")
            aggregate_scores[candidate_id] = (
                _metric_score(
                    pooled,
                    spec.selection_metric,
                    spec.annualization,
                    score_function,
                )
                if int(pooled.notna().sum()) >= spec.minimum_evaluation_observations
                else -math.inf
            )
        ranking = sorted(
            candidate_ids,
            key=lambda candidate_id: (-aggregate_scores[candidate_id], candidate_id),
        )
        selected_id = ranking[0]
        if not np.isfinite(aggregate_scores[selected_id]):
            raise ValueError(
                f"no candidate has enough valid inner observations in outer fold "
                f"{outer_fold.fold_id}"
            )
        selected_by_fold[outer_fold.fold_id] = selected_id
        for rank, candidate_id in enumerate(ranking, start=1):
            finite_fold_scores = np.asarray(candidate_fold_scores[candidate_id], dtype=float)
            finite_fold_scores = finite_fold_scores[np.isfinite(finite_fold_scores)]
            parameter = normalized_candidates[candidate_id]
            pooled = pd.concat(candidate_pieces[candidate_id]).sort_index()
            selection_rows.append(
                {
                    "outer_fold": outer_fold.fold_id,
                    "candidate_id": candidate_id,
                    "rank": rank,
                    "selected": bool(candidate_id == selected_id),
                    "selection_metric": spec.selection_metric,
                    "aggregate_score": (
                        aggregate_scores[candidate_id]
                        if np.isfinite(aggregate_scores[candidate_id])
                        else math.nan
                    ),
                    "mean_inner_fold_score": (
                        float(finite_fold_scores.mean())
                        if len(finite_fold_scores)
                        else math.nan
                    ),
                    "std_inner_fold_score": (
                        float(finite_fold_scores.std(ddof=1))
                        if len(finite_fold_scores) > 1
                        else math.nan
                    ),
                    "valid_inner_folds": int(len(finite_fold_scores)),
                    "inner_validation_observations": int(pooled.notna().sum()),
                    "parameter_payload": _parameter_payload(parameter),
                    "parameter_fingerprint": _parameter_fingerprint(parameter),
                }
            )

        if spec.bootstrap_samples > 0:
            inner_matrix = pd.concat(
                {
                    candidate_id: pd.concat(candidate_pieces[candidate_id]).sort_index()
                    for candidate_id in candidate_ids
                },
                axis=1,
            )
            try:
                check = reality_check(
                    inner_matrix,
                    samples=spec.bootstrap_samples,
                    block_size=spec.bootstrap_block_size,
                    annualization=spec.annualization,
                    seed=spec.random_seed + outer_fold.fold_id,
                )
                reality_rows.append({"outer_fold": outer_fold.fold_id, **check})
            except ValueError as error:
                reality_rows.append(
                    {
                        "outer_fold": outer_fold.fold_id,
                        "method": "centered_joint_circular_block_reality_check_style",
                        "error": str(error),
                    }
                )

        frozen_parameter = normalized_candidates[selected_id]
        outer_returns = _evaluate_returns(
            evaluator,
            frozen_parameter,
            outer_fold.train_index,
            outer_fold.test_index,
        )
        outer_metrics = performance_statistics(
            outer_returns,
            annualization=spec.annualization,
        )
        outer_evaluation_rows.append(
            {
                "outer_fold": outer_fold.fold_id,
                "candidate_id": selected_id,
                "parameter_payload": _parameter_payload(frozen_parameter),
                "parameter_fingerprint": _parameter_fingerprint(frozen_parameter),
                "selection_metric": spec.selection_metric,
                "inner_selection_score": aggregate_scores[selected_id],
                "train_start": _first(outer_fold.train_index),
                "train_end": _last(outer_fold.train_index),
                "test_start": _first(outer_fold.test_index),
                "test_end": _last(outer_fold.test_index),
                **outer_metrics,
            }
        )
        oos_pieces.append(
            pd.DataFrame(
                {
                    "return": outer_returns,
                    "outer_fold": outer_fold.fold_id,
                    "candidate_id": selected_id,
                    "parameter_fingerprint": _parameter_fingerprint(frozen_parameter),
                },
                index=outer_fold.test_index,
            )
        )

    oos = pd.concat(oos_pieces).sort_index()
    if oos.index.has_duplicates:
        raise RuntimeError("outer test windows unexpectedly overlap")
    oos.index.name = chronology.name or "date"
    performance = pd.DataFrame(
        [performance_statistics(oos["return"], annualization=spec.annualization)]
    )
    bootstrap = bootstrap_performance_confidence_intervals(
        oos["return"],
        samples=spec.bootstrap_samples,
        block_size=spec.bootstrap_block_size,
        confidence_level=spec.confidence_level,
        annualization=spec.annualization,
        seed=spec.random_seed,
    )
    pbo_audit = pd.DataFrame()
    if diagnostics is not None:
        pbo_audit, pbo_summary = pbo_style_diagnostic(
            diagnostics,
            outer_folds,
            selected_by_fold,
            metric=spec.selection_metric,
            annualization=spec.annualization,
        )
        reality_rows.append({"outer_fold": math.nan, **pbo_summary})

    return NestedWalkForwardResult(
        spec=spec,
        oos_returns=oos,
        performance=performance,
        fold_audit=pd.concat(fold_audits, ignore_index=True),
        inner_evaluations=pd.DataFrame(inner_evaluation_rows),
        selection_audit=pd.DataFrame(selection_rows),
        outer_evaluations=pd.DataFrame(outer_evaluation_rows),
        bootstrap_confidence_intervals=bootstrap,
        data_snooping_diagnostics=pd.DataFrame(reality_rows),
        pbo_fold_audit=pbo_audit,
    )


def run_nested_walk_forward_from_returns(
    candidate_returns: pd.DataFrame,
    spec: NestedWalkForwardSpec,
    *,
    score_function: ScoreFunction | None = None,
) -> NestedWalkForwardResult:
    """Fast path for cached returns; no strategy/grid cell is rerun.

    Columns are stable candidate identifiers.  The same matrix is passed as the
    optional diagnostic cache, enabling the PBO-style outer-rank audit without
    changing which candidate supplies the reported OOS return.
    """

    if candidate_returns.empty or candidate_returns.shape[1] < 1:
        raise ValueError("candidate_returns must contain at least one candidate")
    frame = candidate_returns.copy()
    frame.index = _validated_index(frame.index)
    frame.columns = frame.columns.map(str)
    if frame.columns.has_duplicates:
        raise ValueError("candidate_returns columns must be unique after string conversion")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    candidates = {str(column): str(column) for column in frame.columns}

    def cached_evaluator(
        column: str,
        _train_index: pd.Index,
        evaluation_index: pd.Index,
    ) -> pd.Series:
        return frame[column].reindex(evaluation_index)

    return run_nested_walk_forward(
        frame.index,
        candidates,
        cached_evaluator,
        spec,
        score_function=score_function,
        diagnostic_returns=frame,
    )


__all__ = [
    "NestedWalkForwardResult",
    "NestedWalkForwardSpec",
    "PurgedFold",
    "ReturnEvaluator",
    "ScoreFunction",
    "bootstrap_performance_confidence_intervals",
    "build_purged_walk_forward_folds",
    "folds_to_audit_table",
    "performance_statistics",
    "pbo_style_diagnostic",
    "reality_check",
    "run_nested_walk_forward",
    "run_nested_walk_forward_from_returns",
]
