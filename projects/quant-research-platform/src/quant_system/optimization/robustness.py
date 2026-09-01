"""Robustness diagnostics."""

from __future__ import annotations

from itertools import combinations
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


def flag_overfit_risk(results: pd.DataFrame, objective_column: str = "objective") -> pd.DataFrame:
    """Flag unstable parameter results with a simple percentile heuristic."""

    if results.empty or objective_column not in results:
        return results
    out = results.copy()
    threshold = out[objective_column].quantile(0.9)
    median = out[objective_column].median()
    out["overfit_risk"] = (out[objective_column] >= threshold) & (out[objective_column] > median * 2)
    return out


def parameter_stability(results: pd.DataFrame, parameter_columns: list[str], objective_column: str = "validation_objective") -> pd.DataFrame:
    """Compute simple local stability by grouping around each parameter value."""

    if results.empty:
        return pd.DataFrame()
    rows = []
    for parameter in parameter_columns:
        if parameter not in results.columns:
            continue
        grouped = results.groupby(parameter)[objective_column].agg(["mean", "std", "count"]).reset_index()
        grouped["parameter"] = parameter
        grouped = grouped.rename(columns={parameter: "value", "mean": "objective_mean", "std": "objective_std"})
        rows.append(grouped[["parameter", "value", "objective_mean", "objective_std", "count"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def estimate_pbo(
    optimization_details: pd.DataFrame,
    *,
    scenario_col: str = "scenario",
    split_col: str = "split_id",
    score_col: str = "validation_objective",
) -> dict[str, Any]:
    """Estimate Probability of Backtest Overfitting from walk-forward results.

    This CSCV-style diagnostic splits validation windows into pseudo train/test
    combinations. For each combination it selects the best scenario in the
    pseudo train half and checks whether that winner ranks in the lower half of
    the pseudo out-of-sample half.
    """

    required = {scenario_col, split_col, score_col}
    if optimization_details.empty or not required.issubset(optimization_details.columns):
        return {"pbo": np.nan, "combinations": 0, "status": "insufficient_data"}
    frame = optimization_details[[scenario_col, split_col, score_col]].copy()
    frame[scenario_col] = frame[scenario_col].astype(str)
    frame[split_col] = frame[split_col].astype(str)
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    pivot = frame.pivot_table(index=scenario_col, columns=split_col, values=score_col, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
    windows = list(pivot.columns)
    if len(pivot) < 2 or len(windows) < 2:
        return {"pbo": np.nan, "combinations": 0, "status": "insufficient_data"}

    train_size = max(1, len(windows) // 2)
    logits: list[float] = []
    selection_details: list[dict[str, Any]] = []
    for train_windows in combinations(windows, train_size):
        test_windows = [window for window in windows if window not in train_windows]
        if not test_windows:
            continue
        train_score = pivot.loc[:, list(train_windows)].mean(axis=1, skipna=True)
        test_score = pivot.loc[:, test_windows].mean(axis=1, skipna=True)
        valid = pd.DataFrame({"train": train_score, "test": test_score}).dropna()
        if len(valid) < 2:
            continue
        winner = str(valid["train"].idxmax())
        ranks = valid["test"].rank(method="average", ascending=True)
        omega = float(ranks.loc[winner] / (len(valid) + 1.0))
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logit = math.log(omega / (1.0 - omega))
        logits.append(logit)
        selection_details.append(
            {
                "train_windows": ",".join(map(str, train_windows)),
                "test_windows": ",".join(map(str, test_windows)),
                "winner": winner,
                "test_rank_percentile": omega,
                "logit": logit,
            }
        )
    if not logits:
        return {"pbo": np.nan, "combinations": 0, "status": "insufficient_data"}
    return {
        "pbo": float(np.mean(np.asarray(logits) < 0.0)),
        "combinations": len(logits),
        "status": "ok",
        "logit_mean": float(np.mean(logits)),
        "selection_details": selection_details,
    }


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    observed_sharpe: float | None = None,
    trials: int = 1,
    periods_per_year: int = 252,
) -> dict[str, float | int | str]:
    """Compute a conservative Deflated Sharpe Ratio probability.

    Defaults are suitable for optimizer governance: ``trials`` should be the
    number of candidate parameter sets in the same selection family.
    """

    values = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(values))
    if n < 30:
        return {"dsr": np.nan, "status": "insufficient_data", "n": n, "trials": int(max(trials, 1))}
    daily_std = float(values.std(ddof=1))
    if daily_std <= 0 or not np.isfinite(daily_std):
        return {"dsr": np.nan, "status": "insufficient_data", "n": n, "trials": int(max(trials, 1))}
    sr = float(observed_sharpe) if observed_sharpe is not None else float(values.mean() / daily_std * math.sqrt(periods_per_year))
    skew = float(values.skew())
    kurtosis = float(values.kurtosis() + 3.0)
    trial_count = int(max(trials, 1))
    denominator = math.sqrt(max(1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr, 1e-12))
    sr_std = denominator / math.sqrt(max(n - 1, 1))
    sr_star = _expected_max_sharpe(0.0, sr_std, trial_count)
    z_score = (sr - sr_star) * math.sqrt(n - 1) / denominator
    return {
        "dsr": float(NormalDist().cdf(z_score)),
        "status": "ok",
        "n": n,
        "trials": trial_count,
        "observed_sharpe": sr,
        "benchmark_sharpe": sr_star,
        "skew": skew,
        "kurtosis": kurtosis,
        "z_score": float(z_score),
    }


def _expected_max_sharpe(mean_sr: float, std_sr: float, trials: int) -> float:
    if trials <= 1 or std_sr <= 0:
        return mean_sr
    normal = NormalDist()
    gamma = 0.5772156649015329
    first = normal.inv_cdf(1.0 - 1.0 / trials)
    second = normal.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return float(mean_sr + std_sr * ((1.0 - gamma) * first + gamma * second))
