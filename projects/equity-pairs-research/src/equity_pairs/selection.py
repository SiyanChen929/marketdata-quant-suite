"""Leakage-safe, diversification-aware selection of pair-strategy sleeves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .backtest import performance_metrics


@dataclass(frozen=True)
class PairSelectionSpec:
    pair_count: int = 10
    maximum_pairs_per_sector: int = 2
    maximum_pairs_per_ticker: int = 1
    minimum_sector_count: int = 6
    minimum_trades: int = 6
    sharpe_weight: float = 0.30
    cagr_weight: float = 0.20
    worst_half_sharpe_weight: float = 0.20
    positive_month_fraction_weight: float = 0.15
    drawdown_weight: float = 0.15


def _validate_spec(spec: PairSelectionSpec) -> None:
    integer_fields = {
        "pair_count": spec.pair_count,
        "maximum_pairs_per_sector": spec.maximum_pairs_per_sector,
        "maximum_pairs_per_ticker": spec.maximum_pairs_per_ticker,
        "minimum_sector_count": spec.minimum_sector_count,
        "minimum_trades": spec.minimum_trades,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integer_fields.values()):
        raise ValueError("pair-selection counts must be positive integers")
    if spec.minimum_sector_count > spec.pair_count:
        raise ValueError("minimum_sector_count cannot exceed pair_count")
    weights = np.array(
        [
            spec.sharpe_weight,
            spec.cagr_weight,
            spec.worst_half_sharpe_weight,
            spec.positive_month_fraction_weight,
            spec.drawdown_weight,
        ],
        dtype=float,
    )
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("selection score weights must be non-negative and sum to one")


def _monthly_positive_fraction(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise TypeError("pair returns must use a DatetimeIndex")
    monthly = (1.0 + clean).groupby(clean.index.to_period("M")).prod() - 1.0
    return float((monthly > 0).mean()) if len(monthly) else np.nan


def build_pair_selection_features(
    pair_returns: pd.DataFrame,
    pair_metadata: pd.DataFrame,
    *,
    trades: pd.DataFrame | None = None,
    spec: PairSelectionSpec = PairSelectionSpec(),
) -> pd.DataFrame:
    """Score pair sleeves using only the return window supplied by the caller.

    The score rewards full-window risk-adjusted return, the weaker half-window,
    positive-month consistency, and shallower drawdown.  All components are
    cross-sectional percentile ranks so no one metric's scale dominates.
    """
    _validate_spec(spec)
    if not isinstance(pair_returns, pd.DataFrame) or pair_returns.empty:
        raise ValueError("pair_returns must be a non-empty DataFrame")
    if not pair_returns.columns.is_unique:
        raise ValueError("pair_returns columns must be unique pair identifiers")
    if not isinstance(pair_returns.index, pd.DatetimeIndex):
        raise TypeError("pair_returns must use a DatetimeIndex")
    required = {"pair", "sector", "dependent", "independent"}
    missing = required - set(pair_metadata.columns)
    if missing:
        raise ValueError(f"pair_metadata missing columns: {', '.join(sorted(missing))}")
    if pair_metadata["pair"].duplicated().any():
        raise ValueError("pair_metadata must contain unique pair identifiers")
    metadata = pair_metadata.drop_duplicates("pair").set_index("pair")
    unknown = pair_returns.columns.difference(metadata.index)
    if len(unknown):
        raise ValueError("pair_metadata missing pairs: " + ", ".join(map(str, unknown)))
    midpoint = len(pair_returns) // 2
    if midpoint < 2 or len(pair_returns) - midpoint < 2:
        raise ValueError("pair_returns needs at least four observations")
    trade_counts = pd.Series(0, index=pair_returns.columns, dtype=int)
    if trades is not None and not trades.empty:
        if not {"pair", "entry_date"}.issubset(trades.columns):
            raise ValueError("trades must contain pair and entry_date")
        entries = trades.copy()
        entries["entry_date"] = pd.to_datetime(entries["entry_date"])
        start, end = pair_returns.index.min(), pair_returns.index.max()
        entries = entries.loc[entries["entry_date"].between(start, end)]
        trade_counts = entries["pair"].value_counts().reindex(pair_returns.columns, fill_value=0).astype(int)
    rows: list[dict[str, Any]] = []
    for pair in pair_returns.columns:
        returns = pd.to_numeric(pair_returns[pair], errors="coerce").fillna(0.0)
        full = performance_metrics(returns)
        first = performance_metrics(returns.iloc[:midpoint])
        second = performance_metrics(returns.iloc[midpoint:])
        half_sharpes = [float(first["sharpe"]), float(second["sharpe"])]
        worst_half = min(half_sharpes) if np.isfinite(half_sharpes).all() else np.nan
        row = metadata.loc[pair]
        rows.append(
            {
                "pair": str(pair),
                "sector": str(row["sector"]),
                "dependent": str(row["dependent"]),
                "independent": str(row["independent"]),
                "tuning_cagr": full["cagr"],
                "tuning_sharpe": full["sharpe"],
                "tuning_max_drawdown": full["max_drawdown"],
                "first_half_sharpe": first["sharpe"],
                "second_half_sharpe": second["sharpe"],
                "worst_half_sharpe": worst_half,
                "positive_month_fraction": _monthly_positive_fraction(returns),
                "tuning_trades": int(trade_counts.loc[pair]),
                "selection_status": row.get("selection_status", "unknown"),
                "formation_pair_pvalue": row.get("pair_pvalue", np.nan),
                "formation_fdr_qvalue": row.get("fdr_qvalue", np.nan),
            }
        )
    features = pd.DataFrame(rows)
    rank_columns = {
        "tuning_sharpe": "sharpe_percentile",
        "tuning_cagr": "cagr_percentile",
        "worst_half_sharpe": "worst_half_sharpe_percentile",
        "positive_month_fraction": "positive_month_percentile",
        "tuning_max_drawdown": "drawdown_percentile",
    }
    for source, destination in rank_columns.items():
        features[destination] = features[source].rank(method="average", pct=True)
    features["selection_score"] = (
        spec.sharpe_weight * features["sharpe_percentile"]
        + spec.cagr_weight * features["cagr_percentile"]
        + spec.worst_half_sharpe_weight * features["worst_half_sharpe_percentile"]
        + spec.positive_month_fraction_weight * features["positive_month_percentile"]
        + spec.drawdown_weight * features["drawdown_percentile"]
    )
    return features.sort_values(
        ["selection_score", "tuning_sharpe", "pair"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def select_diversified_pairs(
    features: pd.DataFrame,
    spec: PairSelectionSpec = PairSelectionSpec(),
) -> pd.DataFrame:
    """Greedily take the highest training score subject to hard diversity caps."""
    _validate_spec(spec)
    required = {
        "pair",
        "sector",
        "dependent",
        "independent",
        "selection_score",
        "tuning_trades",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"features missing columns: {', '.join(sorted(missing))}")
    if features["pair"].astype(str).duplicated().any():
        raise ValueError("features must contain unique pair identifiers")
    ordered = features.loc[
        features["tuning_trades"].ge(spec.minimum_trades)
        & np.isfinite(pd.to_numeric(features["selection_score"], errors="coerce"))
    ].sort_values(
        ["selection_score", "tuning_sharpe", "pair"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    chosen: list[dict[str, Any]] = []
    ticker_counts: dict[str, int] = {}
    sector_counts: dict[str, int] = {}
    for row in ordered.to_dict(orient="records"):
        sector = str(row["sector"])
        tickers = (str(row["dependent"]), str(row["independent"]))
        if sector_counts.get(sector, 0) >= spec.maximum_pairs_per_sector:
            continue
        if any(ticker_counts.get(ticker, 0) >= spec.maximum_pairs_per_ticker for ticker in tickers):
            continue
        chosen.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        for ticker in tickers:
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        if len(chosen) == spec.pair_count:
            break
    if len(chosen) != spec.pair_count:
        raise ValueError(
            f"diversification caps allow only {len(chosen)} of {spec.pair_count} required pairs"
        )
    result = pd.DataFrame(chosen)
    sector_count = int(result["sector"].nunique())
    if sector_count < spec.minimum_sector_count:
        raise ValueError(
            f"greedy selection spans {sector_count} sectors; require {spec.minimum_sector_count}"
        )
    result.insert(0, "selection_rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


def select_constrained_pairs_milp(
    features: pd.DataFrame,
    spec: PairSelectionSpec = PairSelectionSpec(),
    *,
    pair_betas: pd.Series | None = None,
    equal_weight_beta_bound: float | None = None,
    pair_correlations: pd.DataFrame | None = None,
    maximum_pair_correlation: float | None = None,
) -> pd.DataFrame:
    """Solve the exactly-ten selection problem with explicit binary constraints."""
    _validate_spec(spec)
    required = {
        "pair",
        "sector",
        "dependent",
        "independent",
        "selection_score",
        "tuning_trades",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"features missing columns: {', '.join(sorted(missing))}")
    if features["pair"].astype(str).duplicated().any():
        raise ValueError("features must contain unique pair identifiers")
    candidates = features.loc[
        features["tuning_trades"].ge(spec.minimum_trades)
        & np.isfinite(pd.to_numeric(features["selection_score"], errors="coerce"))
    ].copy()
    candidates = candidates.sort_values("pair", kind="mergesort").reset_index(drop=True)
    if len(candidates) < spec.pair_count:
        raise ValueError("fewer eligible candidates than required pairs")
    pairs = candidates["pair"].astype(str).tolist()
    sectors = sorted(candidates["sector"].astype(str).unique())
    pair_count = len(pairs)
    variable_count = pair_count + len(sectors)
    sector_offset = pair_count
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    exactly = np.zeros(variable_count)
    exactly[:pair_count] = 1.0
    rows.append(exactly)
    lower.append(float(spec.pair_count))
    upper.append(float(spec.pair_count))

    all_tickers = sorted(
        set(candidates["dependent"].astype(str))
        | set(candidates["independent"].astype(str))
    )
    for ticker in all_tickers:
        row = np.zeros(variable_count)
        contains = candidates["dependent"].astype(str).eq(ticker) | candidates["independent"].astype(str).eq(ticker)
        row[:pair_count] = contains.to_numpy(dtype=float)
        rows.append(row)
        lower.append(-np.inf)
        upper.append(float(spec.maximum_pairs_per_ticker))

    for sector_index, sector in enumerate(sectors):
        sector_pairs = candidates["sector"].astype(str).eq(sector).to_numpy(dtype=float)
        link_upper = np.zeros(variable_count)
        link_upper[:pair_count] = sector_pairs
        link_upper[sector_offset + sector_index] = -float(spec.maximum_pairs_per_sector)
        rows.append(link_upper)
        lower.append(-np.inf)
        upper.append(0.0)
        link_lower = np.zeros(variable_count)
        link_lower[:pair_count] = sector_pairs
        link_lower[sector_offset + sector_index] = -1.0
        rows.append(link_lower)
        lower.append(0.0)
        upper.append(np.inf)
    minimum_sectors = np.zeros(variable_count)
    minimum_sectors[sector_offset:] = 1.0
    rows.append(minimum_sectors)
    lower.append(float(spec.minimum_sector_count))
    upper.append(np.inf)

    beta_values: pd.Series | None = None
    if equal_weight_beta_bound is not None:
        if pair_betas is None:
            raise ValueError("pair_betas are required for an equal-weight beta bound")
        if not np.isfinite(equal_weight_beta_bound) or equal_weight_beta_bound < 0:
            raise ValueError("equal_weight_beta_bound must be finite and non-negative")
        beta_values = pd.to_numeric(pair_betas, errors="coerce").reindex(pairs)
        if beta_values.isna().any() or not np.isfinite(beta_values.to_numpy()).all():
            raise ValueError("pair_betas must provide finite values for every candidate")
        beta_row = np.zeros(variable_count)
        beta_row[:pair_count] = beta_values.to_numpy(dtype=float)
        rows.append(beta_row)
        magnitude = float(equal_weight_beta_bound) * spec.pair_count
        lower.append(-magnitude)
        upper.append(magnitude)

    if maximum_pair_correlation is not None:
        if pair_correlations is None:
            raise ValueError("pair_correlations are required for a correlation cap")
        if not 0 <= maximum_pair_correlation <= 1:
            raise ValueError("maximum_pair_correlation must be in [0, 1]")
        matrix = pair_correlations.reindex(index=pairs, columns=pairs)
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("pair_correlations must cover every candidate with finite values")
        if not np.allclose(matrix.to_numpy(), matrix.to_numpy().T, atol=1e-10):
            raise ValueError("pair_correlations must be symmetric")
        for left in range(pair_count):
            for right in range(left + 1, pair_count):
                if abs(float(matrix.iloc[left, right])) <= maximum_pair_correlation:
                    continue
                incompatible = np.zeros(variable_count)
                incompatible[left] = 1.0
                incompatible[right] = 1.0
                rows.append(incompatible)
                lower.append(-np.inf)
                upper.append(1.0)

    objective = np.zeros(variable_count)
    # The tiny stable tie-break favors alphabetically earlier pair IDs only
    # when the declared score is exactly tied.
    objective[:pair_count] = -candidates["selection_score"].to_numpy(dtype=float) + np.arange(pair_count) * 1e-10
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={"time_limit": 30.0},
    )
    if not result.success or result.x is None:
        raise ValueError(f"pair-selection constraints are infeasible: {result.message}")
    selected_mask = np.asarray(result.x[:pair_count]) > 0.5
    selected = candidates.loc[selected_mask].copy()
    if len(selected) != spec.pair_count:
        raise RuntimeError("MILP returned an unexpected number of selected pairs")
    selected = selected.sort_values(
        ["selection_score", "tuning_sharpe", "pair"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    if beta_values is not None:
        selected["tuning_market_beta"] = beta_values.reindex(selected["pair"]).to_numpy()
    selected.attrs["optimization_status"] = int(result.status)
    selected.attrs["optimization_message"] = str(result.message)
    selected.attrs["objective_score"] = float(selected["selection_score"].sum())
    return selected


def alpha_tilted_risk_budgets(
    selected_features: pd.DataFrame,
    *,
    minimum_multiplier: float = 0.70,
    maximum_multiplier: float = 1.30,
    tilt_strength: float = 0.20,
) -> pd.Series:
    """Convert training scores into modest, normalized risk-budget tilts."""
    if not {"pair", "selection_score"}.issubset(selected_features.columns):
        raise ValueError("selected_features must contain pair and selection_score")
    if not 0 < minimum_multiplier <= 1 <= maximum_multiplier:
        raise ValueError("risk-budget multipliers must bracket one")
    scores = pd.to_numeric(selected_features.set_index("pair")["selection_score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("selection scores must be finite")
    standard_deviation = float(scores.std(ddof=0))
    zscore = (scores - scores.mean()) / standard_deviation if standard_deviation > 0 else scores * 0.0
    multipliers = (1.0 + tilt_strength * zscore).clip(
        lower=minimum_multiplier,
        upper=maximum_multiplier,
    )
    budgets = multipliers / multipliers.sum()
    budgets.name = "target_risk_budget"
    return budgets
