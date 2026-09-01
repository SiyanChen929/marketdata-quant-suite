"""Forecast shrinkage and operational backtesting for weighted pair sleeves."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .backtest import performance_metrics
from .execution import ExecutableCostModel


@dataclass(frozen=True)
class WeightedPortfolioResult:
    daily: pd.DataFrame
    deployed_weights: pd.DataFrame
    realized_weights: pd.DataFrame
    rebalance_log: pd.DataFrame
    metrics: dict[str, Any]
    execution_log: pd.DataFrame | None = None


def build_gross_name_incidence(pair_metadata: pd.DataFrame) -> pd.DataFrame:
    """Map pair capital to conservative gross-normalized underlying exposure."""
    required = {"pair", "dependent", "independent", "beta"}
    missing = required - set(pair_metadata.columns)
    if missing:
        raise ValueError(f"pair_metadata missing columns: {', '.join(sorted(missing))}")
    if pair_metadata["pair"].duplicated().any():
        raise ValueError("pair_metadata must contain unique pairs")
    names = sorted(
        set(pair_metadata["dependent"].astype(str))
        | set(pair_metadata["independent"].astype(str))
    )
    incidence = pd.DataFrame(
        0.0,
        index=pair_metadata["pair"].astype(str),
        columns=names,
    )
    for row in pair_metadata.to_dict(orient="records"):
        beta = float(row["beta"])
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f"pair {row['pair']} requires a positive finite beta")
        denominator = 1.0 + beta
        incidence.loc[str(row["pair"]), str(row["dependent"])] = 1.0 / denominator
        incidence.loc[str(row["pair"]), str(row["independent"])] = beta / denominator
    return incidence


def empirical_bayes_monthly_forecasts(
    pair_returns: pd.DataFrame,
    *,
    annualization: int = 12,
    maximum_forecast_sharpe: float = 0.50,
) -> pd.DataFrame:
    """Shrink monthly mean returns toward zero and cap implied Sharpe.

    The cross-sectional prior variance removes average sampling variance from
    the dispersion of monthly sample means.  No future observations are used;
    callers are responsible for passing only the estimation window.
    """
    if not isinstance(pair_returns.index, pd.DatetimeIndex):
        raise TypeError("pair_returns must use a DatetimeIndex")
    if pair_returns.empty:
        raise ValueError("pair_returns cannot be empty")
    clean = pair_returns.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    monthly = (1.0 + clean).groupby(clean.index.to_period("M")).prod() - 1.0
    if len(monthly) < 4:
        raise ValueError("at least four monthly observations are required")
    sample_mean = monthly.mean()
    sampling_variance = monthly.var(ddof=1) / len(monthly)
    cross_sectional_variance = float(sample_mean.var(ddof=1)) if len(sample_mean) > 1 else 0.0
    prior_variance = max(0.0, cross_sectional_variance - float(sampling_variance.mean()))
    denominator = prior_variance + sampling_variance
    shrinkage_intensity = pd.Series(
        np.where(denominator > 0, prior_variance / denominator, 0.0),
        index=clean.columns,
        dtype=float,
    )
    shrunk_annual_return = annualization * shrinkage_intensity * sample_mean
    annual_volatility = monthly.std(ddof=1) * math.sqrt(annualization)
    forecast_sharpe = shrunk_annual_return / annual_volatility.replace(0.0, np.nan)
    capped_sharpe = forecast_sharpe.clip(
        lower=-maximum_forecast_sharpe,
        upper=maximum_forecast_sharpe,
    ).fillna(0.0)
    capped_return = capped_sharpe * annual_volatility.fillna(0.0)
    return pd.DataFrame(
        {
            "monthly_sample_mean": sample_mean,
            "sampling_variance_of_mean": sampling_variance,
            "prior_variance": prior_variance,
            "posterior_data_weight": shrinkage_intensity,
            "shrunk_annual_return": shrunk_annual_return,
            "annual_volatility": annual_volatility,
            "forecast_sharpe": forecast_sharpe,
            "capped_forecast_sharpe": capped_sharpe,
            "capped_annual_return": capped_return,
        }
    )


def alpha_tilted_risk_budgets_from_forecasts(
    forecasts: pd.DataFrame,
    *,
    tilt_strength: float = 0.15,
    minimum_multiplier: float = 0.70,
    maximum_multiplier: float = 1.30,
) -> pd.Series:
    """Translate shrunken forecast Sharpe into deliberately modest risk tilts."""
    if "capped_forecast_sharpe" not in forecasts:
        raise ValueError("forecasts must contain capped_forecast_sharpe")
    values = pd.to_numeric(forecasts["capped_forecast_sharpe"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("forecast Sharpe values must be finite")
    scale = float(values.std(ddof=0))
    standardized = (values - values.mean()) / scale if scale > 0 else values * 0.0
    standardized = standardized.clip(-2.0, 2.0)
    multipliers = (1.0 + tilt_strength * standardized).clip(
        minimum_multiplier,
        maximum_multiplier,
    )
    budgets = multipliers / multipliers.sum()
    budgets.name = "target_risk_budget"
    return budgets


def backtest_weighted_sleeves(
    pair_returns: pd.DataFrame,
    target_weights: pd.Series,
    *,
    active_positions: pd.DataFrame | None = None,
    unit_turnover: pd.DataFrame | None = None,
    allocation_cost_bps: float = 5.0,
    rebalance_frequency: str = "monthly",
    no_trade_band: float = 0.01,
    maximum_rebalance_turnover: float = 0.15,
    charge_launch: bool = True,
    charge_liquidation: bool = True,
) -> WeightedPortfolioResult:
    """Backtest fixed target budgets with explicit outer-book mechanics.

    Pair returns already contain the pair backtest's leg turnover and borrow
    charges.  This layer holds sleeve capital between rebalances and charges
    only incremental allocation scaling, launch, and terminal liquidation.
    """
    if pair_returns.empty or not isinstance(pair_returns.index, pd.DatetimeIndex):
        raise ValueError("pair_returns must be non-empty with a DatetimeIndex")
    returns = pair_returns.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weights = pd.to_numeric(target_weights, errors="coerce").reindex(returns.columns)
    if weights.isna().any() or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("target_weights must be finite, non-negative, aligned, and sum to one")
    if not math.isfinite(allocation_cost_bps) or allocation_cost_bps < 0:
        raise ValueError("allocation_cost_bps must be finite and non-negative")
    if not 0 <= no_trade_band <= 1 or not 0 < maximum_rebalance_turnover <= 2:
        raise ValueError("invalid no-trade band or rebalance-turnover limit")
    if rebalance_frequency not in {"monthly", "quarterly", "none"}:
        raise ValueError("rebalance_frequency must be monthly, quarterly, or none")
    positions = (
        active_positions.reindex(index=returns.index, columns=returns.columns).fillna(0.0).ne(0)
        if active_positions is not None
        else pd.DataFrame(True, index=returns.index, columns=returns.columns)
    )
    turnovers = (
        unit_turnover.reindex(index=returns.index, columns=returns.columns).fillna(0.0).clip(lower=0.0)
        if unit_turnover is not None
        else pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    )
    capital = weights.to_numpy(dtype=float).copy()
    target = weights.to_numpy(dtype=float)
    cost_rate = allocation_cost_bps / 10_000.0
    equity_values: list[float] = []
    return_values: list[float] = []
    gross_return_values: list[float] = []
    cost_values: list[float] = []
    deployed_weight_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    rebalance_rows: list[dict[str, Any]] = []
    prior_equity = 1.0
    dates = returns.index
    prior_period: Any = None

    for location, date in enumerate(dates):
        total_before = float(capital.sum())
        current = capital / total_before
        active = positions.iloc[location].to_numpy(dtype=bool)
        allocation_turnover = 0.0
        event = "hold"
        if location == 0 and charge_launch:
            # Unit pair returns may already include some entry turnover on this
            # date. Charge only the missing portion needed to launch an active
            # sleeve from zero capital.
            already_charged = np.minimum(
                turnovers.iloc[location].to_numpy(dtype=float),
                1.0,
            )
            allocation_turnover = float(np.sum(target * active * (1.0 - already_charged)))
            event = "launch"
        elif location > 0 and rebalance_frequency != "none":
            period = date.to_period("M" if rebalance_frequency == "monthly" else "Q")
            if period != prior_period:
                difference = target - current
                if float(np.max(np.abs(difference))) > max(no_trade_band, 1e-12):
                    scaling = min(
                        1.0,
                        maximum_rebalance_turnover / max(float(np.abs(difference).sum()), 1e-18),
                    )
                    proposed = current + scaling * difference
                    proposed = np.maximum(proposed, 0.0)
                    proposed /= proposed.sum()
                    gross_weight_turnover = float(np.abs(proposed - current).sum())
                    allocation_turnover = float(np.sum(np.abs(proposed - current) * active))
                    cost = total_before * allocation_turnover * cost_rate
                    total_before -= cost
                    capital = proposed * total_before
                    current = proposed
                    event = "scheduled_rebalance"
                    rebalance_rows.append(
                        {
                            "date": date,
                            "event": event,
                            "gross_weight_turnover": gross_weight_turnover,
                            "active_allocation_turnover": allocation_turnover,
                            "cost": cost,
                            "blend_fraction": scaling,
                        }
                    )
        if location == 0:
            launch_cost = total_before * allocation_turnover * cost_rate
            total_before -= launch_cost
            capital = current * total_before
            if charge_launch:
                rebalance_rows.append(
                    {
                        "date": date,
                        "event": event,
                        "gross_weight_turnover": allocation_turnover,
                        "active_allocation_turnover": allocation_turnover,
                        "cost": launch_cost,
                        "blend_fraction": 1.0,
                    }
                )
        # These are the capital weights actually exposed to today's pair
        # returns.  End-of-day realized weights are stored separately below.
        deployed_weight_rows.append(current.copy())
        day_returns = returns.iloc[location].to_numpy(dtype=float)
        gross_portfolio_return = float(current @ day_returns)
        capital *= 1.0 + day_returns
        total_after = float(capital.sum())
        day_cost = prior_equity - total_before if location == 0 else 0.0
        if location > 0 and rebalance_rows and rebalance_rows[-1]["date"] == date:
            day_cost = float(rebalance_rows[-1]["cost"])
        if location == len(dates) - 1 and charge_liquidation:
            ending_weights = capital / total_after
            liquidation_turnover = float(np.sum(ending_weights * active))
            liquidation_cost = total_after * liquidation_turnover * cost_rate
            capital *= (total_after - liquidation_cost) / total_after
            total_after = float(capital.sum())
            day_cost += liquidation_cost
            rebalance_rows.append(
                {
                    "date": date,
                    "event": "terminal_liquidation",
                    "gross_weight_turnover": liquidation_turnover,
                    "active_allocation_turnover": liquidation_turnover,
                    "cost": liquidation_cost,
                    "blend_fraction": 1.0,
                }
            )
        equity_values.append(total_after)
        return_values.append(total_after / prior_equity - 1.0)
        gross_return_values.append(gross_portfolio_return)
        cost_values.append(day_cost)
        weight_rows.append(capital / total_after)
        prior_equity = total_after
        prior_period = date.to_period("M" if rebalance_frequency == "monthly" else "Q") if rebalance_frequency != "none" else None

    daily = pd.DataFrame(
        {
            "portfolio_return": return_values,
            "gross_weighted_pair_return": gross_return_values,
            "outer_allocation_cost": cost_values,
            "equity": equity_values,
        },
        index=dates,
    )
    daily.index.name = "date"
    realized_weights = pd.DataFrame(weight_rows, index=dates, columns=returns.columns)
    realized_weights.index.name = "date"
    deployed_weights = pd.DataFrame(
        deployed_weight_rows,
        index=dates,
        columns=returns.columns,
    )
    deployed_weights.index.name = "date"
    log = pd.DataFrame(
        rebalance_rows,
        columns=[
            "date",
            "event",
            "gross_weight_turnover",
            "active_allocation_turnover",
            "cost",
            "blend_fraction",
        ],
    )
    metrics = performance_metrics(daily["portfolio_return"], daily["equity"])
    event_turnover = (
        log.groupby("event")["active_allocation_turnover"].sum()
        if not log.empty
        else pd.Series(dtype=float)
    )
    event_cost = (
        log.groupby("event")["cost"].sum()
        if not log.empty
        else pd.Series(dtype=float)
    )
    metrics.update(
        {
            "outer_allocation_cost_sum": float(daily["outer_allocation_cost"].sum()),
            "outer_allocation_cost_bps_on_initial_capital": float(
                daily["outer_allocation_cost"].sum() * 10_000.0
            ),
            "outer_allocation_turnover_sum": float(log["active_allocation_turnover"].sum()) if not log.empty else 0.0,
            "launch_turnover": float(event_turnover.get("launch", 0.0)),
            "scheduled_rebalance_turnover": float(
                event_turnover.get("scheduled_rebalance", 0.0)
            ),
            "terminal_liquidation_turnover": float(
                event_turnover.get("terminal_liquidation", 0.0)
            ),
            "launch_cost": float(event_cost.get("launch", 0.0)),
            "scheduled_rebalance_cost": float(
                event_cost.get("scheduled_rebalance", 0.0)
            ),
            "terminal_liquidation_cost": float(
                event_cost.get("terminal_liquidation", 0.0)
            ),
            "scheduled_rebalance_count": int(log["event"].eq("scheduled_rebalance").sum()) if not log.empty else 0,
            "launch_cost_charged": bool(charge_launch),
            "terminal_liquidation_cost_charged": bool(charge_liquidation),
            "rebalance_frequency": rebalance_frequency,
        }
    )
    return WeightedPortfolioResult(
        daily=daily,
        deployed_weights=deployed_weights,
        realized_weights=realized_weights,
        rebalance_log=log,
        metrics=metrics,
    )


def backtest_weighted_pair_book(
    signals: pd.DataFrame,
    target_weights: pd.Series,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    transaction_cost_bps: float = 5.0,
    rebalance_frequency: str = "monthly",
    no_trade_band: float = 0.01,
    maximum_rebalance_turnover: float = 0.15,
    charge_terminal_liquidation: bool = True,
    execution_cost_model: ExecutableCostModel | None = None,
    pair_metadata: pd.DataFrame | None = None,
    fail_on_unexecutable: bool = True,
    target_weight_schedule: pd.DataFrame | None = None,
) -> WeightedPortfolioResult:
    """Backtest a weighted pair book from exact pre/post leg notionals.

    Unlike :func:`backtest_weighted_sleeves`, this routine does not scale pair
    returns that already contain transaction costs and then add a second outer
    cost.  It reconstructs each dependent and independent leg in portfolio
    dollars and charges once on the absolute change in those notionals.  This
    handles a signal entry or exit that coincides with a capital rebalance.

    ``signals`` must contain the complete chronological signal history needed
    to calculate leg price returns.  ``start`` and ``end`` are applied only
    after those returns are calculated.  The input ``borrow_cost`` is the
    per-dollar sleeve charge produced by the pair backtester; embedded
    ``transaction_cost`` and ``net_return`` columns are deliberately ignored.

    When ``execution_cost_model`` is supplied, its ticker-level commission,
    spread, nonlinear impact, borrow, financing, cash-yield, short-rebate,
    capacity, and locate logic replaces both the flat transaction-cost rate and
    the signal-level borrow-cost column.  ``pair_metadata`` is then required so
    pair legs can be netted by underlying before costs are evaluated.
    """
    required = {
        "pair",
        "date",
        "price_dependent",
        "price_independent",
        "weight_dependent",
        "weight_independent",
        "gross_return",
        "borrow_cost",
    }
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals missing columns: {', '.join(sorted(missing))}")
    if not math.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    if rebalance_frequency not in {"monthly", "quarterly", "none"}:
        raise ValueError("rebalance_frequency must be monthly, quarterly, or none")
    if not math.isfinite(no_trade_band) or not 0 <= no_trade_band <= 1:
        raise ValueError("no_trade_band must be finite and in [0, 1]")
    if (
        not math.isfinite(maximum_rebalance_turnover)
        or not 0 < maximum_rebalance_turnover <= 2
    ):
        raise ValueError("maximum_rebalance_turnover must be finite and in (0, 2]")
    if execution_cost_model is not None and pair_metadata is None:
        raise ValueError("pair_metadata is required with execution_cost_model")

    work = signals.loc[:, sorted(required)].copy()
    work["pair"] = work["pair"].astype(str)
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    if work.duplicated(["date", "pair"]).any():
        raise ValueError("signals must contain one row per date and pair")
    work = work.sort_values(["pair", "date"], kind="mergesort")
    numeric_columns = sorted(required - {"pair", "date"})
    work[numeric_columns] = work[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if work[numeric_columns].isna().any().any() or not np.isfinite(
        work[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("signal prices, weights, returns, and costs must be finite")
    if (work[["price_dependent", "price_independent"]] <= 0).any().any():
        raise ValueError("signal prices must be positive")
    if (work["borrow_cost"] < 0).any():
        raise ValueError("borrow_cost cannot be negative")

    work["return_dependent"] = work.groupby("pair", sort=False)[
        "price_dependent"
    ].pct_change(fill_method=None).fillna(0.0)
    work["return_independent"] = work.groupby("pair", sort=False)[
        "price_independent"
    ].pct_change(fill_method=None).fillna(0.0)
    if start is not None:
        work = work.loc[work["date"].ge(pd.Timestamp(start))]
    if end is not None:
        work = work.loc[work["date"].le(pd.Timestamp(end))]
    if work.empty:
        raise ValueError("no signal observations remain inside the requested window")

    weights = pd.to_numeric(target_weights, errors="coerce")
    if weights.index.has_duplicates:
        raise ValueError("target_weights index must be unique")
    weights.index = weights.index.astype(str)
    pairs = pd.Index(weights.index, name="pair")
    observed_pairs = pd.Index(work["pair"].unique())
    missing_pairs = pairs.difference(observed_pairs)
    extra_pairs = observed_pairs.difference(pairs)
    if len(missing_pairs) or len(extra_pairs):
        fragments = []
        if len(missing_pairs):
            fragments.append("missing " + ", ".join(missing_pairs))
        if len(extra_pairs):
            fragments.append("unexpected " + ", ".join(extra_pairs))
        raise ValueError("signals and target_weights do not align: " + "; ".join(fragments))
    weights = weights.reindex(pairs)
    if (
        weights.isna().any()
        or not np.isfinite(weights.to_numpy(dtype=float)).all()
        or (weights < 0).any()
        or not np.isclose(float(weights.sum()), 1.0)
    ):
        raise ValueError("target_weights must be finite, non-negative, and sum to one")
    schedule: pd.DataFrame | None = None
    if target_weight_schedule is not None:
        if not isinstance(target_weight_schedule, pd.DataFrame) or target_weight_schedule.empty:
            raise ValueError("target_weight_schedule must be a non-empty DataFrame")
        if not isinstance(target_weight_schedule.index, pd.DatetimeIndex):
            raise TypeError("target_weight_schedule must use a DatetimeIndex")
        if (
            target_weight_schedule.index.has_duplicates
            or not target_weight_schedule.index.is_monotonic_increasing
        ):
            raise ValueError("target_weight_schedule index must be sorted and unique")
        schedule = target_weight_schedule.copy()
        schedule.columns = schedule.columns.astype(str)
        missing_schedule = pairs.difference(schedule.columns)
        extra_schedule = schedule.columns.difference(pairs)
        if len(missing_schedule) or len(extra_schedule):
            raise ValueError("target_weight_schedule columns must match target_weights")
        schedule = schedule.reindex(columns=pairs).apply(pd.to_numeric, errors="coerce")
        if (
            schedule.isna().any().any()
            or not np.isfinite(schedule.to_numpy(dtype=float)).all()
            or (schedule < 0).any().any()
            or not np.allclose(schedule.sum(axis=1).to_numpy(dtype=float), 1.0)
        ):
            raise ValueError(
                "target_weight_schedule rows must be finite, non-negative, and sum to one"
            )

    dependent_names: np.ndarray | None = None
    independent_names: np.ndarray | None = None
    if execution_cost_model is not None:
        assert pair_metadata is not None
        metadata_required = {"pair", "dependent", "independent"}
        metadata_missing = metadata_required - set(pair_metadata.columns)
        if metadata_missing:
            raise ValueError(
                "pair_metadata missing columns: "
                + ", ".join(sorted(metadata_missing))
            )
        metadata = pair_metadata.loc[:, sorted(metadata_required)].copy()
        metadata["pair"] = metadata["pair"].astype(str)
        if metadata["pair"].duplicated().any():
            raise ValueError("pair_metadata must contain unique pairs")
        metadata = metadata.set_index("pair").reindex(pairs)
        if metadata[["dependent", "independent"]].isna().any().any():
            missing_metadata = metadata.index[
                metadata[["dependent", "independent"]].isna().any(axis=1)
            ]
            raise ValueError(
                "pair_metadata missing selected pairs: "
                + ", ".join(missing_metadata.astype(str))
            )
        dependent_names = metadata["dependent"].astype(str).to_numpy()
        independent_names = metadata["independent"].astype(str).to_numpy()

    def aggregate_underlyings(
        dependent_values: np.ndarray,
        independent_values: np.ndarray,
    ) -> pd.Series:
        if dependent_names is None or independent_names is None:
            raise RuntimeError("underlying metadata is unavailable")
        labels = np.concatenate([dependent_names, independent_names])
        values = np.concatenate([dependent_values, independent_values])
        return pd.Series(values, index=labels, dtype=float).groupby(level=0).sum()

    fields = [
        "weight_dependent",
        "weight_independent",
        "gross_return",
        "borrow_cost",
        "return_dependent",
        "return_independent",
    ]
    matrices: dict[str, pd.DataFrame] = {}
    for field in fields:
        matrix = work.pivot(index="date", columns="pair", values=field).sort_index()
        if matrix.index.has_duplicates or not matrix.index.is_monotonic_increasing:
            raise ValueError("signal dates must be sorted and unique after pivoting")
        matrix = matrix.reindex(columns=pairs)
        if matrix.isna().any().any():
            raise ValueError(f"signals do not form a complete date-by-pair panel for {field}")
        matrices[field] = matrix.astype(float)
    dates = matrices["gross_return"].index

    target = weights.to_numpy(dtype=float)
    if schedule is not None and not bool((schedule.index <= dates[0]).any()):
        raise ValueError(
            "target_weight_schedule needs an observation on or before the first backtest date"
        )
    capital = target.copy()
    previous_dependent_dollars = np.zeros(len(pairs), dtype=float)
    previous_independent_dollars = np.zeros(len(pairs), dtype=float)
    cost_rate = transaction_cost_bps / 10_000.0
    prior_equity = 1.0
    prior_period: Any = None
    equity_values: list[float] = []
    return_values: list[float] = []
    gross_return_values: list[float] = []
    transaction_cost_values: list[float] = []
    transaction_cost_rate_values: list[float] = []
    borrow_cost_values: list[float] = []
    borrow_cost_rate_values: list[float] = []
    leg_turnover_values: list[float] = []
    commission_cost_values: list[float] = []
    spread_cost_values: list[float] = []
    impact_cost_values: list[float] = []
    financing_cost_values: list[float] = []
    cash_income_values: list[float] = []
    short_rebate_income_values: list[float] = []
    net_execution_cost_values: list[float] = []
    maximum_participation_values: list[float] = []
    capacity_breach_values: list[int] = []
    locate_failure_values: list[int] = []
    deployed_weight_rows: list[np.ndarray] = []
    realized_weight_rows: list[np.ndarray] = []
    rebalance_rows: list[dict[str, Any]] = []
    execution_rows: list[pd.DataFrame] = []

    for location, date in enumerate(dates):
        total_before = float(capital.sum())
        if total_before <= 0 or not math.isfinite(total_before):
            raise ValueError("portfolio equity became non-positive or non-finite")
        current = capital / total_before
        current_target = target
        if schedule is not None:
            current_target = schedule.loc[schedule.index <= date].iloc[-1].to_numpy(
                dtype=float
            )
        proposed = current.copy()
        event = "hold"
        gross_weight_turnover = 0.0
        blend_fraction = 0.0
        if location == 0:
            proposed = current_target.copy()
            event = "initial_allocation"
            gross_weight_turnover = 1.0
            blend_fraction = 1.0
        elif rebalance_frequency != "none":
            period = date.to_period("M" if rebalance_frequency == "monthly" else "Q")
            if period != prior_period:
                difference = current_target - current
                if float(np.max(np.abs(difference))) > max(no_trade_band, 1e-12):
                    blend_fraction = min(
                        1.0,
                        maximum_rebalance_turnover
                        / max(float(np.abs(difference).sum()), 1e-18),
                    )
                    proposed = np.maximum(current + blend_fraction * difference, 0.0)
                    proposed /= proposed.sum()
                    gross_weight_turnover = float(np.abs(proposed - current).sum())
                    event = "scheduled_rebalance"
        desired_capital = proposed * total_before
        deployed_weight_rows.append(proposed.copy())

        weight_dependent = matrices["weight_dependent"].iloc[location].to_numpy()
        weight_independent = matrices["weight_independent"].iloc[location].to_numpy()
        desired_dependent_dollars = desired_capital * weight_dependent
        desired_independent_dollars = desired_capital * weight_independent
        pair_leg_turnover = (
            np.abs(desired_dependent_dollars - previous_dependent_dollars)
            + np.abs(desired_independent_dollars - previous_independent_dollars)
        )
        pair_gross_pnl = (
            desired_capital
            * matrices["gross_return"].iloc[location].to_numpy(dtype=float)
        )
        commission_cost = 0.0
        spread_cost = 0.0
        impact_cost = 0.0
        financing_cost = 0.0
        cash_income = 0.0
        short_rebate_income = 0.0
        maximum_participation = 0.0
        capacity_breaches = 0
        locate_failures = 0
        if execution_cost_model is None:
            pair_transaction_cost = pair_leg_turnover * cost_rate
            pair_borrow_cost = (
                desired_capital
                * matrices["borrow_cost"].iloc[location].to_numpy(dtype=float)
            )
            daily_base_transaction_cost = float(pair_transaction_cost.sum())
            daily_base_borrow_cost = float(pair_borrow_cost.sum())
            daily_base_net_cost = daily_base_transaction_cost + daily_base_borrow_cost
            daily_base_leg_turnover = float(pair_leg_turnover.sum())
            capital = (
                desired_capital
                + pair_gross_pnl
                - pair_transaction_cost
                - pair_borrow_cost
            )
        else:
            ticker_trades = aggregate_underlyings(
                desired_dependent_dollars - previous_dependent_dollars,
                desired_independent_dollars - previous_independent_dollars,
            )
            ticker_holdings = aggregate_underlyings(
                desired_dependent_dollars,
                desired_independent_dollars,
            )
            breakdown = execution_cost_model.evaluate(
                date,
                ticker_trades,
                ticker_holdings,
                equity=total_before,
            )
            if not breakdown.executable and fail_on_unexecutable:
                failures = breakdown.detail.loc[
                    breakdown.detail["capacity_breach"]
                    | breakdown.detail["locate_failure"],
                    "ticker",
                ].astype(str)
                raise ValueError(
                    f"execution constraints failed on {date.date()}: "
                    + ", ".join(failures)
                )
            daily_base_transaction_cost = breakdown.transaction_cost
            daily_base_borrow_cost = breakdown.borrow_cost
            daily_base_net_cost = breakdown.net_cost
            daily_base_leg_turnover = float(ticker_trades.abs().sum())
            commission_cost = breakdown.commission_cost
            spread_cost = breakdown.spread_cost
            impact_cost = breakdown.impact_cost
            financing_cost = breakdown.financing_cost
            cash_income = breakdown.cash_income
            short_rebate_income = breakdown.short_rebate_income
            maximum_participation = breakdown.maximum_participation_rate
            capacity_breaches = breakdown.capacity_breach_count
            locate_failures = breakdown.locate_failure_count
            execution_rows.append(
                breakdown.detail.assign(date=date, event="regular")
            )
            capital = desired_capital + pair_gross_pnl - daily_base_net_cost * proposed

        previous_dependent_dollars = desired_dependent_dollars * (
            1.0 + matrices["return_dependent"].iloc[location].to_numpy(dtype=float)
        )
        previous_independent_dollars = desired_independent_dollars * (
            1.0 + matrices["return_independent"].iloc[location].to_numpy(dtype=float)
        )
        liquidation_turnover = 0.0
        liquidation_cost = 0.0
        if location == len(dates) - 1 and charge_terminal_liquidation:
            if execution_cost_model is None:
                pair_liquidation_turnover = (
                    np.abs(previous_dependent_dollars)
                    + np.abs(previous_independent_dollars)
                )
                pair_liquidation_cost = pair_liquidation_turnover * cost_rate
                liquidation_turnover = float(pair_liquidation_turnover.sum())
                liquidation_cost = float(pair_liquidation_cost.sum())
                capital -= pair_liquidation_cost
            else:
                ending_holdings = aggregate_underlyings(
                    previous_dependent_dollars,
                    previous_independent_dollars,
                )
                liquidation = execution_cost_model.evaluate(
                    date,
                    -ending_holdings,
                    pd.Series(0.0, index=ending_holdings.index),
                    equity=float(capital.sum()),
                    include_holding_costs=False,
                )
                if not liquidation.executable and fail_on_unexecutable:
                    failures = liquidation.detail.loc[
                        liquidation.detail["capacity_breach"]
                        | liquidation.detail["locate_failure"],
                        "ticker",
                    ].astype(str)
                    raise ValueError(
                        f"terminal execution constraints failed on {date.date()}: "
                        + ", ".join(failures)
                    )
                liquidation_turnover = float(ending_holdings.abs().sum())
                liquidation_cost = liquidation.transaction_cost
                commission_cost += liquidation.commission_cost
                spread_cost += liquidation.spread_cost
                impact_cost += liquidation.impact_cost
                maximum_participation = max(
                    maximum_participation,
                    liquidation.maximum_participation_rate,
                )
                capacity_breaches += liquidation.capacity_breach_count
                locate_failures += liquidation.locate_failure_count
                execution_rows.append(
                    liquidation.detail.assign(
                        date=date,
                        event="terminal_liquidation",
                    )
                )
                capital -= liquidation_cost * (capital / float(capital.sum()))
            previous_dependent_dollars.fill(0.0)
            previous_independent_dollars.fill(0.0)

        total_after = float(capital.sum())
        if total_after <= 0 or not math.isfinite(total_after):
            raise ValueError("portfolio equity became non-positive or non-finite")
        daily_transaction_cost = daily_base_transaction_cost + liquidation_cost
        daily_borrow_cost = daily_base_borrow_cost
        daily_net_execution_cost = daily_base_net_cost + liquidation_cost
        daily_leg_turnover = daily_base_leg_turnover + liquidation_turnover
        equity_values.append(total_after)
        return_values.append(total_after / prior_equity - 1.0)
        gross_return_values.append(float(pair_gross_pnl.sum() / total_before))
        transaction_cost_values.append(daily_transaction_cost)
        transaction_cost_rate_values.append(daily_transaction_cost / total_before)
        borrow_cost_values.append(daily_borrow_cost)
        borrow_cost_rate_values.append(daily_borrow_cost / total_before)
        leg_turnover_values.append(daily_leg_turnover / total_before)
        commission_cost_values.append(commission_cost)
        spread_cost_values.append(spread_cost)
        impact_cost_values.append(impact_cost)
        financing_cost_values.append(financing_cost)
        cash_income_values.append(cash_income)
        short_rebate_income_values.append(short_rebate_income)
        net_execution_cost_values.append(daily_net_execution_cost)
        maximum_participation_values.append(maximum_participation)
        capacity_breach_values.append(capacity_breaches)
        locate_failure_values.append(locate_failures)
        realized_weight_rows.append(capital / total_after)
        if event != "hold":
            rebalance_rows.append(
                {
                    "date": date,
                    "event": event,
                    "gross_weight_turnover": gross_weight_turnover,
                    "active_allocation_turnover": gross_weight_turnover,
                    "cost": daily_transaction_cost,
                    "blend_fraction": blend_fraction,
                }
            )
        if liquidation_turnover > 0:
            rebalance_rows.append(
                {
                    "date": date,
                    "event": "terminal_liquidation",
                    "gross_weight_turnover": 0.0,
                    "active_allocation_turnover": 0.0,
                    "cost": liquidation_cost,
                    "blend_fraction": 1.0,
                }
            )
        prior_equity = total_after
        prior_period = (
            date.to_period("M" if rebalance_frequency == "monthly" else "Q")
            if rebalance_frequency != "none"
            else None
        )

    daily = pd.DataFrame(
        {
            "portfolio_return": return_values,
            "gross_pair_return": gross_return_values,
            "transaction_cost": transaction_cost_values,
            "transaction_cost_rate": transaction_cost_rate_values,
            "borrow_cost": borrow_cost_values,
            "borrow_cost_rate": borrow_cost_rate_values,
            "leg_turnover": leg_turnover_values,
            "commission_cost": commission_cost_values,
            "spread_cost": spread_cost_values,
            "impact_cost": impact_cost_values,
            "financing_cost": financing_cost_values,
            "cash_income": cash_income_values,
            "short_rebate_income": short_rebate_income_values,
            "net_execution_cost": net_execution_cost_values,
            "maximum_participation_rate": maximum_participation_values,
            "capacity_breach_count": capacity_breach_values,
            "locate_failure_count": locate_failure_values,
            "equity": equity_values,
        },
        index=dates,
    )
    daily.index.name = "date"
    deployed_weights = pd.DataFrame(
        deployed_weight_rows,
        index=dates,
        columns=pairs,
    )
    deployed_weights.index.name = "date"
    realized_weights = pd.DataFrame(
        realized_weight_rows,
        index=dates,
        columns=pairs,
    )
    realized_weights.index.name = "date"
    log = pd.DataFrame(
        rebalance_rows,
        columns=[
            "date",
            "event",
            "gross_weight_turnover",
            "active_allocation_turnover",
            "cost",
            "blend_fraction",
        ],
    )
    metrics = performance_metrics(daily["portfolio_return"], daily["equity"])
    metrics.update(
        {
            "transaction_cost_sum": float(daily["transaction_cost"].sum()),
            "transaction_cost_drag_bps": float(
                daily["transaction_cost_rate"].sum() * 10_000.0
            ),
            "borrow_cost_sum": float(daily["borrow_cost"].sum()),
            "borrow_cost_drag_bps": float(daily["borrow_cost_rate"].sum() * 10_000.0),
            "annualized_leg_turnover": float(daily["leg_turnover"].mean() * 252.0),
            "capital_rebalance_turnover_sum": float(
                log.loc[log["event"].eq("scheduled_rebalance"), "gross_weight_turnover"].sum()
            )
            if not log.empty
            else 0.0,
            "scheduled_rebalance_count": int(
                log["event"].eq("scheduled_rebalance").sum()
            )
            if not log.empty
            else 0,
            "rebalance_frequency": rebalance_frequency,
            "no_trade_band": no_trade_band,
            "maximum_rebalance_turnover": maximum_rebalance_turnover,
            "target_weight_schedule_enabled": schedule is not None,
            "target_weight_schedule_observations": len(schedule) if schedule is not None else 0,
            "terminal_liquidation_cost_charged": bool(charge_terminal_liquidation),
            "transaction_cost_model": (
                "ticker-level commission, spread, square-root impact, carry, and capacity"
                if execution_cost_model is not None
                else "exact pre/post underlying-dollar notionals at a flat bps rate"
            ),
            "execution_cost_model_enabled": execution_cost_model is not None,
            "commission_cost_sum": float(daily["commission_cost"].sum()),
            "spread_cost_sum": float(daily["spread_cost"].sum()),
            "impact_cost_sum": float(daily["impact_cost"].sum()),
            "financing_cost_sum": float(daily["financing_cost"].sum()),
            "cash_income_sum": float(daily["cash_income"].sum()),
            "short_rebate_income_sum": float(daily["short_rebate_income"].sum()),
            "net_execution_cost_sum": float(daily["net_execution_cost"].sum()),
            "maximum_participation_rate": float(
                daily["maximum_participation_rate"].max()
            ),
            "capacity_breach_count": int(daily["capacity_breach_count"].sum()),
            "locate_failure_count": int(daily["locate_failure_count"].sum()),
        }
    )
    if execution_cost_model is not None:
        metrics.update(
            {
                "execution_portfolio_notional": execution_cost_model.spec.portfolio_notional,
                "execution_cost_spec": {
                    key: getattr(execution_cost_model.spec, key)
                    for key in execution_cost_model.spec.__dataclass_fields__
                },
                "execution_input_provenance": execution_cost_model.inputs.provenance,
            }
        )
    execution_log = (
        pd.concat(execution_rows, ignore_index=True)
        if execution_rows
        else None
    )
    return WeightedPortfolioResult(
        daily=daily,
        deployed_weights=deployed_weights,
        realized_weights=realized_weights,
        rebalance_log=log,
        metrics=metrics,
        execution_log=execution_log,
    )
