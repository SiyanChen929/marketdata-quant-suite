"""Portfolio risk, diversification, and market-shock diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def equal_weight_market_proxy(prices: pd.DataFrame) -> pd.Series:
    """Build a daily equal-weight return proxy from adjusted closes.

    The cross-sectional average uses only finite, positive-price observations.
    It is intentionally simple and auditable; it is not a point-in-time index.
    """
    numeric = prices.apply(pd.to_numeric, errors="coerce").where(lambda x: x > 0)
    returns = numeric.pct_change(fill_method=None)
    proxy = returns.replace([np.inf, -np.inf], np.nan).mean(axis=1, skipna=True)
    proxy.name = "equal_weight_sp500_proxy_return"
    return proxy


def regression_diagnostics(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    *,
    annualization: int = 252,
) -> dict[str, float | int]:
    """OLS market alpha/beta and correlation for aligned daily returns."""
    aligned = pd.concat(
        [
            pd.to_numeric(strategy_returns, errors="coerce").rename("strategy"),
            pd.to_numeric(market_returns, errors="coerce").rename("market"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 3 or aligned["market"].var(ddof=1) <= 0:
        return {
            "observations": int(len(aligned)),
            "daily_alpha": math.nan,
            "annualized_alpha": math.nan,
            "market_beta": math.nan,
            "market_correlation": math.nan,
            "r_squared": math.nan,
        }
    covariance = float(aligned["strategy"].cov(aligned["market"]))
    beta = covariance / float(aligned["market"].var(ddof=1))
    daily_alpha = float(aligned["strategy"].mean() - beta * aligned["market"].mean())
    correlation = float(aligned["strategy"].corr(aligned["market"]))
    return {
        "observations": int(len(aligned)),
        "daily_alpha": daily_alpha,
        "annualized_alpha": daily_alpha * annualization,
        "market_beta": beta,
        "market_correlation": correlation,
        "r_squared": correlation * correlation,
    }


def shock_diagnostics(
    strategy_returns: pd.DataFrame,
    market_returns: pd.Series,
    *,
    shock_threshold: float = 0.02,
) -> pd.DataFrame:
    """Summarize strategy behavior on broad-market shock days."""
    market = pd.to_numeric(market_returns, errors="coerce").rename("market_return")
    data = strategy_returns.apply(pd.to_numeric, errors="coerce").join(market, how="inner")
    masks = {
        f"market_down_at_least_{shock_threshold:.0%}": data["market_return"] <= -shock_threshold,
        f"market_up_at_least_{shock_threshold:.0%}": data["market_return"] >= shock_threshold,
        f"absolute_market_move_at_least_{shock_threshold:.0%}": data["market_return"].abs() >= shock_threshold,
        "all_days": data["market_return"].notna(),
    }
    rows: list[dict[str, Any]] = []
    for regime, mask in masks.items():
        for portfolio in strategy_returns.columns:
            values = data.loc[mask, portfolio].dropna()
            rows.append(
                {
                    "regime": regime,
                    "portfolio": portfolio,
                    "observations": int(len(values)),
                    "mean_daily_return": float(values.mean()) if len(values) else math.nan,
                    "median_daily_return": float(values.median()) if len(values) else math.nan,
                    "cumulative_return": float((1.0 + values).prod() - 1.0) if len(values) else math.nan,
                    "positive_day_fraction": float((values > 0).mean()) if len(values) else math.nan,
                    "worst_day": float(values.min()) if len(values) else math.nan,
                    "best_day": float(values.max()) if len(values) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def pair_correlation_diagnostics(
    pair_returns: pd.DataFrame,
    active_positions: pd.DataFrame | None = None,
    *,
    minimum_active_overlap: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Return unconditional matrix, active-overlap observations, and summaries."""
    returns = pair_returns.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    unconditional = returns.corr()
    columns = list(returns.columns)
    active_rows: list[dict[str, Any]] = []
    if active_positions is not None:
        active = active_positions.reindex(index=returns.index, columns=columns).fillna(0).ne(0)
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1 :]:
                mask = active[left] & active[right]
                overlap = int(mask.sum())
                if overlap < minimum_active_overlap:
                    continue
                corr = returns.loc[mask, left].corr(returns.loc[mask, right])
                active_rows.append(
                    {
                        "pair_a": left,
                        "pair_b": right,
                        "active_overlap_days": overlap,
                        "active_return_correlation": float(corr) if pd.notna(corr) else math.nan,
                    }
                )
    active_frame = pd.DataFrame(active_rows)
    upper = unconditional.to_numpy(dtype=float)[np.triu_indices(len(columns), k=1)]
    finite_upper = upper[np.isfinite(upper)]
    active_values = (
        pd.to_numeric(active_frame.get("active_return_correlation"), errors="coerce").dropna().to_numpy()
        if not active_frame.empty
        else np.array([], dtype=float)
    )

    def _summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
        if not len(values):
            return {
                f"{prefix}_count": 0,
                f"{prefix}_mean": math.nan,
                f"{prefix}_median": math.nan,
                f"{prefix}_p90": math.nan,
                f"{prefix}_maximum": math.nan,
                f"{prefix}_absolute_mean": math.nan,
                f"{prefix}_absolute_median": math.nan,
                f"{prefix}_absolute_p90": math.nan,
                f"{prefix}_absolute_maximum": math.nan,
            }
        absolute = np.abs(values)
        return {
            f"{prefix}_count": int(len(values)),
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_p90": float(np.quantile(values, 0.90)),
            f"{prefix}_maximum": float(np.max(values)),
            f"{prefix}_absolute_mean": float(np.mean(absolute)),
            f"{prefix}_absolute_median": float(np.median(absolute)),
            f"{prefix}_absolute_p90": float(np.quantile(absolute, 0.90)),
            f"{prefix}_absolute_maximum": float(np.max(absolute)),
        }

    summary = {
        **_summary(finite_upper, "unconditional_correlation"),
        **_summary(active_values, "active_correlation"),
        "minimum_active_overlap": int(minimum_active_overlap),
    }
    return unconditional, active_frame, summary


def overlap_correlation_diagnostics(
    unconditional: pd.DataFrame,
    active_correlations: pd.DataFrame,
    selected_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label pair-of-pair correlations by whether they reuse an underlying."""
    pair_legs = {
        str(row["pair"]): {str(row["dependent"]), str(row["independent"])}
        for row in selected_pairs.to_dict(orient="records")
    }
    columns = [column for column in unconditional.columns if column in pair_legs]
    active_lookup: dict[tuple[str, str], tuple[int, float]] = {}
    if not active_correlations.empty:
        for row in active_correlations.to_dict(orient="records"):
            key = tuple(sorted((str(row["pair_a"]), str(row["pair_b"]))))
            active_lookup[key] = (
                int(row["active_overlap_days"]),
                float(row["active_return_correlation"]),
            )
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            key = tuple(sorted((left, right)))
            overlap_days, active_correlation = active_lookup.get(key, (0, math.nan))
            shared = sorted(pair_legs[left] & pair_legs[right])
            rows.append(
                {
                    "pair_a": left,
                    "pair_b": right,
                    "shares_underlying": bool(shared),
                    "shared_tickers": ", ".join(shared),
                    "unconditional_return_correlation": float(unconditional.loc[left, right]),
                    "active_overlap_days": overlap_days,
                    "active_return_correlation": active_correlation,
                }
            )
    detail = pd.DataFrame(
        rows,
        columns=[
            "pair_a",
            "pair_b",
            "shares_underlying",
            "shared_tickers",
            "unconditional_return_correlation",
            "active_overlap_days",
            "active_return_correlation",
        ],
    )
    summary_rows: list[dict[str, Any]] = []
    if detail.empty:
        return detail, pd.DataFrame()
    for shared, group in detail.groupby("shares_underlying", dropna=False):
        unconditional_values = pd.to_numeric(
            group["unconditional_return_correlation"], errors="coerce"
        ).dropna()
        active_values = pd.to_numeric(
            group["active_return_correlation"], errors="coerce"
        ).dropna()
        summary_rows.append(
            {
                "group": "shared_underlying" if shared else "disjoint_underlyings",
                "pair_of_pair_count": int(len(group)),
                "mean_absolute_unconditional_correlation": float(unconditional_values.abs().mean()),
                "median_absolute_unconditional_correlation": float(unconditional_values.abs().median()),
                "active_correlation_count": int(len(active_values)),
                "mean_absolute_active_correlation": float(active_values.abs().mean()),
                "median_absolute_active_correlation": float(active_values.abs().median()),
                "maximum_absolute_active_correlation": float(active_values.abs().max()),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def portfolio_exposure_diagnostics(
    signals: pd.DataFrame,
    selected_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate pair weights into daily portfolio and name-level exposures."""
    required = {
        "date",
        "pair",
        "weight_dependent",
        "weight_independent",
    }
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals missing columns: {', '.join(sorted(missing))}")
    legs = selected_pairs.drop_duplicates("pair").set_index("pair")[["dependent", "independent"]]
    working = signals.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.loc[working["pair"].isin(legs.index)].copy()
    working = working.join(legs, on="pair", how="left")
    pair_count = int(working["pair"].nunique())
    if pair_count == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    allocation = 1.0 / pair_count
    dependent = working[["date", "pair", "dependent", "weight_dependent"]].rename(
        columns={"dependent": "ticker", "weight_dependent": "exposure"}
    )
    independent = working[["date", "pair", "independent", "weight_independent"]].rename(
        columns={"independent": "ticker", "weight_independent": "exposure"}
    )
    name_exposure = pd.concat([dependent, independent], ignore_index=True)
    name_exposure["exposure"] = pd.to_numeric(name_exposure["exposure"], errors="coerce").fillna(0.0) * allocation
    name_exposure = (
        name_exposure.groupby(["date", "ticker"], as_index=False)["exposure"]
        .sum()
        .sort_values(["date", "ticker"])
    )
    sleeve_daily = working.assign(
        sleeve_gross=(
            pd.to_numeric(working["weight_dependent"], errors="coerce").fillna(0.0).abs()
            + pd.to_numeric(working["weight_independent"], errors="coerce").fillna(0.0).abs()
        ) * allocation
    ).groupby("date")["sleeve_gross"].sum()
    daily = name_exposure.assign(
        gross=lambda x: x["exposure"].abs(),
        long=lambda x: x["exposure"].clip(lower=0),
        short=lambda x: -x["exposure"].clip(upper=0),
    ).groupby("date").agg(
        gross_exposure=("gross", "sum"),
        net_exposure=("exposure", "sum"),
        long_exposure=("long", "sum"),
        short_exposure=("short", "sum"),
        maximum_absolute_name_exposure=("gross", "max"),
        active_names=("gross", lambda values: int((values > 0).sum())),
    )
    daily = daily.rename(columns={"gross_exposure": "netted_gross_exposure"})
    daily["sleeve_gross_exposure"] = sleeve_daily.reindex(daily.index).fillna(0.0)
    daily["net_to_gross"] = daily["net_exposure"] / daily["netted_gross_exposure"].replace(0, np.nan)
    daily.index.name = "date"
    daily = daily.reset_index()

    ticker_counts = pd.concat(
        [selected_pairs[["pair", "dependent"]].rename(columns={"dependent": "ticker"}),
         selected_pairs[["pair", "independent"]].rename(columns={"independent": "ticker"})],
        ignore_index=True,
    )
    overlap = (
        ticker_counts.groupby("ticker")
        .agg(pair_count=("pair", "nunique"), pairs=("pair", lambda values: ", ".join(sorted(set(values)))))
        .reset_index()
        .sort_values(["pair_count", "ticker"], ascending=[False, True])
    )
    summary = pd.DataFrame(
        [
            {
                "selected_pairs": pair_count,
                "unique_names": int(overlap["ticker"].nunique()),
                "names_used_more_than_once": int((overlap["pair_count"] > 1).sum()),
                "maximum_pair_reuse": int(overlap["pair_count"].max()),
                "mean_sleeve_gross_exposure": float(daily["sleeve_gross_exposure"].mean()),
                "maximum_sleeve_gross_exposure": float(daily["sleeve_gross_exposure"].max()),
                "mean_netted_gross_exposure": float(daily["netted_gross_exposure"].mean()),
                "maximum_netted_gross_exposure": float(daily["netted_gross_exposure"].max()),
                "mean_net_exposure": float(daily["net_exposure"].mean()),
                "mean_absolute_net_exposure": float(daily["net_exposure"].abs().mean()),
                "maximum_absolute_net_exposure": float(daily["net_exposure"].abs().max()),
                "mean_maximum_absolute_name_exposure": float(daily["maximum_absolute_name_exposure"].mean()),
                "maximum_absolute_name_exposure": float(daily["maximum_absolute_name_exposure"].max()),
            }
        ]
    )
    return daily, name_exposure, summary


def ticker_overlap_table(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    """List how often each underlying name appears across selected pairs."""
    ticker_counts = pd.concat(
        [selected_pairs[["pair", "dependent"]].rename(columns={"dependent": "ticker"}),
         selected_pairs[["pair", "independent"]].rename(columns={"independent": "ticker"})],
        ignore_index=True,
    )
    return (
        ticker_counts.groupby("ticker")
        .agg(pair_count=("pair", "nunique"), pairs=("pair", lambda values: ", ".join(sorted(set(values)))))
        .reset_index()
        .sort_values(["pair_count", "ticker"], ascending=[False, True])
    )
