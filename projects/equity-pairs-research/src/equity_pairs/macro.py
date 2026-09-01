"""FRED-derived regime variables and ex-post performance attribution."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def derive_macro_variables(observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy().sort_index()
    if {"DGS10", "DGS2"}.issubset(result.columns):
        result["yield_curve_2s10s"] = result["DGS10"] - result["DGS2"]
    if "CPIAUCSL" in result:
        monthly_cpi = result["CPIAUCSL"].dropna()
        cpi_yoy = monthly_cpi.pct_change(12, fill_method=None) * 100.0
        result["cpi_yoy"] = np.nan
        result.loc[cpi_yoy.index, "cpi_yoy"] = cpi_yoy
    if "UNRATE" in result:
        monthly_unemployment = result["UNRATE"].dropna()
        unemployment_change = monthly_unemployment.diff(12)
        result["unemployment_12m_change"] = np.nan
        result.loc[unemployment_change.index, "unemployment_12m_change"] = unemployment_change
    return result


def align_macro_regimes(macro: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill current-vintage FRED data for explicitly ex-post attribution."""
    aligned = macro.reindex(macro.index.union(dates)).sort_index().ffill().reindex(dates)
    regimes = pd.DataFrame(index=dates)
    if "yield_curve_2s10s" in aligned:
        regimes["yield_curve_regime"] = np.select(
            [aligned["yield_curve_2s10s"] < 0, aligned["yield_curve_2s10s"].notna()],
            ["inverted", "non_inverted"],
            default="unknown",
        )
    if "VIXCLS" in aligned:
        regimes["volatility_regime"] = np.select(
            [aligned["VIXCLS"] >= 25, aligned["VIXCLS"].notna()],
            ["vix_at_least_25", "vix_below_25"],
            default="unknown",
        )
    if "cpi_yoy" in aligned:
        regimes["inflation_regime"] = np.select(
            [aligned["cpi_yoy"] >= 3, aligned["cpi_yoy"].notna()],
            ["cpi_at_least_3pct", "cpi_below_3pct"],
            default="unknown",
        )
    if "USREC" in aligned:
        regimes["business_cycle_regime"] = np.select(
            [aligned["USREC"] >= 0.5, aligned["USREC"].notna()],
            ["nber_recession", "nber_expansion"],
            default="unknown",
        )
    if "DFF" in aligned:
        six_month_change = aligned["DFF"].diff(126)
        regimes["policy_rate_regime"] = np.select(
            [six_month_change > 0.50, six_month_change < -0.50, six_month_change.notna()],
            ["rising", "falling", "stable"],
            default="unknown",
        )
    return regimes


def regime_attribution(portfolio_returns: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Compute conditional daily/annualized performance by ex-post macro regime."""
    if portfolio_returns.empty or macro.empty:
        return pd.DataFrame()
    regimes = align_macro_regimes(macro, portfolio_returns.index)
    rows: list[dict[str, object]] = []
    for portfolio in portfolio_returns.columns:
        returns = pd.to_numeric(portfolio_returns[portfolio], errors="coerce").fillna(0.0)
        for dimension in regimes.columns:
            for regime, dates in regimes.groupby(dimension, dropna=False).groups.items():
                sample = returns.loc[dates]
                if sample.empty:
                    continue
                standard_deviation = float(sample.std(ddof=1)) if len(sample) > 1 else math.nan
                rows.append(
                    {
                        "portfolio": portfolio,
                        "regime_dimension": dimension,
                        "regime": str(regime),
                        "trading_days": int(len(sample)),
                        "fraction_of_test": float(len(sample) / len(returns)),
                        "average_daily_return": float(sample.mean()),
                        "conditional_annualized_return": float(sample.mean() * 252.0),
                        "conditional_annualized_volatility": standard_deviation * math.sqrt(252.0)
                        if np.isfinite(standard_deviation)
                        else math.nan,
                        "conditional_sharpe": float(sample.mean() / standard_deviation * math.sqrt(252.0))
                        if np.isfinite(standard_deviation) and standard_deviation > 0
                        else math.nan,
                        "positive_day_fraction": float((sample > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)
