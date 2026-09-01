"""Point-in-time fundamental scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.config import FundamentalsConfig
from quant_system.fundamentals.factors import FACTOR_GROUPS, LOWER_IS_BETTER


def score_fundamentals(fundamentals: pd.DataFrame, config: FundamentalsConfig) -> pd.DataFrame:
    """Score fundamentals with sector-neutral z-scores where data exists."""

    if fundamentals.empty:
        return pd.DataFrame(columns=["date", "symbol", "fundamental_score"])
    out = fundamentals.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["fundamental_known_date"] = pd.to_datetime(out.get("known_date", out["date"]), errors="coerce").dt.normalize()
    out["fundamental_known_date"] = out["fundamental_known_date"].fillna(out["date"])
    out["sector"] = out.get("sector", "UNKNOWN").fillna("UNKNOWN")
    available_factor_count = pd.Series(0, index=out.index, dtype="float64")
    possible_factor_count = 0
    for group, factors in FACTOR_GROUPS.items():
        group_scores = []
        group_available = pd.Series(0, index=out.index, dtype="float64")
        group_possible = len(factors)
        possible_factor_count += group_possible
        for factor in factors:
            if factor not in out.columns:
                continue
            values = pd.to_numeric(out[factor], errors="coerce")
            group_available += values.notna().astype(float)
            available_factor_count += values.notna().astype(float)
            values = _winsorize(values, config.winsorize_pct)
            z = out.assign(_v=values).groupby(["date", "sector"])["_v"].transform(_zscore)
            if factor in LOWER_IS_BETTER:
                z = -z
            group_scores.append(z)
        if group_scores:
            out[f"{group}_score"] = _z_to_score(pd.concat(group_scores, axis=1).mean(axis=1))
            out[f"{group}_factor_coverage"] = (group_available / max(group_possible, 1)).clip(0, 1)
        else:
            out[f"{group}_score"] = 50.0
            out[f"{group}_factor_coverage"] = 0.0
    out["fundamental_score"] = 0.0
    for group, weight in config.weights.items():
        out["fundamental_score"] += float(weight) * out.get(f"{group}_score", 50.0)
    out["fundamental_score"] = out["fundamental_score"].fillna(50.0).clip(0, 100)
    out["fundamental_factor_coverage"] = (available_factor_count / max(possible_factor_count, 1)).clip(0, 1)
    out["fundamental_missing_group_count"] = sum(out[f"{group}_factor_coverage"].le(0).astype(int) for group in FACTOR_GROUPS)
    low_coverage = out["fundamental_factor_coverage"] < 0.25
    out["moat_score"] = (
        0.55 * out["quality_score"].fillna(50.0)
        + 0.25 * out["balance_sheet_score"].fillna(50.0)
        + 0.20 * out["revision_score"].fillna(50.0)
    ).clip(0, 100)
    out.loc[low_coverage, "moat_score"] = np.nan
    out["fundamental_data_quality"] = np.where(
        low_coverage,
        "earnings_proxy_low_coverage",
        "broad_fundamental_coverage",
    )
    keep = [
        "date",
        "symbol",
        "sector",
        "industry",
        "fundamental_known_date",
        "fundamental_score",
        "fundamental_factor_coverage",
        "fundamental_missing_group_count",
        "fundamental_data_quality",
        "moat_score",
        *[f"{group}_score" for group in FACTOR_GROUPS],
        *[f"{group}_factor_coverage" for group in FACTOR_GROUPS],
    ]
    return out[[column for column in keep if column in out.columns]].sort_values(["date", "symbol"])


def align_fundamentals_to_prices(scored: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """As-of merge fundamental scores onto daily price dates without future fill."""

    if scored.empty:
        return pd.DataFrame(columns=["date", "symbol", "fundamental_score"])
    dates = prices[["date", "symbol"]].drop_duplicates().sort_values(["symbol", "date"])
    scored = scored.sort_values(["symbol", "date"])
    frames = []
    for symbol, left in dates.groupby("symbol"):
        right = scored[scored["symbol"] == symbol]
        if right.empty:
            continue
        aligned = pd.merge_asof(left.sort_values("date"), right.sort_values("date"), on="date", by="symbol", direction="backward")
        frames.append(aligned)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", "fundamental_score"])
    if "fundamental_score" in out.columns:
        out = out[out["fundamental_score"].notna()].copy()
    if "fundamental_known_date" in out.columns:
        out["fundamental_data_age_days"] = (pd.to_datetime(out["date"]) - pd.to_datetime(out["fundamental_known_date"])).dt.days
    return out


def _winsorize(values: pd.Series, pct: float) -> pd.Series:
    if values.dropna().empty or pct <= 0:
        return values
    lo, hi = values.quantile([pct, 1 - pct])
    return values.clip(lo, hi)


def _zscore(values: pd.Series) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def _z_to_score(z: pd.Series) -> pd.Series:
    return (50.0 + 15.0 * z).clip(0, 100)
