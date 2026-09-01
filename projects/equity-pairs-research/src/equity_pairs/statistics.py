"""Cointegration diagnostics and sector-level pair selection.

The implementation deliberately separates full-window descriptive rankings from
formation-window selections.  The former answers "what was most cointegrated?";
only the latter is eligible for the held-out trading test.
"""

from __future__ import annotations

import itertools
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import CollinearityWarning, InterpolationWarning
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class PairScreenSettings:
    minimum_observations: int = 756
    minimum_history_ratio: float = 0.90
    alpha: float = 0.05
    maximum_half_life_days: int = 252
    adf_maxlag: int = 5
    exclude_same_issuer: bool = True
    workers: int = 4


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values, preserving NaNs and order."""
    values = np.asarray(list(pvalues), dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(values)
    valid = values[valid_mask]
    if valid.size == 0:
        return result
    order = np.argsort(valid, kind="stable")
    ranked = valid[order]
    m = float(valid.size)
    adjusted = ranked * m / np.arange(1, valid.size + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid_mask] = restored
    return result


def fit_log_hedge(dependent: np.ndarray, independent: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS log-price hedge relation: log(y) = alpha + beta * log(x) + error."""
    y = np.log(np.asarray(dependent, dtype=float))
    x = np.log(np.asarray(independent, dtype=float))
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - alpha - beta * x
    return float(alpha), float(beta), residual


def estimate_half_life(spread: np.ndarray) -> float:
    """Estimate discrete OU/AR(1) half-life in trading days."""
    values = np.asarray(spread, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return math.nan
    lagged = values[:-1]
    delta = np.diff(values)
    design = np.column_stack([np.ones(lagged.size), lagged])
    slope = float(np.linalg.lstsq(design, delta, rcond=None)[0][1])
    if not np.isfinite(slope) or slope >= 0:
        return math.inf
    return float(-math.log(2.0) / slope)


def hurst_exponent(spread: np.ndarray) -> float:
    """Estimate the generalized Hurst exponent from lagged-difference scaling."""
    values = np.asarray(spread, dtype=float)
    values = values[np.isfinite(values)]
    max_lag = min(100, max(3, values.size // 4))
    lags = np.unique(np.geomspace(2, max_lag, num=min(20, max_lag - 1)).astype(int))
    usable_lags: list[int] = []
    scales: list[float] = []
    for lag in lags:
        scale = float(np.std(values[lag:] - values[:-lag], ddof=1))
        if np.isfinite(scale) and scale > 0:
            usable_lags.append(int(lag))
            scales.append(scale)
    if len(scales) < 3:
        return math.nan
    return float(np.polyfit(np.log(usable_lags), np.log(scales), 1)[0])


def _safe_adf(values: np.ndarray, maxlag: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < max(30, maxlag + 10) or np.std(values) <= 1e-12:
        return math.nan, math.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            statistic, pvalue, *_ = adfuller(
                values,
                maxlag=maxlag,
                regression="c",
                autolag="BIC",
            )
        return float(statistic), float(pvalue)
    except (ValueError, np.linalg.LinAlgError):
        return math.nan, math.nan


def integration_diagnostics(prices: pd.DataFrame, maxlag: int = 5) -> pd.DataFrame:
    """Check that log levels look I(1): unit root in levels, stationary differences."""
    rows: list[dict[str, object]] = []
    for ticker in prices.columns:
        series = pd.to_numeric(prices[ticker], errors="coerce").dropna()
        series = series[series > 0]
        logs = np.log(series.to_numpy(dtype=float))
        level_stat, level_p = _safe_adf(logs, maxlag)
        return_stat, return_p = _safe_adf(np.diff(logs), maxlag)
        rows.append(
            {
                "ticker": str(ticker),
                "observations": int(len(series)),
                "level_adf_statistic": level_stat,
                "level_adf_pvalue": level_p,
                "return_adf_statistic": return_stat,
                "return_adf_pvalue": return_p,
                "looks_i1": bool(
                    np.isfinite(level_p)
                    and np.isfinite(return_p)
                    and level_p > 0.05
                    and return_p < 0.05
                ),
            }
        )
    return pd.DataFrame(rows)


def _safe_coint(y: np.ndarray, x: np.ndarray, maxlag: int) -> tuple[float, float]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CollinearityWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            statistic, pvalue, _ = coint(
                np.log(y),
                np.log(x),
                trend="c",
                maxlag=maxlag,
                autolag="BIC",
            )
        return float(statistic), float(pvalue)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return math.nan, math.nan


def evaluate_pair(
    ticker_a: str,
    ticker_b: str,
    sector: str,
    prices: pd.DataFrame,
    expected_observations: int,
    settings: PairScreenSettings,
    i1_lookup: dict[str, bool] | None = None,
) -> dict[str, object]:
    """Evaluate both Engle-Granger orientations and retain a multiplicity-safe pair p-value."""
    aligned = prices[[ticker_a, ticker_b]].apply(pd.to_numeric, errors="coerce").dropna()
    aligned = aligned[(aligned[ticker_a] > 0) & (aligned[ticker_b] > 0)]
    nobs = int(len(aligned))
    history_ratio = nobs / expected_observations if expected_observations else 0.0
    base: dict[str, object] = {
        "sector": sector,
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "pair": f"{ticker_a}__{ticker_b}",
        "observations": nobs,
        "history_ratio": history_ratio,
    }
    if nobs < settings.minimum_observations or history_ratio < settings.minimum_history_ratio:
        return {
            **base,
            "screen_status": "insufficient_history",
            "dependent": None,
            "independent": None,
            "alpha": math.nan,
            "beta": math.nan,
            "eg_statistic_ab": math.nan,
            "eg_pvalue_ab": math.nan,
            "eg_statistic_ba": math.nan,
            "eg_pvalue_ba": math.nan,
            "pair_pvalue": math.nan,
            "half_life_days": math.nan,
            "hurst_exponent": math.nan,
            "return_correlation": math.nan,
            "positive_hedge_ratio": False,
            "both_look_i1": False,
        }

    y_a = aligned[ticker_a].to_numpy(dtype=float)
    y_b = aligned[ticker_b].to_numpy(dtype=float)
    stat_ab, p_ab = _safe_coint(y_a, y_b, settings.adf_maxlag)
    stat_ba, p_ba = _safe_coint(y_b, y_a, settings.adf_maxlag)
    if not np.isfinite(p_ab) and not np.isfinite(p_ba):
        return {
            **base,
            "screen_status": "test_failure",
            "dependent": None,
            "independent": None,
            "alpha": math.nan,
            "beta": math.nan,
            "eg_statistic_ab": stat_ab,
            "eg_pvalue_ab": p_ab,
            "eg_statistic_ba": stat_ba,
            "eg_pvalue_ba": p_ba,
            "pair_pvalue": math.nan,
            "half_life_days": math.nan,
            "hurst_exponent": math.nan,
            "return_correlation": math.nan,
            "positive_hedge_ratio": False,
            "both_look_i1": False,
        }

    choose_ab = not np.isfinite(p_ba) or (np.isfinite(p_ab) and p_ab <= p_ba)
    if choose_ab:
        dependent, independent = ticker_a, ticker_b
        dep_values, ind_values = y_a, y_b
    else:
        dependent, independent = ticker_b, ticker_a
        dep_values, ind_values = y_b, y_a
    alpha, beta, residual = fit_log_hedge(dep_values, ind_values)
    finite_pvalues = [p for p in (p_ab, p_ba) if np.isfinite(p)]
    # Bonferroni-correct the data-dependent choice between the two orientations.
    pair_pvalue = min(1.0, 2.0 * min(finite_pvalues))
    log_returns = np.diff(np.log(aligned[[ticker_a, ticker_b]].to_numpy(dtype=float)), axis=0)
    return_correlation = (
        float(np.corrcoef(log_returns.T)[0, 1]) if log_returns.shape[0] >= 3 else math.nan
    )
    i1_lookup = i1_lookup or {}
    return {
        **base,
        "screen_status": "ok",
        "dependent": dependent,
        "independent": independent,
        "alpha": alpha,
        "beta": beta,
        "eg_statistic_ab": stat_ab,
        "eg_pvalue_ab": p_ab,
        "eg_statistic_ba": stat_ba,
        "eg_pvalue_ba": p_ba,
        "pair_pvalue": pair_pvalue,
        "half_life_days": estimate_half_life(residual),
        "hurst_exponent": hurst_exponent(residual),
        "return_correlation": return_correlation,
        "positive_hedge_ratio": bool(np.isfinite(beta) and beta > 0),
        "both_look_i1": bool(i1_lookup.get(ticker_a, False) and i1_lookup.get(ticker_b, False)),
    }


def screen_sector_pairs(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    settings: PairScreenSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exhaustively screen every within-sector pair present in the price matrix."""
    if not {"ticker", "sector"}.issubset(universe.columns):
        raise ValueError("universe must contain ticker and sector columns")
    prices = prices.sort_index()
    integration = integration_diagnostics(prices, maxlag=settings.adf_maxlag)
    i1_lookup = integration.set_index("ticker")["looks_i1"].to_dict()
    available = set(map(str, prices.columns))
    work: list[tuple[str, str, str]] = []
    columns = ["ticker", "sector"]
    issuer_column = next((name for name in ("cik", "issuer_id") if name in universe.columns), None)
    if issuer_column:
        columns.append(issuer_column)
    clean_universe = universe[columns].dropna(subset=["ticker", "sector"]).drop_duplicates("ticker")
    issuer_lookup: dict[str, str] = {}
    if issuer_column:
        for ticker, issuer in clean_universe.set_index("ticker")[issuer_column].items():
            if pd.notna(issuer) and str(issuer).strip() and str(issuer).strip().lower() not in {"nan", "<na>", "none"}:
                issuer_lookup[str(ticker)] = str(issuer).strip()
    for sector, group in clean_universe.groupby("sector", sort=True):
        tickers = sorted(set(map(str, group["ticker"])) & available)
        for a, b in itertools.combinations(tickers, 2):
            same_issuer = issuer_lookup.get(a) and issuer_lookup.get(a) == issuer_lookup.get(b)
            if settings.exclude_same_issuer and same_issuer:
                continue
            work.append((a, b, str(sector)))

    def run(item: tuple[str, str, str]) -> dict[str, object]:
        a, b, sector = item
        return evaluate_pair(
            a,
            b,
            sector,
            prices,
            expected_observations=len(prices),
            settings=settings,
            i1_lookup=i1_lookup,
        )

    if settings.workers == 1:
        rows = [run(item) for item in work]
    else:
        with ThreadPoolExecutor(max_workers=settings.workers) as executor:
            rows = list(executor.map(run, work))
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics, integration

    diagnostics["fdr_qvalue"] = np.nan
    for _, index in diagnostics.groupby("sector", sort=False).groups.items():
        diagnostics.loc[index, "fdr_qvalue"] = benjamini_hochberg(
            diagnostics.loc[index, "pair_pvalue"]
        )
    diagnostics["fdr_significant"] = diagnostics["fdr_qvalue"] <= settings.alpha
    diagnostics["eligible"] = (
        diagnostics["screen_status"].eq("ok")
        & diagnostics["fdr_significant"]
        & diagnostics["positive_hedge_ratio"]
        & diagnostics["both_look_i1"]
        & diagnostics["half_life_days"].between(2.0, settings.maximum_half_life_days)
        & diagnostics["hurst_exponent"].lt(0.50)
    )
    diagnostics = diagnostics.sort_values(
        ["sector", "pair_pvalue", "half_life_days", "pair"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    diagnostics["cointegration_rank_in_sector"] = (
        diagnostics.groupby("sector")["pair_pvalue"].rank(method="first", ascending=True).astype("Int64")
    )
    return diagnostics, integration


def select_top_pairs(diagnostics: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Select strict candidates first, then deterministic fallbacks to fill each sector."""
    if diagnostics.empty:
        return diagnostics.copy()
    selected_groups: list[pd.DataFrame] = []
    for sector, group in diagnostics.groupby("sector", sort=True):
        tradable = group["screen_status"].eq("ok") & group["pair_pvalue"].notna()
        if "positive_hedge_ratio" in group:
            tradable &= group["positive_hedge_ratio"].fillna(False)
        if "alpha" in group:
            tradable &= np.isfinite(pd.to_numeric(group["alpha"], errors="coerce"))
        if "beta" in group:
            tradable &= np.isfinite(pd.to_numeric(group["beta"], errors="coerce"))
        usable = group[tradable].copy()
        usable = usable.sort_values(
            ["eligible", "pair_pvalue", "half_life_days", "pair"],
            ascending=[False, True, True, True],
            na_position="last",
        ).head(top_n)
        usable["selection_status"] = np.where(usable["eligible"], "strict", "fallback")
        usable["selection_rank_in_sector"] = np.arange(1, len(usable) + 1)
        selected_groups.append(usable)
    if not selected_groups:
        return diagnostics.iloc[0:0].copy()
    return pd.concat(selected_groups, ignore_index=True)


def descriptive_top_pairs(diagnostics: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the literal lowest-p-value pairs per sector for descriptive reporting."""
    if diagnostics.empty:
        return diagnostics.copy()
    usable = diagnostics[diagnostics["screen_status"].eq("ok") & diagnostics["pair_pvalue"].notna()]
    top = (
        usable.sort_values(["sector", "pair_pvalue", "half_life_days", "pair"])
        .groupby("sector", sort=True, group_keys=False)
        .head(top_n)
        .copy()
    )
    top["descriptive_rank_in_sector"] = top.groupby("sector").cumcount() + 1
    return top.reset_index(drop=True)
