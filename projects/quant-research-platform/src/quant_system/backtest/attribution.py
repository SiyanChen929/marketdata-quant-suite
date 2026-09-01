"""Factor and strategy-family attribution for research backtests."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


FACTOR_LABELS: dict[str, str] = {
    "final_score": "综合分",
    "technical_score": "技术总分",
    "relative_strength_score": "相对强弱",
    "theme_score": "主题分",
    "fundamental_score": "基本面分",
    "event_risk_score": "事件风险",
    "momentum_return_pct": "动量收益",
    "momentum_risk_adj": "风险调整动量",
    "distance_from_high_pct": "距高点/新高贴近度",
    "volume_expansion": "量能扩张",
    "breakout_score": "突破分",
    "breakout_volume_score": "突破量能分",
    "breakout_rs_score": "突破相对强弱",
    "retest_score": "回踩分",
    "trend_price_vs_slow_pct": "趋势均线距离",
    "trend_adx": "ADX趋势强度",
}

FACTOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "momentum": ("momentum_return_pct", "momentum_risk_adj", "distance_from_high_pct", "volume_expansion"),
    "breakout_retest": ("breakout_score", "breakout_volume_score", "breakout_rs_score", "retest_score"),
    "trend": ("trend_price_vs_slow_pct", "trend_adx"),
    "ensemble_topline": ("final_score", "technical_score", "relative_strength_score"),
    "fundamental_event": ("fundamental_score", "event_risk_score"),
}

FAMILY_LABELS: dict[str, str] = {
    "momentum": "动量",
    "breakout_retest": "突破/回踩",
    "trend": "趋势",
    "ensemble_topline": "综合/技术/相对强弱",
    "fundamental_event": "基本面/事件",
}

SCORE_PATTERNS: dict[str, str] = {
    "momentum_return_pct": r"(?:^|[|\s])return=([-+0-9.]+)%",
    "momentum_risk_adj": r"(?:^|[;\s|])risk_adj=([-+0-9.]+)",
    "distance_from_high_pct": r"(?:^|[;\s|])(?:near_high|prior_high_gap)=([-+0-9.]+)%",
    "volume_expansion": r"(?:^|[;\s|])volume_exp=([-+0-9.]+)",
    "breakout_score": r"(?:^|[;\s|])breakout=([-+0-9.]+)",
    "breakout_volume_score": r"breakout=[^|]*?volume=([-+0-9.]+)",
    "breakout_rs_score": r"breakout=[^|]*?rs=([-+0-9.]+)",
    "retest_score": r"(?:^|[;\s|])(?:retest|retest_score)=([-+0-9.]+)",
    "trend_price_vs_slow_pct": r"(?:^|[;\s|])price_vs_slow=([-+0-9.]+)%",
    "trend_adx": r"(?:^|[;\s|])adx=([-+0-9.]+)",
}

DEFAULT_FACTORS: tuple[str, ...] = tuple(FACTOR_LABELS)


def factor_correlation_table(signals: pd.DataFrame, factors: Iterable[str] = DEFAULT_FACTORS) -> pd.DataFrame:
    """Return high absolute factor correlations for collinearity diagnostics."""

    if signals.empty:
        return pd.DataFrame(columns=["factor_a", "factor_b", "rank_correlation", "abs_correlation"])
    panel = _parse_score_decomposition(signals.copy())
    available = [factor for factor in factors if factor in panel.columns and panel[factor].notna().sum() >= 100]
    if len(available) < 2:
        return pd.DataFrame(columns=["factor_a", "factor_b", "rank_correlation", "abs_correlation"])
    ranked = panel[available].rank()
    corr = ranked.corr(method="pearson")
    rows: list[dict] = []
    for idx, factor_a in enumerate(available):
        for factor_b in available[idx + 1 :]:
            value = corr.loc[factor_a, factor_b]
            if pd.notna(value):
                rows.append(
                    {
                        "factor_a": factor_a,
                        "factor_b": factor_b,
                        "rank_correlation": float(value),
                        "abs_correlation": abs(float(value)),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def compute_attribution_tables(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    train_start: str | None = None,
    train_end: str | None = None,
    validation_start: str | None = None,
    validation_end: str | None = None,
    forward_start: str | None = None,
    horizons: Iterable[int] = (1, 5, 21),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return factor IC and strategy-family IC tables.

    IC is computed cross-sectionally by date with rank correlation between a
    factor available at signal date t and forward returns from t to t+h.
    """

    panel = build_factor_forward_return_panel(signals, prices, horizons=horizons)
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    periods = _periods(train_start, train_end, validation_start, validation_end, forward_start)
    factor_rows: list[pd.DataFrame] = []
    family_rows: list[pd.DataFrame] = []
    for name, start, end in periods:
        period_panel = _date_slice(panel, start, end)
        if period_panel.empty:
            continue
        factor_rows.append(factor_ic_table(period_panel, period=name, horizons=horizons))
        family_rows.append(strategy_family_ic_table(period_panel, period=name, horizons=horizons))
    factor_ic = pd.concat(factor_rows, ignore_index=True) if factor_rows else pd.DataFrame()
    family_ic = pd.concat(family_rows, ignore_index=True) if family_rows else pd.DataFrame()
    return factor_ic, family_ic


def build_factor_forward_return_panel(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 5, 21),
) -> pd.DataFrame:
    """Merge signal factors with forward returns."""

    required = {"date", "symbol"}
    if signals.empty or prices.empty or not required.issubset(signals.columns) or not required.issubset(prices.columns):
        return pd.DataFrame()
    panel = signals.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["symbol"] = panel["symbol"].astype(str)
    panel = _parse_score_decomposition(panel)
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px = prices[["date", "symbol", price_col]].copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px["symbol"] = px["symbol"].astype(str)
    px = px.drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"])
    wide = px.pivot(index="date", columns="symbol", values=price_col).sort_index()
    for horizon in horizons:
        forward_matrix = wide.shift(-int(horizon)) / wide - 1.0
        try:
            forward = forward_matrix.stack(future_stack=True).dropna().rename(f"fwd_{int(horizon)}d").reset_index()
        except TypeError:
            forward = forward_matrix.stack(dropna=True).rename(f"fwd_{int(horizon)}d").reset_index()
        panel = panel.merge(forward, on=["date", "symbol"], how="left")
    return panel


def factor_ic_table(
    panel: pd.DataFrame,
    *,
    period: str = "full",
    horizons: Iterable[int] = (1, 5, 21),
    factors: Iterable[str] = DEFAULT_FACTORS,
) -> pd.DataFrame:
    """Summarize daily rank IC for each factor and horizon."""

    rows: list[dict] = []
    for factor in factors:
        if factor not in panel.columns or panel[factor].notna().sum() < 100:
            continue
        for horizon in horizons:
            ret_col = f"fwd_{int(horizon)}d"
            if ret_col not in panel.columns:
                continue
            ics = daily_rank_ic(panel, factor, ret_col)
            if len(ics) < 20:
                continue
            rows.append(_summarize_ic(ics, period=period, factor=factor, label=FACTOR_LABELS.get(factor, factor), horizon=f"{int(horizon)}d"))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["period", "horizon_days", "mean_rank_ic"], ascending=[True, True, False])
    return out


def strategy_family_ic_table(
    panel: pd.DataFrame,
    *,
    period: str = "full",
    horizons: Iterable[int] = (1, 5, 21),
) -> pd.DataFrame:
    """Average component daily ICs into strategy-family attribution."""

    rows: list[dict] = []
    for family, factors in FACTOR_FAMILIES.items():
        for horizon in horizons:
            daily_components = []
            ret_col = f"fwd_{int(horizon)}d"
            for factor in factors:
                if factor in panel.columns and ret_col in panel.columns:
                    ics = daily_rank_ic(panel, factor, ret_col)
                    if len(ics) >= 20:
                        daily_components.append(ics.rename(factor))
            if not daily_components:
                continue
            family_ic = pd.concat(daily_components, axis=1).mean(axis=1, skipna=True).dropna()
            if len(family_ic) < 20:
                continue
            rows.append(_summarize_ic(family_ic, period=period, factor=family, label=FAMILY_LABELS.get(family, family), horizon=f"{int(horizon)}d"))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.rename(columns={"factor": "family"})
        out = out.sort_values(["period", "horizon_days", "mean_rank_ic"], ascending=[True, True, False])
    return out


def daily_rank_ic(panel: pd.DataFrame, factor: str, ret_col: str) -> pd.Series:
    """Compute per-date Spearman rank IC using pandas ranks."""

    values: dict[pd.Timestamp, float] = {}
    subset = panel[["date", factor, ret_col]].dropna()
    for date, day in subset.groupby("date", sort=True):
        if len(day) < 10 or day[factor].nunique() < 3 or day[ret_col].nunique() < 3:
            continue
        ic = day[factor].rank().corr(day[ret_col].rank())
        if pd.notna(ic):
            values[pd.Timestamp(date)] = float(ic)
    return pd.Series(values, dtype=float).sort_index()


def _parse_score_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    text = out.get("score_decomposition", pd.Series("", index=out.index)).fillna("").astype(str)
    for column, pattern in SCORE_PATTERNS.items():
        out[column] = pd.to_numeric(text.str.extract(pattern, expand=False), errors="coerce")
    return out


def _summarize_ic(ics: pd.Series, *, period: str, factor: str, label: str, horizon: str) -> dict:
    mean = float(ics.mean())
    std = float(ics.std(ddof=1))
    return {
        "period": period,
        "factor": factor,
        "label": label,
        "horizon": horizon,
        "horizon_days": int(str(horizon).replace("d", "")),
        "mean_rank_ic": mean,
        "median_rank_ic": float(ics.median()),
        "ic_t_stat": mean / (std / np.sqrt(len(ics))) if std > 0 else np.nan,
        "positive_ic_rate": float((ics > 0).mean()),
        "n_days": int(len(ics)),
    }


def _periods(
    train_start: str | None,
    train_end: str | None,
    validation_start: str | None,
    validation_end: str | None,
    forward_start: str | None,
) -> list[tuple[str, str | None, str | None]]:
    periods: list[tuple[str, str | None, str | None]] = [("full", None, None)]
    if train_start or train_end:
        periods.append(("train", train_start, train_end))
    if validation_start or validation_end:
        periods.append(("validation", validation_start, validation_end))
    if forward_start:
        periods.append(("forward_current", forward_start, None))
    return periods


def _date_slice(panel: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = panel
    if start:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.copy()
