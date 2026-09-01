"""Institutional-style portfolio attribution tables."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


ATTRIBUTION_DIMENSIONS = ("primary_theme", "sector", "industry", "symbol", "strategy_family", "entry_reason")
FACTOR_COLUMNS = ("final_score", "technical_score", "relative_strength_score", "fundamental_score", "event_risk_score")


def compute_institutional_attribution(result) -> dict[str, pd.DataFrame]:
    """Compute attribution by theme, industry, symbol, strategy, factor, and entry reason."""

    positions = _positions(result)
    prices = _prices(result)
    targets = _metadata(result)
    if positions.empty or prices.empty:
        return {name: pd.DataFrame() for name in [*ATTRIBUTION_DIMENSIONS, "factor_quintile"]}
    contribution_panel = _contribution_panel(positions, prices, targets)
    if contribution_panel.empty:
        return {name: pd.DataFrame() for name in [*ATTRIBUTION_DIMENSIONS, "factor_quintile"]}
    tables: dict[str, pd.DataFrame] = {}
    for dimension in ATTRIBUTION_DIMENSIONS:
        tables[dimension] = _bucket_table(contribution_panel, dimension)
    tables["factor_quintile"] = _factor_quintile_table(contribution_panel)
    return tables


def compute_period_institutional_attribution(
    result,
    periods: dict[str, tuple[str | None, str | None]],
) -> dict[str, pd.DataFrame]:
    """Compute attribution tables inside named as-of research periods.

    These artifacts are used for strategy evolution. They keep theme/symbol
    selection honest by separating train, validation, and forward contribution
    instead of ranking drivers on the full period after the fact.
    """

    positions = _positions(result)
    prices = _prices(result)
    targets = _metadata(result)
    empty = {name: pd.DataFrame() for name in [*ATTRIBUTION_DIMENSIONS, "factor_quintile"]}
    if positions.empty or prices.empty:
        return empty
    panel = _contribution_panel(positions, prices, targets)
    if panel.empty:
        return empty

    tables: dict[str, list[pd.DataFrame]] = {name: [] for name in [*ATTRIBUTION_DIMENSIONS, "factor_quintile"]}
    for period_name, (start, end) in periods.items():
        period_panel = _filter_period(panel, start, end)
        if period_panel.empty:
            continue
        for dimension in ATTRIBUTION_DIMENSIONS:
            table = _bucket_table(period_panel, dimension)
            if not table.empty:
                table.insert(0, "period", period_name)
                tables[dimension].append(table)
        factor_table = _factor_quintile_table(period_panel)
        if not factor_table.empty:
            factor_table.insert(0, "period", period_name)
            tables["factor_quintile"].append(factor_table)
    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in tables.items()
    }


def _filter_period(panel: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = panel.copy()
    dates = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    mask = dates.notna()
    if start:
        mask &= dates.ge(pd.Timestamp(start).normalize())
    if end:
        mask &= dates.le(pd.Timestamp(end).normalize())
    return out.loc[mask].copy()


def _positions(result) -> pd.DataFrame:
    data = result.positions if isinstance(getattr(result, "positions", None), pd.DataFrame) else pd.DataFrame()
    if data.empty or not {"date", "symbol"}.issubset(data.columns):
        return pd.DataFrame()
    out = data.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    if "weight" not in out.columns:
        if "market_value" in out.columns:
            equity = result.equity_curve if isinstance(getattr(result, "equity_curve", None), pd.DataFrame) else pd.DataFrame()
            if not equity.empty and {"date", "equity"}.issubset(equity.columns):
                eq = equity[["date", "equity"]].copy()
                eq["date"] = pd.to_datetime(eq["date"], errors="coerce").dt.normalize()
                out = out.merge(eq, on="date", how="left")
            if "equity" in out.columns:
                out["weight"] = pd.to_numeric(out["market_value"], errors="coerce") / pd.to_numeric(out["equity"], errors="coerce").replace(0, np.nan)
            else:
                out["weight"] = 0.0
        else:
            out["weight"] = 0.0
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out[["date", "symbol", "weight"]].dropna(subset=["date", "symbol"])


def _prices(result) -> pd.DataFrame:
    data = result.prices if isinstance(getattr(result, "prices", None), pd.DataFrame) else pd.DataFrame()
    if data.empty or not {"date", "symbol"}.issubset(data.columns):
        return pd.DataFrame()
    price_col = "adj_close" if "adj_close" in data.columns else "close"
    out = data[["date", "symbol", price_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=["date", "symbol", price_col]).sort_values(["symbol", "date"])
    out["daily_return"] = out.groupby("symbol")[price_col].pct_change().fillna(0.0)
    return out[["date", "symbol", "daily_return"]]


def _metadata(result) -> pd.DataFrame:
    targets = result.targets if isinstance(getattr(result, "targets", None), pd.DataFrame) else pd.DataFrame()
    signals = result.signals if isinstance(getattr(result, "signals", None), pd.DataFrame) else pd.DataFrame()
    frames = []
    for frame in [targets, signals]:
        if frame.empty or not {"date", "symbol"}.issubset(frame.columns):
            continue
        keep = [
            "date",
            "symbol",
            "primary_theme",
            "sector",
            "industry",
            "reason_for_entry",
            *FACTOR_COLUMNS,
        ]
        available = [col for col in keep if col in frame.columns]
        item = frame[available].copy()
        item["date"] = pd.to_datetime(item["date"], errors="coerce").dt.normalize()
        item["symbol"] = item["symbol"].astype(str).str.upper()
        frames.append(item)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "symbol"], keep="first")
    for column in ("primary_theme", "sector", "industry", "reason_for_entry"):
        if column not in out.columns:
            out[column] = "unknown"
        out[column] = out[column].fillna("unknown").replace("", "unknown")
    out["strategy_family"] = out["reason_for_entry"].map(_strategy_family)
    out["entry_reason"] = out["reason_for_entry"].map(_entry_reason)
    return out


def _contribution_panel(positions: pd.DataFrame, prices: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    panel = positions.merge(prices, on=["date", "symbol"], how="left")
    panel["daily_return"] = pd.to_numeric(panel["daily_return"], errors="coerce").fillna(0.0)
    panel["contribution"] = panel["weight"] * panel["daily_return"]
    if not metadata.empty:
        panel = panel.merge(metadata, on=["date", "symbol"], how="left")
    for column in ("primary_theme", "sector", "industry", "strategy_family", "entry_reason"):
        if column not in panel:
            panel[column] = "unknown"
        panel[column] = panel[column].fillna("unknown").replace("", "unknown")
    return panel


def _bucket_table(panel: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if dimension not in panel.columns:
        return pd.DataFrame()
    grouped = panel.groupby(dimension, dropna=False)
    out = grouped.agg(
        days=("date", "nunique"),
        avg_net_weight=("weight", "mean"),
        avg_gross_weight=("weight", lambda s: s.abs().mean()),
        total_contribution=("contribution", "sum"),
        avg_daily_contribution=("contribution", "mean"),
        hit_rate=("contribution", lambda s: float((s > 0).mean())),
        worst_daily_contribution=("contribution", "min"),
        best_daily_contribution=("contribution", "max"),
        symbols=("symbol", "nunique"),
    ).reset_index()
    out = out.rename(columns={dimension: "bucket"})
    out["dimension"] = dimension
    return out.sort_values("total_contribution", ascending=False).reset_index(drop=True)


def _factor_quintile_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for factor in FACTOR_COLUMNS:
        if factor not in panel.columns:
            continue
        data = panel[["date", factor, "weight", "contribution", "symbol"]].copy()
        data[factor] = pd.to_numeric(data[factor], errors="coerce")
        data = data.dropna(subset=[factor])
        if data.empty:
            continue
        data["quintile"] = data.groupby("date")[factor].transform(_safe_quintile)
        for quintile, frame in data.dropna(subset=["quintile"]).groupby("quintile"):
            rows.append(
                {
                    "factor": factor,
                    "quintile": int(quintile),
                    "rows": len(frame),
                    "symbols": frame["symbol"].nunique(),
                    "avg_weight": frame["weight"].mean(),
                    "total_contribution": frame["contribution"].sum(),
                    "hit_rate": float((frame["contribution"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["factor", "quintile"]).reset_index(drop=True) if rows else pd.DataFrame()


def _safe_quintile(values: pd.Series) -> pd.Series:
    if values.nunique(dropna=True) < 3:
        return pd.Series(np.nan, index=values.index)
    try:
        return pd.qcut(values.rank(method="first"), 5, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=values.index)


def _strategy_family(reason: str) -> str:
    text = str(reason).lower()
    if "breakout" in text or "retest" in text:
        return "breakout_retest"
    if "trend" in text:
        return "trend"
    if "momentum" in text:
        return "momentum"
    if "mean reversion" in text:
        return "mean_reversion"
    if "crowding" in text:
        return "risk_crowding"
    return "unknown"


def _entry_reason(reason: str) -> str:
    text = str(reason)
    if not text or text.lower() == "nan":
        return "unknown"
    first = re.split(r"[|.;]", text, maxsplit=1)[0].strip()
    return first[:80] if first else "unknown"
