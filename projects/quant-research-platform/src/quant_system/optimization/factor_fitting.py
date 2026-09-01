"""Regularized factor fitting without forward-period leakage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quant_system.config import AppConfig
from quant_system.optimization.factor_mining import build_factor_mining_panel, mine_factors
from quant_system.strategies.base import StrategyContext


def fit_factor_model(
    features: pd.DataFrame,
    context: StrategyContext,
    config: AppConfig,
    *,
    horizon: int = 5,
    max_factors: int = 8,
    ridge_alpha: float = 10.0,
    min_names_per_date: int = 20,
    allow_negative_weights: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fit a regularized composite factor on train only and evaluate OOS."""

    mining = mine_factors(features, context, config, horizons=(horizon,), max_factors=80, min_names_per_date=min_names_per_date)
    panel = mining.get("panel", pd.DataFrame())
    selected = mining.get("selected_decorrelated", pd.DataFrame())
    if panel.empty or selected.empty:
        empty = pd.DataFrame()
        return {"weights": empty, "performance": empty, "daily_ic": empty, "latest_scores": empty, "mining_summary": mining.get("summary", empty)}
    candidates = _candidate_rows(selected, max_factors=max_factors)
    if candidates.empty:
        empty = pd.DataFrame()
        return {"weights": empty, "performance": empty, "daily_ic": empty, "latest_scores": empty, "mining_summary": mining.get("summary", empty)}
    factors = candidates["factor"].astype(str).tolist()
    directions = dict(zip(candidates["factor"].astype(str), candidates["direction"].astype(str)))
    model_panel = _build_model_panel(panel, factors, directions, horizon=horizon)
    train = _date_slice(model_panel, config.regime.train_start, config.regime.train_end).dropna(subset=[f"fwd_{horizon}d"])
    if train.empty:
        empty = pd.DataFrame()
        return {"weights": empty, "performance": empty, "daily_ic": empty, "latest_scores": empty, "mining_summary": mining.get("summary", empty)}
    priors = dict(zip(candidates["factor"].astype(str), pd.to_numeric(candidates["selection_score"], errors="coerce").fillna(0.0)))
    weights = _fit_ridge_weights(train, factors, horizon=horizon, ridge_alpha=ridge_alpha, priors=priors, allow_negative_weights=allow_negative_weights)
    scored = _score_panel(model_panel, weights)
    performance, daily_ic = _evaluate_composite(scored, config, horizon=horizon, min_names_per_date=min_names_per_date)
    latest_scores = _latest_scores(scored, top_n=80)
    weights_table = _weights_table(weights, candidates, ridge_alpha=ridge_alpha, horizon=horizon, allow_negative_weights=allow_negative_weights)
    return {
        "weights": weights_table,
        "performance": performance,
        "daily_ic": daily_ic,
        "latest_scores": latest_scores,
        "mining_summary": mining.get("summary", pd.DataFrame()),
        "selected_decorrelated": selected,
    }


def save_factor_fit_report(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    """Persist factor fit artifacts and an HTML report."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(out / f"{name}.csv", index=False)
    (out / "factor_fit_report.html").write_text(_html_report(tables), encoding="utf-8")


def _candidate_rows(selected: pd.DataFrame, *, max_factors: int) -> pd.DataFrame:
    data = selected.copy()
    if "decorrelation_status" in data:
        data = data[data["decorrelation_status"].eq("keep")]
    if "candidate_status" in data:
        data = data[data["candidate_status"].eq("candidate")]
    data = data.sort_values("selection_score", ascending=False)
    return data.head(int(max_factors)).reset_index(drop=True)


def _build_model_panel(panel: pd.DataFrame, factors: list[str], directions: dict[str, str], *, horizon: int) -> pd.DataFrame:
    keep = ["date", "symbol", f"fwd_{horizon}d", *factors]
    data = panel[[col for col in keep if col in panel.columns]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    for factor in factors:
        if factor not in data:
            data[factor] = np.nan
        raw = pd.to_numeric(data[factor], errors="coerce")
        z = raw.groupby(data["date"]).transform(_winsorized_zscore)
        if directions.get(factor) == "low_is_good":
            z = -z
        data[f"{factor}__z"] = z
    return data


def _fit_ridge_weights(
    train: pd.DataFrame,
    factors: list[str],
    *,
    horizon: int,
    ridge_alpha: float,
    priors: dict[str, float],
    allow_negative_weights: bool,
) -> dict[str, float]:
    zcols = [f"{factor}__z" for factor in factors]
    data = train[["date", f"fwd_{horizon}d", *zcols]].copy()
    data[f"fwd_{horizon}d"] = pd.to_numeric(data[f"fwd_{horizon}d"], errors="coerce")
    y = data[f"fwd_{horizon}d"] - data.groupby("date")[f"fwd_{horizon}d"].transform("mean")
    x = data[zcols].fillna(0.0).to_numpy(dtype=float)
    yv = y.fillna(0.0).to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(yv)
    x = x[valid]
    yv = yv[valid]
    if x.size == 0:
        return {factor: 0.0 for factor in factors}
    xtx = x.T @ x
    penalty = float(ridge_alpha) * np.eye(len(factors))
    raw = np.linalg.pinv(xtx + penalty) @ x.T @ yv
    if not allow_negative_weights:
        raw = np.clip(raw, 0.0, None)
        if raw.sum() <= 1e-12:
            raw = np.array([max(float(priors.get(factor, 0.0)), 0.0) for factor in factors], dtype=float)
    if np.abs(raw).sum() > 1e-12:
        raw = raw / np.abs(raw).sum()
    return dict(zip(factors, raw.astype(float)))


def _score_panel(panel: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = panel.copy()
    score = pd.Series(0.0, index=out.index)
    for factor, weight in weights.items():
        score = score + float(weight) * out.get(f"{factor}__z", 0.0).fillna(0.0)
    out["fitted_factor_score"] = score
    out["fitted_factor_rank"] = out.groupby("date")["fitted_factor_score"].rank(pct=True)
    return out


def _evaluate_composite(
    scored: pd.DataFrame,
    config: AppConfig,
    *,
    horizon: int,
    min_names_per_date: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = [
        ("full", None, None, False),
        ("train", config.regime.train_start, config.regime.train_end, True),
        ("validation", config.regime.validation_start, config.regime.validation_end, False),
        ("forward", config.regime.forward_start, None, False),
    ]
    ret_col = f"fwd_{horizon}d"
    daily_rows: list[dict] = []
    perf_rows: list[dict] = []
    for period, start, end, used_for_fit in periods:
        data = _date_slice(scored, start, end)
        daily_ic = _daily_ic(data, "fitted_factor_score", ret_col, min_names_per_date=min_names_per_date)
        spread = _daily_quantile_spread(data, "fitted_factor_score", ret_col, min_names_per_date=min_names_per_date)
        for date, value in daily_ic.items():
            daily_rows.append({"period": period, "date": date, "daily_rank_ic": value, "used_for_fit": used_for_fit})
        if len(daily_ic) < 10:
            continue
        perf_rows.append(
            {
                "period": period,
                "horizon": f"{horizon}d",
                "used_for_fit": used_for_fit,
                "mean_rank_ic": float(daily_ic.mean()),
                "median_rank_ic": float(daily_ic.median()),
                "ic_t_stat": _t_stat(daily_ic),
                "positive_ic_rate": float((daily_ic > 0).mean()),
                "mean_top_bottom_spread": float(spread.mean()) if len(spread) else np.nan,
                "spread_t_stat": _t_stat(spread) if len(spread) else np.nan,
                "n_days": int(len(daily_ic)),
                "selection_note": "Weights fitted only on train. Validation/forward are out-of-sample diagnostics.",
            }
        )
    return pd.DataFrame(perf_rows), pd.DataFrame(daily_rows)


def _latest_scores(scored: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    latest_date = scored["date"].max()
    latest = scored[scored["date"].eq(latest_date)].copy()
    cols = ["date", "symbol", "fitted_factor_score", "fitted_factor_rank"]
    return latest[cols].sort_values("fitted_factor_score", ascending=False).head(int(top_n)).reset_index(drop=True)


def _weights_table(
    weights: dict[str, float],
    candidates: pd.DataFrame,
    *,
    ridge_alpha: float,
    horizon: int,
    allow_negative_weights: bool,
) -> pd.DataFrame:
    rows = [
        {
            "factor": factor,
            "fitted_weight": weight,
            "abs_weight": abs(weight),
            "ridge_alpha": ridge_alpha,
            "horizon": f"{horizon}d",
            "allow_negative_weights": allow_negative_weights,
        }
        for factor, weight in weights.items()
    ]
    out = pd.DataFrame(rows)
    if not candidates.empty:
        out = out.merge(candidates, on="factor", how="left", suffixes=("", "_candidate"))
    return out.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def _winsorized_zscore(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() < 5:
        return pd.Series(np.nan, index=values.index)
    lo = x.quantile(0.01)
    hi = x.quantile(0.99)
    x = x.clip(lo, hi)
    std = x.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return ((x - x.mean()) / std).clip(-3.0, 3.0)


def _daily_ic(data: pd.DataFrame, factor: str, ret_col: str, *, min_names_per_date: int) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    subset = data[["date", factor, ret_col]].dropna()
    for date, day in subset.groupby("date", sort=True):
        if len(day) < min_names_per_date or day[factor].nunique() < 5 or day[ret_col].nunique() < 5:
            continue
        ic = day[factor].rank().corr(day[ret_col].rank())
        if pd.notna(ic):
            values[pd.Timestamp(date)] = float(ic)
    return pd.Series(values, dtype=float).sort_index()


def _daily_quantile_spread(data: pd.DataFrame, factor: str, ret_col: str, *, min_names_per_date: int) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    subset = data[["date", factor, ret_col]].dropna()
    for date, day in subset.groupby("date", sort=True):
        if len(day) < min_names_per_date or day[factor].nunique() < 5:
            continue
        ranks = day[factor].rank(method="first")
        try:
            bucket = pd.qcut(ranks, 5, labels=False) + 1
        except ValueError:
            continue
        top = day.loc[bucket.eq(5), ret_col].mean()
        bottom = day.loc[bucket.eq(1), ret_col].mean()
        if pd.notna(top) and pd.notna(bottom):
            values[pd.Timestamp(date)] = float(top - bottom)
    return pd.Series(values, dtype=float).sort_index()


def _date_slice(data: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = data
    if start:
        out = out[out["date"].ge(pd.to_datetime(start))]
    if end:
        out = out[out["date"].le(pd.to_datetime(end))]
    return out


def _t_stat(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    return float(values.mean() / (std / np.sqrt(len(values)))) if std > 0 else np.nan


def _html_report(tables: dict[str, pd.DataFrame]) -> str:
    weights = tables.get("weights", pd.DataFrame())
    performance = tables.get("performance", pd.DataFrame())
    latest = tables.get("latest_scores", pd.DataFrame())
    selected = tables.get("selected_decorrelated", pd.DataFrame())
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Factor Fit Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 28px; color: #172033; }}
    .notice {{ background: #fff7e6; border: 1px solid #ffd58a; padding: 12px 14px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #e4e8f0; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>持续因子拟合报告</h1>
  <p class="notice">权重只在 train period 拟合；validation/forward/current 只用于验收和监控。不要用 forward 结果反向改参数。</p>
  <h2>拟合表现</h2>{_format_table(performance, 20)}
  <h2>拟合权重</h2>{_format_table(weights, 30)}
  <h2>最新综合因子 Top</h2>{_format_table(latest, 80)}
  <h2>原始去共线候选池</h2>{_format_table(selected, 50)}
</body>
</html>"""


def _format_table(data: pd.DataFrame, n: int) -> str:
    if data.empty:
        return "<p>无足够数据。</p>"
    table = data.head(n).copy()
    for column in table.columns:
        if table[column].dtype.kind in "fc":
            table[column] = pd.to_numeric(table[column], errors="coerce").map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return table.to_html(index=False, escape=False)
