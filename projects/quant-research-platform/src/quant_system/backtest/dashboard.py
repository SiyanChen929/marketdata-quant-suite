"""Read-only view models for the internal backtest dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN_TABLES = (
    "equity_curve",
    "trades",
    "positions",
    "signals",
    "period_metrics",
    "daily_data_quality_summary",
)


def discover_runs(root: str | Path = "runs", limit: int = 100) -> pd.DataFrame:
    """Return recent complete backtest runs without traversing heavy artifacts."""

    base = Path(root)
    rows: list[dict[str, object]] = []
    if not base.exists():
        return pd.DataFrame(columns=["run_id", "path", "modified_at"])
    for metrics_path in base.glob("*/metrics.json"):
        run_dir = metrics_path.parent
        if not (run_dir / "equity_curve.csv").exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
        rows.append(
            {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "modified_at": pd.Timestamp(metrics_path.stat().st_mtime, unit="s", tz="UTC"),
                **{key: metrics.get(key) for key in ("total_return", "cagr", "sharpe", "max_drawdown", "average_turnover")},
            }
        )
    if not rows:
        return pd.DataFrame(columns=["run_id", "path", "modified_at"])
    return pd.DataFrame(rows).sort_values("modified_at", ascending=False).head(max(1, int(limit))).reset_index(drop=True)


def load_run(run_dir: str | Path) -> dict[str, object]:
    """Load the light dashboard artifacts for a saved run."""

    root = Path(run_dir)
    metrics: dict[str, object] = {}
    metrics_path = root / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
    result: dict[str, object] = {"run_id": root.name, "path": root, "metrics": metrics}
    for name in RUN_TABLES:
        path = root / f"{name}.csv"
        if not path.exists():
            result[name] = pd.DataFrame()
            continue
        frame = pd.read_csv(path)
        for column in ("date", "datetime"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], format="mixed", errors="coerce")
        result[name] = frame
    return result


def enrich_equity_curve(equity: pd.DataFrame, rolling_window: int = 63) -> pd.DataFrame:
    """Add drawdown and rolling risk diagnostics to an equity curve."""

    if equity.empty or not {"date", "equity"}.issubset(equity.columns):
        return pd.DataFrame()
    out = equity.copy().sort_values("date")
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out = out.dropna(subset=["date", "equity"])
    out["daily_return"] = out["equity"].pct_change().replace([np.inf, -np.inf], np.nan)
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    window = max(5, int(rolling_window))
    rolling = out["daily_return"].rolling(window, min_periods=max(5, window // 3))
    out["rolling_volatility"] = rolling.std(ddof=0) * np.sqrt(252.0)
    out["rolling_sharpe"] = rolling.mean() / rolling.std(ddof=0).replace(0, np.nan) * np.sqrt(252.0)
    out["normalized_equity"] = out["equity"] / out["equity"].iloc[0] * 100.0 if not out.empty else pd.Series(dtype=float)
    return out


def comparison_curves(run_dirs: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build normalized equity curves and metric rows for selected runs."""

    curves: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        loaded = load_run(run_dir)
        curve = enrich_equity_curve(loaded["equity_curve"])
        run_id = str(loaded["run_id"])
        if not curve.empty:
            selected = curve[["date", "normalized_equity", "drawdown"]].copy()
            selected["run_id"] = run_id
            curves.append(selected)
        metric_rows.append({"run_id": run_id, **dict(loaded["metrics"])})
    return (
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["date", "normalized_equity", "drawdown", "run_id"]),
        pd.DataFrame(metric_rows),
    )


def monthly_return_matrix(equity: pd.DataFrame) -> pd.DataFrame:
    """Return year-by-month strategy returns for a heatmap."""

    curve = enrich_equity_curve(equity)
    if curve.empty:
        return pd.DataFrame()
    month_end = curve.set_index("date")["equity"].resample("ME").last()
    returns = month_end.pct_change()
    table = returns.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    matrix = table.pivot(index="year", columns="month", values="return")
    return matrix.reindex(columns=range(1, 13))


def trade_activity(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trading notional and execution costs by day."""

    if trades.empty or "date" not in trades.columns:
        return pd.DataFrame()
    data = trades.copy()
    quantity = pd.to_numeric(data.get("quantity", 0.0), errors="coerce").fillna(0.0).abs()
    price = pd.to_numeric(data.get("price", 0.0), errors="coerce").fillna(0.0)
    data["notional"] = quantity * price
    data["fee"] = pd.to_numeric(data.get("fee", 0.0), errors="coerce").fillna(0.0)
    data["estimated_liquidity_cost"] = pd.to_numeric(data.get("estimated_liquidity_cost", 0.0), errors="coerce").fillna(0.0)
    data["trade_date"] = pd.to_datetime(data["date"]).dt.normalize()
    return (
        data.groupby("trade_date", as_index=False)
        .agg(trades=("symbol", "size"), notional=("notional", "sum"), fees=("fee", "sum"), liquidity_cost=("estimated_liquidity_cost", "sum"))
    )
