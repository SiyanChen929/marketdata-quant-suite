"""Regime-aware reporting helpers."""

from __future__ import annotations

import pandas as pd

from quant_system.backtest.metrics import equity_metrics


def period_metrics(equity_curve: pd.DataFrame, periods: dict[str, tuple[str, str | None]]) -> pd.DataFrame:
    """Compute metrics by named train/validation/forward periods."""

    rows = []
    if equity_curve.empty:
        return pd.DataFrame()
    curve = equity_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    for name, (start, end) in periods.items():
        mask = curve["date"] >= pd.Timestamp(start)
        if end:
            mask &= curve["date"] <= pd.Timestamp(end)
        metrics = equity_metrics(curve.loc[mask])
        metrics["period"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)
