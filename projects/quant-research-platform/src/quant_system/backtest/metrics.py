"""Backtest metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_metrics(equity_curve: pd.DataFrame) -> dict[str, float]:
    """Compute standard performance metrics from daily equity."""

    if equity_curve.empty or len(equity_curve) < 2:
        return {}
    curve = equity_curve.sort_values("date").copy()
    equity = curve["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    dates = pd.to_datetime(curve["date"])
    trailing_start = dates.iloc[-1] - pd.DateOffset(years=1)
    trailing_curve = curve.loc[dates >= trailing_start]
    trailing_one_year_return = (
        trailing_curve["equity"].astype(float).iloc[-1] / trailing_curve["equity"].astype(float).iloc[0] - 1.0
        if len(trailing_curve) >= 2
        else total_return
    )
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 252)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    vol = returns.std(ddof=0) * np.sqrt(252)
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0.0
    sortino = cagr / downside if downside > 0 else 0.0
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1.0
    max_dd = drawdowns.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    win_rate = float((returns > 0).mean())
    avg_win = float(returns[returns > 0].mean()) if (returns > 0).any() else 0.0
    avg_loss = float(returns[returns < 0].mean()) if (returns < 0).any() else 0.0
    profit_factor = float(returns[returns > 0].sum() / abs(returns[returns < 0].sum())) if (returns < 0).any() else 0.0
    return {
        "total_return": float(total_return),
        "trailing_one_year_return": float(trailing_one_year_return),
        "cagr": float(cagr),
        "annualized_volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_dd),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "average_gross_exposure": float(curve.get("gross_exposure", pd.Series([0])).mean()),
        "average_net_exposure": float(curve.get("net_exposure", pd.Series([0])).mean()),
        "average_turnover": float(curve.get("turnover", pd.Series([0])).mean()),
        "long_contribution": float(curve.get("long_pnl", pd.Series([0])).sum()),
        "short_contribution": float(curve.get("short_pnl", pd.Series([0])).sum()),
    }


def monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Monthly returns table."""

    if equity_curve.empty:
        return pd.DataFrame(columns=["month", "return"])
    curve = equity_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    month_end = curve.set_index("date")["equity"].resample("ME").last()
    return month_end.pct_change().fillna(0.0).rename("return").reset_index().rename(columns={"date": "month"})


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown for a price/equity series."""

    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())
