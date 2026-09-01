"""Pre-trade risk checks for live order tickets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.config import LiveTradingConfig


def preflight_orders(orders: pd.DataFrame, config: LiveTradingConfig) -> pd.DataFrame:
    """Attach pass/block decisions and reasons to order tickets."""

    if orders.empty:
        return orders
    data = orders.copy()
    rows = []
    total_notional = float(pd.to_numeric(data.get("notional"), errors="coerce").fillna(0.0).sum())
    equity = _first_equity(data)
    turnover_limit = equity * float(config.max_daily_turnover_pct_equity) if np.isfinite(equity) else np.inf
    total_limit = min(float(config.max_total_order_notional), turnover_limit)
    priority = _numeric_series(data, "priority_score")
    if priority.isna().all():
        priority = _numeric_series(data, "final_score")
    data["_preflight_priority"] = priority.fillna(_numeric_series(data, "notional").fillna(0.0))
    data = data.sort_values(["_preflight_priority", "notional"], ascending=[False, False]).reset_index(drop=True)
    used_budget = 0.0
    for _, row in data.iterrows():
        reasons = _row_block_reasons(row, config)
        notional = _float(row.get("notional"))
        if np.isfinite(notional) and used_budget + notional > total_limit:
            reasons.append(f"daily_order_budget_exhausted:{used_budget + notional:.0f}>{total_limit:.0f}")
        if not reasons and np.isfinite(notional):
            used_budget += notional
        rows.append(
            {
                "client_order_id": row.get("client_order_id"),
                "symbol": row.get("symbol"),
                "risk_pass": not reasons,
                "risk_status": "pass" if not reasons else "blocked",
                "risk_reasons": "|".join(reasons),
                "daily_order_budget_used_before": used_budget - notional if not reasons and np.isfinite(notional) else used_budget,
                "daily_order_budget_limit": total_limit,
                "total_requested_notional": total_notional,
            }
        )
    checks = pd.DataFrame(rows)
    return data.drop(columns=["_preflight_priority"], errors="ignore").merge(checks, on=["client_order_id", "symbol"], how="left")


def approved_orders(preflight: pd.DataFrame) -> pd.DataFrame:
    """Return only orders that passed preflight."""

    if preflight.empty or "risk_pass" not in preflight:
        return pd.DataFrame()
    return preflight[preflight["risk_pass"].fillna(False).astype(bool)].copy()


def _row_block_reasons(row: pd.Series, config: LiveTradingConfig) -> list[str]:
    reasons: list[str] = []
    notional = _float(row.get("notional"))
    target_weight = abs(_float(row.get("target_weight")))
    event_risk = _float(row.get("event_risk_score"))
    overnight_risk = _float(row.get("overnight_gap_risk_score"))
    minute_risk = _float(row.get("minute_pretrade_risk_score"))
    if not np.isfinite(notional) or notional <= 0:
        reasons.append("invalid_notional")
    if np.isfinite(notional) and notional < float(config.min_order_notional):
        reasons.append("below_min_order_notional")
    if np.isfinite(notional) and notional > float(config.max_order_notional):
        reasons.append("above_max_order_notional")
    if np.isfinite(target_weight) and target_weight > float(config.max_single_symbol_weight):
        reasons.append("above_max_single_symbol_weight")
    if str(row.get("order_type", "")).lower() == "market" and not config.allow_market_orders:
        reasons.append("market_orders_disabled")
    if config.block_if_data_quality_flagged and bool(row.get("data_quality_blocked", False)):
        reasons.append("data_quality_blocked")
    if np.isfinite(event_risk) and event_risk > float(config.block_if_event_risk_above):
        reasons.append("event_risk_too_high")
    if np.isfinite(overnight_risk) and overnight_risk > float(config.block_if_overnight_gap_risk_above):
        reasons.append("overnight_gap_risk_too_high")
    if np.isfinite(minute_risk) and minute_risk > float(config.block_if_minute_pretrade_risk_above):
        reasons.append("minute_pretrade_risk_too_high")
    if config.reduce_only_when_risk_flagged and reasons and str(row.get("side")) in {"buy", "sell_short"}:
        reasons.append("new_risk_not_reduce_only")
    return reasons


def _first_equity(data: pd.DataFrame) -> float:
    if "account_equity" not in data:
        return np.nan
    values = pd.to_numeric(data["account_equity"], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce")


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
