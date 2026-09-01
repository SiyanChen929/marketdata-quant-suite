"""Build executable order tickets from the latest trading blotter."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import numpy as np
import pandas as pd

from quant_system.config import LiveTradingConfig


def build_order_tickets(blotter: pd.DataFrame, config: LiveTradingConfig, *, equity: float | None = None) -> pd.DataFrame:
    """Translate latest manual trading actions into normalized order tickets."""

    if blotter.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    account_equity = float(equity) if equity is not None and np.isfinite(equity) else np.nan
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _, row in blotter.iterrows():
        symbol = str(row.get("symbol", "")).upper().strip()
        price = _float(row.get("close"))
        notional = _float(row.get("estimated_trade_notional"))
        if not symbol or not np.isfinite(price) or price <= 0 or not np.isfinite(notional):
            continue
        abs_notional = abs(notional)
        if abs_notional < float(config.min_order_notional):
            continue
        side = _side_from_delta(notional, _float(row.get("current_weight")), _float(row.get("target_weight")))
        quantity = abs_notional / price
        limit_price = _limit_price(price, side, config.limit_price_buffer_bps)
        if config.max_order_notional > 0:
            abs_notional = min(abs_notional, float(config.max_order_notional))
            quantity = abs_notional / price
        client_order_id = _client_order_id(timestamp, symbol, side, quantity, limit_price)
        rows.append(
            {
                "client_order_id": client_order_id,
                "created_at_utc": timestamp,
                "symbol": symbol,
                "side": side,
                "quantity": float(quantity),
                "notional": float(abs_notional),
                "signed_notional": float(notional),
                "order_type": "market" if config.allow_market_orders and config.default_order_type == "market" else "limit",
                "limit_price": float(limit_price),
                "time_in_force": str(config.tif).lower(),
                "signal_date": row.get("signal_date"),
                "action_label": row.get("action_label"),
                "target_weight": _float(row.get("target_weight")),
                "current_weight": _float(row.get("current_weight")),
                "delta_weight": _float(row.get("delta_weight")),
                "estimated_shares": _float(row.get("estimated_shares")),
                "reference_price": price,
                "account_equity": account_equity,
                "priority_score": _float(row.get("priority_score")),
                "final_score": _float(row.get("final_score")),
                "relative_strength_score": _float(row.get("relative_strength_score")),
                "fundamental_score": _float(row.get("fundamental_score")),
                "event_risk_score": _float(row.get("event_risk_score")),
                "overnight_gap_risk_score": _float(row.get("overnight_gap_risk_score")),
                "minute_pretrade_risk_score": _float(row.get("minute_pretrade_risk_score")),
                "primary_theme": row.get("primary_theme", ""),
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "decision_note": row.get("decision_note", ""),
            }
        )
    return pd.DataFrame(rows)


def _side_from_delta(notional: float, current_weight: float, target_weight: float) -> str:
    if notional > 0:
        return "buy" if target_weight >= 0 else "buy_to_cover"
    return "sell" if current_weight > 0 else "sell_short"


def _limit_price(price: float, side: str, buffer_bps: float) -> float:
    buffer = float(buffer_bps) / 10_000.0
    if side in {"buy", "buy_to_cover"}:
        return round(price * (1.0 + buffer), 4)
    return round(price * (1.0 - buffer), 4)


def _client_order_id(timestamp: str, symbol: str, side: str, quantity: float, limit_price: float) -> str:
    payload = f"{timestamp}|{symbol}|{side}|{quantity:.4f}|{limit_price:.4f}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return f"QS-{timestamp}-{symbol}-{digest}"


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
