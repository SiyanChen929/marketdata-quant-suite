"""Live trading plan and paper submission workflow."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path

import pandas as pd

from quant_system.config import AppConfig
from quant_system.live.broker import BrokerAdapter, UnconfiguredLiveBroker
from quant_system.live.orders import build_order_tickets
from quant_system.live.paper import PaperBrokerAdapter
from quant_system.live.risk import approved_orders, preflight_orders


def create_live_plan(config: AppConfig, *, run_id: str = "latest") -> Path:
    """Create an auditable order plan from an existing research run."""

    run_dir = _resolve_run_dir(config.output_dir, run_id)
    blotter = _load_blotter(run_dir)
    equity = _load_latest_equity(run_dir, fallback=float(config.live.paper_cash))
    orders = build_order_tickets(blotter, config.live, equity=equity)
    orders = _merge_quality_fields(orders, blotter)
    preflight = preflight_orders(orders, config.live)
    broker = broker_for(config)
    account = broker.account()
    out = Path(config.live.order_ticket_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    blotter.to_csv(out / "source_blotter.csv", index=False)
    orders.to_csv(out / "order_tickets.csv", index=False)
    preflight.to_csv(out / "preflight_orders.csv", index=False)
    approved_orders(preflight).to_csv(out / "approved_orders.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_run": str(run_dir),
                "broker": config.live.broker,
                "mode": config.live.mode,
                "account": asdict(account),
                "orders": int(len(orders)),
                "approved_orders": int(len(approved_orders(preflight))),
                "blocked_orders": int(len(preflight) - len(approved_orders(preflight))),
                "live_safety": {
                    "manual_approval_required": config.live.require_manual_approval,
                    "allow_market_orders": config.live.allow_market_orders,
                    "real_order_submission_enabled": _real_order_submission_enabled(config),
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _write_plan_html(out, preflight, account)
    return out


def submit_live_plan(config: AppConfig, *, plan_dir: str | Path, confirm_live: bool = False) -> Path:
    """Submit approved orders to the configured broker.

    Real broker submission is blocked unless the config, environment, and CLI
    confirmation all explicitly allow it.
    """

    plan = Path(plan_dir)
    preflight_path = plan / "preflight_orders.csv"
    if not preflight_path.exists():
        raise FileNotFoundError(f"Missing preflight file: {preflight_path}")
    preflight = pd.read_csv(preflight_path)
    orders = approved_orders(preflight)
    if orders.empty:
        raise RuntimeError("No approved orders to submit.")
    if config.live.broker != "paper" and not (confirm_live and _real_order_submission_enabled(config)):
        raise RuntimeError("Real broker submission blocked. Use paper broker first, or set live.enabled=true, LIVE_TRADING_ENABLED=true, and --confirm-live.")
    broker = broker_for(config)
    results = broker.submit_orders(orders)
    out = plan / "submitted_orders.csv"
    results.to_csv(out, index=False)
    return out


def broker_for(config: AppConfig) -> BrokerAdapter:
    """Return the configured broker adapter."""

    broker_name = str(config.live.broker).lower()
    if broker_name == "paper":
        return PaperBrokerAdapter(config.live)
    return UnconfiguredLiveBroker(broker_name)


def _resolve_run_dir(output_dir: str, run_id: str) -> Path:
    if run_id == "latest":
        return Path(output_dir) / "latest"
    path = Path(run_id)
    if path.exists():
        return path
    return Path(output_dir) / run_id


def _load_blotter(run_dir: Path) -> pd.DataFrame:
    for name in ("premarket_plan.csv", "latest_manual_trading_signals.csv"):
        path = run_dir / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No trading blotter found in {run_dir}. Run backtest/report first.")


def _load_latest_equity(run_dir: Path, *, fallback: float) -> float:
    path = run_dir / "equity_curve.csv"
    if not path.exists():
        return fallback
    data = pd.read_csv(path)
    if data.empty or "equity" not in data:
        return fallback
    value = pd.to_numeric(data["equity"], errors="coerce").dropna()
    return float(value.iloc[-1]) if not value.empty else fallback


def _merge_quality_fields(orders: pd.DataFrame, blotter: pd.DataFrame) -> pd.DataFrame:
    if orders.empty or blotter.empty or "symbol" not in blotter:
        return orders
    quality_cols = [
        "symbol",
        "blocked_from_trading",
        "block_reason",
        "status",
        "execution_condition",
        "risk_instruction",
    ]
    available = [col for col in quality_cols if col in blotter.columns]
    if len(available) <= 1:
        return orders
    quality = blotter[available].copy()
    quality["symbol"] = quality["symbol"].astype(str).str.upper()
    return orders.merge(quality.drop_duplicates("symbol"), on="symbol", how="left")


def _real_order_submission_enabled(config: AppConfig) -> bool:
    return bool(config.live.enabled and os.environ.get("LIVE_TRADING_ENABLED", "").lower() == "true")


def _write_plan_html(out: Path, preflight: pd.DataFrame, account) -> None:
    table = preflight.copy()
    if not table.empty:
        keep = [
            "risk_status",
            "risk_reasons",
            "symbol",
            "side",
            "quantity",
            "notional",
            "order_type",
            "limit_price",
            "priority_score",
            "target_weight",
            "delta_weight",
            "daily_order_budget_used_before",
            "daily_order_budget_limit",
            "final_score",
            "event_risk_score",
            "minute_pretrade_risk_score",
            "decision_note",
        ]
        table = table[[col for col in keep if col in table.columns]]
    html = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Live Trading Plan</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 28px; color: #172033; }}
    .notice {{ background: #fff7e6; border: 1px solid #ffd58a; padding: 12px 14px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e4e8f0; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>实盘前订单计划</h1>
  <p class="notice">默认安全模式：只有通过 preflight 的订单才会进入 approved_orders.csv；paper broker 不会发送真实订单。</p>
  <p>Account: {account.account_id} | Mode: {account.mode} | Equity: {account.equity:,.2f} | Buying Power: {account.buying_power:,.2f}</p>
  {table.to_html(index=False, escape=False) if not table.empty else '<p>No orders.</p>'}
</body>
</html>"""
    (out / "live_plan.html").write_text(html, encoding="utf-8")
