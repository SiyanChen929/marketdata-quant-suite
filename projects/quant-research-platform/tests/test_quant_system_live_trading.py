from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_system.config import AppConfig, LiveTradingConfig
from quant_system.live.orders import build_order_tickets
from quant_system.live.risk import approved_orders, preflight_orders
from quant_system.live.runner import create_live_plan, submit_live_plan


def test_build_order_tickets_uses_limit_prices_and_caps_notional() -> None:
    blotter = pd.DataFrame(
        [
            {
                "signal_date": "2026-05-12",
                "symbol": "AAA",
                "estimated_trade_notional": 50_000,
                "target_weight": 0.08,
                "current_weight": 0.02,
                "delta_weight": 0.06,
                "close": 100,
                "final_score": 80,
            }
        ]
    )

    orders = build_order_tickets(blotter, LiveTradingConfig(max_order_notional=10_000, limit_price_buffer_bps=25), equity=100_000)

    assert len(orders) == 1
    assert orders.iloc[0]["side"] == "buy"
    assert orders.iloc[0]["notional"] == 10_000
    assert orders.iloc[0]["limit_price"] == 100.25


def test_preflight_blocks_high_event_risk_new_buys() -> None:
    orders = pd.DataFrame(
        [
            {
                "client_order_id": "1",
                "symbol": "AAA",
                "side": "buy",
                "notional": 5_000,
                "target_weight": 0.05,
                "order_type": "limit",
                "event_risk_score": 90,
                "account_equity": 100_000,
            }
        ]
    )

    checked = preflight_orders(orders, LiveTradingConfig(block_if_event_risk_above=70))

    assert checked.iloc[0]["risk_status"] == "blocked"
    assert "event_risk_too_high" in checked.iloc[0]["risk_reasons"]
    assert approved_orders(checked).empty


def test_preflight_approves_highest_priority_until_daily_budget() -> None:
    orders = pd.DataFrame(
        [
            {"client_order_id": "low", "symbol": "LOW", "side": "buy", "notional": 7_000, "target_weight": 0.05, "order_type": "limit", "priority_score": 1, "account_equity": 100_000},
            {"client_order_id": "high", "symbol": "HIGH", "side": "buy", "notional": 7_000, "target_weight": 0.05, "order_type": "limit", "priority_score": 10, "account_equity": 100_000},
        ]
    )

    checked = preflight_orders(
        orders,
        LiveTradingConfig(max_total_order_notional=10_000, max_daily_turnover_pct_equity=1.0),
    )

    approved = approved_orders(checked)
    assert approved["symbol"].tolist() == ["HIGH"]
    blocked = checked[checked["symbol"].eq("LOW")].iloc[0]
    assert "daily_order_budget_exhausted" in blocked["risk_reasons"]


def test_live_plan_and_paper_submit(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "latest"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "signal_date": "2026-05-12",
                "symbol": "AAA",
                "action_label": "加仓",
                "estimated_trade_notional": 5_000,
                "target_weight": 0.05,
                "current_weight": 0.0,
                "delta_weight": 0.05,
                "close": 100,
                "final_score": 80,
                "event_risk_score": 0,
            }
        ]
    ).to_csv(run_dir / "premarket_plan.csv", index=False)
    pd.DataFrame([{"date": "2026-05-12", "equity": 100_000}]).to_csv(run_dir / "equity_curve.csv", index=False)
    config = AppConfig(
        output_dir=str(tmp_path / "runs"),
        live=LiveTradingConfig(order_ticket_dir=str(tmp_path / "live"), min_order_notional=250),
    )

    plan = create_live_plan(config)
    submitted = submit_live_plan(config, plan_dir=plan)

    assert (plan / "preflight_orders.csv").exists()
    assert (plan / "approved_orders.csv").exists()
    assert submitted.exists()
    statuses = pd.read_csv(submitted)
    assert statuses.iloc[0]["broker_status"] == "paper_submitted"


def test_real_submit_requires_explicit_enable(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    pd.DataFrame([{"client_order_id": "1", "symbol": "AAA", "risk_pass": True, "notional": 1_000}]).to_csv(plan / "preflight_orders.csv", index=False)
    config = AppConfig(live=LiveTradingConfig(broker="real_broker", enabled=False))

    with pytest.raises(RuntimeError, match="Real broker submission blocked"):
        submit_live_plan(config, plan_dir=plan, confirm_live=True)
