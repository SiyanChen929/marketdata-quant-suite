"""Live trading monitor for paper and broker-backed execution."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from quant_marketdata import MarketDataStore, normalize_bars

from quant_system.config import AppConfig
from quant_system.live.runner import broker_for


def run_live_monitor(
    config: AppConfig,
    *,
    plan_dir: str | Path = "latest",
    out_dir: str | Path | None = None,
    interval_seconds: int = 60,
    iterations: int = 1,
) -> Path:
    """Run the live monitor once or in a polling loop."""

    plan = resolve_plan_dir(config, plan_dir)
    output = Path(out_dir) if out_dir else plan / "monitor"
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    while True:
        write_live_monitor_snapshot(config, plan, output)
        count += 1
        if iterations > 0 and count >= iterations:
            break
        time.sleep(max(1, int(interval_seconds)))
    return output


def write_live_monitor_snapshot(config: AppConfig, plan_dir: Path, out_dir: Path) -> dict[str, object]:
    """Write one monitor snapshot and return its manifest."""

    snapshot_time = datetime.now().isoformat(timespec="seconds")
    preflight = _read_csv(plan_dir / "preflight_orders.csv")
    approved = _read_csv(plan_dir / "approved_orders.csv")
    submitted = _read_csv(plan_dir / "submitted_orders.csv")
    latest_run = Path(config.output_dir) / "latest"
    quality = _read_csv(latest_run / "daily_data_quality_detail.csv")
    risk_summary = _read_csv(latest_run / "portfolio_risk_model_summary.csv")
    stress = _read_csv(latest_run / "portfolio_risk_stress_scenarios.csv")
    latest_prices = _configured_latest_prices(config, _order_symbols(preflight))
    broker_state = _broker_snapshot(config)
    orders = _order_monitor_table(preflight, approved, submitted, latest_prices)
    alerts = _monitor_alerts(orders, quality, risk_summary, stress, broker_state)
    positions = broker_state.get("positions", pd.DataFrame())
    orders.to_csv(out_dir / "live_order_monitor.csv", index=False)
    alerts.to_csv(out_dir / "live_alerts.csv", index=False)
    if isinstance(positions, pd.DataFrame):
        positions.to_csv(out_dir / "broker_positions.csv", index=False)
    manifest = {
        "snapshot_time": snapshot_time,
        "plan_dir": str(plan_dir),
        "broker": config.live.broker,
        "account": broker_state.get("account", {}),
        "broker_error": broker_state.get("error", ""),
        "orders": int(len(orders)),
        "approved_orders": int(orders.get("approved", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not orders.empty else 0,
        "submitted_orders": int(orders.get("submitted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not orders.empty else 0,
        "alerts": int(len(alerts)),
        "latest_report": str(latest_run / "report.html"),
    }
    (out_dir / "live_monitor.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (out_dir / "live_monitor.html").write_text(_html(manifest, orders, alerts, risk_summary, stress, positions), encoding="utf-8")
    return manifest


def resolve_plan_dir(config: AppConfig, plan_dir: str | Path) -> Path:
    """Resolve a live plan path, accepting 'latest'."""

    if str(plan_dir) != "latest":
        return Path(plan_dir)
    root = Path(config.live.order_ticket_dir)
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No live plan directories found under {root}. Run live-plan first.")
    return sorted(candidates)[-1]


def _broker_snapshot(config: AppConfig) -> dict[str, object]:
    broker = broker_for(config)
    try:
        account = asdict(broker.account())
        positions = broker.positions()
        return {"account": account, "positions": positions, "error": ""}
    except Exception as exc:
        return {"account": {}, "positions": pd.DataFrame(), "error": str(exc)}


def _order_monitor_table(preflight: pd.DataFrame, approved: pd.DataFrame, submitted: pd.DataFrame, latest_prices: pd.DataFrame) -> pd.DataFrame:
    if preflight.empty:
        return pd.DataFrame()
    orders = preflight.copy()
    orders["client_order_id"] = orders["client_order_id"].astype(str)
    approved_ids = set(approved.get("client_order_id", pd.Series(dtype=str)).astype(str)) if not approved.empty else set()
    submitted_ids = set(submitted.get("client_order_id", pd.Series(dtype=str)).astype(str)) if not submitted.empty else set()
    orders["approved"] = orders["client_order_id"].isin(approved_ids)
    orders["submitted"] = orders["client_order_id"].isin(submitted_ids)
    if not submitted.empty:
        submit_cols = [col for col in ["client_order_id", "broker_order_id", "broker_status", "broker_message"] if col in submitted.columns]
        orders = orders.merge(submitted[submit_cols].drop_duplicates("client_order_id"), on="client_order_id", how="left")
    if not latest_prices.empty:
        orders = orders.merge(latest_prices, on="symbol", how="left")
        orders["price_drift_pct"] = pd.to_numeric(orders["latest_close"], errors="coerce") / pd.to_numeric(orders["reference_price"], errors="coerce") - 1.0
    orders["monitor_status"] = orders.apply(_order_status, axis=1)
    keep = [
        "monitor_status",
        "risk_status",
        "risk_reasons",
        "symbol",
        "side",
        "quantity",
        "notional",
        "order_type",
        "limit_price",
        "reference_price",
        "latest_close",
        "price_drift_pct",
        "approved",
        "submitted",
        "broker_status",
        "broker_message",
        "final_score",
        "event_risk_score",
        "minute_pretrade_risk_score",
        "decision_note",
    ]
    return orders[[col for col in keep if col in orders.columns]]


def _monitor_alerts(
    orders: pd.DataFrame,
    quality: pd.DataFrame,
    risk_summary: pd.DataFrame,
    stress: pd.DataFrame,
    broker_state: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if broker_state.get("error"):
        rows.append(_alert("critical", "broker_unavailable", str(broker_state["error"])))
    if not orders.empty:
        approved_unsubmitted = orders[orders["approved"].fillna(False).astype(bool) & ~orders["submitted"].fillna(False).astype(bool)]
        if not approved_unsubmitted.empty:
            rows.append(_alert("warning", "approved_orders_not_submitted", f"{len(approved_unsubmitted)} approved orders are not submitted."))
        drift = (
            pd.to_numeric(orders["price_drift_pct"], errors="coerce")
            if "price_drift_pct" in orders
            else pd.Series(index=orders.index, dtype="float64")
        )
        if drift.abs().gt(0.02).any():
            symbols = ", ".join(orders.loc[drift.abs().gt(0.02), "symbol"].astype(str).head(10))
            rows.append(_alert("warning", "price_drift_gt_2pct", f"Reference-to-latest price drift exceeds 2% for: {symbols}"))
    if not quality.empty and "blocked_from_trading" in quality:
        blocked = quality[quality["blocked_from_trading"].fillna(False).astype(bool)]
        if not blocked.empty:
            rows.append(_alert("critical", "data_quality_blocks", f"{len(blocked)} symbols blocked by data quality checks."))
    if not risk_summary.empty:
        flags = str(risk_summary.get("risk_flags", pd.Series([""])).fillna("").iloc[0])
        action = str(risk_summary.get("suggested_risk_action", pd.Series([""])).fillna("").iloc[0])
        if flags and flags != "within_guardrails":
            rows.append(_alert("warning", "portfolio_risk_flags", f"{flags}. {action}"))
    if not stress.empty and "estimated_pnl_pct_equity" in stress:
        worst = pd.to_numeric(stress["estimated_pnl_pct_equity"], errors="coerce").min()
        if pd.notna(worst) and worst < -0.05:
            rows.append(_alert("warning", "stress_loss_gt_5pct", f"Worst listed stress scenario is {worst:.2%}."))
    return pd.DataFrame(rows)


def _alert(severity: str, alert_type: str, message: str) -> dict[str, object]:
    return {"timestamp": datetime.now().isoformat(timespec="seconds"), "severity": severity, "alert_type": alert_type, "message": message}


def _order_status(row: pd.Series) -> str:
    if not bool(row.get("approved", False)):
        return "blocked_by_preflight"
    if bool(row.get("submitted", False)):
        status = str(row.get("broker_status", "submitted"))
        return status or "submitted"
    return "approved_pending_submit"


def _latest_prices(path: str | Path) -> pd.DataFrame:
    return _latest_prices_from_frame(_read_csv(Path(path)))


def _configured_latest_prices(config: AppConfig, symbols: list[str]) -> pd.DataFrame:
    """Load monitor prices from the configured source without formal fallbacks."""

    if not _uses_shared_marketdata(config):
        return _latest_prices(config.data_path)
    if not symbols:
        return pd.DataFrame()

    bars = MarketDataStore().read_bars(
        symbols=symbols,
        start=config.start_date,
        end=config.end_date,
        finality="confirmed",
        resolution="D",
    )
    bars = normalize_bars(
        bars,
        source="marketdata.app",
        finality="confirmed",
    )
    return _latest_prices_from_frame(bars)


def _latest_prices_from_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or not {"date", "symbol"}.issubset(data.columns):
        return pd.DataFrame()
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    price_col = "adj_close" if "adj_close" in data.columns else "close"
    latest = data.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
    return latest[["symbol", price_col, "date"]].rename(columns={price_col: "latest_close", "date": "latest_price_date"})


def _order_symbols(preflight: pd.DataFrame) -> list[str]:
    if preflight.empty or "symbol" not in preflight:
        return []
    return sorted(
        {
            symbol
            for symbol in preflight["symbol"].astype(str).str.strip().str.upper()
            if symbol and symbol not in {"NAN", "NONE"}
        }
    )


def _uses_shared_marketdata(config: AppConfig) -> bool:
    return str(config.data_provider).strip().lower() in {"marketdata", "marketdata.app"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _html(manifest: dict[str, object], orders: pd.DataFrame, alerts: pd.DataFrame, risk_summary: pd.DataFrame, stress: pd.DataFrame, positions: object) -> str:
    account = manifest.get("account", {})
    if not isinstance(account, dict):
        account = {}
    positions_df = positions if isinstance(positions, pd.DataFrame) else pd.DataFrame()
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="30">
  <title>Live Monitor</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 28px; color: #172033; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e4e8f0; border-radius: 8px; padding: 12px; background: #fbfcfe; }}
    .notice {{ background: #fff7e6; border: 1px solid #ffd58a; padding: 12px 14px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0 24px; }}
    th, td {{ border-bottom: 1px solid #e4e8f0; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>实时实盘监控</h1>
  <p class="notice">刷新时间：{manifest.get('snapshot_time')}。当前 broker={manifest.get('broker')}；真实券商未配置时只显示 paper/unconfigured 状态，不代表真实成交。</p>
  <div class="grid">
    <div class="card"><strong>订单</strong><br>{manifest.get('orders', 0)}</div>
    <div class="card"><strong>已批准</strong><br>{manifest.get('approved_orders', 0)}</div>
    <div class="card"><strong>已提交</strong><br>{manifest.get('submitted_orders', 0)}</div>
    <div class="card"><strong>警报</strong><br>{manifest.get('alerts', 0)}</div>
  </div>
  <h2>账户</h2>
  <p>Account: {account.get('account_id', '')} | Equity: {account.get('equity', '')} | Cash: {account.get('cash', '')} | Buying Power: {account.get('buying_power', '')}</p>
  <h2>警报</h2>{alerts.to_html(index=False, escape=False) if not alerts.empty else '<p>暂无警报。</p>'}
  <h2>订单监控</h2>{orders.to_html(index=False, escape=False) if not orders.empty else '<p>暂无订单。</p>'}
  <h2>组合风险</h2>{risk_summary.to_html(index=False, escape=False) if not risk_summary.empty else '<p>暂无组合风险摘要。</p>'}
  <h2>压力情景</h2>{stress.to_html(index=False, escape=False) if not stress.empty else '<p>暂无压力情景。</p>'}
  <h2>Broker 持仓</h2>{positions_df.to_html(index=False, escape=False) if not positions_df.empty else '<p>暂无 broker 持仓或 broker 未配置。</p>'}
</body>
</html>"""
