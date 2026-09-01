from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from quant_marketdata import DataContractError, MarketDataStore

from quant_system.config import AppConfig, LiveTradingConfig
from quant_system.live.monitor import _configured_latest_prices, run_live_monitor


def test_live_monitor_writes_snapshot_and_alerts(tmp_path: Path) -> None:
    plan = tmp_path / "live" / "20260513_100000"
    plan.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "client_order_id": "1",
                "risk_status": "pass",
                "symbol": "AAA",
                "side": "buy",
                "quantity": 10,
                "notional": 1000,
                "order_type": "limit",
                "limit_price": 101,
                "reference_price": 100,
                "final_score": 80,
            },
            {
                "client_order_id": "2",
                "risk_status": "blocked",
                "risk_reasons": "event_risk_too_high",
                "symbol": "BBB",
                "side": "buy",
                "quantity": 5,
                "notional": 500,
                "order_type": "limit",
                "limit_price": 51,
                "reference_price": 50,
                "final_score": 70,
            },
        ]
    ).to_csv(plan / "preflight_orders.csv", index=False)
    pd.DataFrame([{"client_order_id": "1", "symbol": "AAA"}]).to_csv(plan / "approved_orders.csv", index=False)
    latest = tmp_path / "runs" / "latest"
    latest.mkdir(parents=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2026-05-13", "adj_close": 103},
            {"symbol": "BBB", "date": "2026-05-13", "adj_close": 50},
        ]
    ).to_csv(tmp_path / "prices.csv", index=False)
    pd.DataFrame([{"symbol": "AAA", "blocked_from_trading": False, "status": "ok"}]).to_csv(latest / "daily_data_quality_detail.csv", index=False)
    pd.DataFrame([{"risk_flags": "benchmark_beta_high:1.7", "suggested_risk_action": "wait"}]).to_csv(latest / "portfolio_risk_model_summary.csv", index=False)
    pd.DataFrame([{"scenario": "down", "estimated_pnl_pct_equity": -0.06}]).to_csv(latest / "portfolio_risk_stress_scenarios.csv", index=False)
    config = AppConfig(
        output_dir=str(tmp_path / "runs"),
        data_path=str(tmp_path / "prices.csv"),
        live=LiveTradingConfig(order_ticket_dir=str(tmp_path / "live")),
    )

    out = run_live_monitor(config, plan_dir=plan, iterations=1)

    assert (out / "live_monitor.html").exists()
    orders = pd.read_csv(out / "live_order_monitor.csv")
    alerts = pd.read_csv(out / "live_alerts.csv")
    assert "approved_pending_submit" in set(orders["monitor_status"])
    assert "approved_orders_not_submitted" in set(alerts["alert_type"])
    assert "stress_loss_gt_5pct" in set(alerts["alert_type"])


def test_live_monitor_resolves_latest_plan(tmp_path: Path) -> None:
    older = tmp_path / "live" / "20260512_100000"
    newer = tmp_path / "live" / "20260513_100000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    pd.DataFrame([{"client_order_id": "1", "risk_status": "blocked", "symbol": "AAA", "notional": 100, "reference_price": 10}]).to_csv(
        newer / "preflight_orders.csv", index=False
    )
    pd.DataFrame(columns=["client_order_id"]).to_csv(newer / "approved_orders.csv", index=False)
    latest = tmp_path / "runs" / "latest"
    latest.mkdir(parents=True)
    config = AppConfig(output_dir=str(tmp_path / "runs"), live=LiveTradingConfig(order_ticket_dir=str(tmp_path / "live")))

    out = run_live_monitor(config, plan_dir="latest", iterations=1)

    manifest = pd.read_json(out / "live_monitor.json", typ="series")
    assert str(newer) in manifest["plan_dir"]


def test_marketdata_live_prices_use_confirmed_store_not_legacy_csv(monkeypatch, tmp_path: Path) -> None:
    shared_data = tmp_path / "shared-marketdata"
    monkeypatch.setenv("QUANT_DATA_HOME", str(shared_data))
    MarketDataStore().write_bars(
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "symbol": ["AAA", "AAA"],
                "open": [10.0, 10.5],
                "high": [10.8, 11.2],
                "low": [9.8, 10.3],
                "close": [10.6, 11.0],
                "volume": [1000, 1200],
                "source": ["marketdata.app", "marketdata.app"],
                "finality": ["confirmed", "confirmed"],
            }
        ),
        finality="confirmed",
    )
    legacy_csv = tmp_path / "legacy.csv"
    pd.DataFrame(
        [{"date": "2024-01-04", "symbol": "AAA", "adj_close": 999.0}]
    ).to_csv(legacy_csv, index=False)
    config = AppConfig(
        data_provider="marketdata",
        data_path=str(legacy_csv),
        start_date="2024-01-01",
    )

    latest = _configured_latest_prices(config, ["AAA"])

    assert latest.iloc[0]["latest_close"] == 11.0
    assert latest.iloc[0]["latest_price_date"] == pd.Timestamp("2024-01-03")


@pytest.mark.parametrize(
    ("column", "value"),
    [("source", "legacy.csv"), ("finality", "provisional")],
)
def test_marketdata_live_prices_reject_wrong_provenance(monkeypatch, column, value) -> None:
    bars = pd.DataFrame(
        {
            "date": ["2024-01-03"],
            "symbol": ["AAA"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.8],
            "close": [10.6],
            "volume": [1000],
            "source": ["marketdata.app"],
            "finality": ["confirmed"],
        }
    )
    bars[column] = value

    class FakeStore:
        def read_bars(self, **kwargs):
            assert kwargs["symbols"] == ["AAA"]
            assert kwargs["finality"] == "confirmed"
            assert kwargs["resolution"] == "D"
            return bars

    monkeypatch.setattr("quant_system.live.monitor.MarketDataStore", FakeStore)

    with pytest.raises(DataContractError):
        _configured_latest_prices(
            AppConfig(data_provider="marketdata"),
            ["AAA"],
        )
