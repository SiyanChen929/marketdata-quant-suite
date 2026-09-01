import pandas as pd

from quant_system.config import AppConfig, PortfolioConfig
from quant_system.optimization.holding_timing_audit import (
    build_holding_timing_audit,
    save_holding_timing_audit,
    summarize_holding_timing_audit,
)


def _config() -> AppConfig:
    return AppConfig(
        portfolio=PortfolioConfig(
            rebalance="daily",
            construction="score_weighted",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            max_total_positions=1,
            min_target_weight=0.0,
        )
    )


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "signal": 1,
                "final_score": 90,
                "technical_score": 88,
                "relative_strength_score": 92,
                "theme_score": 75,
                "event_risk_score": 5,
                "overnight_gap_risk_score": 10,
                "primary_theme": "optical",
                "theme_active": True,
                "benchmark_risk_on": True,
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "signal": 1,
                "final_score": 70,
                "relative_strength_score": 65,
                "theme_score": 60,
                "primary_theme": "optical",
            },
            {
                "date": "2024-01-03",
                "symbol": "AAA",
                "signal": 1,
                "final_score": 69,
                "technical_score": 78,
                "relative_strength_score": 91,
                "theme_score": 73,
                "event_risk_score": 7,
                "overnight_gap_risk_score": 12,
                "primary_theme": "optical",
                "theme_active": True,
                "benchmark_risk_on": True,
            },
            {
                "date": "2024-01-03",
                "symbol": "BBB",
                "signal": 1,
                "final_score": 95,
                "relative_strength_score": 80,
                "theme_score": 66,
                "primary_theme": "optical",
            },
        ]
    )


def _prices() -> pd.DataFrame:
    rows = []
    values = {
        "AAA": [100, 100, 110, 120],
        "BBB": [100, 100, 101, 103],
    }
    for symbol, prices in values.items():
        for date, price in zip(pd.date_range("2024-01-02", periods=4), prices):
            rows.append({"date": date, "symbol": symbol, "adj_close": price, "close": price})
    return pd.DataFrame(rows)


def test_holding_timing_audit_detects_rank_exit_with_missed_upside() -> None:
    audit = build_holding_timing_audit(_signals(), _prices(), _config(), horizon=2, min_weight_drop=0.01)

    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["symbol"] == "AAA"
    assert row["exit_type"] == "full_exit"
    assert row["prev_weight"] == 1.0
    assert row["new_weight"] == 0.0
    assert row["forward_return"] > 0.19
    assert row["hypothetical_hold_lift"] > 0.19
    assert bool(row["early_exit_pass"]) is True


def test_holding_timing_summary_and_report(tmp_path) -> None:
    audit = build_holding_timing_audit(_signals(), _prices(), _config(), horizon=2, min_weight_drop=0.01)
    summary = summarize_holding_timing_audit(audit, _config())

    full = summary[summary["period"].eq("full")].iloc[0]
    assert full["n_events"] == 1
    assert full["positive_forward_rate"] == 1.0

    out = save_holding_timing_audit(audit, summary, tmp_path)
    assert (out / "holding_timing_audit.csv").exists()
    assert (out / "holding_timing_summary.csv").exists()
    assert (out / "holding_timing_audit.html").exists()
