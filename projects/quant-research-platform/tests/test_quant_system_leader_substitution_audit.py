import pandas as pd

from quant_system.config import AppConfig, PortfolioConfig
from quant_system.optimization.leader_substitution_audit import (
    build_leader_substitution_audit,
    save_leader_substitution_audit,
    summarize_leader_substitution_audit,
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
                "final_score": 82,
                "relative_strength_score": 70,
                "theme_score": 60,
                "event_risk_score": 3,
                "overnight_gap_risk_score": 8,
                "primary_theme": "optical",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "signal": 1,
                "final_score": 78,
                "relative_strength_score": 88,
                "theme_score": 72,
                "event_risk_score": 5,
                "overnight_gap_risk_score": 10,
                "primary_theme": "optical",
            },
            {
                "date": "2024-01-02",
                "symbol": "CCC",
                "signal": 1,
                "final_score": 77,
                "relative_strength_score": 95,
                "theme_score": 80,
                "event_risk_score": 50,
                "overnight_gap_risk_score": 10,
                "primary_theme": "optical",
            },
            {
                "date": "2024-01-02",
                "symbol": "DDD",
                "signal": 1,
                "final_score": 76,
                "relative_strength_score": 90,
                "theme_score": 75,
                "event_risk_score": 2,
                "overnight_gap_risk_score": 8,
                "primary_theme": "software",
            },
        ]
    )


def _prices() -> pd.DataFrame:
    rows = []
    for symbol, values in {
        "AAA": [100, 101, 100],
        "BBB": [100, 103, 112],
        "CCC": [100, 105, 120],
        "DDD": [100, 101, 110],
    }.items():
        for date, price in zip(pd.date_range("2024-01-02", periods=3), values):
            rows.append({"date": date, "symbol": symbol, "adj_close": price, "close": price})
    return pd.DataFrame(rows)


def test_leader_substitution_audit_finds_same_theme_lift_and_respects_risk() -> None:
    audit = build_leader_substitution_audit(_signals(), _prices(), _config(), horizon=2)

    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["selected_symbol"] == "AAA"
    assert row["rejected_symbol"] == "BBB"
    assert row["theme"] == "optical"
    assert row["replacement_lift"] > 0.10
    assert bool(row["replacement_pass"]) is True


def test_leader_substitution_summary_and_report(tmp_path) -> None:
    audit = build_leader_substitution_audit(_signals(), _prices(), _config(), horizon=2)
    summary = summarize_leader_substitution_audit(audit, _config())

    full = summary[summary["period"].eq("full")].iloc[0]
    assert full["n_pairs"] == 1
    assert full["positive_lift_rate"] == 1.0

    out = save_leader_substitution_audit(audit, summary, tmp_path)
    assert (out / "leader_substitution_audit.csv").exists()
    assert (out / "leader_substitution_summary.csv").exists()
    assert (out / "leader_substitution_audit.html").exists()
