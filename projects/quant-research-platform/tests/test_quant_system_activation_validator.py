import pandas as pd

from quant_system.config import AppConfig, PortfolioConfig
from quant_system.optimization.activation_validator import (
    save_activation_validation_report,
    selected_row_activation_summary,
)


def _config() -> AppConfig:
    return AppConfig(
        portfolio=PortfolioConfig(
            rebalance="daily",
            construction="score_weighted",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=0.60,
            max_total_positions=2,
            min_target_weight=0.0,
        )
    )


def test_selected_row_activation_detects_changed_names_and_weights() -> None:
    config = _config()
    baseline = pd.DataFrame(
        [
            {"date": "2024-01-03", "symbol": "AAA", "signal": 1, "final_score": 90},
            {"date": "2024-01-03", "symbol": "BBB", "signal": 1, "final_score": 80},
            {"date": "2024-01-03", "symbol": "CCC", "signal": 0, "final_score": 70},
            {"date": "2024-01-04", "symbol": "AAA", "signal": 1, "final_score": 88},
            {"date": "2024-01-04", "symbol": "BBB", "signal": 1, "final_score": 77},
            {"date": "2024-01-04", "symbol": "CCC", "signal": 0, "final_score": 75},
        ]
    )
    candidate = baseline.copy()
    candidate.loc[candidate["symbol"].eq("CCC"), "signal"] = 1
    candidate.loc[candidate["symbol"].eq("CCC"), "final_score"] = 95
    candidate.loc[candidate["symbol"].eq("BBB"), "signal"] = 0

    summary = selected_row_activation_summary(baseline, candidate, config, label="mutation")
    validation = summary[summary["period"].eq("validation")].iloc[0]

    assert bool(validation["activation_pass"]) is True
    assert validation["selected_changed_dates"] == 2
    assert validation["target_changed_dates"] == 2
    assert validation["max_abs_target_weight_diff"] > 0
    assert validation["score_changed_rows"] > 0


def test_selected_row_activation_noops_are_flagged_false() -> None:
    config = _config()
    signals = pd.DataFrame(
        [
            {"date": "2024-01-03", "symbol": "AAA", "signal": 1, "final_score": 90},
            {"date": "2024-01-03", "symbol": "BBB", "signal": 1, "final_score": 80},
        ]
    )

    summary = selected_row_activation_summary(signals, signals.copy(), config, label="noop")
    validation = summary[summary["period"].eq("validation")].iloc[0]

    assert bool(validation["activation_pass"]) is False
    assert validation["selected_changed_dates"] == 0
    assert validation["target_changed_dates"] == 0


def test_selected_row_activation_can_limit_date_window() -> None:
    config = _config()
    baseline = pd.DataFrame(
        [
            {"date": "2024-01-03", "symbol": "AAA", "signal": 1, "final_score": 90},
            {"date": "2024-01-04", "symbol": "AAA", "signal": 1, "final_score": 90},
            {"date": "2024-01-05", "symbol": "AAA", "signal": 1, "final_score": 90},
        ]
    )
    candidate = baseline.copy()
    candidate.loc[candidate["date"].eq("2024-01-04"), "final_score"] = 95

    summary = selected_row_activation_summary(
        baseline,
        candidate,
        config,
        start_date="2024-01-04",
        end_date="2024-01-05",
        max_dates=1,
    )
    validation = summary[summary["period"].eq("validation")].iloc[0]

    assert validation["n_dates"] == 1
    assert validation["score_changed_rows"] == 1


def test_save_activation_validation_report_writes_csv_and_html(tmp_path) -> None:
    summary = pd.DataFrame([{"label": "x", "period": "full", "activation_pass": True}])

    out = save_activation_validation_report(summary, tmp_path)

    assert (out / "activation_validation.csv").exists()
    assert (out / "activation_validation.html").exists()
