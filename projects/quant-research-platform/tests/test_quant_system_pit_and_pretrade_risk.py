from __future__ import annotations

import pandas as pd

from quant_system.config import FundamentalsConfig, RegimeConfig
from quant_system.data.intraday_provider import CSVIntradayDataProvider
from quant_system.events.earnings import align_earnings_to_dates
from quant_system.events.event_risk import score_event_risk
from quant_system.fundamentals.scoring import align_fundamentals_to_prices, score_fundamentals
from quant_system.portfolio.crowding import apply_crowding_risk_to_targets
from quant_system.portfolio.gap_risk import (
    apply_gap_beta_guard_to_targets,
    apply_overnight_gap_risk_to_targets,
    detect_overnight_gap_risk,
)
from quant_system.regime.intraday_risk import apply_minute_pretrade_budget_to_targets, detect_minute_pretrade_risk


def test_fundamentals_align_only_after_known_date() -> None:
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": [pd.Timestamp("2024-02-15")],
            "known_date": [pd.Timestamp("2024-02-15")],
            "sector": ["Tech"],
            "industry": ["Semis"],
            "revenue_yoy_growth": [0.25],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-14", "2024-02-15", "2024-02-16"]),
            "symbol": ["AAA", "AAA", "AAA"],
        }
    )

    aligned = align_fundamentals_to_prices(score_fundamentals(fundamentals, FundamentalsConfig()), prices)

    assert pd.Timestamp("2024-02-14") not in set(aligned["date"])
    assert aligned.loc[aligned["date"].eq(pd.Timestamp("2024-02-15")), "fundamental_known_date"].iloc[0] == pd.Timestamp("2024-02-15")
    assert aligned.loc[aligned["date"].eq(pd.Timestamp("2024-02-16")), "fundamental_data_age_days"].iloc[0] == 1


def test_event_alignment_respects_known_date_and_expected_move() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01", "2024-05-03", "2024-05-06"]),
            "symbol": ["AAA", "AAA", "AAA"],
        }
    )
    earnings = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "event_date": [pd.Timestamp("2024-05-06")],
            "known_date": [pd.Timestamp("2024-05-03")],
            "event_risk_score": [40.0],
            "expected_move": [0.20],
        }
    )

    aligned = score_event_risk(align_earnings_to_dates(prices, earnings))

    assert pd.isna(aligned.loc[aligned["date"].eq(pd.Timestamp("2024-05-01")), "days_to_earnings"].iloc[0])
    known = aligned[aligned["date"].eq(pd.Timestamp("2024-05-03"))].iloc[0]
    assert known["days_to_earnings"] == 3
    assert known["expected_move"] == 0.20
    assert known["event_risk_score"] > 40.0


def test_crowding_overlay_caps_theme_exposure(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {"date": "2024-01-02", "symbol": "AAA", "target_weight": 0.35, "sector": "Tech", "industry": "Semis", "reason_for_entry": ""},
            {"date": "2024-01-02", "symbol": "BBB", "target_weight": 0.35, "sector": "Tech", "industry": "Semis", "reason_for_entry": ""},
            {"date": "2024-01-02", "symbol": "CCC", "target_weight": 0.10, "sector": "Energy", "industry": "Gas", "reason_for_entry": ""},
        ]
    )

    out = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(crowding_overlay=True, max_theme_gross_exposure=0.40, max_sector_gross_exposure=1.0, max_industry_gross_exposure=1.0),
        themes,
    )

    semis_gross = out[out["primary_theme"].eq("semis")]["target_weight"].abs().sum()
    assert round(semis_gross, 6) == 0.40
    assert (out[out["primary_theme"].eq("semis")]["crowding_multiplier"] < 1.0).all()


def test_crowding_overlay_supports_per_theme_caps(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB]\n  optical: [CCC, DDD]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {"date": "2024-01-02", "symbol": "AAA", "target_weight": 0.35, "reason_for_entry": ""},
            {"date": "2024-01-02", "symbol": "BBB", "target_weight": 0.35, "reason_for_entry": ""},
            {"date": "2024-01-02", "symbol": "CCC", "target_weight": 0.20, "reason_for_entry": ""},
            {"date": "2024-01-02", "symbol": "DDD", "target_weight": 0.20, "reason_for_entry": ""},
        ]
    )

    out = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.50,
            theme_gross_exposure_caps={"semis": 0.30, "optical": 0.25},
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
        ),
        themes,
    )

    gross = out.groupby("primary_theme")["target_weight"].apply(lambda s: round(s.abs().sum(), 6)).to_dict()
    assert gross["semis"] == 0.30
    assert gross["optical"] == 0.25


def test_crowding_leader_relief_preserves_theme_cap_but_shifts_weight_to_top_rs_names(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB, CCC]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "target_weight": 0.30,
                "theme_score": 90.0,
                "relative_strength_score": 96.0,
                "benchmark_risk_on": True,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "target_weight": 0.25,
                "theme_score": 88.0,
                "relative_strength_score": 92.0,
                "benchmark_risk_on": True,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "CCC",
                "target_weight": 0.15,
                "theme_score": 70.0,
                "relative_strength_score": 55.0,
                "benchmark_risk_on": True,
                "reason_for_entry": "",
            },
        ]
    )

    baseline = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(crowding_overlay=True, max_theme_gross_exposure=0.40, max_sector_gross_exposure=1.0, max_industry_gross_exposure=1.0),
        themes,
    ).set_index("symbol")
    relieved = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.40,
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
            leader_crowding_relief_overlay=True,
            leader_crowding_relief_min_relative_strength_score=85.0,
            leader_crowding_relief_min_theme_score=80.0,
            leader_crowding_relief_weight=0.50,
            leader_crowding_relief_max_names_per_bucket=2,
            leader_crowding_relief_max_weight_per_symbol=0.02,
            leader_crowding_relief_max_bucket_weight_restore=0.03,
        ),
        themes,
    ).set_index("symbol")

    assert round(relieved["target_weight"].abs().sum(), 6) == 0.40
    assert relieved.loc["AAA", "target_weight"] > baseline.loc["AAA", "target_weight"]
    assert relieved.loc["BBB", "target_weight"] > baseline.loc["BBB", "target_weight"]
    assert relieved.loc["CCC", "target_weight"] < baseline.loc["CCC", "target_weight"]
    assert bool(relieved.loc["AAA", "crowding_relief_applied"])
    assert "\"role\": \"leader\"" in relieved.loc["AAA", "crowding_relief_log"]


def test_crowding_leader_relief_stays_off_when_benchmark_is_not_risk_on(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "target_weight": 0.35,
                "theme_score": 90.0,
                "relative_strength_score": 96.0,
                "benchmark_risk_on": False,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "target_weight": 0.25,
                "theme_score": 88.0,
                "relative_strength_score": 92.0,
                "benchmark_risk_on": False,
                "reason_for_entry": "",
            },
        ]
    )

    out = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.40,
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
            leader_crowding_relief_overlay=True,
        ),
        themes,
    )

    assert not out["crowding_relief_applied"].any()
    assert set(out["crowding_relief_circuit_breaker"]) == {"benchmark_not_risk_on"}


def test_crowding_leader_relief_can_use_low_risk_non_risk_on_exception(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB, CCC]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "target_weight": 0.28,
                "final_score": 78.0,
                "theme_score": 90.0,
                "relative_strength_score": 96.0,
                "benchmark_risk_on": False,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "target_weight": 0.22,
                "final_score": 75.0,
                "theme_score": 88.0,
                "relative_strength_score": 92.0,
                "benchmark_risk_on": False,
                "event_risk_score": 10.0,
                "overnight_gap_risk_score": 20.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "CCC",
                "target_weight": 0.10,
                "final_score": 60.0,
                "theme_score": 70.0,
                "relative_strength_score": 55.0,
                "benchmark_risk_on": False,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "reason_for_entry": "",
            },
        ]
    )

    baseline = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(crowding_overlay=True, max_theme_gross_exposure=0.40, max_sector_gross_exposure=1.0, max_industry_gross_exposure=1.0),
        themes,
    ).set_index("symbol")
    relieved = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.40,
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
            leader_crowding_relief_overlay=True,
            leader_crowding_relief_require_benchmark_risk_on=True,
            leader_crowding_relief_allow_non_risk_on_exception=True,
            leader_crowding_relief_min_relative_strength_score=85.0,
            leader_crowding_relief_min_theme_score=80.0,
            leader_crowding_relief_min_final_score_exception=72.0,
            leader_crowding_relief_max_event_risk_score_circuit_breaker=20.0,
            leader_crowding_relief_max_overnight_gap_risk_score_circuit_breaker=25.0,
            leader_crowding_relief_min_base_multiplier_circuit_breaker=0.66,
            leader_crowding_relief_weight=0.50,
            leader_crowding_relief_max_names_per_bucket=2,
            leader_crowding_relief_max_weight_per_symbol=0.02,
            leader_crowding_relief_max_bucket_weight_restore=0.03,
        ),
        themes,
    ).set_index("symbol")

    assert round(relieved["target_weight"].abs().sum(), 6) == 0.40
    assert relieved.loc["AAA", "target_weight"] > baseline.loc["AAA", "target_weight"]
    assert relieved.loc["BBB", "target_weight"] > baseline.loc["BBB", "target_weight"]
    assert relieved.loc["CCC", "target_weight"] < baseline.loc["CCC", "target_weight"]
    assert "\"activation_mode\": \"non_risk_on_exception\"" in relieved.loc["AAA", "crowding_relief_log"]


def test_crowding_leader_relief_non_risk_on_exception_respects_gap_circuit_breaker(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "target_weight": 0.35,
                "final_score": 80.0,
                "theme_score": 90.0,
                "relative_strength_score": 96.0,
                "benchmark_risk_on": False,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 45.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "target_weight": 0.25,
                "final_score": 78.0,
                "theme_score": 88.0,
                "relative_strength_score": 92.0,
                "benchmark_risk_on": False,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 40.0,
                "reason_for_entry": "",
            },
        ]
    )

    out = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.40,
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
            leader_crowding_relief_overlay=True,
            leader_crowding_relief_allow_non_risk_on_exception=True,
            leader_crowding_relief_min_final_score_exception=72.0,
            leader_crowding_relief_max_event_risk_score_circuit_breaker=20.0,
            leader_crowding_relief_max_overnight_gap_risk_score_circuit_breaker=25.0,
            leader_crowding_relief_min_base_multiplier_circuit_breaker=0.66,
        ),
        themes,
    )

    assert not out["crowding_relief_applied"].any()
    assert set(out["crowding_relief_circuit_breaker"]) == {"non_risk_on_exception_blocked"}


def test_crowding_leader_relief_same_theme_peer_exception_can_activate_when_benchmark_not_risk_on(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB, CCC]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "target_weight": 0.24,
                "final_score": 80.0,
                "theme_score": 90.0,
                "relative_strength_score": 96.0,
                "mom_return": 0.32,
                "benchmark_risk_on": False,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "target_weight": 0.22,
                "final_score": 78.0,
                "theme_score": 88.0,
                "relative_strength_score": 92.0,
                "mom_return": 0.24,
                "benchmark_risk_on": False,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "CCC",
                "target_weight": 0.14,
                "final_score": 66.0,
                "theme_score": 72.0,
                "relative_strength_score": 70.0,
                "mom_return": 0.08,
                "benchmark_risk_on": False,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
        ]
    )

    baseline = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(crowding_overlay=True, max_theme_gross_exposure=0.40, max_sector_gross_exposure=1.0, max_industry_gross_exposure=1.0),
        themes,
    ).set_index("symbol")
    relieved = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.40,
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
            leader_crowding_relief_overlay=True,
            leader_crowding_relief_same_theme_peer_exception_overlay=True,
            leader_crowding_relief_min_relative_strength_score=88.0,
            leader_crowding_relief_min_theme_score=80.0,
            leader_crowding_relief_same_theme_peer_min_count=3,
            leader_crowding_relief_same_theme_peer_min_active_share=0.66,
            leader_crowding_relief_same_theme_peer_min_avg_mom_return=0.18,
            leader_crowding_relief_min_base_multiplier_circuit_breaker=0.66,
            leader_crowding_relief_weight=0.50,
            leader_crowding_relief_max_names_per_bucket=2,
            leader_crowding_relief_max_weight_per_symbol=0.02,
            leader_crowding_relief_max_bucket_weight_restore=0.03,
        ),
        themes,
    ).set_index("symbol")

    assert round(relieved["target_weight"].abs().sum(), 6) == 0.40
    assert relieved.loc["AAA", "target_weight"] > baseline.loc["AAA", "target_weight"]
    assert relieved.loc["BBB", "target_weight"] > baseline.loc["BBB", "target_weight"]
    assert relieved.loc["CCC", "target_weight"] < baseline.loc["CCC", "target_weight"]
    assert "\"activation_mode\": \"same_theme_peer_exception\"" in relieved.loc["AAA", "crowding_relief_log"]


def test_crowding_leader_relief_same_theme_peer_restore_requires_broad_active_theme(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB, CCC]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "target_weight": 0.30,
                "final_score": 82.0,
                "theme_score": 90.0,
                "relative_strength_score": 96.0,
                "mom_return": 0.20,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "target_weight": 0.18,
                "final_score": 58.0,
                "theme_score": 55.0,
                "relative_strength_score": 60.0,
                "mom_return": 0.02,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "CCC",
                "target_weight": 0.12,
                "final_score": 54.0,
                "theme_score": 50.0,
                "relative_strength_score": 58.0,
                "mom_return": 0.01,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
        ]
    )

    out = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.40,
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
            leader_crowding_relief_overlay=True,
            leader_crowding_relief_same_theme_peer_restore_overlay=True,
            leader_crowding_relief_min_relative_strength_score=85.0,
            leader_crowding_relief_min_theme_score=80.0,
            leader_crowding_relief_same_theme_peer_restore_min_count=3,
            leader_crowding_relief_same_theme_peer_restore_min_active_share=0.50,
            leader_crowding_relief_same_theme_peer_restore_min_avg_mom_return=0.12,
        ),
        themes,
    )

    assert not out["crowding_relief_applied"].any()
    assert set(out["crowding_relief_circuit_breaker"]) == {"same_theme_peer_restore_blocked"}


def test_crowding_leader_relief_controlled_pullback_priority_prefers_reset_leader(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB, CCC]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "target_weight": 0.26,
                "final_score": 86.0,
                "theme_score": 92.0,
                "relative_strength_score": 98.0,
                "mom_return": 0.24,
                "distance_to_high_252": -0.02,
                "retest_distance_ma_pct": 0.11,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "target_weight": 0.22,
                "final_score": 76.0,
                "theme_score": 90.0,
                "relative_strength_score": 95.0,
                "mom_return": 0.05,
                "distance_to_high_252": -0.08,
                "retest_distance_ma_pct": 0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "CCC",
                "target_weight": 0.12,
                "final_score": 68.0,
                "theme_score": 82.0,
                "relative_strength_score": 89.0,
                "mom_return": 0.02,
                "distance_to_high_252": -0.09,
                "retest_distance_ma_pct": 0.03,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "reason_for_entry": "",
            },
        ]
    )

    baseline = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.40,
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
            leader_crowding_relief_overlay=True,
            leader_crowding_relief_min_relative_strength_score=88.0,
            leader_crowding_relief_min_theme_score=80.0,
            leader_crowding_relief_weight=0.50,
            leader_crowding_relief_max_names_per_bucket=1,
            leader_crowding_relief_max_weight_per_symbol=0.03,
            leader_crowding_relief_max_bucket_weight_restore=0.03,
        ),
        themes,
    ).set_index("symbol")
    pullback = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.40,
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
            leader_crowding_relief_overlay=True,
            leader_crowding_relief_min_relative_strength_score=88.0,
            leader_crowding_relief_min_theme_score=80.0,
            leader_crowding_relief_weight=0.50,
            leader_crowding_relief_max_names_per_bucket=1,
            leader_crowding_relief_max_weight_per_symbol=0.03,
            leader_crowding_relief_max_bucket_weight_restore=0.03,
            leader_crowding_relief_controlled_pullback_overlay=True,
            leader_crowding_relief_controlled_pullback_min_final_score=70.0,
            leader_crowding_relief_controlled_pullback_min_distance_from_high=0.04,
            leader_crowding_relief_controlled_pullback_max_distance_from_high=0.12,
            leader_crowding_relief_controlled_pullback_max_above_ma20_pct=0.06,
            leader_crowding_relief_controlled_pullback_max_mom_return=0.10,
        ),
        themes,
    ).set_index("symbol")

    assert round(pullback["target_weight"].abs().sum(), 6) == 0.40
    assert baseline.loc["AAA", "target_weight"] > pullback.loc["AAA", "target_weight"]
    assert pullback.loc["BBB", "target_weight"] > baseline.loc["BBB", "target_weight"]
    assert "\"controlled_pullback_eligible\": true" in pullback.loc["BBB", "crowding_relief_log"]


def test_crowding_leader_relief_same_theme_peer_restore_logs_qualified_restore(tmp_path) -> None:
    themes = tmp_path / "themes.yml"
    themes.write_text("themes:\n  semis: [AAA, BBB, CCC]\n", encoding="utf-8")
    targets = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "target_weight": 0.24,
                "final_score": 82.0,
                "theme_score": 90.0,
                "relative_strength_score": 96.0,
                "mom_return": 0.24,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "target_weight": 0.22,
                "final_score": 79.0,
                "theme_score": 88.0,
                "relative_strength_score": 92.0,
                "mom_return": 0.18,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
            {
                "date": "2024-01-02",
                "symbol": "CCC",
                "target_weight": 0.14,
                "final_score": 64.0,
                "theme_score": 72.0,
                "relative_strength_score": 70.0,
                "mom_return": 0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 15.0,
                "reason_for_entry": "",
            },
        ]
    )

    relieved = apply_crowding_risk_to_targets(
        targets,
        RegimeConfig(
            crowding_overlay=True,
            max_theme_gross_exposure=0.40,
            max_sector_gross_exposure=1.0,
            max_industry_gross_exposure=1.0,
            leader_crowding_relief_overlay=True,
            leader_crowding_relief_same_theme_peer_restore_overlay=True,
            leader_crowding_relief_min_relative_strength_score=88.0,
            leader_crowding_relief_min_theme_score=80.0,
            leader_crowding_relief_same_theme_peer_restore_min_count=3,
            leader_crowding_relief_same_theme_peer_restore_min_active_share=0.50,
            leader_crowding_relief_same_theme_peer_restore_min_avg_mom_return=0.12,
            leader_crowding_relief_weight=0.50,
            leader_crowding_relief_max_names_per_bucket=2,
            leader_crowding_relief_max_weight_per_symbol=0.02,
            leader_crowding_relief_max_bucket_weight_restore=0.03,
        ),
        themes,
    ).set_index("symbol")

    assert round(relieved["target_weight"].abs().sum(), 6) == 0.40
    assert relieved.loc["AAA", "target_weight"] > 0.16
    assert relieved.loc["BBB", "target_weight"] > 0.1466666667
    assert relieved.loc["CCC", "target_weight"] < 0.0933333334
    assert "\"same_theme_peer_restore_qualified\": true" in relieved.loc["AAA", "crowding_relief_log"]


def test_overnight_gap_risk_uses_historical_gaps_before_target_execution() -> None:
    dates = pd.bdate_range("2024-01-01", periods=35)
    rows = []
    close = 100.0
    for idx, date in enumerate(dates):
        gap = 0.12 if idx in {20, 25, 30} else 0.0
        open_price = close * (1 + gap)
        close = open_price
        rows.append({"date": date, "symbol": "AAA", "open": open_price, "high": open_price, "low": open_price, "close": close, "adj_close": close})
    prices = pd.DataFrame(rows)
    config = RegimeConfig(overnight_gap_risk_overlay=True, overnight_gap_reduce_score=30, overnight_gap_reduce_multiplier=0.5)

    risk = detect_overnight_gap_risk(prices, config)
    assert risk.iloc[-1]["overnight_gap_risk_score"] >= 30
    targets = pd.DataFrame({"date": [dates[-1]], "symbol": ["AAA"], "target_weight": [0.20]})
    out = apply_overnight_gap_risk_to_targets(targets, prices, config)
    assert out["target_weight"].iloc[0] == 0.10


def test_gap_beta_guard_reduces_long_targets_with_position_circuit_breaker() -> None:
    dates = pd.bdate_range("2024-01-01", periods=90)
    rows = []
    spy_close = 100.0
    aaa_close = 100.0
    for idx, date in enumerate(dates):
        spy_open = spy_close * (1.0 + (0.003 if idx % 7 == 0 else 0.0))
        spy_close = spy_open * (1.0 + (0.002 if idx % 2 == 0 else -0.001))
        rows.append(
            {
                "date": date,
                "symbol": "SPY",
                "open": spy_open,
                "high": max(spy_open, spy_close),
                "low": min(spy_open, spy_close),
                "close": spy_close,
                "adj_close": spy_close,
            }
        )
        gap = 0.06 if idx in {45, 52, 61, 75} else 0.0
        aaa_open = aaa_close * (1.0 + gap)
        aaa_close = aaa_open * (1.0 + (0.015 if idx % 2 == 0 else -0.010))
        rows.append(
            {
                "date": date,
                "symbol": "AAA",
                "open": aaa_open,
                "high": max(aaa_open, aaa_close),
                "low": min(aaa_open, aaa_close),
                "close": aaa_close,
                "adj_close": aaa_close,
            }
        )
    prices = pd.DataFrame(rows)
    config = RegimeConfig(
        overnight_gap_risk_overlay=True,
        overnight_gap_reduce_score=200,
        overnight_gap_cash_score=300,
        gap_beta_guard_overlay=True,
        gap_beta_min_gap_score=10,
        gap_beta_min_gap_vol_20d=0.001,
        gap_beta_min_realized_vol_63d=0.01,
        gap_beta_min_beta_63d=0.0,
        gap_beta_reduce_multiplier=0.20,
        gap_beta_min_multiplier=0.40,
        gap_beta_max_position_cut=0.40,
    )
    targets = pd.DataFrame(
        {
            "date": [dates[-1], dates[-1]],
            "symbol": ["AAA", "AAA"],
            "target_weight": [0.20, -0.20],
            "reason_for_entry": ["long", "short"],
        }
    )

    out = apply_overnight_gap_risk_to_targets(targets, prices, config)

    long_weight = out.loc[out["target_weight"].gt(0), "target_weight"].iloc[0]
    short_weight = out.loc[out["target_weight"].lt(0), "target_weight"].iloc[0]
    assert round(long_weight, 6) == 0.12
    assert round(short_weight, 6) == -0.20
    assert out.loc[out["target_weight"].gt(0), "gap_beta_guard_multiplier"].iloc[0] == 0.60
    assert "gap_beta_guard" in out.loc[out["target_weight"].gt(0), "reason_for_entry"].iloc[0]


def test_gap_beta_guard_can_require_benchmark_risk_off() -> None:
    config = RegimeConfig(
        gap_beta_guard_overlay=True,
        gap_beta_guard_require_benchmark_risk_off=True,
        gap_beta_min_gap_score=80,
        gap_beta_min_realized_vol_63d=0.50,
        gap_beta_min_beta_63d=1.50,
        gap_beta_reduce_multiplier=0.50,
        gap_beta_max_position_cut=0.60,
    )
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["RISKON", "RISKOFF"],
            "target_weight": [0.20, 0.20],
            "overnight_gap_risk_score": [95, 95],
            "overnight_gap_vol_20d": [0.06, 0.06],
            "realized_vol_63d": [0.80, 0.80],
            "beta_to_benchmark_63d": [2.0, 2.0],
            "benchmark_risk_on": [True, False],
            "reason_for_entry": ["risk-on name", "risk-off name"],
        }
    )

    out = apply_gap_beta_guard_to_targets(targets, config)

    weights = dict(zip(out["symbol"], out["target_weight"], strict=False))
    assert weights["RISKON"] == 0.20
    assert weights["RISKOFF"] == 0.10
    reason = out.loc[out["symbol"].eq("RISKOFF"), "reason_for_entry"].iloc[0]
    assert "benchmark risk-off" in reason


def test_minute_pretrade_budget_reduces_next_session_targets() -> None:
    date = pd.Timestamp("2024-01-02")
    rows = []
    for symbol in ["SPY", "AAA", "BBB", "CCC"]:
        rows.extend(
            [
                {"date": date, "datetime": date + pd.Timedelta(hours=14), "symbol": symbol, "open": 100, "high": 100, "low": 95, "close": 96, "volume": 1000},
                {"date": date, "datetime": date + pd.Timedelta(hours=15), "symbol": symbol, "open": 96, "high": 97, "low": 94, "close": 95, "volume": 2000},
            ]
        )
    intraday = pd.DataFrame(rows)
    config = RegimeConfig(
        minute_pretrade_budget_overlay=True,
        minute_pretrade_reduce_score=30,
        minute_pretrade_cash_score=200,
        minute_pretrade_reduce_multiplier=0.5,
        intraday_benchmark_drop_reduce=-0.03,
        intraday_breadth_down_threshold=0.50,
    )

    risk = detect_minute_pretrade_risk(intraday, "SPY", config)
    assert risk.iloc[0]["minute_pretrade_risk_score"] >= 30
    targets = pd.DataFrame({"date": [date], "symbol": ["AAA"], "target_weight": [0.20], "reason_for_entry": ["test"]})
    out = apply_minute_pretrade_budget_to_targets(targets, intraday, "SPY", config)
    assert out["target_weight"].iloc[0] == 0.10
    assert "Minute-level pretrade budget" in out["reason_for_entry"].iloc[0]


def test_csv_intraday_provider_keeps_mixed_date_formats(tmp_path) -> None:
    path = tmp_path / "intraday.csv"
    pd.DataFrame(
        [
            {"symbol": "SPY", "datetime": "2026-05-08 13:30:00", "date": "2026-05-08", "open": 100, "high": 101, "low": 100, "close": 101, "volume": 1000},
            {"symbol": "SPY", "datetime": "2026-05-11 13:30:00", "date": "2026-05-11 00:00:00", "open": 101, "high": 102, "low": 101, "close": 102, "volume": 1000},
            {"symbol": "AAA", "datetime": "2026-05-08 13:30:00", "date": "2026-05-08", "open": 50, "high": 51, "low": 50, "close": 51, "volume": 1000},
            {"symbol": "AAA", "datetime": "2026-05-11 13:30:00", "date": "2026-05-11 00:00:00", "open": 51, "high": 52, "low": 51, "close": 52, "volume": 1000},
        ]
    ).to_csv(path, index=False)

    out = CSVIntradayDataProvider(path).get_bulk_intraday(["SPY", "AAA"], "2026-05-01", None)

    assert len(out) == 4
    assert out["date"].nunique() == 2
    assert set(out["date"].dt.strftime("%Y-%m-%d")) == {"2026-05-08", "2026-05-11"}
