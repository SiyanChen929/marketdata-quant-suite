from __future__ import annotations

import pandas as pd

from quant_system.config import PortfolioConfig
from quant_system.portfolio.construction import construct_target_weights


def test_score_volatility_construction_downweights_high_vol_names() -> None:
    signals = pd.DataFrame(
        [
            {"date": "2024-01-02", "symbol": "LOWVOL", "signal": 1.0, "final_score": 90.0, "vol_20d": 0.02},
            {"date": "2024-01-02", "symbol": "HIGHVOL", "signal": 1.0, "final_score": 91.0, "vol_20d": 0.12},
        ]
    )

    targets = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="score_volatility",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
        ),
    ).set_index("symbol")

    assert targets.loc["LOWVOL", "target_weight"] > targets.loc["HIGHVOL", "target_weight"]
    assert round(targets["target_weight"].sum(), 10) == 1.0


def test_equal_weight_construction_ignores_score_edge() -> None:
    signals = pd.DataFrame(
        [
            {"date": "2024-01-02", "symbol": "AAA", "signal": 1.0, "final_score": 70.0},
            {"date": "2024-01-02", "symbol": "BBB", "signal": 1.0, "final_score": 95.0},
        ]
    )

    targets = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="equal_weight",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
        ),
    )

    assert set(round(weight, 6) for weight in targets["target_weight"]) == {0.5}


def test_leader_addon_tilts_existing_longs_with_caps() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "LEADER",
                "signal": 1.0,
                "final_score": 92.0,
                "relative_strength_score": 90.0,
                "theme_score": 80.0,
                "mom_return": 0.30,
                "benchmark_risk_on": True,
                "vol_20d": 0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "SOLID",
                "signal": 1.0,
                "final_score": 91.0,
                "relative_strength_score": 75.0,
                "theme_score": 70.0,
                "mom_return": 0.25,
                "benchmark_risk_on": True,
                "vol_20d": 0.04,
            },
        ]
    )
    base = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="score_volatility",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
        ),
    ).set_index("symbol")
    tilted = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="score_volatility",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=0.70,
            leader_addon_overlay=True,
            leader_addon_multiplier=1.50,
            leader_addon_min_final_score=90.0,
            leader_addon_min_relative_strength_score=85.0,
            leader_addon_min_theme_score=75.0,
            leader_addon_min_mom_return=0.20,
            leader_addon_max_names_per_date=1,
        ),
    ).set_index("symbol")

    assert tilted.loc["LEADER", "target_weight"] > base.loc["LEADER", "target_weight"]
    assert tilted.loc["LEADER", "target_weight"] <= 0.70
    assert bool(tilted.loc["LEADER", "leader_addon_eligible"]) is True
    assert bool(tilted.loc["SOLID", "leader_addon_eligible"]) is False


def test_conviction_sizing_overlay_makes_long_weights_more_convex_without_changing_gross() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": f"AAA{i}",
                "signal": 1.0,
                "final_score": 60.0 + i * 5.0,
                "relative_strength_score": 60.0 + i * 4.0,
                "theme_score": 55.0 + i,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
            }
            for i in range(8)
        ]
    )
    base_config = PortfolioConfig(
        construction="score_weighted",
        target_gross_exposure=1.0,
        target_net_exposure=1.0,
        max_position_weight=1.0,
    )
    convex_config = PortfolioConfig(
        construction="score_weighted",
        target_gross_exposure=1.0,
        target_net_exposure=1.0,
        max_position_weight=1.0,
        conviction_sizing_overlay=True,
        conviction_sizing_power=1.4,
        conviction_sizing_max_multiplier=1.5,
        conviction_sizing_min_names=8,
        conviction_sizing_min_final_score=60.0,
        conviction_sizing_min_relative_strength_score=55.0,
        conviction_sizing_min_theme_score=50.0,
    )

    base = construct_target_weights(signals, base_config).set_index("symbol")
    convex = construct_target_weights(signals, convex_config).set_index("symbol")

    assert convex.loc["AAA7", "target_weight"] > base.loc["AAA7", "target_weight"]
    assert convex.loc["AAA0", "target_weight"] < base.loc["AAA0", "target_weight"]
    assert round(convex["target_weight"].abs().sum(), 10) == 1.0
    assert convex.loc["AAA7", "conviction_sizing_multiplier"] > convex.loc["AAA0", "conviction_sizing_multiplier"]
    assert bool(convex.loc["AAA7", "conviction_sizing_eligible"]) is True


def test_conviction_sizing_overlay_respects_gap_risk_circuit_breaker() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": f"AAA{i}",
                "signal": 1.0,
                "final_score": 70.0 + i,
                "relative_strength_score": 70.0 + i,
                "theme_score": 60.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 50.0 if i == 7 else 10.0,
            }
            for i in range(8)
        ]
    )

    out = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="score_weighted",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            conviction_sizing_overlay=True,
            conviction_sizing_max_overnight_gap_risk_score_circuit_breaker=35.0,
            conviction_sizing_min_names=8,
        ),
    ).set_index("symbol")

    assert out.loc["AAA7", "conviction_sizing_multiplier"] == 1.0
    assert bool(out.loc["AAA7", "conviction_sizing_eligible"]) is False


def test_conviction_sizing_theme_peer_support_blocks_isolated_theme_leader() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "SEMIA",
                "signal": 1.0,
                "final_score": 96.0,
                "relative_strength_score": 92.0,
                "theme_score": 78.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.22,
                "theme_active": True,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-02",
                "symbol": "SEMIB",
                "signal": 1.0,
                "final_score": 92.0,
                "relative_strength_score": 88.0,
                "theme_score": 72.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.18,
                "theme_active": True,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-02",
                "symbol": "SEMIC",
                "signal": 1.0,
                "final_score": 90.0,
                "relative_strength_score": 86.0,
                "theme_score": 70.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.16,
                "theme_active": True,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-02",
                "symbol": "ISO1",
                "signal": 1.0,
                "final_score": 95.0,
                "relative_strength_score": 94.0,
                "theme_score": 84.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.24,
                "theme_active": True,
                "primary_theme": "robotics",
            },
            {
                "date": "2024-01-02",
                "symbol": "ISO2",
                "signal": 1.0,
                "final_score": 89.0,
                "relative_strength_score": 78.0,
                "theme_score": 68.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.08,
                "theme_active": True,
                "primary_theme": "robotics",
            },
            {
                "date": "2024-01-02",
                "symbol": "OTHER1",
                "signal": 1.0,
                "final_score": 80.0,
                "relative_strength_score": 70.0,
                "theme_score": 58.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.06,
                "theme_active": True,
                "primary_theme": "power",
            },
            {
                "date": "2024-01-02",
                "symbol": "OTHER2",
                "signal": 1.0,
                "final_score": 79.0,
                "relative_strength_score": 69.0,
                "theme_score": 57.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.05,
                "theme_active": True,
                "primary_theme": "power",
            },
            {
                "date": "2024-01-02",
                "symbol": "OTHER3",
                "signal": 1.0,
                "final_score": 78.0,
                "relative_strength_score": 68.0,
                "theme_score": 56.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.04,
                "theme_active": True,
                "primary_theme": "power",
            },
            {
                "date": "2024-01-02",
                "symbol": "OTHER4",
                "signal": 1.0,
                "final_score": 77.0,
                "relative_strength_score": 67.0,
                "theme_score": 55.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.03,
                "theme_active": True,
                "primary_theme": "power",
            },
            {
                "date": "2024-01-02",
                "symbol": "OTHER5",
                "signal": 1.0,
                "final_score": 76.0,
                "relative_strength_score": 66.0,
                "theme_score": 54.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.02,
                "theme_active": True,
                "primary_theme": "power",
            },
        ]
    )

    out = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="score_weighted",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            conviction_sizing_overlay=True,
            conviction_sizing_power=1.3,
            conviction_sizing_max_multiplier=1.35,
            conviction_sizing_min_names=10,
            conviction_sizing_min_final_score=60.0,
            conviction_sizing_min_relative_strength_score=55.0,
            conviction_sizing_min_theme_score=50.0,
            conviction_sizing_require_same_theme_peer_support=True,
            conviction_sizing_same_theme_peer_min_count=3,
            conviction_sizing_same_theme_peer_min_share=0.45,
            conviction_sizing_same_theme_peer_min_avg_relative_strength_score=80.0,
            conviction_sizing_same_theme_peer_min_avg_mom_return=0.10,
        ),
    ).set_index("symbol")

    assert bool(out.loc["SEMIA", "conviction_sizing_eligible"]) is True
    assert bool(out.loc["SEMIA", "conviction_sizing_same_theme_peer_support"]) is True
    assert out.loc["SEMIA", "conviction_sizing_theme_peer_count"] == 3
    assert out.loc["SEMIA", "conviction_sizing_circuit_breaker"] == "applied"
    assert out.loc["SEMIA", "conviction_sizing_multiplier"] > 1.0
    assert bool(out.loc["ISO1", "conviction_sizing_eligible"]) is False
    assert bool(out.loc["ISO1", "conviction_sizing_same_theme_peer_support"]) is False
    assert out.loc["ISO1", "conviction_sizing_circuit_breaker"] == "same_theme_peer_count_low"
    assert out.loc["ISO1", "conviction_sizing_multiplier"] == 1.0
    assert '"require_same_theme_peer_support": true' in out.loc["ISO1", "conviction_sizing_log"]


def test_conviction_sizing_allowed_themes_blocks_unattributed_theme() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": symbol,
                "signal": 1.0,
                "final_score": final_score,
                "relative_strength_score": 88.0,
                "theme_score": 70.0,
                "benchmark_risk_on": True,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 10.0,
                "mom_return": 0.16,
                "theme_active": True,
                "primary_theme": theme,
            }
            for symbol, final_score, theme in [
                ("SEMIA", 96.0, "semiconductors"),
                ("SEMIB", 94.0, "semiconductors"),
                ("SEMIC", 92.0, "semiconductors"),
                ("SOFTA", 95.0, "ai_software_data"),
                ("SOFTB", 93.0, "ai_software_data"),
                ("SOFTC", 91.0, "ai_software_data"),
                ("RETA", 98.0, "retail"),
                ("RETB", 90.0, "retail"),
                ("RETC", 89.0, "retail"),
                ("RETD", 88.0, "retail"),
            ]
        ]
    )

    out = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="score_weighted",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            conviction_sizing_overlay=True,
            conviction_sizing_allowed_themes=("semiconductors", "ai_software_data"),
            conviction_sizing_require_same_theme_peer_support=True,
            conviction_sizing_same_theme_peer_min_count=3,
            conviction_sizing_same_theme_peer_min_share=0.30,
            conviction_sizing_same_theme_peer_min_avg_relative_strength_score=80.0,
            conviction_sizing_same_theme_peer_min_avg_mom_return=0.10,
            conviction_sizing_min_names=10,
        ),
    ).set_index("symbol")

    assert bool(out.loc["SEMIA", "conviction_sizing_eligible"]) is True
    assert out.loc["SEMIA", "conviction_sizing_multiplier"] > 1.0
    assert bool(out.loc["RETA", "conviction_sizing_eligible"]) is False
    assert out.loc["RETA", "conviction_sizing_multiplier"] == 1.0
    assert out.loc["RETA", "conviction_sizing_circuit_breaker"] == "theme_not_allowed"
    assert '"allowed_themes": ["ai_software_data", "semiconductors"]' in out.loc["RETA", "conviction_sizing_log"]


def test_leader_addon_controlled_pullback_requires_reset_and_logs_breaker() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "RESET",
                "signal": 1.0,
                "final_score": 95.0,
                "relative_strength_score": 93.0,
                "theme_score": 72.0,
                "mom_return": 0.24,
                "benchmark_risk_on": True,
                "distance_to_high_252": -0.06,
                "retest_distance_ma_pct": 0.03,
                "event_risk_score": 0.0,
                "overnight_gap_risk_score": 0.0,
                "vol_20d": 0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "EXTENDED",
                "signal": 1.0,
                "final_score": 94.0,
                "relative_strength_score": 92.0,
                "theme_score": 70.0,
                "mom_return": 0.23,
                "benchmark_risk_on": True,
                "distance_to_high_252": -0.01,
                "retest_distance_ma_pct": 0.12,
                "event_risk_score": 0.0,
                "overnight_gap_risk_score": 0.0,
                "vol_20d": 0.04,
            },
        ]
    )
    out = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="score_volatility",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            leader_addon_overlay=True,
            leader_addon_multiplier=1.15,
            leader_addon_min_final_score=74.0,
            leader_addon_min_relative_strength_score=90.0,
            leader_addon_min_theme_score=60.0,
            leader_addon_min_mom_return=0.20,
            leader_addon_max_names_per_date=2,
            leader_addon_require_controlled_pullback=True,
            leader_addon_min_drawdown_from_high=0.02,
            leader_addon_max_drawdown_from_high=0.15,
            leader_addon_max_above_ma20_pct_circuit_breaker=0.08,
            leader_addon_max_event_risk_score_circuit_breaker=20.0,
            leader_addon_max_overnight_gap_risk_score_circuit_breaker=30.0,
        ),
    ).set_index("symbol")

    assert bool(out.loc["RESET", "leader_addon_eligible"]) is True
    assert out.loc["RESET", "leader_addon_circuit_breaker"] == "applied"
    assert "\"controlled_pullback_required\": true" in out.loc["RESET", "leader_addon_log"]
    assert bool(out.loc["EXTENDED", "leader_addon_eligible"]) is False
    assert out.loc["EXTENDED", "leader_addon_circuit_breaker"] == "above_ma20_too_extended"


def test_leader_addon_same_theme_peer_support_blocks_isolated_theme_leaders() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "LEADER_A",
                "signal": 1.0,
                "final_score": 96.0,
                "relative_strength_score": 94.0,
                "theme_score": 82.0,
                "mom_return": 0.28,
                "benchmark_risk_on": True,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "vol_20d": 0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "LEADER_B",
                "signal": 1.0,
                "final_score": 90.0,
                "relative_strength_score": 88.0,
                "theme_score": 76.0,
                "mom_return": 0.22,
                "benchmark_risk_on": True,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "vol_20d": 0.05,
            },
            {
                "date": "2024-01-02",
                "symbol": "ISOLATED",
                "signal": 1.0,
                "final_score": 95.0,
                "relative_strength_score": 93.0,
                "theme_score": 84.0,
                "mom_return": 0.26,
                "benchmark_risk_on": True,
                "theme_active": True,
                "primary_theme": "robotics",
                "vol_20d": 0.04,
            },
        ]
    )
    out = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="score_volatility",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            leader_addon_overlay=True,
            leader_addon_multiplier=1.20,
            leader_addon_min_final_score=72.0,
            leader_addon_min_relative_strength_score=84.0,
            leader_addon_min_theme_score=64.0,
            leader_addon_min_mom_return=0.12,
            leader_addon_max_names_per_date=3,
            leader_addon_require_same_theme_peer_support=True,
            leader_addon_same_theme_peer_min_count=2,
            leader_addon_same_theme_peer_min_share=0.25,
        ),
    ).set_index("symbol")

    assert bool(out.loc["LEADER_A", "leader_addon_eligible"]) is True
    assert bool(out.loc["LEADER_A", "leader_addon_same_theme_peer_support"]) is True
    assert out.loc["LEADER_A", "leader_addon_theme_peer_count"] == 2
    assert bool(out.loc["ISOLATED", "leader_addon_eligible"]) is False
    assert bool(out.loc["ISOLATED", "leader_addon_same_theme_peer_support"]) is False
    assert out.loc["ISOLATED", "leader_addon_circuit_breaker"] == "same_theme_peer_support_low"
    assert "\"same_theme_peer_support_required\": true" in out.loc["ISOLATED", "leader_addon_log"]


def test_leader_addon_gap_quality_support_prefers_benign_gap_theme_leaders() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "SAFE_A",
                "signal": 1.0,
                "final_score": 96.0,
                "relative_strength_score": 94.0,
                "theme_score": 82.0,
                "mom_return": 0.24,
                "benchmark_risk_on": True,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "distance_to_high_252": -0.05,
                "retest_distance_ma_pct": 0.03,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 12.0,
                "vol_20d": 0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "SAFE_B",
                "signal": 1.0,
                "final_score": 93.0,
                "relative_strength_score": 91.0,
                "theme_score": 78.0,
                "mom_return": 0.21,
                "benchmark_risk_on": True,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "distance_to_high_252": -0.06,
                "retest_distance_ma_pct": 0.02,
                "event_risk_score": 4.0,
                "overnight_gap_risk_score": 14.0,
                "vol_20d": 0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "HOT",
                "signal": 1.0,
                "final_score": 97.0,
                "relative_strength_score": 95.0,
                "theme_score": 84.0,
                "mom_return": 0.26,
                "benchmark_risk_on": True,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "distance_to_high_252": -0.04,
                "retest_distance_ma_pct": 0.04,
                "event_risk_score": 5.0,
                "overnight_gap_risk_score": 35.0,
                "vol_20d": 0.04,
            },
        ]
    )
    out = construct_target_weights(
        signals,
        PortfolioConfig(
            construction="score_volatility",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            leader_addon_overlay=True,
            leader_addon_multiplier=1.18,
            leader_addon_min_final_score=74.0,
            leader_addon_min_relative_strength_score=88.0,
            leader_addon_min_theme_score=62.0,
            leader_addon_min_mom_return=0.16,
            leader_addon_max_names_per_date=2,
            leader_addon_require_controlled_pullback=True,
            leader_addon_min_drawdown_from_high=0.03,
            leader_addon_max_drawdown_from_high=0.14,
            leader_addon_require_same_theme_peer_support=True,
            leader_addon_same_theme_peer_min_count=2,
            leader_addon_same_theme_peer_min_share=0.25,
            leader_addon_gap_quality_support_overlay=True,
            leader_addon_gap_quality_max_symbol_gap_risk_score=18.0,
            leader_addon_gap_quality_same_theme_max_gap_risk_score=16.0,
            leader_addon_gap_quality_same_theme_min_count=2,
            leader_addon_gap_quality_same_theme_min_share=0.34,
            leader_addon_max_event_risk_score_circuit_breaker=20.0,
            leader_addon_max_overnight_gap_risk_score_circuit_breaker=40.0,
        ),
    ).set_index("symbol")

    assert bool(out.loc["SAFE_A", "leader_addon_eligible"]) is True
    assert bool(out.loc["SAFE_A", "leader_addon_gap_quality_support"]) is True
    assert out.loc["SAFE_A", "leader_addon_gap_quality_peer_count"] == 2
    assert bool(out.loc["HOT", "leader_addon_eligible"]) is False
    assert out.loc["HOT", "leader_addon_circuit_breaker"] == "gap_quality_symbol_gap_high"
    assert "\"gap_quality_required\": true" in out.loc["SAFE_A", "leader_addon_log"]


def test_leader_persistence_stability_overlay_retains_small_trim_for_stable_leader() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "signal": 1.0,
                "final_score": 100.0,
                "relative_strength_score": 95.0,
                "theme_score": 80.0,
                "theme_active": True,
                "benchmark_risk_on": True,
                "mom_return": 0.30,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "signal": 1.0,
                "final_score": 90.0,
                "relative_strength_score": 92.0,
                "theme_score": 78.0,
                "theme_active": True,
                "benchmark_risk_on": True,
                "mom_return": 0.24,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-02",
                "symbol": "CCC",
                "signal": 1.0,
                "final_score": 80.0,
                "relative_strength_score": 70.0,
                "theme_score": 55.0,
                "theme_active": False,
                "benchmark_risk_on": True,
                "mom_return": 0.05,
                "primary_theme": "other",
            },
            {
                "date": "2024-01-03",
                "symbol": "AAA",
                "signal": 1.0,
                "final_score": 94.0,
                "relative_strength_score": 92.0,
                "theme_score": 80.0,
                "theme_active": True,
                "benchmark_risk_on": True,
                "mom_return": 0.28,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-03",
                "symbol": "BBB",
                "signal": 1.0,
                "final_score": 93.0,
                "relative_strength_score": 91.0,
                "theme_score": 77.0,
                "theme_active": True,
                "benchmark_risk_on": True,
                "mom_return": 0.23,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-03",
                "symbol": "CCC",
                "signal": 1.0,
                "final_score": 80.0,
                "relative_strength_score": 70.0,
                "theme_score": 55.0,
                "theme_active": False,
                "benchmark_risk_on": True,
                "mom_return": 0.05,
                "primary_theme": "other",
            },
        ]
    )
    base = construct_target_weights(
        signals,
        PortfolioConfig(
            rebalance="daily_throttled",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            throttle_min_weight_change=0.0,
            throttle_min_score_change=0.0,
            throttle_min_new_weight=0.0,
        ),
    )
    stable = construct_target_weights(
        signals,
        PortfolioConfig(
            rebalance="daily_throttled",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            throttle_min_weight_change=0.0,
            throttle_min_score_change=0.0,
            throttle_min_new_weight=0.0,
            leader_persistence_overlay=True,
            leader_persistence_stability_overlay=True,
            leader_persistence_min_final_score=68.0,
            leader_persistence_min_relative_strength_score=88.0,
            leader_persistence_min_theme_score=62.0,
            leader_persistence_min_mom_return=0.15,
            leader_persistence_min_theme_peer_count=2,
            leader_persistence_min_theme_peer_share=0.40,
            leader_persistence_retain_fraction_of_cut=0.50,
            leader_persistence_max_weight_bonus=0.02,
            leader_persistence_min_desired_weight_fraction=0.40,
            leader_persistence_max_final_score_drop=6.0,
            leader_persistence_max_relative_strength_drop=5.0,
        ),
    )

    target_date = pd.Timestamp("2024-01-03")
    base_aaa = base[(base["date"] == target_date) & (base["symbol"] == "AAA")].iloc[0]
    stable_aaa = stable[(stable["date"] == target_date) & (stable["symbol"] == "AAA")].iloc[0]

    assert stable_aaa["target_weight"] > base_aaa["target_weight"]
    assert bool(stable_aaa["leader_persistence_applied"]) is True
    assert stable_aaa["leader_persistence_circuit_breaker"] == "applied"


def test_leader_persistence_stability_overlay_blocks_large_score_decay() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "signal": 1.0,
                "final_score": 100.0,
                "relative_strength_score": 95.0,
                "theme_score": 80.0,
                "theme_active": True,
                "benchmark_risk_on": True,
                "mom_return": 0.30,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "signal": 1.0,
                "final_score": 90.0,
                "relative_strength_score": 92.0,
                "theme_score": 78.0,
                "theme_active": True,
                "benchmark_risk_on": True,
                "mom_return": 0.24,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-03",
                "symbol": "AAA",
                "signal": 1.0,
                "final_score": 87.0,
                "relative_strength_score": 84.0,
                "theme_score": 80.0,
                "theme_active": True,
                "benchmark_risk_on": True,
                "mom_return": 0.28,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-03",
                "symbol": "BBB",
                "signal": 1.0,
                "final_score": 86.0,
                "relative_strength_score": 91.0,
                "theme_score": 77.0,
                "theme_active": True,
                "benchmark_risk_on": True,
                "mom_return": 0.23,
                "primary_theme": "semiconductors",
            },
            {
                "date": "2024-01-02",
                "symbol": "CCC",
                "signal": 1.0,
                "final_score": 80.0,
                "relative_strength_score": 70.0,
                "theme_score": 55.0,
                "theme_active": False,
                "benchmark_risk_on": True,
                "mom_return": 0.05,
                "primary_theme": "other",
            },
            {
                "date": "2024-01-03",
                "symbol": "CCC",
                "signal": 1.0,
                "final_score": 80.0,
                "relative_strength_score": 70.0,
                "theme_score": 55.0,
                "theme_active": False,
                "benchmark_risk_on": True,
                "mom_return": 0.05,
                "primary_theme": "other",
            },
        ]
    )
    out = construct_target_weights(
        signals,
        PortfolioConfig(
            rebalance="daily_throttled",
            target_gross_exposure=1.0,
            target_net_exposure=1.0,
            max_position_weight=1.0,
            throttle_min_weight_change=0.0,
            throttle_min_score_change=0.0,
            throttle_min_new_weight=0.0,
            leader_persistence_overlay=True,
            leader_persistence_stability_overlay=True,
            leader_persistence_min_final_score=68.0,
            leader_persistence_min_relative_strength_score=80.0,
            leader_persistence_min_theme_score=62.0,
            leader_persistence_min_mom_return=0.15,
            leader_persistence_min_theme_peer_count=2,
            leader_persistence_min_theme_peer_share=1.0,
            leader_persistence_retain_fraction_of_cut=0.50,
            leader_persistence_max_weight_bonus=0.02,
            leader_persistence_min_desired_weight_fraction=0.40,
            leader_persistence_max_final_score_drop=6.0,
            leader_persistence_max_relative_strength_drop=5.0,
        ),
    )

    aaa = out[(out["date"] == pd.Timestamp("2024-01-03")) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(aaa["leader_persistence_applied"]) is False
    assert aaa["leader_persistence_circuit_breaker"] == "final_score_drop_too_large"
