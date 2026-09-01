from __future__ import annotations

import pandas as pd

from quant_system.config import PortfolioConfig
from quant_system.portfolio.construction import construct_target_weights, rebalance_dates


def test_daily_throttled_uses_daily_signal_dates() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert rebalance_dates(dates, "daily_throttled") == set(dates)


def test_daily_throttled_keeps_full_active_portfolio_when_one_weight_changes() -> None:
    signals = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "A", "signal": 1, "final_score": 90},
            {"date": "2024-01-01", "symbol": "B", "signal": -1, "final_score": 20},
            {"date": "2024-01-02", "symbol": "A", "signal": 1, "final_score": 90},
            {"date": "2024-01-02", "symbol": "B", "signal": -1, "final_score": 20.1},
            {"date": "2024-01-02", "symbol": "C", "signal": 1, "final_score": 95},
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1,
        target_net_exposure=0,
        max_position_weight=0.50,
        throttle_min_weight_change=0.20,
        throttle_min_new_weight=0.01,
    )
    out = construct_target_weights(signals, config)
    day2 = out[out["date"] == pd.Timestamp("2024-01-02")]
    assert set(day2["symbol"]) == {"A", "B", "C"}


def test_daily_throttled_refreshes_context_for_unchanged_active_rows() -> None:
    signals = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "A", "signal": 1, "final_score": 90, "event_risk_score": 0, "days_to_earnings": 10},
            {"date": "2024-01-01", "symbol": "B", "signal": -1, "final_score": 20, "event_risk_score": 0, "days_to_earnings": 10},
            {"date": "2024-01-02", "symbol": "A", "signal": 1, "final_score": 90, "event_risk_score": 80, "days_to_earnings": 1},
            {"date": "2024-01-02", "symbol": "B", "signal": -1, "final_score": 20, "event_risk_score": 0, "days_to_earnings": 10},
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1,
        target_net_exposure=0,
        max_position_weight=0.50,
        throttle_min_weight_change=0.20,
        throttle_min_score_change=7.0,
        throttle_min_new_weight=0.01,
    )
    out = construct_target_weights(signals, config)
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]
    assert day2_a["event_risk_score"] == 80
    assert day2_a["days_to_earnings"] == 1


def test_daily_throttled_leader_hold_buffer_keeps_small_residual_for_strong_dropped_long() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 92,
                "relative_strength_score": 90,
                "theme_score": 80,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 0,
                "final_score": 88,
                "relative_strength_score": 89,
                "theme_score": 78,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {"date": "2024-01-02", "symbol": "B", "signal": 1, "final_score": 95},
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1,
        target_net_exposure=1,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_hold_buffer_overlay=True,
        leader_hold_buffer_min_final_score=80,
        leader_hold_buffer_min_relative_strength_score=85,
        leader_hold_buffer_min_theme_score=70,
        leader_hold_buffer_weight_fraction=0.5,
        leader_hold_buffer_max_weight=0.30,
        leader_hold_buffer_max_days=3,
    )

    out = construct_target_weights(signals, config)
    day1_a = out[(out["date"] == pd.Timestamp("2024-01-01")) & (out["symbol"] == "A")].iloc[0]
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert day1_a["target_weight"] == 1.0
    assert 0.0 < day2_a["target_weight"] < day1_a["target_weight"]
    assert bool(day2_a["leader_hold_buffer_applied"]) is True
    assert int(day2_a["leader_hold_buffer_days"]) == 1
    assert day2_a["leader_hold_buffer_circuit_breaker"] == "applied"
    assert "Leader hold buffer retained" in day2_a["reason_for_entry"]


def test_daily_throttled_leader_hold_buffer_respects_risk_circuit_breaker() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 92,
                "relative_strength_score": 90,
                "theme_score": 80,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 0,
                "final_score": 88,
                "relative_strength_score": 89,
                "theme_score": 78,
                "benchmark_risk_on": False,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {"date": "2024-01-02", "symbol": "B", "signal": 1, "final_score": 95},
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1,
        target_net_exposure=1,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_hold_buffer_overlay=True,
        leader_hold_buffer_min_final_score=80,
        leader_hold_buffer_min_relative_strength_score=85,
        leader_hold_buffer_min_theme_score=70,
        leader_hold_buffer_weight_fraction=0.5,
        leader_hold_buffer_max_weight=0.30,
        leader_hold_buffer_max_days=3,
        leader_hold_buffer_require_benchmark_risk_on=True,
    )

    out = construct_target_weights(signals, config)
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert day2_a["target_weight"] == 0.0
    assert day2_a["reason_for_exit"] == "Dropped from daily-throttled target set."
    assert bool(day2_a["leader_hold_buffer_applied"]) is False
    assert day2_a["leader_hold_buffer_circuit_breaker"] == "benchmark_not_risk_on"


def test_daily_throttled_leader_hold_buffer_high_conviction_requires_momentum() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 94,
                "relative_strength_score": 94,
                "theme_score": 72,
                "mom_return": 0.25,
                "distance_to_high_252": -0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 0,
                "final_score": 91,
                "relative_strength_score": 92,
                "theme_score": 70,
                "mom_return": 0.05,
                "distance_to_high_252": -0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {"date": "2024-01-02", "symbol": "B", "signal": 1, "final_score": 95},
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1,
        target_net_exposure=1,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_hold_buffer_overlay=True,
        leader_hold_buffer_min_final_score=70,
        leader_hold_buffer_min_relative_strength_score=90,
        leader_hold_buffer_min_theme_score=65,
        leader_hold_buffer_min_mom_return=0.15,
        leader_hold_buffer_max_drawdown_from_high=0.18,
        leader_hold_buffer_weight_fraction=0.7,
        leader_hold_buffer_max_weight=0.35,
        leader_hold_buffer_max_days=5,
    )

    out = construct_target_weights(signals, config)
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert day2_a["target_weight"] == 0.0
    assert bool(day2_a["leader_hold_buffer_applied"]) is False
    assert day2_a["leader_hold_buffer_circuit_breaker"] == "momentum_return_low"


def test_daily_throttled_leader_delayed_exit_keeps_rank_exit_with_strong_trend() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 94,
                "relative_strength_score": 94,
                "theme_score": 72,
                "mom_return": 0.25,
                "distance_to_high_252": -0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 0,
                "final_score": 91,
                "relative_strength_score": 92,
                "theme_score": 70,
                "mom_return": 0.20,
                "distance_to_high_252": -0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {"date": "2024-01-02", "symbol": "B", "signal": 1, "final_score": 95},
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1,
        target_net_exposure=1,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_delayed_exit_overlay=True,
        leader_delayed_exit_min_final_score=70,
        leader_delayed_exit_min_relative_strength_score=90,
        leader_delayed_exit_min_theme_score=65,
        leader_delayed_exit_min_mom_return=0.15,
        leader_delayed_exit_max_drawdown_from_high=0.18,
        leader_delayed_exit_retain_fraction=0.85,
        leader_delayed_exit_max_weight=0.90,
        leader_delayed_exit_max_days=3,
    )

    out = construct_target_weights(signals, config)
    day1_a = out[(out["date"] == pd.Timestamp("2024-01-01")) & (out["symbol"] == "A")].iloc[0]
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert day1_a["target_weight"] == 1.0
    assert day2_a["target_weight"] == 0.85
    assert bool(day2_a["leader_delayed_exit_applied"]) is True
    assert int(day2_a["leader_delayed_exit_days"]) == 1
    assert day2_a["leader_delayed_exit_circuit_breaker"] == "applied"
    assert "Leader delayed-exit kept" in day2_a["reason_for_entry"]


def test_daily_throttled_leader_delayed_exit_respects_momentum_breaker() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 94,
                "relative_strength_score": 94,
                "theme_score": 72,
                "mom_return": 0.25,
                "distance_to_high_252": -0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 0,
                "final_score": 91,
                "relative_strength_score": 92,
                "theme_score": 70,
                "mom_return": 0.03,
                "distance_to_high_252": -0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {"date": "2024-01-02", "symbol": "B", "signal": 1, "final_score": 95},
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1,
        target_net_exposure=1,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_delayed_exit_overlay=True,
        leader_delayed_exit_min_final_score=70,
        leader_delayed_exit_min_relative_strength_score=90,
        leader_delayed_exit_min_theme_score=65,
        leader_delayed_exit_min_mom_return=0.15,
        leader_delayed_exit_max_drawdown_from_high=0.18,
        leader_delayed_exit_retain_fraction=0.85,
        leader_delayed_exit_max_weight=0.90,
        leader_delayed_exit_max_days=3,
    )

    out = construct_target_weights(signals, config)
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert day2_a["target_weight"] == 0.0
    assert bool(day2_a["leader_delayed_exit_applied"]) is False
    assert day2_a["leader_delayed_exit_circuit_breaker"] == "momentum_return_low"


def test_daily_throttled_leader_delayed_exit_stability_blocks_large_score_drop() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 94,
                "relative_strength_score": 95,
                "theme_score": 72,
                "mom_return": 0.25,
                "distance_to_high_252": -0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 0,
                "final_score": 83,
                "relative_strength_score": 92,
                "theme_score": 70,
                "mom_return": 0.20,
                "distance_to_high_252": -0.04,
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
            },
            {"date": "2024-01-02", "symbol": "B", "signal": 1, "final_score": 95},
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1,
        target_net_exposure=1,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_delayed_exit_overlay=True,
        leader_delayed_exit_stability_overlay=True,
        leader_delayed_exit_min_final_score=70,
        leader_delayed_exit_min_relative_strength_score=90,
        leader_delayed_exit_min_theme_score=65,
        leader_delayed_exit_min_mom_return=0.15,
        leader_delayed_exit_max_drawdown_from_high=0.18,
        leader_delayed_exit_retain_fraction=0.85,
        leader_delayed_exit_max_weight=0.90,
        leader_delayed_exit_max_days=3,
        leader_delayed_exit_max_final_score_drop=6.0,
        leader_delayed_exit_max_relative_strength_drop=5.0,
    )

    out = construct_target_weights(signals, config)
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert day2_a["target_weight"] == 0.0
    assert bool(day2_a["leader_delayed_exit_applied"]) is False
    assert day2_a["leader_delayed_exit_circuit_breaker"] == "final_score_drop_high"


def test_daily_throttled_leader_persistence_softens_reduction_for_strong_same_theme_leader() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 92,
                "relative_strength_score": 94,
                "theme_score": 83,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.32,
            },
            {
                "date": "2024-01-01",
                "symbol": "B",
                "signal": 1,
                "final_score": 88,
                "relative_strength_score": 91,
                "theme_score": 80,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.24,
            },
            {
                "date": "2024-01-01",
                "symbol": "C",
                "signal": 1,
                "final_score": 84,
                "relative_strength_score": 76,
                "theme_score": 58,
                "theme_active": False,
                "primary_theme": "other",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 5,
                "mom_return": 0.05,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 1,
                "final_score": 90,
                "relative_strength_score": 95,
                "theme_score": 84,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.30,
            },
            {
                "date": "2024-01-02",
                "symbol": "B",
                "signal": 1,
                "final_score": 89,
                "relative_strength_score": 92,
                "theme_score": 82,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.22,
            },
            {
                "date": "2024-01-02",
                "symbol": "C",
                "signal": 1,
                "final_score": 98,
                "relative_strength_score": 80,
                "theme_score": 56,
                "theme_active": False,
                "primary_theme": "other",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 5,
                "mom_return": 0.08,
            },
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1.0,
        target_net_exposure=1.0,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_persistence_overlay=True,
        leader_persistence_min_final_score=68.0,
        leader_persistence_min_relative_strength_score=88.0,
        leader_persistence_min_theme_score=62.0,
        leader_persistence_min_mom_return=0.15,
        leader_persistence_min_theme_peer_count=2,
        leader_persistence_min_theme_peer_share=0.40,
        leader_persistence_retain_fraction_of_cut=0.75,
        leader_persistence_max_weight_bonus=0.40,
    )

    out = construct_target_weights(signals, config)
    day1_a = out[(out["date"] == pd.Timestamp("2024-01-01")) & (out["symbol"] == "A")].iloc[0]
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert day2_a["target_weight"] > 0.40
    assert day2_a["target_weight"] < day1_a["target_weight"]
    assert bool(day2_a["leader_persistence_applied"]) is True
    assert day2_a["leader_persistence_circuit_breaker"] == "applied"
    assert day2_a["leader_persistence_incremental_weight"] > 0.0
    assert "Leader persistence retained" in day2_a["reason_for_entry"]


def test_daily_throttled_leader_persistence_respects_gap_risk_circuit_breaker() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 92,
                "relative_strength_score": 94,
                "theme_score": 83,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.32,
            },
            {
                "date": "2024-01-01",
                "symbol": "B",
                "signal": 1,
                "final_score": 88,
                "relative_strength_score": 91,
                "theme_score": 80,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.24,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 1,
                "final_score": 90,
                "relative_strength_score": 95,
                "theme_score": 84,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 60,
                "mom_return": 0.30,
            },
            {
                "date": "2024-01-02",
                "symbol": "B",
                "signal": 1,
                "final_score": 89,
                "relative_strength_score": 92,
                "theme_score": 82,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.22,
            },
            {
                "date": "2024-01-02",
                "symbol": "C",
                "signal": 1,
                "final_score": 98,
                "relative_strength_score": 80,
                "theme_score": 56,
                "theme_active": False,
                "primary_theme": "other",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 5,
                "mom_return": 0.08,
            },
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1.0,
        target_net_exposure=1.0,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_persistence_overlay=True,
        leader_persistence_min_theme_peer_count=2,
        leader_persistence_min_theme_peer_share=0.40,
        leader_persistence_max_overnight_gap_risk_score_circuit_breaker=30.0,
    )

    out = construct_target_weights(signals, config)
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert bool(day2_a["leader_persistence_applied"]) is False
    assert day2_a["leader_persistence_circuit_breaker"] == "gap_risk_high"


def test_daily_throttled_leader_persistence_peer_strength_reward_keeps_more_weight() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 92,
                "relative_strength_score": 95,
                "theme_score": 83,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.32,
                "distance_to_high_252": -0.05,
            },
            {
                "date": "2024-01-01",
                "symbol": "B",
                "signal": 1,
                "final_score": 89,
                "relative_strength_score": 94,
                "theme_score": 82,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 8,
                "mom_return": 0.26,
                "distance_to_high_252": -0.04,
            },
            {
                "date": "2024-01-01",
                "symbol": "C",
                "signal": 1,
                "final_score": 88,
                "relative_strength_score": 93,
                "theme_score": 81,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 8,
                "mom_return": 0.24,
                "distance_to_high_252": -0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 1,
                "final_score": 90,
                "relative_strength_score": 95,
                "theme_score": 84,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.30,
                "distance_to_high_252": -0.05,
            },
            {
                "date": "2024-01-02",
                "symbol": "B",
                "signal": 1,
                "final_score": 89,
                "relative_strength_score": 93,
                "theme_score": 82,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.22,
                "distance_to_high_252": -0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "C",
                "signal": 1,
                "final_score": 88,
                "relative_strength_score": 92,
                "theme_score": 81,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 8,
                "mom_return": 0.21,
                "distance_to_high_252": -0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "D",
                "signal": 1,
                "final_score": 98,
                "relative_strength_score": 80,
                "theme_score": 56,
                "theme_active": False,
                "primary_theme": "other",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 5,
                "mom_return": 0.08,
                "distance_to_high_252": -0.01,
            },
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1.0,
        target_net_exposure=1.0,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_persistence_overlay=True,
        leader_persistence_min_theme_peer_count=2,
        leader_persistence_min_theme_peer_share=0.40,
        leader_persistence_retain_fraction_of_cut=0.35,
        leader_persistence_max_weight_bonus=0.12,
        leader_persistence_peer_strength_reward_overlay=True,
        leader_persistence_peer_strength_reward_min_theme_peer_count=3,
        leader_persistence_peer_strength_reward_min_theme_peer_share=0.55,
        leader_persistence_peer_strength_reward_min_relative_strength_score=92.0,
        leader_persistence_peer_strength_reward_retain_fraction_boost=0.20,
        leader_persistence_peer_strength_reward_max_weight_bonus=0.10,
        leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker=0.14,
    )

    out = construct_target_weights(signals, config)
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert bool(day2_a["leader_persistence_applied"]) is True
    assert bool(day2_a["leader_persistence_peer_strength_reward_applied"]) is True
    assert day2_a["leader_persistence_peer_strength_reward_circuit_breaker"] == "applied"
    assert day2_a["leader_persistence_incremental_weight"] > 0.18


def test_daily_throttled_leader_persistence_peer_strength_reward_respects_drawdown_breaker() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "A",
                "signal": 1,
                "final_score": 92,
                "relative_strength_score": 95,
                "theme_score": 83,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.32,
                "distance_to_high_252": -0.20,
            },
            {
                "date": "2024-01-01",
                "symbol": "B",
                "signal": 1,
                "final_score": 89,
                "relative_strength_score": 94,
                "theme_score": 82,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 8,
                "mom_return": 0.26,
                "distance_to_high_252": -0.04,
            },
            {
                "date": "2024-01-01",
                "symbol": "C",
                "signal": 1,
                "final_score": 88,
                "relative_strength_score": 93,
                "theme_score": 81,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 8,
                "mom_return": 0.24,
                "distance_to_high_252": -0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "A",
                "signal": 1,
                "final_score": 90,
                "relative_strength_score": 95,
                "theme_score": 84,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.30,
                "distance_to_high_252": -0.20,
            },
            {
                "date": "2024-01-02",
                "symbol": "B",
                "signal": 1,
                "final_score": 89,
                "relative_strength_score": 93,
                "theme_score": 82,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 10,
                "mom_return": 0.22,
                "distance_to_high_252": -0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "C",
                "signal": 1,
                "final_score": 88,
                "relative_strength_score": 92,
                "theme_score": 81,
                "theme_active": True,
                "primary_theme": "semiconductors",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 8,
                "mom_return": 0.21,
                "distance_to_high_252": -0.04,
            },
            {
                "date": "2024-01-02",
                "symbol": "D",
                "signal": 1,
                "final_score": 98,
                "relative_strength_score": 80,
                "theme_score": 56,
                "theme_active": False,
                "primary_theme": "other",
                "benchmark_risk_on": True,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 5,
                "mom_return": 0.08,
                "distance_to_high_252": -0.01,
            },
        ]
    )
    config = PortfolioConfig(
        rebalance="daily_throttled",
        target_gross_exposure=1.0,
        target_net_exposure=1.0,
        max_position_weight=1.0,
        throttle_min_weight_change=0.0,
        throttle_min_new_weight=0.0,
        leader_persistence_overlay=True,
        leader_persistence_min_theme_peer_count=2,
        leader_persistence_min_theme_peer_share=0.40,
        leader_persistence_retain_fraction_of_cut=0.35,
        leader_persistence_max_weight_bonus=0.12,
        leader_persistence_peer_strength_reward_overlay=True,
        leader_persistence_peer_strength_reward_min_theme_peer_count=3,
        leader_persistence_peer_strength_reward_min_theme_peer_share=0.55,
        leader_persistence_peer_strength_reward_min_relative_strength_score=92.0,
        leader_persistence_peer_strength_reward_retain_fraction_boost=0.20,
        leader_persistence_peer_strength_reward_max_weight_bonus=0.10,
        leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker=0.14,
    )

    out = construct_target_weights(signals, config)
    day2_a = out[(out["date"] == pd.Timestamp("2024-01-02")) & (out["symbol"] == "A")].iloc[0]

    assert bool(day2_a["leader_persistence_applied"]) is True
    assert bool(day2_a["leader_persistence_peer_strength_reward_applied"]) is False
    assert day2_a["leader_persistence_peer_strength_reward_circuit_breaker"] == "drawdown_from_high_high"
    assert 0.10 < day2_a["leader_persistence_incremental_weight"] < 0.13
