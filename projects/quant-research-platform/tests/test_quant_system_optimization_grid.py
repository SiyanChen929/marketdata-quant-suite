from __future__ import annotations

import warnings

from quant_system.optimization.parameter_search import default_research_scenarios, objective_score
from quant_system.cli import _aggregate_walk_forward_results, _apply_return_promotion_guards, _best_config_from_results, _filter_scenarios
from quant_system.config import AppConfig
import pandas as pd


def test_default_research_scenarios_include_rebalance_and_strategy_profiles() -> None:
    scenarios = default_research_scenarios("quick")
    assert scenarios
    first = scenarios[0]
    assert {"scenario", "rebalance_profile", "strategy_profile", "portfolio", "strategies"}.issubset(first)
    assert "momentum" in first["strategies"]
    assert "lookback_returns" in first["strategies"]["momentum"]


def test_walk_forward_aggregation_avoids_fragmentation_warning() -> None:
    detail = pd.DataFrame(
        [
            {
                "split_id": split_id,
                "scenario": "scenario_a",
                "rebalance_profile": "daily",
                "strategy_profile": "tech_rs",
                "constraint_profile": "compound",
                "event_profile": "half_reduce",
                "rebalance": "daily_throttled",
                "construction": "score_weighted",
                "target_gross_exposure": 1.5,
                "target_net_exposure": 1.5,
                "momentum_custom_final_score_weights": True,
                "momentum_custom_final_score_blend": 1.0,
                "lookback_returns": 21,
                "skip_recent_days": 3,
                "long_quantile": 0.25,
                "short_quantile": 0.0,
                "full_cagr": 0.10 + split_id * 0.01,
                "full_max_drawdown": -0.20,
                "full_sharpe": 0.80,
                "full_turnover": 0.30,
                "train_cagr": 0.08,
                "train_max_drawdown": -0.18,
                "train_sharpe": 0.70,
                "train_turnover": 0.25,
                "train_objective": 0.12,
                "train_dsr": 0.60,
                "validation_cagr": 0.15,
                "validation_max_drawdown": -0.15,
                "validation_sharpe": 0.90,
                "validation_turnover": 0.28,
                "validation_objective": 0.20 + split_id * 0.01,
                "validation_dsr": 0.70,
                "objective_gap": -0.08,
                "trade_count": 10,
                "validation_pass": split_id != 0,
            }
            for split_id in range(3)
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        out = _aggregate_walk_forward_results(detail)

    assert not [warning for warning in caught if issubclass(warning.category, pd.errors.PerformanceWarning)]
    assert out.loc[0, "validation_pass_rate"] == 2 / 3
    assert bool(out.loc[0, "validation_pass"]) is True


def test_research_profile_includes_pure_daily_diagnostic() -> None:
    scenarios = default_research_scenarios("research")
    rebalances = {scenario["portfolio"]["rebalance"] for scenario in scenarios}
    assert "daily" in rebalances
    assert "daily_throttled" in rebalances


def test_aggressive_profile_searches_execution_and_position_constraints() -> None:
    scenarios = default_research_scenarios("aggressive")
    assert scenarios
    assert {scenario["constraint_profile"] for scenario in scenarios} >= {
        "aggressive_realistic",
        "aggressive_high_turnover",
        "aggressive_capacity_guarded",
    }
    assert any(scenario["execution"].get("daily_turnover_cap") == 0.50 for scenario in scenarios)
    assert any(scenario["portfolio"].get("max_total_positions") == 80 for scenario in scenarios)
    assert any(scenario["risk"].get("min_holding_days") == 0 for scenario in scenarios)
    assert any(scenario["regime"].get("exposure_overlay") is True for scenario in scenarios)
    assert any(scenario["regime"].get("risk_off_exposure_multiplier") == 0.0 for scenario in scenarios)
    assert any(scenario["regime"].get("analog_momentum_overlay") is True for scenario in scenarios)
    assert any(
        scenario["portfolio"].get("target_gross_exposure", 0) > scenario["portfolio"].get("target_net_exposure", 0)
        for scenario in scenarios
    )


def test_cycle_return_profile_prioritizes_high_ai_cycle_exposure() -> None:
    scenarios = default_research_scenarios("cycle_return")
    assert scenarios
    assert {scenario["constraint_profile"] for scenario in scenarios} >= {
        "cycle_long_only_compound_guarded",
        "cycle_long_only_conviction_sizing",
        "cycle_long_only_intraday_leader_exception",
        "cycle_long_only_regime_leader_exception",
        "cycle_long_only_conviction_sizing_theme_peer_support",
        "cycle_long_only_conviction_sizing_attributed_themes",
        "cycle_long_only_leader_addon",
        "cycle_long_only_leader_addon_gap_quality",
        "cycle_long_only_leader_hold_buffer",
        "cycle_long_only_leader_crowding_pullback_exception",
        "cycle_long_only_leader_crowding_peer_restore",
        "cycle_long_only_leader_crowding_peer_exception",
        "cycle_long_only_leader_delayed_exit_rank_credit",
        "cycle_long_only_leader_persistence",
        "cycle_long_only_leader_persistence_stable",
        "cycle_max_return_guarded",
        "cycle_max_return_guarded_leader_relief",
        "cycle_max_return_higher_turnover",
        "cycle_max_return_higher_turnover_leader_relief",
    }
    assert any(scenario["portfolio"].get("target_gross_exposure") == 2.0 for scenario in scenarios)
    assert any(scenario["portfolio"].get("target_gross_exposure") == scenario["portfolio"].get("target_net_exposure") for scenario in scenarios)
    assert any(scenario["portfolio"].get("target_net_exposure") < scenario["portfolio"].get("target_gross_exposure") for scenario in scenarios)
    conviction_sizing = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_conviction_sizing"]
    assert conviction_sizing
    assert all(scenario["portfolio"].get("conviction_sizing_overlay") is True for scenario in conviction_sizing)
    assert all(scenario["portfolio"].get("conviction_sizing_max_multiplier", 0.0) <= 1.40 for scenario in conviction_sizing)
    conviction_sizing_theme_peer = [
        scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_conviction_sizing_theme_peer_support"
    ]
    assert conviction_sizing_theme_peer
    assert all(scenario["portfolio"].get("conviction_sizing_overlay") is True for scenario in conviction_sizing_theme_peer)
    assert all(scenario["portfolio"].get("conviction_sizing_require_same_theme_peer_support") is True for scenario in conviction_sizing_theme_peer)
    assert all(scenario["portfolio"].get("conviction_sizing_same_theme_peer_min_count", 0) >= 3 for scenario in conviction_sizing_theme_peer)
    conviction_sizing_attributed = [
        scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_conviction_sizing_attributed_themes"
    ]
    assert conviction_sizing_attributed
    assert all(scenario["portfolio"].get("conviction_sizing_overlay") is True for scenario in conviction_sizing_attributed)
    assert all(scenario["portfolio"].get("conviction_sizing_require_same_theme_peer_support") is True for scenario in conviction_sizing_attributed)
    assert all("semiconductors" in scenario["portfolio"].get("conviction_sizing_allowed_themes", ()) for scenario in conviction_sizing_attributed)
    assert all(scenario["portfolio"].get("conviction_sizing_max_multiplier", 0.0) <= 1.25 for scenario in conviction_sizing_attributed)
    intraday_exception = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_intraday_leader_exception"]
    assert intraday_exception
    assert all(scenario["regime"].get("intraday_leader_exception_overlay") is True for scenario in intraday_exception)
    assert all(scenario["regime"].get("intraday_leader_exception_restore_multiplier", 0.0) <= 0.85 for scenario in intraday_exception)
    assert all(scenario["regime"].get("intraday_leader_exception_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 25.0 for scenario in intraday_exception)
    regime_exception = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_regime_leader_exception"]
    assert regime_exception
    assert all(scenario["regime"].get("regime_leader_exception_overlay") is True for scenario in regime_exception)
    assert all(scenario["regime"].get("regime_leader_exception_restore_multiplier", 0.0) <= 0.75 for scenario in regime_exception)
    assert all(scenario["regime"].get("regime_leader_exception_min_regime_score", 0.0) >= 50.0 for scenario in regime_exception)
    assert all(scenario["regime"].get("regime_leader_exception_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 25.0 for scenario in regime_exception)
    assert any(scenario["strategies"]["momentum"].get("short_quantile") == 0.0 for scenario in scenarios)
    assert any(scenario["strategies"]["momentum"].get("short_quantile") > 0.0 for scenario in scenarios)
    long_only = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only"]
    assert long_only
    assert all(scenario["strategies"]["momentum"].get("short_quantile") == 0.0 for scenario in long_only)
    assert all(scenario["strategies"]["momentum"].get("long_pullback_reset_inclusion_overlay") is True for scenario in long_only)
    assert all(scenario["strategies"]["momentum"].get("pullback_reset_inclusion_max_names_per_date") <= 2 for scenario in long_only)
    theme_strength_reset = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_theme_strength_reset"]
    assert theme_strength_reset
    assert all(scenario["strategies"]["momentum"].get("theme_strength_delta_overlay") is True for scenario in theme_strength_reset)
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_theme_strength_support_overlay") is True
        for scenario in theme_strength_reset
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_min_theme_strength_delta_score", 0.0) >= 35.0
        for scenario in theme_strength_reset
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_theme_strength_score_rank_credit", 0.0) <= 0.10
        for scenario in theme_strength_reset
    )
    theme_strength = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_theme_strength_delta"]
    assert theme_strength
    assert all(scenario["strategies"]["momentum"].get("theme_strength_delta_overlay") is True for scenario in theme_strength)
    assert all(scenario["strategies"]["momentum"].get("theme_strength_delta_min_peer_count", 0) >= 3 for scenario in theme_strength)
    assert all(scenario["strategies"]["momentum"].get("theme_strength_delta_max_score_boost", 0.0) <= 1.5 for scenario in theme_strength)
    assert all(
        scenario["strategies"]["momentum"].get("theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 35.0
        for scenario in theme_strength
    )
    controlled = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_tech_rs_controlled"]
    assert controlled
    assert all(scenario["strategies"]["momentum"].get("momentum_custom_final_score_weights") is True for scenario in controlled)
    assert all(scenario["strategies"]["momentum"].get("momentum_custom_score_controlled_entry_overlay") is True for scenario in controlled)
    assert all(scenario["strategies"]["momentum"].get("momentum_custom_score_max_above_ma20_pct", 1.0) <= 0.10 for scenario in controlled)
    assert all(
        scenario["strategies"]["momentum"].get("momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 30.0
        for scenario in controlled
    )
    short_term_reset = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_short_term_reset"]
    assert short_term_reset
    assert all(scenario["strategies"]["momentum"].get("pullback_short_term_reset_overlay") is True for scenario in short_term_reset)
    assert all(
        scenario["strategies"]["momentum"].get("pullback_short_term_reset_max_above_ma20_pct", 1.0) <= 0.03
        for scenario in short_term_reset
    )
    assert all(scenario["strategies"]["momentum"].get("pullback_short_term_reset_max_ret_5d", 1.0) <= 0.01 for scenario in short_term_reset)
    assert all(scenario["strategies"]["momentum"].get("pullback_short_term_reset_max_ret_10d", 1.0) <= 0.02 for scenario in short_term_reset)
    assert all(
        scenario["strategies"]["momentum"].get("pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 30.0
        for scenario in short_term_reset
    )
    post_earnings_drift = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_post_earnings_drift"]
    assert post_earnings_drift
    assert all(scenario["strategies"]["momentum"].get("post_earnings_drift_overlay") is True for scenario in post_earnings_drift)
    assert all(
        scenario["strategies"]["momentum"].get("post_earnings_drift_min_surprise_eps_pct", 0.0) >= 8.0
        for scenario in post_earnings_drift
    )
    assert all(
        scenario["strategies"]["momentum"].get("post_earnings_drift_max_days_since_earnings", 999) <= 6
        for scenario in post_earnings_drift
    )
    assert all(
        scenario["strategies"]["momentum"].get("post_earnings_drift_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 30.0
        for scenario in post_earnings_drift
    )
    assert all(
        scenario["strategies"]["momentum"].get("post_earnings_drift_require_theme_active") is True
        for scenario in post_earnings_drift
    )
    activation_reset = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_activation_reset_support"]
    assert activation_reset
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_activation_support_overlay") is True
        for scenario in activation_reset
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_activation_score_rank_credit", 1.0) <= 0.08
        for scenario in activation_reset
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_min_volume_expansion", 0.0) >= 1.05
        for scenario in activation_reset
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 30.0
        for scenario in activation_reset
    )
    short_term_volume = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_short_term_volume_tilt"]
    assert short_term_volume
    assert all(scenario["strategies"]["momentum"].get("short_term_volume_tilt_overlay") is True for scenario in short_term_volume)
    assert all(scenario["strategies"]["momentum"].get("short_term_volume_tilt_max_score_boost", 10.0) <= 1.75 for scenario in short_term_volume)
    assert all(scenario["strategies"]["momentum"].get("short_term_volume_tilt_min_volume_expansion", 0.0) >= 1.0 for scenario in short_term_volume)
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 35.0
        for scenario in short_term_volume
    )
    crowding_peer_exception = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_crowding_peer_exception"]
    assert crowding_peer_exception
    assert all(scenario["regime"].get("leader_crowding_relief_overlay") is True for scenario in crowding_peer_exception)
    assert all(
        scenario["regime"].get("leader_crowding_relief_same_theme_peer_exception_overlay") is True
        for scenario in crowding_peer_exception
    )
    assert all(
        scenario["regime"].get("leader_crowding_relief_same_theme_peer_min_active_share", 0.0) >= 0.45
        for scenario in crowding_peer_exception
    )
    assert all(
        scenario["regime"].get("leader_crowding_relief_same_theme_peer_min_avg_mom_return", 0.0) >= 0.18
        for scenario in crowding_peer_exception
    )
    crowding_pullback_exception = [
        scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_crowding_pullback_exception"
    ]
    assert crowding_pullback_exception
    assert all(
        scenario["regime"].get("leader_crowding_relief_controlled_pullback_overlay") is True
        for scenario in crowding_pullback_exception
    )
    assert all(
        scenario["regime"].get("leader_crowding_relief_controlled_pullback_min_distance_from_high", 0.0) >= 0.04
        for scenario in crowding_pullback_exception
    )
    assert all(
        scenario["regime"].get("leader_crowding_relief_controlled_pullback_max_above_ma20_pct", 1.0) <= 0.06
        for scenario in crowding_pullback_exception
    )
    crowding_peer_restore = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_crowding_peer_restore"]
    assert crowding_peer_restore
    assert all(scenario["regime"].get("leader_crowding_relief_overlay") is True for scenario in crowding_peer_restore)
    assert all(
        scenario["regime"].get("leader_crowding_relief_same_theme_peer_restore_overlay") is True
        for scenario in crowding_peer_restore
    )
    assert all(
        scenario["regime"].get("leader_crowding_relief_same_theme_peer_restore_min_active_share", 0.0) >= 0.50
        for scenario in crowding_peer_restore
    )
    assert all(
        scenario["regime"].get("leader_crowding_relief_same_theme_peer_restore_min_avg_mom_return", 0.0) >= 0.12
        for scenario in crowding_peer_restore
    )
    short_term_volume_lift = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_short_term_volume_lift"]
    assert short_term_volume_lift
    assert all(scenario["strategies"]["momentum"].get("short_term_volume_tilt_overlay") is True for scenario in short_term_volume_lift)
    assert all(scenario["strategies"]["momentum"].get("short_term_volume_tilt_max_score_boost", 0.0) <= 2.5 for scenario in short_term_volume_lift)
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_min_volume_expansion", 0.0) >= 1.15
        for scenario in short_term_volume_lift
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 28.0
        for scenario in short_term_volume_lift
    )
    short_term_volume_adx_relaxed = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_short_term_volume_adx_relaxed"
    ]
    assert short_term_volume_adx_relaxed
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_min_adx_circuit_breaker") == 12.0
        for scenario in short_term_volume_adx_relaxed
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_require_benchmark_risk_on") is True
        for scenario in short_term_volume_adx_relaxed
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_max_event_risk_score_circuit_breaker", 100.0) <= 18.0
        for scenario in short_term_volume_adx_relaxed
    )
    short_term_volume_peer_confirmed = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_short_term_volume_peer_confirmed"
    ]
    assert short_term_volume_peer_confirmed
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_same_theme_peer_confirmation_overlay") is True
        for scenario in short_term_volume_peer_confirmed
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_same_theme_peer_min_count", 0) >= 2
        for scenario in short_term_volume_peer_confirmed
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_same_theme_peer_min_share", 0.0) >= 0.50
        for scenario in short_term_volume_peer_confirmed
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_same_theme_peer_min_avg_mom_return", 0.0) >= 0.10
        for scenario in short_term_volume_peer_confirmed
    )
    short_term_volume_peer_supported = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_short_term_volume_peer_supported"
    ]
    assert short_term_volume_peer_supported
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_same_theme_peer_volume_substitution_overlay") is True
        for scenario in short_term_volume_peer_supported
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion", 0.0) >= 0.70
        for scenario in short_term_volume_peer_supported
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall", 1.0) <= 0.35
        for scenario in short_term_volume_peer_supported
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale", 1.0) <= 0.60
        for scenario in short_term_volume_peer_supported
    )
    leader_persistence_peer_reward = [
        scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_persistence_peer_reward"
    ]
    assert leader_persistence_peer_reward
    assert all(
        scenario["portfolio"].get("leader_persistence_peer_strength_reward_overlay") is True
        for scenario in leader_persistence_peer_reward
    )
    assert all(
        scenario["portfolio"].get("leader_persistence_peer_strength_reward_min_theme_peer_count", 0) >= 3
        for scenario in leader_persistence_peer_reward
    )
    assert all(
        scenario["portfolio"].get("leader_persistence_peer_strength_reward_min_theme_peer_share", 0.0) >= 0.55
        for scenario in leader_persistence_peer_reward
    )
    assert all(
        scenario["portfolio"].get("leader_persistence_peer_strength_reward_max_weight_bonus", 0.0) <= 0.0075
        for scenario in leader_persistence_peer_reward
    )
    short_term_volume_peer_exception = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_short_term_volume_peer_exception"
    ]
    assert short_term_volume_peer_exception
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_non_risk_on_exception_overlay") is True
        for scenario in short_term_volume_peer_exception
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count", 0) >= 3
        for scenario in short_term_volume_peer_exception
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share", 0.0) >= 0.60
        for scenario in short_term_volume_peer_exception
    )
    assert all(
        scenario["strategies"]["momentum"].get("short_term_volume_tilt_non_risk_on_exception_score_boost_scale", 1.0) <= 0.50
        for scenario in short_term_volume_peer_exception
    )
    boundary_credit = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_boundary_rs_theme_credit"]
    assert boundary_credit
    assert all(scenario["strategies"]["momentum"].get("boundary_rs_theme_credit_overlay") is True for scenario in boundary_credit)
    assert all(
        scenario["strategies"]["momentum"].get("boundary_rs_theme_credit_min_relative_strength_score", 0.0) >= 88.0
        for scenario in boundary_credit
    )
    assert all(
        scenario["strategies"]["momentum"].get("boundary_rs_theme_credit_min_theme_score", 0.0) >= 70.0
        for scenario in boundary_credit
    )
    assert all(
        scenario["strategies"]["momentum"].get("boundary_rs_theme_credit_rank_buffer_below", 0.0) <= 0.12
        for scenario in boundary_credit
    )
    assert all(
        scenario["strategies"]["momentum"].get("boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 25.0
        for scenario in boundary_credit
    )
    exit_quality_credit = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_exit_quality_rank_credit"]
    assert exit_quality_credit
    assert all(scenario["strategies"]["momentum"].get("exit_quality_rank_credit_overlay") is True for scenario in exit_quality_credit)
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_min_relative_strength_score", 0.0) >= 88.0
        for scenario in exit_quality_credit
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_max_score_rank_credit", 1.0) <= 0.04
        for scenario in exit_quality_credit
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_require_benchmark_risk_on") is True
        for scenario in exit_quality_credit
    )
    exit_quality_theme_support = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_exit_quality_theme_support"
    ]
    assert exit_quality_theme_support
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_theme_support_overlay") is True
        for scenario in exit_quality_theme_support
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_theme_support_min_peer_count", 0) >= 2
        for scenario in exit_quality_theme_support
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_theme_support_min_combined_delta", -1.0) >= 0.0
        for scenario in exit_quality_theme_support
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_require_theme_active") is True
        for scenario in exit_quality_theme_support
    )
    exit_quality_reset_breadth = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_exit_quality_reset_breadth"
    ]
    assert exit_quality_reset_breadth
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_overlay") is True
        for scenario in exit_quality_reset_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_short_term_reset_overlay") is True
        for scenario in exit_quality_reset_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_min_reset_score", 0.0) >= 20.0
        for scenario in exit_quality_reset_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold", 1.0) <= 0.08
        for scenario in exit_quality_reset_breadth
    )
    exit_quality_reset_theme_peer_breadth = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_exit_quality_reset_theme_peer_breadth"
    ]
    assert exit_quality_reset_theme_peer_breadth
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay") is True
        for scenario in exit_quality_reset_theme_peer_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_rank_buffer_below", 1.0) <= 0.045
        for scenario in exit_quality_reset_theme_peer_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_peer_min_count", 0) >= 2
        for scenario in exit_quality_reset_theme_peer_breadth
    )
    exit_quality_reset_theme_peer_reset_breadth = [
        scenario
        for scenario in scenarios
        if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_exit_quality_reset_theme_peer_reset_breadth"
    ]
    assert exit_quality_reset_theme_peer_reset_breadth
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay")
        is True
        for scenario in exit_quality_reset_theme_peer_reset_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_rank_buffer_below", 1.0) <= 0.04
        for scenario in exit_quality_reset_theme_peer_reset_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share", 0.0)
        >= 0.34
        for scenario in exit_quality_reset_theme_peer_reset_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count", 0)
        >= 2
        for scenario in exit_quality_reset_theme_peer_reset_breadth
    )
    exit_quality_reset_theme_peer_leader_guard = [
        scenario
        for scenario in scenarios
        if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_exit_quality_reset_theme_peer_leader_guard"
    ]
    assert exit_quality_reset_theme_peer_leader_guard
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay") is True
        for scenario in exit_quality_reset_theme_peer_leader_guard
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct", 0.0)
        >= 0.78
        for scenario in exit_quality_reset_theme_peer_leader_guard
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap", 999.0)
        <= 6.0
        for scenario in exit_quality_reset_theme_peer_leader_guard
    )
    exit_quality_reset_theme_peer_quality = [
        scenario
        for scenario in scenarios
        if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_exit_quality_reset_theme_peer_quality"
    ]
    assert exit_quality_reset_theme_peer_quality
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay")
        is True
        for scenario in exit_quality_reset_theme_peer_quality
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs", 0.0)
        >= 84.0
        for scenario in exit_quality_reset_theme_peer_quality
    )
    assert all(
        scenario["strategies"]["momentum"].get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay")
        is True
        for scenario in exit_quality_reset_theme_peer_quality
    )
    breadth_accel = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_theme_breadth_accel"]
    assert breadth_accel
    assert all(scenario["strategies"]["momentum"].get("theme_breadth_acceleration_overlay") is True for scenario in breadth_accel)
    assert all(scenario["strategies"]["momentum"].get("theme_breadth_acceleration_min_theme_peer_count", 0) >= 3 for scenario in breadth_accel)
    assert all(scenario["strategies"]["momentum"].get("theme_breadth_acceleration_max_score_boost", 0.0) <= 2.0 for scenario in breadth_accel)
    assert all(
        scenario["strategies"]["momentum"].get("theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 35.0
        for scenario in breadth_accel
    )
    gap_continuation = [
        scenario
        for scenario in scenarios
        if scenario["strategy_profile"]
        in {
            "ai_cycle_pullback_long_only_gap_adjusted_continuation",
            "ai_cycle_pullback_long_only_gap_adjusted_continuation_strict",
            "ai_cycle_pullback_long_only_gap_adjusted_continuation_theme_peer_quality",
            "ai_cycle_pullback_long_only_gap_adjusted_continuation_theme_peer_quality_strict",
        }
    ]
    assert gap_continuation
    assert all(scenario["strategies"]["momentum"].get("gap_adjusted_continuation_overlay") is True for scenario in gap_continuation)
    assert all(
        scenario["strategies"]["momentum"].get("gap_adjusted_continuation_max_gap_risk_score_circuit_breaker", 100.0) <= 28.0
        for scenario in gap_continuation
    )
    assert all(
        scenario["strategies"]["momentum"].get("gap_adjusted_continuation_max_event_risk_score_circuit_breaker", 100.0) <= 18.0
        for scenario in gap_continuation
    )
    assert all(scenario["strategies"]["momentum"].get("gap_adjusted_continuation_max_score_boost", 0.0) <= 2.5 for scenario in gap_continuation)
    peer_quality_gap = [
        scenario
        for scenario in gap_continuation
        if "theme_peer_quality" in scenario["strategy_profile"]
    ]
    assert peer_quality_gap
    assert all(
        scenario["strategies"]["momentum"].get("gap_adjusted_continuation_same_theme_peer_quality_overlay") is True
        for scenario in peer_quality_gap
    )
    assert all(
        scenario["strategies"]["momentum"].get("gap_adjusted_continuation_same_theme_peer_min_count", 0) >= 3
        for scenario in peer_quality_gap
    )
    assert all(
        scenario["strategies"]["momentum"].get("gap_adjusted_continuation_same_theme_peer_min_avg_rs", 0.0) >= 82.0
        for scenario in peer_quality_gap
    )
    medium_term_pullback = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_medium_term_leader_pullback"
    ]
    assert medium_term_pullback
    assert all(scenario["strategies"]["momentum"].get("medium_term_leader_pullback_overlay") is True for scenario in medium_term_pullback)
    assert all(
        scenario["strategies"]["momentum"].get("medium_term_leader_pullback_min_ret_100d_rank", 0.0) >= 0.60
        for scenario in medium_term_pullback
    )
    assert all(
        scenario["strategies"]["momentum"].get("medium_term_leader_pullback_max_ret_5d", 1.0) <= 0.015
        for scenario in medium_term_pullback
    )
    assert all(
        scenario["strategies"]["momentum"].get("medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 28.0
        for scenario in medium_term_pullback
    )
    assert all(
        scenario["strategies"]["momentum"].get("medium_term_leader_pullback_max_score_boost", 0.0) <= 2.0
        for scenario in medium_term_pullback
    )
    theme_tilt = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_theme_leader_tilt"]
    assert theme_tilt
    assert all(scenario["strategies"]["momentum"].get("theme_leader_tilt_overlay") is True for scenario in theme_tilt)
    assert all(scenario["strategies"]["momentum"].get("theme_leader_tilt_min_theme_peer_count", 0) >= 2 for scenario in theme_tilt)
    assert all(scenario["strategies"]["momentum"].get("theme_leader_tilt_min_within_theme_rank_pct", 0.0) >= 0.75 for scenario in theme_tilt)
    assert all(scenario["strategies"]["momentum"].get("theme_leader_tilt_max_score_boost", 0.0) <= 2.5 for scenario in theme_tilt)
    assert all(
        scenario["strategies"]["momentum"].get("theme_leader_tilt_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 30.0
        for scenario in theme_tilt
    )
    compound_leader = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_compound_leader_credit"]
    assert compound_leader
    assert all(scenario["strategies"]["momentum"].get("compound_leader_score_credit_overlay") is True for scenario in compound_leader)
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_min_126d_voladj_rank", 0.0) >= 0.75
        for scenario in compound_leader
    )
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_min_252d_voladj_rank", 0.0) >= 0.78
        for scenario in compound_leader
    )
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_max_ret_10d", 1.0) <= 0.05
        for scenario in compound_leader
    )
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 25.0
        for scenario in compound_leader
    )
    compound_leader_volume_confirmed = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_compound_leader_volume_confirmed"
    ]
    assert compound_leader_volume_confirmed
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_volume_confirmation_overlay") is True
        for scenario in compound_leader_volume_confirmed
    )
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_min_ret_100d_rank", 0.0) >= 0.65
        for scenario in compound_leader_volume_confirmed
    )
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_min_volume_expansion", 0.0) >= 1.0
        for scenario in compound_leader_volume_confirmed
    )
    compound_leader_boundary = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_compound_leader_boundary"
    ]
    assert compound_leader_boundary
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_boundary_overlay") is True
        for scenario in compound_leader_boundary
    )
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_rank_buffer_below", 1.0) <= 0.08
        for scenario in compound_leader_boundary
    )
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_rank_buffer_above", 1.0) <= 0.02
        for scenario in compound_leader_boundary
    )
    assert all(
        scenario["strategies"]["momentum"].get("compound_leader_score_credit_max_score_boost", 0.0) <= 1.5
        for scenario in compound_leader_boundary
    )
    boundary_compound_promotion = [
        scenario
        for scenario in scenarios
        if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_boundary_compound_pullback_promotion"
    ]
    assert boundary_compound_promotion
    assert all(
        scenario["strategies"]["momentum"].get("boundary_rank_promotion_overlay") is True
        for scenario in boundary_compound_promotion
    )
    assert all(
        scenario["strategies"]["momentum"].get("boundary_rank_promotion_compound_pullback_overlay") is True
        for scenario in boundary_compound_promotion
    )
    assert all(
        scenario["strategies"]["momentum"].get("boundary_rank_promotion_compound_pullback_min_ret_100d_rank", 0.0) >= 0.55
        for scenario in boundary_compound_promotion
    )
    assert all(
        scenario["strategies"]["momentum"].get("boundary_rank_promotion_compound_pullback_max_volume_expansion", 9.0) <= 1.10
        for scenario in boundary_compound_promotion
    )
    blended = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_tech_rs_blended"]
    assert blended
    assert all(scenario["strategies"]["momentum"].get("momentum_custom_final_score_weights") is True for scenario in blended)
    assert all(scenario["strategies"]["momentum"].get("momentum_custom_final_score_blend") == 0.65 for scenario in blended)
    blend90 = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_tech_rs_blend90"]
    assert blend90
    assert all(scenario["strategies"]["momentum"].get("momentum_custom_final_score_weights") is True for scenario in blend90)
    assert all(scenario["strategies"]["momentum"].get("momentum_custom_final_score_blend") == 0.90 for scenario in blend90)
    weighted = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_tech_rs_weighted"]
    assert weighted
    assert all(scenario["strategies"]["momentum"].get("momentum_custom_final_score_weights") is True for scenario in weighted)
    assert all(scenario["strategies"]["momentum"].get("momentum_final_score_technical_weight") == 0.42 for scenario in weighted)
    assert all(scenario["strategies"]["momentum"].get("momentum_final_score_relative_strength_weight") == 0.33 for scenario in weighted)
    assert all(scenario["strategies"]["momentum"].get("momentum_final_score_fundamental_weight") == 0.05 for scenario in weighted)
    assert all(scenario["strategies"]["momentum"].get("momentum_custom_final_score_blend") == 1.0 for scenario in weighted)
    reclaim = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_reclaim"]
    assert reclaim
    assert all(scenario["strategies"]["momentum"].get("short_quantile") == 0.0 for scenario in reclaim)
    assert all(scenario["strategies"]["momentum"].get("long_pullback_reclaim_overlay") is True for scenario in reclaim)
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reclaim_min_recent_below_ma20_pct", 0.0) >= 0.01
        for scenario in reclaim
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reclaim_max_above_ma20_pct", 1.0) <= 0.03
        for scenario in reclaim
    )
    substitution = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_same_theme_substitution"]
    assert substitution
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_same_theme_substitution_overlay") is True
        for scenario in substitution
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_same_theme_substitution_min_relative_strength_edge", 0.0) >= 4.0
        for scenario in substitution
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 25.0
        for scenario in substitution
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker", 1.0) <= 0.05
        for scenario in substitution
    )
    substitution_breadth = [
        scenario
        for scenario in scenarios
        if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_same_theme_substitution_breadth_guarded"
    ]
    assert substitution_breadth
    assert all(
        scenario["strategies"]["momentum"].get(
            "pullback_reset_same_theme_substitution_require_theme_breadth_not_deteriorating"
        )
        is True
        for scenario in substitution_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get(
            "pullback_reset_same_theme_substitution_theme_breadth_active_share_threshold",
            1.0,
        )
        <= 0.40
        for scenario in substitution_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get(
            "pullback_reset_same_theme_substitution_theme_breadth_shortfall_threshold",
            0.0,
        )
        >= 0.08
        for scenario in substitution_breadth
    )
    substitution_activation = [
        scenario
        for scenario in scenarios
        if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_same_theme_substitution_activation_guarded"
    ]
    assert substitution_activation
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_activation_support_overlay") is True
        for scenario in substitution_activation
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_same_theme_substitution_require_activation_support") is True
        for scenario in substitution_activation
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_same_theme_substitution_require_theme_breadth_not_deteriorating")
        is True
        for scenario in substitution_activation
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 35.0
        for scenario in reclaim
    )
    reclaim_reset = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_reclaim_reset_support"]
    assert reclaim_reset
    assert all(scenario["strategies"]["momentum"].get("long_pullback_reclaim_overlay") is True for scenario in reclaim_reset)
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_reclaim_support_overlay") is True
        for scenario in reclaim_reset
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_min_reclaim_score", 0.0) >= 20.0
        for scenario in reclaim_reset
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_reclaim_score_rank_credit", 0.0) <= 0.10
        for scenario in reclaim_reset
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_require_theme_breadth_not_deteriorating") is True
        for scenario in reclaim_reset
    )
    reset_breadth = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_reset_breadth_expansion"]
    assert reset_breadth
    assert all(scenario["strategies"]["momentum"].get("short_quantile") == 0.0 for scenario in reset_breadth)
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_require_theme_breadth_expansion") is True
        for scenario in reset_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_theme_breadth_min_active_share", 0.0) >= 0.45
        for scenario in reset_breadth
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_theme_breadth_expansion_threshold", 0.0) >= 0.03
        for scenario in reset_breadth
    )
    reset_rs = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_rs_accel_reset"]
    assert reset_rs
    assert all(scenario["strategies"]["momentum"].get("short_quantile") == 0.0 for scenario in reset_rs)
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_require_rs_acceleration") is True
        for scenario in reset_rs
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_min_current_rs_score", 0.0) >= 78.0
        for scenario in reset_rs
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reset_inclusion_min_rs_acceleration", 0.0) >= 5.0
        for scenario in reset_rs
    )
    disciplined = [scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_breadth_disciplined"]
    assert disciplined
    assert all(scenario["strategies"]["momentum"].get("short_quantile") == 0.0 for scenario in disciplined)
    assert all(scenario["strategies"]["momentum"].get("long_reentry_discipline_overlay") is True for scenario in disciplined)
    assert all(
        scenario["strategies"]["momentum"].get("reentry_discipline_min_pullback_volume_reset_score_exemption", 0.0) > 0.0
        for scenario in disciplined
    )
    assert all(scenario["strategies"]["momentum"].get("reentry_discipline_require_theme_deterioration") is True for scenario in disciplined)
    assert all(
        scenario["strategies"]["momentum"].get("reentry_discipline_theme_score_deterioration_threshold", 100.0) <= 64.0
        for scenario in disciplined
    )
    assert all(
        scenario["strategies"]["momentum"].get("reentry_discipline_require_theme_breadth_deterioration") is True
        for scenario in disciplined
    )
    assert all(
        scenario["strategies"]["momentum"].get("reentry_discipline_theme_breadth_active_share_threshold", 1.0) <= 0.40
        for scenario in disciplined
    )
    assert all(
        scenario["strategies"]["momentum"].get("reentry_discipline_theme_breadth_shortfall_threshold", 0.0) >= 0.05
        for scenario in disciplined
    )
    assert any(scenario["strategies"]["momentum"].get("long_pullback_entry_overlay") is True for scenario in scenarios)
    assert any(scenario["strategies"]["momentum"].get("long_pullback_volume_contraction_overlay") is True for scenario in scenarios)
    assert any(scenario["strategies"]["momentum"].get("long_reentry_discipline_overlay") is True for scenario in scenarios)
    assert any(scenario["strategies"]["momentum"].get("short_rebound_avoidance_overlay") is True for scenario in scenarios)
    assert all(scenario["strategies"]["momentum"].get("pullback_volume_contraction_max_score_boost", 0.0) <= 3.0 for scenario in long_only)
    assert all(scenario["strategies"]["momentum"].get("pullback_volume_contraction_require_benchmark_risk_on", True) is True for scenario in long_only)
    assert all(scenario["strategies"]["momentum"].get("pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker", 35.0) <= 35.0 for scenario in long_only)
    assert all(scenario["objective"].get("trailing_one_year_weight", 0.0) > 0.0 for scenario in scenarios)
    assert all(scenario["objective"].get("max_drawdown_penalty", 1.0) <= 0.10 for scenario in scenarios)
    leader_addon = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_addon"]
    assert leader_addon
    assert all(scenario["portfolio"].get("leader_addon_overlay") is True for scenario in leader_addon)
    assert all(scenario["portfolio"].get("leader_addon_multiplier", 1.0) <= 1.25 for scenario in leader_addon)
    assert all(scenario["portfolio"].get("leader_addon_max_names_per_date", 0) <= 5 for scenario in leader_addon)
    gap_quality_addon = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_addon_gap_quality"]
    assert gap_quality_addon
    assert all(scenario["portfolio"].get("leader_addon_overlay") is True for scenario in gap_quality_addon)
    assert all(scenario["portfolio"].get("leader_addon_gap_quality_support_overlay") is True for scenario in gap_quality_addon)
    assert all(scenario["portfolio"].get("leader_addon_require_controlled_pullback") is True for scenario in gap_quality_addon)
    assert all(scenario["portfolio"].get("leader_addon_gap_quality_max_symbol_gap_risk_score", 100.0) <= 18.0 for scenario in gap_quality_addon)
    assert all(scenario["portfolio"].get("leader_addon_gap_quality_same_theme_min_share", 0.0) >= 0.34 for scenario in gap_quality_addon)
    pullback_addon = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_addon_pullback"]
    assert pullback_addon
    assert all(scenario["portfolio"].get("leader_addon_overlay") is True for scenario in pullback_addon)
    assert all(scenario["portfolio"].get("leader_addon_require_controlled_pullback") is True for scenario in pullback_addon)
    assert all(scenario["portfolio"].get("leader_addon_min_relative_strength_score", 0.0) >= 90.0 for scenario in pullback_addon)
    assert all(scenario["portfolio"].get("leader_addon_min_mom_return", 0.0) >= 0.20 for scenario in pullback_addon)
    assert all(scenario["portfolio"].get("leader_addon_max_names_per_date", 99) <= 2 for scenario in pullback_addon)
    assert all(scenario["portfolio"].get("leader_addon_max_above_ma20_pct_circuit_breaker", 1.0) <= 0.08 for scenario in pullback_addon)
    theme_peer_addon = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_addon_theme_peer"]
    assert theme_peer_addon
    assert all(scenario["portfolio"].get("leader_addon_overlay") is True for scenario in theme_peer_addon)
    assert all(scenario["portfolio"].get("leader_addon_require_same_theme_peer_support") is True for scenario in theme_peer_addon)
    assert all(scenario["portfolio"].get("leader_addon_same_theme_peer_min_count", 0) >= 2 for scenario in theme_peer_addon)
    assert all(scenario["portfolio"].get("leader_addon_same_theme_peer_min_share", 0.0) >= 0.25 for scenario in theme_peer_addon)
    assert all(scenario["portfolio"].get("leader_addon_max_names_per_date", 99) <= 3 for scenario in theme_peer_addon)
    hold_buffer = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_hold_buffer"]
    assert hold_buffer
    assert all(scenario["portfolio"].get("leader_hold_buffer_overlay") is True for scenario in hold_buffer)
    assert all(scenario["portfolio"].get("leader_hold_buffer_max_days", 0) <= 3 for scenario in hold_buffer)
    assert all(scenario["portfolio"].get("leader_hold_buffer_max_weight", 1.0) <= 0.02 for scenario in hold_buffer)
    assert all(scenario["portfolio"].get("target_gross_exposure") == 1.50 for scenario in hold_buffer)
    assert all(scenario["portfolio"].get("target_net_exposure") == 1.50 for scenario in hold_buffer)
    assert all(
        scenario["portfolio"].get("leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker", 100.0) <= 30.0
        for scenario in hold_buffer
    )
    delayed_exit = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_delayed_exit_rank_credit"]
    assert delayed_exit
    assert all(scenario["portfolio"].get("leader_delayed_exit_overlay") is True for scenario in delayed_exit)
    assert all(scenario["portfolio"].get("leader_delayed_exit_min_relative_strength_score", 0.0) >= 90.0 for scenario in delayed_exit)
    assert all(scenario["portfolio"].get("leader_delayed_exit_min_mom_return", 0.0) >= 0.15 for scenario in delayed_exit)
    assert all(scenario["portfolio"].get("leader_delayed_exit_retain_fraction", 0.0) >= 0.80 for scenario in delayed_exit)
    assert all(scenario["portfolio"].get("leader_delayed_exit_max_weight", 1.0) <= 0.08 for scenario in delayed_exit)
    assert all(scenario["portfolio"].get("leader_delayed_exit_stability_overlay") is True for scenario in delayed_exit)
    assert all(scenario["portfolio"].get("leader_delayed_exit_max_final_score_drop", 999.0) <= 6.0 for scenario in delayed_exit)
    assert all(
        scenario["portfolio"].get("leader_delayed_exit_max_relative_strength_drop", 999.0) <= 5.0
        for scenario in delayed_exit
    )
    persistence = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_persistence"]
    assert persistence
    assert all(scenario["portfolio"].get("leader_persistence_overlay") is True for scenario in persistence)
    assert all(scenario["portfolio"].get("leader_persistence_min_theme_peer_count", 0) >= 2 for scenario in persistence)
    assert all(scenario["portfolio"].get("leader_persistence_min_theme_peer_share", 0.0) >= 0.40 for scenario in persistence)
    assert all(scenario["portfolio"].get("leader_persistence_max_weight_bonus", 1.0) <= 0.015 for scenario in persistence)
    stable_persistence = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_persistence_stable"]
    assert stable_persistence
    assert all(scenario["portfolio"].get("leader_persistence_stability_overlay") is True for scenario in stable_persistence)


def test_cycle_return_profile_includes_reclaim_same_theme_gap_confirmed_scenario() -> None:
    scenarios = default_research_scenarios("cycle_return")
    reclaim_gap_confirmed = [
        scenario for scenario in scenarios if scenario["strategy_profile"] == "ai_cycle_pullback_long_only_reclaim_same_theme_gap_confirmed"
    ]

    assert reclaim_gap_confirmed
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reclaim_same_theme_gap_confirmation_overlay") is True
        for scenario in reclaim_gap_confirmed
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reclaim_same_theme_gap_confirmation_min_peer_count", 0) >= 3
        for scenario in reclaim_gap_confirmed
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reclaim_same_theme_gap_confirmation_min_active_share", 0.0) >= 0.50
        for scenario in reclaim_gap_confirmed
    )
    assert all(
        scenario["strategies"]["momentum"].get("pullback_reclaim_same_theme_gap_confirmation_max_avg_gap_risk_score", 100.0) <= 22.0
        for scenario in reclaim_gap_confirmed
    )


def test_filter_scenarios_can_target_late_cycle_return_branches() -> None:
    scenarios = default_research_scenarios("cycle_return")

    filtered = _filter_scenarios(scenarios, ["same_theme_substitution"])

    assert filtered
    assert len(filtered) < len(scenarios)
    assert all("same_theme_substitution" in scenario["strategy_profile"] for scenario in filtered)
    stable_persistence = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_persistence_stable"]
    assert stable_persistence
    assert all(scenario["portfolio"].get("leader_persistence_min_desired_weight_fraction", 0.0) >= 0.40 for scenario in stable_persistence)
    assert all(scenario["portfolio"].get("leader_persistence_max_final_score_drop", 100.0) <= 6.0 for scenario in stable_persistence)
    assert all(scenario["portfolio"].get("leader_persistence_max_relative_strength_drop", 100.0) <= 5.0 for scenario in stable_persistence)
    compound_relief = [scenario for scenario in scenarios if scenario["constraint_profile"] == "cycle_long_only_leader_crowding_relief"]
    assert compound_relief
    assert all(scenario["portfolio"].get("target_gross_exposure") == 1.50 for scenario in compound_relief)
    assert all(scenario["portfolio"].get("target_net_exposure") == 1.50 for scenario in compound_relief)
    assert all(scenario["regime"].get("leader_crowding_relief_overlay") is True for scenario in compound_relief)
    assert all(scenario["regime"].get("leader_crowding_relief_max_bucket_weight_restore") == 0.025 for scenario in compound_relief)
    relief = [scenario for scenario in scenarios if scenario["regime"].get("leader_crowding_relief_overlay")]
    assert relief
    assert all(scenario["regime"].get("leader_crowding_relief_max_bucket_weight_restore", 0.0) > 0.0 for scenario in relief)


def test_best_config_from_results_preserves_short_term_volume_peer_confirmation_params() -> None:
    results = pd.DataFrame(
        [
            {
                "rebalance": "daily_throttled",
                "throttle_min_weight_change": 0.015,
                "throttle_min_score_change": 5.0,
                "throttle_min_new_weight": 0.005,
                "target_gross_exposure": 1.5,
                "target_net_exposure": 1.5,
                "max_total_positions": 55,
                "min_target_weight": 0.0075,
                "daily_turnover_cap": 0.50,
                "max_adv_participation": 0.05,
                "market_impact_bps_per_1pct_adv": 1.0,
                "min_trade_notional": 250,
                "min_holding_days": 1,
                "max_drawdown_reduce_exposure": 0.18,
                "max_drawdown_cash_mode": 0.30,
                "drawdown_reduction_multiplier": 0.35,
                "drawdown_reset": "quarterly",
                "regime_exposure_overlay": True,
                "risk_off_exposure_multiplier": 0.0,
                "partial_risk_on_exposure_multiplier": 0.5,
                "partial_exposure_regime_score": 45.0,
                "full_exposure_regime_score": 60.0,
                "analog_momentum_overlay": False,
                "analog_inactive_exposure_multiplier": 1.0,
                "analog_active_exposure_multiplier": 1.0,
                "analog_quantile": 0.80,
                "momentum_weight": 0.80,
                "breakout_weight": 0.15,
                "trend_weight": 0.05,
                "lookback_returns": 21,
                "skip_recent_days": 3,
                "long_quantile": 0.25,
                "short_quantile": 0.0,
                "momentum_max_positions": 35,
                "long_pullback_entry_overlay": True,
                "pullback_min_relative_strength_score": 72.0,
                "pullback_min_theme_score": 58.0,
                "pullback_min_drawdown_from_high": 0.03,
                "pullback_max_drawdown_from_high": 0.18,
                "pullback_max_above_ma20_pct": 0.07,
                "pullback_max_score_boost": 5.0,
                "short_term_volume_tilt_overlay": True,
                "short_term_volume_tilt_max_ret_5d": 0.01,
                "short_term_volume_tilt_min_volume_expansion": 1.15,
                "short_term_volume_tilt_min_relative_strength_score": 60.0,
                "short_term_volume_tilt_min_theme_score": 55.0,
                "short_term_volume_tilt_max_above_ma20_pct": 0.06,
                "short_term_volume_tilt_require_benchmark_risk_on": True,
                "short_term_volume_tilt_require_theme_active": False,
                "short_term_volume_tilt_require_price_above_ma50": True,
                "short_term_volume_tilt_max_event_risk_score_circuit_breaker": 18.0,
                "short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker": 28.0,
                "short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker": 0.0,
                "short_term_volume_tilt_min_adx_circuit_breaker": 18.0,
                "short_term_volume_tilt_max_score_boost": 2.5,
                "short_term_volume_tilt_same_theme_peer_confirmation_overlay": True,
                "short_term_volume_tilt_same_theme_peer_min_count": 2,
                "short_term_volume_tilt_same_theme_peer_min_share": 0.50,
                "short_term_volume_tilt_same_theme_peer_min_avg_mom_return": 0.10,
                "short_term_volume_tilt_same_theme_peer_volume_substitution_overlay": True,
                "short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion": 0.70,
                "short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall": 0.35,
                "short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale": 0.60,
                "short_term_volume_tilt_non_risk_on_exception_overlay": True,
                "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count": 3,
                "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share": 0.60,
                "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return": 0.14,
                "short_term_volume_tilt_non_risk_on_exception_score_boost_scale": 0.50,
                "gap_adjusted_continuation_overlay": True,
                "gap_adjusted_continuation_min_drawdown_from_high": 0.03,
                "gap_adjusted_continuation_max_drawdown_from_high": 0.16,
                "gap_adjusted_continuation_max_above_ma20_pct": 0.05,
                "gap_adjusted_continuation_min_relative_strength_score": 72.0,
                "gap_adjusted_continuation_min_theme_score": 58.0,
                "gap_adjusted_continuation_min_adx_circuit_breaker": 18.0,
                "gap_adjusted_continuation_max_event_risk_score_circuit_breaker": 18.0,
                "gap_adjusted_continuation_max_gap_risk_score_circuit_breaker": 28.0,
                "gap_adjusted_continuation_benign_gap_risk_score": 14.0,
                "gap_adjusted_continuation_gap_risk_lookback_days": 10,
                "gap_adjusted_continuation_min_gap_risk_improvement": 3.0,
                "gap_adjusted_continuation_min_volume_expansion": 0.60,
                "gap_adjusted_continuation_max_volume_expansion": 1.35,
                "gap_adjusted_continuation_min_benchmark_ret63d_circuit_breaker": 0.0,
                "gap_adjusted_continuation_require_benchmark_risk_on": True,
                "gap_adjusted_continuation_require_theme_active": False,
                "gap_adjusted_continuation_same_theme_peer_quality_overlay": True,
                "gap_adjusted_continuation_same_theme_peer_min_count": 3,
                "gap_adjusted_continuation_same_theme_peer_min_share": 0.45,
                "gap_adjusted_continuation_same_theme_peer_min_avg_rs": 82.0,
                "gap_adjusted_continuation_max_score_boost": 2.5,
                "medium_term_leader_pullback_overlay": True,
                "medium_term_leader_pullback_min_ret_100d_rank": 0.60,
                "medium_term_leader_pullback_max_ret_5d": 0.015,
                "medium_term_leader_pullback_max_ret_10d": 0.03,
                "medium_term_leader_pullback_min_drawdown_from_high": 0.02,
                "medium_term_leader_pullback_max_drawdown_from_high": 0.16,
                "medium_term_leader_pullback_max_above_ma20_pct": 0.04,
                "medium_term_leader_pullback_max_volume_expansion": 1.15,
                "medium_term_leader_pullback_min_relative_strength_score": 68.0,
                "medium_term_leader_pullback_min_theme_score": 56.0,
                "medium_term_leader_pullback_min_mom_return": 0.05,
                "medium_term_leader_pullback_require_benchmark_risk_on": True,
                "medium_term_leader_pullback_require_theme_active": False,
                "medium_term_leader_pullback_require_price_above_ma50": True,
                "medium_term_leader_pullback_max_event_risk_score_circuit_breaker": 18.0,
                "medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker": 28.0,
                "medium_term_leader_pullback_min_benchmark_ret63d_circuit_breaker": 0.0,
                "medium_term_leader_pullback_min_adx_circuit_breaker": 18.0,
                "medium_term_leader_pullback_max_score_boost": 2.0,
                "boundary_rank_promotion_overlay": True,
                "boundary_rank_promotion_rank_buffer_below": 0.08,
                "boundary_rank_promotion_min_relative_strength_score": 84.0,
                "boundary_rank_promotion_min_theme_score": 64.0,
                "boundary_rank_promotion_min_technical_score": 54.0,
                "boundary_rank_promotion_min_mom_return": 0.05,
                "boundary_rank_promotion_min_drawdown_from_high": 0.02,
                "boundary_rank_promotion_max_drawdown_from_high": 0.16,
                "boundary_rank_promotion_max_above_ma20_pct": 0.04,
                "boundary_rank_promotion_max_final_score_deficit": 4.0,
                "boundary_rank_promotion_max_promotions_per_date": 1,
                "boundary_rank_promotion_promoted_score_step": 0.12,
                "boundary_rank_promotion_require_benchmark_risk_on": True,
                "boundary_rank_promotion_require_theme_active": True,
                "boundary_rank_promotion_require_price_above_ma50": True,
                "boundary_rank_promotion_max_event_risk_score_circuit_breaker": 18.0,
                "boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker": 28.0,
                "boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker": 0.0,
                "boundary_rank_promotion_min_adx_circuit_breaker": 18.0,
                "boundary_rank_promotion_compound_pullback_overlay": True,
                "boundary_rank_promotion_compound_pullback_min_ret_100d_rank": 0.55,
                "boundary_rank_promotion_compound_pullback_min_252d_voladj_rank": 0.55,
                "boundary_rank_promotion_compound_pullback_max_ret_5d": 0.015,
                "boundary_rank_promotion_compound_pullback_max_ret_10d": 0.03,
                "boundary_rank_promotion_compound_pullback_max_volume_expansion": 1.10,
                "breakout_window": 20,
                "breakout_max_holding_days": 30,
                "trend_ma_fast": 20,
                "trend_ma_slow": 100,
                "validation_objective": 0.20,
                "train_objective": 0.15,
                "validation_pass": True,
            }
        ]
    )

    config = _best_config_from_results(AppConfig(), results)

    assert config is not None
    assert config.strategies["momentum"]["short_term_volume_tilt_overlay"] is True
    assert config.strategies["momentum"]["short_term_volume_tilt_same_theme_peer_confirmation_overlay"] is True
    assert config.strategies["momentum"]["short_term_volume_tilt_same_theme_peer_min_count"] == 2
    assert config.strategies["momentum"]["short_term_volume_tilt_same_theme_peer_min_share"] == 0.50
    assert config.strategies["momentum"]["short_term_volume_tilt_same_theme_peer_min_avg_mom_return"] == 0.10
    assert config.strategies["momentum"]["short_term_volume_tilt_same_theme_peer_volume_substitution_overlay"] is True
    assert config.strategies["momentum"]["short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion"] == 0.70
    assert config.strategies["momentum"]["short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall"] == 0.35
    assert config.strategies["momentum"]["short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale"] == 0.60
    assert config.strategies["momentum"]["short_term_volume_tilt_non_risk_on_exception_overlay"] is True
    assert config.strategies["momentum"]["short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count"] == 3
    assert config.strategies["momentum"]["short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share"] == 0.60
    assert config.strategies["momentum"]["short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return"] == 0.14
    assert config.strategies["momentum"]["short_term_volume_tilt_non_risk_on_exception_score_boost_scale"] == 0.50
    assert config.strategies["momentum"]["gap_adjusted_continuation_overlay"] is True
    assert config.strategies["momentum"]["gap_adjusted_continuation_max_gap_risk_score_circuit_breaker"] == 28.0
    assert config.strategies["momentum"]["gap_adjusted_continuation_benign_gap_risk_score"] == 14.0
    assert config.strategies["momentum"]["gap_adjusted_continuation_gap_risk_lookback_days"] == 10
    assert config.strategies["momentum"]["gap_adjusted_continuation_same_theme_peer_quality_overlay"] is True
    assert config.strategies["momentum"]["gap_adjusted_continuation_same_theme_peer_min_count"] == 3
    assert config.strategies["momentum"]["gap_adjusted_continuation_same_theme_peer_min_share"] == 0.45
    assert config.strategies["momentum"]["gap_adjusted_continuation_same_theme_peer_min_avg_rs"] == 82.0
    assert config.strategies["momentum"]["gap_adjusted_continuation_max_score_boost"] == 2.5
    assert config.strategies["momentum"]["medium_term_leader_pullback_overlay"] is True
    assert config.strategies["momentum"]["medium_term_leader_pullback_min_ret_100d_rank"] == 0.60
    assert config.strategies["momentum"]["medium_term_leader_pullback_max_ret_5d"] == 0.015
    assert config.strategies["momentum"]["medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker"] == 28.0
    assert config.strategies["momentum"]["medium_term_leader_pullback_max_score_boost"] == 2.0
    assert config.strategies["momentum"]["boundary_rank_promotion_overlay"] is True
    assert config.strategies["momentum"]["boundary_rank_promotion_compound_pullback_overlay"] is True
    assert config.strategies["momentum"]["boundary_rank_promotion_compound_pullback_min_ret_100d_rank"] == 0.55
    assert config.strategies["momentum"]["boundary_rank_promotion_compound_pullback_min_252d_voladj_rank"] == 0.55
    assert config.strategies["momentum"]["boundary_rank_promotion_compound_pullback_max_volume_expansion"] == 1.10


def test_return_objective_can_prioritize_trailing_return_with_guardrails() -> None:
    steady = {
        "cagr": 0.20,
        "total_return": 0.40,
        "trailing_one_year_return": 0.20,
        "max_drawdown": -0.10,
        "average_turnover": 0.10,
    }
    accelerating = {
        "cagr": 0.18,
        "total_return": 0.38,
        "trailing_one_year_return": 0.50,
        "max_drawdown": -0.12,
        "average_turnover": 0.12,
    }

    assert objective_score(accelerating, cagr_weight=0.30, total_return_weight=0.25, trailing_one_year_weight=0.45) > objective_score(
        steady,
        cagr_weight=0.30,
        total_return_weight=0.25,
        trailing_one_year_weight=0.45,
    )


def test_sharpe15_profile_uses_quality_gates_without_forward_data() -> None:
    scenarios = default_research_scenarios("sharpe15")
    assert scenarios
    assert {scenario["constraint_profile"] for scenario in scenarios} >= {
        "quality_guarded",
        "quality_offense",
        "quality_high_conviction",
    }
    assert any(scenario["strategies"]["momentum"].get("require_long_theme_active") is True for scenario in scenarios)
    assert any(scenario["strategies"]["momentum"].get("short_only_when_benchmark_risk_off") is True for scenario in scenarios)
    assert all(scenario["strategies"]["momentum"].get("short_quantile", 0.0) <= 0.05 for scenario in scenarios)
    assert all(scenario["regime"].get("exposure_overlay") is True for scenario in scenarios)


def test_best_config_from_results_uses_validation_winner() -> None:
    results = pd.DataFrame(
        [
            {
                "rebalance": "weekly",
                "throttle_min_weight_change": 0.02,
                "throttle_min_score_change": 7.0,
                "throttle_min_new_weight": 0.005,
                "target_gross_exposure": 1.0,
                "target_net_exposure": 0.5,
                "max_total_positions": 40,
                "min_target_weight": 0.01,
                "daily_turnover_cap": 0.20,
                "max_adv_participation": 0.05,
                "market_impact_bps_per_1pct_adv": 1.0,
                "min_trade_notional": 250,
                "min_holding_days": 3,
                "max_drawdown_reduce_exposure": 0.12,
                "max_drawdown_cash_mode": 0.20,
                "drawdown_reduction_multiplier": 0.5,
                "drawdown_reset": "yearly",
                "regime_exposure_overlay": False,
                "risk_off_exposure_multiplier": 1.0,
                "partial_risk_on_exposure_multiplier": 0.75,
                "partial_exposure_regime_score": 45.0,
                "full_exposure_regime_score": 60.0,
                "analog_momentum_overlay": False,
                "analog_inactive_exposure_multiplier": 1.0,
                "analog_active_exposure_multiplier": 1.0,
                "analog_quantile": 0.80,
                "momentum_weight": 0.5,
                "breakout_weight": 0.3,
                "trend_weight": 0.2,
                "lookback_returns": 63,
                "skip_recent_days": 5,
                "long_quantile": 0.25,
                "short_quantile": 0.25,
                "momentum_max_positions": 20,
                "breakout_window": 55,
                "breakout_max_holding_days": 60,
                "trend_ma_fast": 50,
                "trend_ma_slow": 200,
                "validation_objective": 0.01,
                "train_objective": 0.02,
                "validation_pass": True,
            },
            {
                "rebalance": "daily_throttled",
                "throttle_min_weight_change": 0.03,
                "throttle_min_score_change": 10.0,
                "throttle_min_new_weight": 0.01,
                "target_gross_exposure": 1.25,
                "target_net_exposure": 0.9,
                "max_total_positions": 80,
                "min_target_weight": 0.005,
                "leader_addon_overlay": True,
                "leader_addon_multiplier": 1.25,
                "leader_addon_min_final_score": 70.0,
                "leader_addon_min_relative_strength_score": 82.0,
                "leader_addon_min_theme_score": 65.0,
                "leader_addon_min_mom_return": 0.10,
                "leader_addon_max_names_per_date": 5,
                "leader_addon_require_benchmark_risk_on": True,
                "leader_addon_require_controlled_pullback": True,
                "leader_addon_require_same_theme_peer_support": True,
                "leader_addon_same_theme_peer_min_count": 2,
                "leader_addon_same_theme_peer_min_share": 0.25,
                "leader_addon_min_drawdown_from_high": 0.02,
                "leader_addon_max_drawdown_from_high": 0.15,
                "leader_addon_max_above_ma20_pct_circuit_breaker": 0.08,
                "leader_addon_max_event_risk_score_circuit_breaker": 20.0,
                "leader_addon_max_overnight_gap_risk_score_circuit_breaker": 30.0,
                "leader_hold_buffer_overlay": True,
                "leader_hold_buffer_min_final_score": 68.0,
                "leader_hold_buffer_min_relative_strength_score": 85.0,
                "leader_hold_buffer_min_theme_score": 62.0,
                "leader_hold_buffer_weight_fraction": 0.50,
                "leader_hold_buffer_max_weight": 0.02,
                "leader_hold_buffer_max_days": 3,
                "leader_hold_buffer_require_benchmark_risk_on": True,
                "leader_hold_buffer_max_event_risk_score_circuit_breaker": 20.0,
                "leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker": 30.0,
                "leader_delayed_exit_overlay": True,
                "leader_delayed_exit_min_final_score": 70.0,
                "leader_delayed_exit_min_relative_strength_score": 90.0,
                "leader_delayed_exit_min_theme_score": 65.0,
                "leader_delayed_exit_min_mom_return": 0.15,
                "leader_delayed_exit_max_drawdown_from_high": 0.18,
                "leader_delayed_exit_retain_fraction": 0.85,
                "leader_delayed_exit_max_weight": 0.08,
                "leader_delayed_exit_max_days": 3,
                "leader_delayed_exit_require_benchmark_risk_on": True,
                "leader_delayed_exit_require_theme_active": False,
                "leader_delayed_exit_max_event_risk_score_circuit_breaker": 20.0,
                "leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
                "leader_delayed_exit_stability_overlay": True,
                "leader_delayed_exit_max_final_score_drop": 6.0,
                "leader_delayed_exit_max_relative_strength_drop": 5.0,
                "leader_persistence_overlay": True,
                "leader_persistence_min_final_score": 68.0,
                "leader_persistence_min_relative_strength_score": 88.0,
                "leader_persistence_min_theme_score": 62.0,
                "leader_persistence_min_mom_return": 0.15,
                "leader_persistence_min_theme_peer_count": 2,
                "leader_persistence_min_theme_peer_share": 0.40,
                "leader_persistence_retain_fraction_of_cut": 0.35,
                "leader_persistence_max_weight_bonus": 0.015,
                "leader_persistence_require_theme_active": True,
                "leader_persistence_require_benchmark_risk_on": True,
                "leader_persistence_max_event_risk_score_circuit_breaker": 20.0,
                "leader_persistence_max_overnight_gap_risk_score_circuit_breaker": 30.0,
                "leader_persistence_stability_overlay": True,
                "leader_persistence_min_desired_weight_fraction": 0.40,
                "leader_persistence_max_final_score_drop": 6.0,
                "leader_persistence_max_relative_strength_drop": 5.0,
                "leader_persistence_peer_strength_reward_overlay": True,
                "leader_persistence_peer_strength_reward_min_theme_peer_count": 3,
                "leader_persistence_peer_strength_reward_min_theme_peer_share": 0.55,
                "leader_persistence_peer_strength_reward_min_relative_strength_score": 92.0,
                "leader_persistence_peer_strength_reward_retain_fraction_boost": 0.20,
                "leader_persistence_peer_strength_reward_max_weight_bonus": 0.0075,
                "leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker": 0.14,
                "daily_turnover_cap": 0.50,
                "max_adv_participation": 0.10,
                "market_impact_bps_per_1pct_adv": 0.75,
                "min_trade_notional": 100,
                "min_holding_days": 0,
                "max_drawdown_reduce_exposure": 0.22,
                "max_drawdown_cash_mode": 0.35,
                "drawdown_reduction_multiplier": 0.25,
                "drawdown_reset": "quarterly",
                "regime_exposure_overlay": True,
                "risk_off_exposure_multiplier": 0.2,
                "partial_risk_on_exposure_multiplier": 0.65,
                "partial_exposure_regime_score": 45.0,
                "full_exposure_regime_score": 60.0,
                "analog_momentum_overlay": True,
                "analog_inactive_exposure_multiplier": 0.50,
                "analog_active_exposure_multiplier": 1.0,
                "analog_quantile": 0.80,
                "leader_crowding_relief_overlay": True,
                "leader_crowding_relief_min_relative_strength_score": 88.0,
                "leader_crowding_relief_min_theme_score": 60.0,
                "leader_crowding_relief_weight": 0.35,
                "leader_crowding_relief_max_names_per_bucket": 2,
                "leader_crowding_relief_max_weight_per_symbol": 0.0125,
                "leader_crowding_relief_max_bucket_weight_restore": 0.025,
                "leader_crowding_relief_require_benchmark_risk_on": True,
                "leader_crowding_relief_controlled_pullback_overlay": True,
                "leader_crowding_relief_controlled_pullback_min_final_score": 64.0,
                "leader_crowding_relief_controlled_pullback_min_distance_from_high": 0.04,
                "leader_crowding_relief_controlled_pullback_max_distance_from_high": 0.18,
                "leader_crowding_relief_controlled_pullback_max_above_ma20_pct": 0.06,
                "leader_crowding_relief_controlled_pullback_max_mom_return": 0.12,
                "momentum_weight": 0.6,
                "breakout_weight": 0.25,
                "trend_weight": 0.15,
                "lookback_returns": 126,
                "skip_recent_days": 10,
                "long_quantile": 0.2,
                "short_quantile": 0.2,
                "momentum_max_positions": 15,
                "momentum_custom_final_score_weights": True,
                "momentum_final_score_technical_weight": 0.42,
                "momentum_final_score_relative_strength_weight": 0.33,
                "momentum_final_score_theme_weight": 0.20,
                "momentum_final_score_fundamental_weight": 0.05,
                "momentum_final_score_event_risk_penalty": 0.05,
                "momentum_custom_final_score_blend": 0.65,
                "momentum_custom_score_controlled_entry_overlay": True,
                "momentum_custom_score_min_relative_strength_score": 62.0,
                "momentum_custom_score_min_theme_score": 58.0,
                "momentum_custom_score_max_above_ma20_pct": 0.10,
                "momentum_custom_score_max_event_risk_score_circuit_breaker": 20.0,
                "momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker": 30.0,
                "momentum_custom_score_min_benchmark_ret63d_circuit_breaker": 0.0,
                "momentum_custom_score_require_benchmark_risk_on": True,
                "momentum_custom_score_require_theme_active": True,
                "momentum_custom_score_require_price_above_ma50": True,
                "compound_leader_score_credit_overlay": True,
                "compound_leader_score_credit_min_126d_voladj_rank": 0.78,
                "compound_leader_score_credit_min_252d_voladj_rank": 0.80,
                "compound_leader_score_credit_min_final_score": 60.0,
                "compound_leader_score_credit_min_relative_strength_score": 80.0,
                "compound_leader_score_credit_min_theme_score": 60.0,
                "compound_leader_score_credit_min_mom_return": 0.05,
                "compound_leader_score_credit_min_theme_peer_count": 2,
                "compound_leader_score_credit_max_ret_10d": 0.04,
                "compound_leader_score_credit_max_drawdown_from_high": 0.12,
                "compound_leader_score_credit_max_above_ma20_pct": 0.08,
                "compound_leader_score_credit_boundary_overlay": True,
                "compound_leader_score_credit_rank_buffer_below": 0.08,
                "compound_leader_score_credit_rank_buffer_above": 0.02,
                "compound_leader_score_credit_require_benchmark_risk_on": True,
                "compound_leader_score_credit_require_theme_active": True,
                "compound_leader_score_credit_require_price_above_ma50": True,
                "compound_leader_score_credit_max_event_risk_score_circuit_breaker": 15.0,
                "compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker": 25.0,
                "compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
                "compound_leader_score_credit_min_adx_circuit_breaker": 20.0,
                "compound_leader_score_credit_max_score_boost": 1.75,
                "compound_leader_score_credit_volume_confirmation_overlay": True,
                "compound_leader_score_credit_min_ret_100d_rank": 0.65,
                "compound_leader_score_credit_min_volume_expansion": 1.0,
                "long_pullback_entry_overlay": True,
                "pullback_min_relative_strength_score": 70.0,
                "pullback_min_theme_score": 60.0,
                "pullback_min_drawdown_from_high": 0.03,
                "pullback_max_drawdown_from_high": 0.18,
                "pullback_max_above_ma20_pct": 0.08,
                "pullback_max_score_boost": 5.0,
                "long_pullback_volume_contraction_overlay": True,
                "pullback_volume_contraction_max_volume_expansion": 0.90,
                "pullback_volume_contraction_min_drawdown_from_high": 0.03,
                "pullback_volume_contraction_max_drawdown_from_high": 0.16,
                "pullback_volume_contraction_max_above_ma20_pct": 0.03,
                "pullback_volume_contraction_min_relative_strength_score": 68.0,
                "pullback_volume_contraction_min_theme_score": 58.0,
                "pullback_volume_contraction_require_benchmark_risk_on": True,
                "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 35.0,
                "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
                "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
                "pullback_volume_contraction_max_score_boost": 3.0,
                "long_pullback_reset_inclusion_overlay": True,
                "pullback_reset_inclusion_min_score": 50.0,
                "pullback_reset_inclusion_max_names_per_date": 2,
                "pullback_reset_inclusion_min_final_score": 52.0,
                "pullback_reset_inclusion_min_score_rank": 0.55,
                "pullback_reset_inclusion_activation_support_overlay": True,
                "pullback_reset_inclusion_activation_min_score": 35.0,
                "pullback_reset_inclusion_activation_score_rank_credit": 0.08,
                "pullback_reset_inclusion_require_rs_acceleration": True,
                "pullback_reset_inclusion_rs_acceleration_lookback_days": 10,
                "pullback_reset_inclusion_min_current_rs_score": 78.0,
                "pullback_reset_inclusion_min_rs_acceleration": 5.0,
                "pullback_reset_same_theme_substitution_overlay": True,
                "pullback_reset_same_theme_substitution_min_relative_strength_edge": 4.0,
                "pullback_reset_same_theme_substitution_min_theme_score_edge": 2.0,
                "pullback_reset_same_theme_substitution_min_reset_score": 25.0,
                "pullback_reset_same_theme_substitution_min_reclaim_score": 25.0,
                "pullback_reset_same_theme_substitution_min_theme_strength_delta_score": 0.0,
                "pullback_reset_same_theme_substitution_max_final_score_deficit": 4.0,
                "pullback_reset_same_theme_substitution_max_score_rank_gap": 0.08,
                "pullback_reset_same_theme_substitution_max_event_risk_score_circuit_breaker": 20.0,
                "pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker": 25.0,
                "pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker": 0.045,
                "pullback_reset_same_theme_substitution_max_promotions_per_date": 1,
                "pullback_reset_same_theme_substitution_require_theme_active": True,
                "pullback_reset_same_theme_substitution_require_activation_support": True,
                "pullback_reset_same_theme_substitution_require_theme_breadth_not_deteriorating": True,
                "pullback_reset_same_theme_substitution_theme_breadth_lookback_days": 10,
                "pullback_reset_same_theme_substitution_theme_breadth_active_share_threshold": 0.40,
                "pullback_reset_same_theme_substitution_theme_breadth_shortfall_threshold": 0.08,
                "boundary_rs_theme_credit_overlay": True,
                "boundary_rs_theme_credit_rank_buffer_below": 0.10,
                "boundary_rs_theme_credit_rank_buffer_above": 0.03,
                "boundary_rs_theme_credit_min_relative_strength_score": 88.0,
                "boundary_rs_theme_credit_min_theme_score": 70.0,
                "boundary_rs_theme_credit_min_technical_score": 54.0,
                "boundary_rs_theme_credit_min_mom_return": 0.0,
                "boundary_rs_theme_credit_min_drawdown_from_high": 0.0,
                "boundary_rs_theme_credit_max_drawdown_from_high": 0.12,
                "boundary_rs_theme_credit_max_above_ma20_pct": 0.08,
                "boundary_rs_theme_credit_require_benchmark_risk_on": True,
                "boundary_rs_theme_credit_require_theme_active": True,
                "boundary_rs_theme_credit_require_price_above_ma50": True,
                "boundary_rs_theme_credit_max_event_risk_score_circuit_breaker": 15.0,
                "boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker": 25.0,
                "boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
                "boundary_rs_theme_credit_min_adx_circuit_breaker": 18.0,
                "boundary_rs_theme_credit_max_score_boost": 1.75,
                "exit_quality_rank_credit_overlay": True,
                "exit_quality_rank_credit_rank_buffer_below": 0.07,
                "exit_quality_rank_credit_rank_buffer_above": 0.01,
                "exit_quality_rank_credit_min_final_score": 68.0,
                "exit_quality_rank_credit_min_relative_strength_score": 88.0,
                "exit_quality_rank_credit_min_theme_score": 62.0,
                "exit_quality_rank_credit_min_technical_score": 60.0,
                "exit_quality_rank_credit_min_mom_return": 0.12,
                "exit_quality_rank_credit_max_drawdown_from_high": 0.18,
                "exit_quality_rank_credit_max_above_ma20_pct": 0.08,
                "exit_quality_rank_credit_require_benchmark_risk_on": True,
                "exit_quality_rank_credit_require_theme_active": False,
                "exit_quality_rank_credit_require_price_above_ma50": True,
                "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": 20.0,
                "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
                "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
                "exit_quality_rank_credit_min_adx_circuit_breaker": 18.0,
                "exit_quality_rank_credit_max_score_rank_credit": 0.04,
                "exit_quality_rank_credit_reset_support_overlay": True,
                "exit_quality_rank_credit_reset_support_min_reset_score": 20.0,
                "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days": 10,
                "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold": 0.42,
                "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold": 0.08,
                "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share": 0.40,
                "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion": False,
                "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold": 0.02,
                "exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay": True,
                "exit_quality_rank_credit_reset_support_same_theme_peer_min_count": 2,
                "exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay": True,
                "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share": 0.34,
                "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count": 2,
                "exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay": True,
                "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs": 84.0,
                "exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay": True,
                "exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct": 0.78,
                "exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap": 6.0,
                "exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count": 2,
                "reentry_discipline_require_theme_breadth_deterioration": True,
                "reentry_discipline_theme_breadth_lookback_days": 10,
                "reentry_discipline_theme_breadth_active_share_threshold": 0.38,
                "reentry_discipline_theme_breadth_shortfall_threshold": 0.05,
                "breakout_window": 100,
                "breakout_max_holding_days": 90,
                "trend_ma_fast": 20,
                "trend_ma_slow": 100,
                "validation_objective": 0.05,
                "train_objective": 0.01,
                "validation_pass": True,
            },
        ]
    )
    config = _best_config_from_results(AppConfig(), results)
    assert config is not None
    assert config.portfolio.rebalance == "daily_throttled"
    assert config.portfolio.throttle_min_weight_change == 0.03
    assert config.portfolio.target_gross_exposure == 1.25
    assert config.portfolio.max_total_positions == 80
    assert config.portfolio.leader_addon_overlay is True
    assert config.portfolio.leader_addon_multiplier == 1.25
    assert config.portfolio.leader_addon_min_relative_strength_score == 82.0
    assert config.portfolio.leader_addon_max_names_per_date == 5
    assert config.portfolio.leader_addon_require_controlled_pullback is True
    assert config.portfolio.leader_addon_require_same_theme_peer_support is True
    assert config.portfolio.leader_addon_same_theme_peer_min_count == 2
    assert config.portfolio.leader_addon_same_theme_peer_min_share == 0.25
    assert config.portfolio.leader_addon_min_drawdown_from_high == 0.02
    assert config.portfolio.leader_addon_max_drawdown_from_high == 0.15
    assert config.portfolio.leader_addon_max_above_ma20_pct_circuit_breaker == 0.08
    assert config.portfolio.leader_addon_max_event_risk_score_circuit_breaker == 20.0
    assert config.portfolio.leader_addon_max_overnight_gap_risk_score_circuit_breaker == 30.0
    assert config.portfolio.leader_hold_buffer_overlay is True
    assert config.portfolio.leader_hold_buffer_min_final_score == 68.0
    assert config.portfolio.leader_hold_buffer_min_relative_strength_score == 85.0
    assert config.portfolio.leader_hold_buffer_min_theme_score == 62.0
    assert config.portfolio.leader_hold_buffer_max_days == 3
    assert config.portfolio.leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker == 30.0
    assert config.portfolio.leader_delayed_exit_overlay is True
    assert config.portfolio.leader_delayed_exit_min_relative_strength_score == 90.0
    assert config.portfolio.leader_delayed_exit_min_mom_return == 0.15
    assert config.portfolio.leader_delayed_exit_retain_fraction == 0.85
    assert config.portfolio.leader_delayed_exit_max_weight == 0.08
    assert config.portfolio.leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker == 30.0
    assert config.portfolio.leader_delayed_exit_stability_overlay is True
    assert config.portfolio.leader_delayed_exit_max_final_score_drop == 6.0
    assert config.portfolio.leader_delayed_exit_max_relative_strength_drop == 5.0
    assert config.portfolio.leader_persistence_overlay is True
    assert config.portfolio.leader_persistence_min_relative_strength_score == 88.0
    assert config.portfolio.leader_persistence_min_theme_peer_count == 2
    assert config.portfolio.leader_persistence_max_weight_bonus == 0.015
    assert config.portfolio.leader_persistence_stability_overlay is True
    assert config.portfolio.leader_persistence_min_desired_weight_fraction == 0.40
    assert config.portfolio.leader_persistence_max_final_score_drop == 6.0
    assert config.portfolio.leader_persistence_max_relative_strength_drop == 5.0
    assert config.portfolio.leader_persistence_peer_strength_reward_overlay is True
    assert config.portfolio.leader_persistence_peer_strength_reward_min_theme_peer_count == 3
    assert config.portfolio.leader_persistence_peer_strength_reward_min_theme_peer_share == 0.55
    assert config.portfolio.leader_persistence_peer_strength_reward_min_relative_strength_score == 92.0
    assert config.portfolio.leader_persistence_peer_strength_reward_retain_fraction_boost == 0.20
    assert config.portfolio.leader_persistence_peer_strength_reward_max_weight_bonus == 0.0075
    assert config.portfolio.leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker == 0.14
    assert config.execution.daily_turnover_cap == 0.50
    assert config.risk.min_holding_days == 0
    assert config.risk.max_drawdown_reduce_exposure == 0.22
    assert config.risk.max_drawdown_cash_mode == 0.35
    assert config.risk.drawdown_reduction_multiplier == 0.25
    assert config.risk.drawdown_reset == "quarterly"
    assert config.regime.exposure_overlay is True
    assert config.regime.risk_off_exposure_multiplier == 0.2
    assert config.regime.analog_momentum_overlay is True
    assert config.regime.analog_inactive_exposure_multiplier == 0.50
    assert config.regime.leader_crowding_relief_overlay is True
    assert config.regime.leader_crowding_relief_min_relative_strength_score == 88.0
    assert config.regime.leader_crowding_relief_max_bucket_weight_restore == 0.025
    assert config.regime.leader_crowding_relief_controlled_pullback_overlay is True
    assert config.regime.leader_crowding_relief_controlled_pullback_min_distance_from_high == 0.04
    assert config.regime.leader_crowding_relief_controlled_pullback_max_mom_return == 0.12
    assert config.strategies["momentum"]["long_pullback_entry_overlay"] is True
    assert config.strategies["momentum"]["pullback_min_relative_strength_score"] == 70.0
    assert config.strategies["momentum"]["long_pullback_volume_contraction_overlay"] is True
    assert config.strategies["momentum"]["pullback_volume_contraction_max_volume_expansion"] == 0.90
    assert config.strategies["momentum"]["pullback_volume_contraction_require_benchmark_risk_on"] is True
    assert config.strategies["momentum"]["pullback_volume_contraction_max_score_boost"] == 3.0
    assert config.strategies["momentum"]["long_pullback_reset_inclusion_overlay"] is True
    assert config.strategies["momentum"]["pullback_reset_inclusion_max_names_per_date"] == 2
    assert config.strategies["momentum"]["pullback_reset_inclusion_min_score_rank"] == 0.55
    assert config.strategies["momentum"]["pullback_reset_inclusion_activation_support_overlay"] is True
    assert config.strategies["momentum"]["pullback_reset_inclusion_activation_min_score"] == 35.0
    assert config.strategies["momentum"]["pullback_reset_inclusion_activation_score_rank_credit"] == 0.08
    assert config.strategies["momentum"]["pullback_reset_inclusion_require_rs_acceleration"] is True
    assert config.strategies["momentum"]["pullback_reset_inclusion_rs_acceleration_lookback_days"] == 10
    assert config.strategies["momentum"]["pullback_reset_inclusion_min_current_rs_score"] == 78.0
    assert config.strategies["momentum"]["pullback_reset_inclusion_min_rs_acceleration"] == 5.0
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_overlay"] is True
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_min_relative_strength_edge"] == 4.0
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_max_score_rank_gap"] == 0.08
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker"] == 0.045
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_max_promotions_per_date"] == 1
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_require_theme_active"] is True
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_require_activation_support"] is True
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_require_theme_breadth_not_deteriorating"] is True
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_theme_breadth_lookback_days"] == 10
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_theme_breadth_active_share_threshold"] == 0.40
    assert config.strategies["momentum"]["pullback_reset_same_theme_substitution_theme_breadth_shortfall_threshold"] == 0.08
    assert config.strategies["momentum"]["compound_leader_score_credit_overlay"] is True
    assert config.strategies["momentum"]["compound_leader_score_credit_min_126d_voladj_rank"] == 0.78
    assert config.strategies["momentum"]["compound_leader_score_credit_min_252d_voladj_rank"] == 0.80
    assert config.strategies["momentum"]["compound_leader_score_credit_min_theme_peer_count"] == 2
    assert config.strategies["momentum"]["compound_leader_score_credit_max_ret_10d"] == 0.04
    assert config.strategies["momentum"]["compound_leader_score_credit_boundary_overlay"] is True
    assert config.strategies["momentum"]["compound_leader_score_credit_rank_buffer_below"] == 0.08
    assert config.strategies["momentum"]["compound_leader_score_credit_rank_buffer_above"] == 0.02
    assert config.strategies["momentum"]["compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker"] == 25.0
    assert config.strategies["momentum"]["compound_leader_score_credit_max_score_boost"] == 1.75
    assert config.strategies["momentum"]["compound_leader_score_credit_volume_confirmation_overlay"] is True
    assert config.strategies["momentum"]["compound_leader_score_credit_min_ret_100d_rank"] == 0.65
    assert config.strategies["momentum"]["compound_leader_score_credit_min_volume_expansion"] == 1.0
    assert config.strategies["momentum"]["boundary_rs_theme_credit_overlay"] is True
    assert config.strategies["momentum"]["boundary_rs_theme_credit_rank_buffer_below"] == 0.10
    assert config.strategies["momentum"]["boundary_rs_theme_credit_min_relative_strength_score"] == 88.0
    assert config.strategies["momentum"]["boundary_rs_theme_credit_min_theme_score"] == 70.0
    assert config.strategies["momentum"]["boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker"] == 25.0
    assert config.strategies["momentum"]["boundary_rs_theme_credit_max_score_boost"] == 1.75
    assert config.strategies["momentum"]["exit_quality_rank_credit_overlay"] is True
    assert config.strategies["momentum"]["exit_quality_rank_credit_min_relative_strength_score"] == 88.0
    assert config.strategies["momentum"]["exit_quality_rank_credit_max_score_rank_credit"] == 0.04
    assert config.strategies["momentum"]["exit_quality_rank_credit_require_theme_active"] is False
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_overlay"] is True
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_min_reset_score"] == 20.0
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_theme_breadth_lookback_days"] == 10
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold"] == 0.42
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold"] == 0.08
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_theme_breadth_min_active_share"] == 0.40
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_require_theme_breadth_expansion"] is False
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold"] == 0.02
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay"] is True
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_peer_min_count"] == 2
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay"] is True
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share"] == 0.34
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count"] == 2
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay"] is True
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs"] == 84.0
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay"] is True
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct"] == 0.78
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap"] == 6.0
    assert config.strategies["momentum"]["exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count"] == 2
    assert config.strategies["momentum"]["reentry_discipline_require_theme_breadth_deterioration"] is True
    assert config.strategies["momentum"]["reentry_discipline_theme_breadth_lookback_days"] == 10
    assert config.strategies["momentum"]["reentry_discipline_theme_breadth_active_share_threshold"] == 0.38
    assert config.strategies["momentum"]["reentry_discipline_theme_breadth_shortfall_threshold"] == 0.05
    assert config.strategies["momentum"]["momentum_custom_final_score_weights"] is True
    assert config.strategies["momentum"]["momentum_final_score_technical_weight"] == 0.42
    assert config.strategies["momentum"]["momentum_final_score_relative_strength_weight"] == 0.33
    assert config.strategies["momentum"]["momentum_custom_final_score_blend"] == 0.65
    assert config.strategies["momentum"]["momentum_custom_score_controlled_entry_overlay"] is True
    assert config.strategies["momentum"]["momentum_custom_score_min_relative_strength_score"] == 62.0
    assert config.strategies["momentum"]["momentum_custom_score_min_theme_score"] == 58.0
    assert config.strategies["momentum"]["momentum_custom_score_max_above_ma20_pct"] == 0.10
    assert config.strategies["momentum"]["momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker"] == 30.0
    assert config.strategies["momentum"]["lookback_returns"] == 126
    assert config.strategies["breakout"]["breakout_window"] == 100


def test_cycle_return_promotion_guard_blocks_weak_full_period_short_drag() -> None:
    warnings: list[str] = []
    rows = pd.DataFrame(
        [
            {
                "scenario": "weak_high_gross",
                "validation_pass": True,
                "validation_objective": 0.30,
                "train_objective": 0.20,
                "full_total_return": 0.35,
                "full_sharpe": 0.40,
                "validation_cagr": 0.25,
                "full_short_contribution": -50_000.0,
            },
            {
                "scenario": "durable_long_only",
                "validation_pass": True,
                "validation_objective": 0.20,
                "train_objective": 0.15,
                "full_total_return": 1.50,
                "full_sharpe": 0.90,
                "validation_cagr": 0.20,
                "full_short_contribution": 0.0,
            },
        ]
    )

    guarded = _apply_return_promotion_guards(rows, warnings, profile="cycle_return", initial_capital=100_000)

    weak = guarded[guarded["scenario"] == "weak_high_gross"].iloc[0]
    durable = guarded[guarded["scenario"] == "durable_long_only"].iloc[0]
    assert bool(weak["validation_pass_before_promotion_guard"]) is True
    assert bool(weak["validation_pass"]) is False
    assert "validation_sharpe_below_guard" in weak["promotion_guard_reason"]
    assert "short_drag_above_guard" in weak["promotion_guard_reason"]
    assert bool(durable["promotion_guard_pass"]) is True
    assert bool(durable["validation_pass"]) is True
    assert warnings
