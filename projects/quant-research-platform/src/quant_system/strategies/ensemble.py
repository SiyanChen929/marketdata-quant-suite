"""Ensemble signal composition."""

from __future__ import annotations

import pandas as pd


def combine_strategy_signals(signals: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Combine strategy signals into a single signal table."""

    frames: list[pd.DataFrame] = []
    for name, frame in signals.items():
        if frame.empty:
            continue
        weight = float(weights.get(name, 0.0))
        final_score = frame["final_score"] if "final_score" in frame else 50.0
        temp = frame.assign(
            strategy=name,
            strategy_weight=weight,
            weighted_signal=frame["signal"] * weight,
            weighted_score=final_score * weight,
        )
        frames.append(temp)
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True, copy=False).copy()
    aggregations = {
        "signal": ("weighted_signal", "sum"),
        "final_score": ("weighted_score", "sum"),
    }
    for column, reducer in {
        "technical_score": "mean",
        "relative_strength_score": "mean",
        "theme_score": "mean",
        "fundamental_score": "mean",
        "event_risk_score": "mean",
        "reason_for_entry": lambda values: " | ".join(str(v) for v in values if str(v)),
        "score_decomposition": lambda values: " | ".join(str(v) for v in values if str(v)),
    }.items():
        if column in stacked.columns:
            aggregations[column] = (column, reducer)
    for column, reducer in {
        "days_to_earnings": "min",
        "stop_loss_atr": "mean",
        "trailing_stop_atr": "mean",
        "take_profit_r_multiple": "mean",
        "max_holding_days": "max",
        "fundamental_factor_coverage": "mean",
        "fundamental_missing_group_count": "mean",
        "fundamental_data_quality": "first",
        "moat_score": "mean",
        "growth_score": "mean",
        "quality_score": "mean",
        "balance_sheet_score": "mean",
        "valuation_score": "mean",
        "revision_score": "mean",
        "mom_return": "mean",
        "mom_risk_adjusted": "mean",
        "mom_distance_high": "mean",
        "mom_volume_expansion": "mean",
        "distance_to_high_252": "mean",
        "distance_to_prior_high_252": "mean",
        "breakout_component": "mean",
        "volume_component": "mean",
        "rs_component": "mean",
        "retest_component": "mean",
        "retest_distance_ma_pct": "mean",
        "trend_price_vs_slow_pct": "mean",
        "trend_adx": "mean",
        "mean_reversion_pressure": "mean",
        "rsi_14": "mean",
        "zscore_20": "mean",
        "pullback_short_term_reset_score": "mean",
        "pullback_short_term_reset_boost": "mean",
        "pullback_short_term_reset_ret_5d": "mean",
        "pullback_short_term_reset_ret_10d": "mean",
        "pullback_short_term_reset_blocked": "max",
        "short_term_volume_tilt_score": "mean",
        "short_term_volume_tilt_boost": "mean",
        "short_term_volume_tilt_ret_5d": "mean",
        "short_term_volume_tilt_volume_expansion": "mean",
        "short_term_volume_tilt_candidate": "max",
        "short_term_volume_tilt_circuit_ok": "max",
        "short_term_volume_tilt_peer_confirmed": "max",
        "short_term_volume_tilt_blocked": "max",
        "short_term_volume_tilt_block_reason": "first",
        "short_term_volume_tilt_same_theme_peer_count": "mean",
        "short_term_volume_tilt_same_theme_peer_confirm_share": "mean",
        "short_term_volume_tilt_same_theme_peer_avg_mom_return": "mean",
        "boundary_rs_theme_credit_score": "mean",
        "boundary_rs_theme_credit_boost": "mean",
        "boundary_rs_theme_credit_base_rank": "mean",
        "boundary_rs_theme_credit_blocked": "max",
        "boundary_rs_theme_credit_eligible": "max",
        "exit_quality_rank_credit_score": "mean",
        "exit_quality_rank_credit": "mean",
        "exit_quality_base_rank": "mean",
        "exit_quality_effective_score_rank": "mean",
        "exit_quality_rank_credit_blocked": "max",
        "exit_quality_rank_credit_eligible": "max",
        "exit_quality_theme_support_score": "mean",
        "exit_quality_theme_support_eligible": "max",
        "gap_adjusted_continuation_score": "mean",
        "gap_adjusted_continuation_boost": "mean",
        "gap_adjusted_continuation_gap_risk": "mean",
        "gap_adjusted_continuation_gap_improvement": "mean",
        "gap_adjusted_continuation_eligible": "max",
        "gap_adjusted_continuation_blocked": "max",
        "pullback_reclaim_score": "mean",
        "pullback_reclaim_boost": "mean",
        "pullback_reclaim_recent_below_ma20_pct": "mean",
        "pullback_reclaim_blocked": "max",
        "pullback_reset_reclaim_score": "mean",
        "pullback_reset_reclaim_supported": "max",
        "pullback_reset_reclaim_score_rank_credit": "mean",
        "pullback_reset_theme_breadth_safe": "max",
        "pullback_reset_theme_breadth_shortfall": "mean",
        "theme_strength_delta_score": "mean",
        "theme_strength_delta_boost": "mean",
        "theme_strength_score_prior": "mean",
        "theme_strength_delta": "mean",
        "theme_strength_peer_rs_prior": "mean",
        "theme_strength_peer_rs_delta": "mean",
        "theme_strength_peer_count": "mean",
        "theme_strength_delta_eligible": "max",
        "theme_breadth_acceleration_score": "mean",
        "theme_breadth_acceleration_boost": "mean",
        "theme_breadth_active_share_prior": "mean",
        "theme_breadth_acceleration": "mean",
        "theme_breadth_peer_count": "mean",
        "theme_breadth_acceleration_eligible": "max",
        "theme_leader_tilt_score": "mean",
        "theme_leader_tilt_boost": "mean",
        "theme_leader_tilt_theme_rank_pct": "mean",
        "theme_leader_tilt_theme_peer_count": "mean",
        "theme_leader_tilt_eligible": "max",
    }.items():
        if column in stacked.columns:
            aggregations[column] = (column, reducer)
    grouped = _aggregate_signal_columns(stacked, aggregations)
    grouped["signal"] = grouped["signal"].clip(-1.0, 1.0)
    return grouped


def _aggregate_signal_columns(stacked: pd.DataFrame, aggregations: dict, batch_size: int = 24) -> pd.DataFrame:
    """Aggregate wide signal frames without pandas fragmented-frame warnings."""

    keys = ["date", "symbol"]
    items = list(aggregations.items())
    grouped: pd.DataFrame | None = None
    for start in range(0, len(items), batch_size):
        chunk = dict(items[start : start + batch_size])
        part = stacked.groupby(keys, as_index=False, sort=False).agg(**chunk)
        if grouped is None:
            grouped = part
        else:
            grouped = grouped.merge(part, on=keys, how="left", sort=False)
    if grouped is None:
        return stacked[keys].drop_duplicates().reset_index(drop=True)
    return grouped
