"""Cross-sectional momentum long/short strategy."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_system.strategies.base import Strategy, StrategyContext, attach_context_scores, event_entries_allowed, final_score, output_columns

LOGGER = logging.getLogger(__name__)


class CrossSectionalMomentumLongShort(Strategy):
    """Rank securities by delayed, risk-adjusted momentum."""

    name = "momentum"

    def generate_signals(self, features: pd.DataFrame, context: StrategyContext) -> pd.DataFrame:
        """Generate long/short momentum signals.

        Momentum uses close-to-close information through date t and is executed
        by the engine no earlier than t+1.
        """

        params = self.params
        lookback = int(params.get("lookback_returns", 63))
        skip = int(params.get("skip_recent_days", 5))
        long_q = float(params.get("long_quantile", 0.2))
        short_q = float(params.get("short_quantile", 0.2))
        max_positions = int(params.get("max_positions", 50))
        out = features.sort_values(["symbol", "date"]).copy()
        grouped = out.groupby("symbol", group_keys=False)
        delayed_price = grouped["adj_close"].shift(skip)
        base_price = grouped["adj_close"].shift(lookback + skip)
        out["mom_return"] = delayed_price / base_price - 1.0
        out["mom_vol"] = grouped["return_1d"].transform(lambda s: s.shift(skip).rolling(lookback, min_periods=max(10, lookback // 3)).std())
        out["mom_risk_adjusted"] = out["mom_return"] / out["mom_vol"].replace(0, np.nan)
        out["mom_distance_high"] = out.get("distance_to_prior_high_252", out["distance_to_high_252"])
        out["mom_volume_expansion"] = out["volume_expansion"].replace([np.inf, -np.inf], np.nan)
        by_date = out.groupby("date")
        out["return_rank"] = by_date["mom_return"].rank(pct=True).fillna(0.5)
        out["risk_rank"] = by_date["mom_risk_adjusted"].rank(pct=True).fillna(0.5)
        out["high_rank"] = by_date["mom_distance_high"].rank(pct=True).fillna(0.5)
        out["volume_rank"] = by_date["mom_volume_expansion"].rank(pct=True).fillna(0.5)
        out["technical_score"] = (
            0.45 * out["return_rank"]
            + 0.30 * out["risk_rank"]
            + 0.15 * out["high_rank"]
            + 0.10 * out["volume_rank"]
        ) * 100.0
        out["relative_strength_score"] = out.get("rs_63d", pd.Series(index=out.index, dtype=float)).groupby(out["date"]).rank(pct=True) * 100.0
        out = attach_context_scores(out, context)
        out = _attach_benchmark_regime(out, context.benchmark)
        out["final_score"] = _momentum_final_score(out, params)
        out = _apply_theme_strength_delta_overlay(out, params)
        out = _apply_theme_breadth_acceleration_overlay(out, params)
        out = _apply_theme_leader_tilt_overlay(out, params)
        out = _apply_compound_leader_score_credit_overlay(out, params)
        out = _apply_medium_term_leader_pullback_overlay(out, params)
        out = _apply_post_earnings_drift_overlay(out, params)
        out = _apply_boundary_rs_theme_credit_overlay(out, params)
        out = _apply_pullback_entry_overlay(out, params)
        out = _apply_pullback_volume_contraction_overlay(out, params)
        out = _apply_gap_adjusted_continuation_overlay(out, params)
        out = _apply_pullback_short_term_reset_overlay(out, params)
        out = _apply_short_term_volume_tilt_overlay(out, params)
        out = _apply_pullback_reclaim_overlay(out, params)
        out = _apply_reentry_discipline_overlay(out, params)
        valid = out.dropna(subset=["technical_score"]).copy()
        valid = _apply_boundary_rank_promotion_overlay(valid, params, long_q)
        valid["score_rank"] = valid.groupby("date")["final_score"].rank(pct=True)
        valid = _apply_exit_quality_rank_credit_overlay(valid, params, long_q)
        valid["signal"] = 0.0
        long_rank = pd.to_numeric(valid.get("exit_quality_effective_score_rank", valid["score_rank"]), errors="coerce").fillna(valid["score_rank"])
        long_mask = (long_rank >= 1.0 - long_q) & (valid["fundamental_score"].fillna(50.0) >= float(params.get("long_min_fundamental_score", 40)))
        long_mask &= _long_quality_gate(valid, params)
        reset_include_mask = _long_pullback_reset_inclusion_gate(valid, params)
        event_allowed = event_entries_allowed(valid, params)
        substitution_promote_mask, substitution_demote_mask = _apply_pullback_reset_same_theme_substitution_overlay(
            valid,
            params,
            long_mask,
            reset_include_mask,
            event_allowed,
        )
        final_long_mask = ((long_mask | reset_include_mask) & ~substitution_demote_mask) | substitution_promote_mask
        short_mask = (valid["score_rank"] <= short_q) & (valid["fundamental_score"].fillna(50.0) <= float(params.get("short_max_fundamental_score", 60)))
        short_mask &= _short_quality_gate(valid, params)
        valid.loc[final_long_mask & event_allowed, "signal"] = 1.0
        valid.loc[short_mask & event_allowed, "signal"] = -1.0
        valid["momentum_gate_reason"] = valid.apply(lambda row: _gate_reason(row, params), axis=1)
        valid["score_decomposition"] = valid.apply(
            lambda row: (
                f"return={_fmt_pct(row.get('mom_return'))}; "
                f"risk_adj={_fmt_num(row.get('mom_risk_adjusted'))}; "
                f"prior_high_gap={_fmt_pct(row.get('mom_distance_high'))}; "
                f"volume_exp={_fmt_num(row.get('mom_volume_expansion'))}; "
                f"custom_blend={_fmt_num(row.get('momentum_custom_score_effective_blend'))}; "
                f"custom_gate={int(bool(row.get('momentum_custom_score_controlled_entry_eligible', False)))}; "
                f"boundary_credit={_fmt_num(row.get('boundary_rs_theme_credit_score'))}; "
                f"boundary_rank={_fmt_num(row.get('boundary_rs_theme_credit_base_rank'))}; "
                f"boundary_block={int(bool(row.get('boundary_rs_theme_credit_blocked', False)))}; "
                f"boundary_promo_score={_fmt_num(row.get('boundary_rank_promotion_score'))}; "
                f"boundary_promo_rank={_fmt_num(row.get('boundary_rank_promotion_base_rank'))}; "
                f"boundary_promo_gap={_fmt_num(row.get('boundary_rank_promotion_score_gap'))}; "
                f"boundary_promo={int(bool(row.get('boundary_rank_promotion_promoted', False)))}; "
                f"boundary_promo_block={int(bool(row.get('boundary_rank_promotion_blocked', False)))}; "
                f"boundary_promo_compound={int(bool(row.get('boundary_rank_promotion_compound_pullback_eligible', False)))}; "
                f"boundary_promo_ret100_rank={_fmt_num(row.get('boundary_rank_promotion_compound_pullback_ret_100d_rank'))}; "
                f"boundary_promo_252_rank={_fmt_num(row.get('boundary_rank_promotion_compound_pullback_252d_voladj_rank'))}; "
                f"boundary_promo_ret5={_fmt_pct(row.get('boundary_rank_promotion_compound_pullback_ret_5d'))}; "
                f"boundary_promo_ret10={_fmt_pct(row.get('boundary_rank_promotion_compound_pullback_ret_10d'))}; "
                f"boundary_promo_volume={_fmt_num(row.get('boundary_rank_promotion_compound_pullback_volume_expansion'))}; "
                f"exit_quality_credit={_fmt_num(row.get('exit_quality_rank_credit_score'))}; "
                f"exit_quality_rank={_fmt_num(row.get('exit_quality_base_rank'))}; "
                f"exit_quality_effective_rank={_fmt_num(row.get('exit_quality_effective_score_rank'))}; "
                f"exit_quality_block={int(bool(row.get('exit_quality_rank_credit_blocked', False)))}; "
                f"exit_quality_theme_support={_fmt_num(row.get('exit_quality_theme_support_score'))}; "
                f"exit_quality_theme_gate={int(bool(row.get('exit_quality_theme_support_eligible', False)))}; "
                f"exit_quality_reset_support={_fmt_num(row.get('exit_quality_reset_support_score'))}; "
                f"exit_quality_reset_gate={int(bool(row.get('exit_quality_reset_support_eligible', False)))}; "
                f"exit_quality_reset_breadth={_fmt_num(row.get('exit_quality_reset_breadth_active_share'))}; "
                f"exit_quality_reset_peer_reset_share={_fmt_num(row.get('exit_quality_reset_breadth_same_theme_peer_reset_share'))}; "
                f"exit_quality_reset_peer_reset_count={int(row.get('exit_quality_reset_breadth_same_theme_peer_reset_count', 0) or 0)}; "
                f"exit_quality_reset_peer_reset_rs_mean={_fmt_num(row.get('exit_quality_reset_breadth_same_theme_peer_reset_rs_mean'))}; "
                f"exit_quality_reset_leader_rank={_fmt_num(row.get('exit_quality_reset_same_theme_leader_rank_pct'))}; "
                f"exit_quality_reset_leader_gap={_fmt_num(row.get('exit_quality_reset_same_theme_leader_score_gap'))}; "
                f"exit_quality_reset_leader_gate={int(bool(row.get('exit_quality_reset_same_theme_leader_guard_eligible', False)))}; "
                f"pullback_entry={_fmt_num(row.get('pullback_entry_score'))}; "
                f"pullback_volume_reset={_fmt_num(row.get('pullback_volume_contraction_score'))}; "
                f"gap_continuation={_fmt_num(row.get('gap_adjusted_continuation_score'))}; "
                f"gap_continuation_risk={_fmt_num(row.get('gap_adjusted_continuation_gap_risk'))}; "
                f"gap_continuation_improve={_fmt_num(row.get('gap_adjusted_continuation_gap_improvement'))}; "
                f"gap_continuation_peer_count={int(row.get('gap_adjusted_continuation_same_theme_peer_count', 0) or 0)}; "
                f"gap_continuation_peer_share={_fmt_num(row.get('gap_adjusted_continuation_same_theme_peer_share'))}; "
                f"gap_continuation_peer_rs={_fmt_num(row.get('gap_adjusted_continuation_same_theme_peer_avg_rs'))}; "
                f"gap_continuation_peer_gate={int(bool(row.get('gap_adjusted_continuation_same_theme_peer_quality_ok', False)))}; "
                f"gap_continuation_block={int(bool(row.get('gap_adjusted_continuation_blocked', False)))}; "
                f"pullback_short_term_reset={_fmt_num(row.get('pullback_short_term_reset_score'))}; "
                f"pullback_ret5={_fmt_pct(row.get('pullback_short_term_reset_ret_5d'))}; "
                f"pullback_ret10={_fmt_pct(row.get('pullback_short_term_reset_ret_10d'))}; "
                f"pullback_short_term_block={int(bool(row.get('pullback_short_term_reset_blocked', False)))}; "
                f"short_term_volume_tilt={_fmt_num(row.get('short_term_volume_tilt_score'))}; "
                f"short_term_volume_ret5={_fmt_pct(row.get('short_term_volume_tilt_ret_5d'))}; "
                f"short_term_volume={_fmt_num(row.get('short_term_volume_tilt_volume_expansion'))}; "
                f"short_term_volume_block={int(bool(row.get('short_term_volume_tilt_blocked', False)))}; "
                f"short_term_volume_mode={row.get('short_term_volume_tilt_activation_mode', 'off')}; "
                f"pullback_reclaim={_fmt_num(row.get('pullback_reclaim_score'))}; "
                f"reclaim_recent_below={_fmt_pct(row.get('pullback_reclaim_recent_below_ma20_pct'))}; "
                f"reclaim_theme_gap_ok={int(bool(row.get('pullback_reclaim_same_theme_gap_confirmation_ok', False)))}; "
                f"reclaim_theme_peers={int(row.get('pullback_reclaim_same_theme_peer_count', 0) or 0)}; "
                f"reclaim_theme_share={_fmt_num(row.get('pullback_reclaim_same_theme_active_share'))}; "
                f"reclaim_theme_gap={_fmt_num(row.get('pullback_reclaim_same_theme_avg_gap_risk'))}; "
                f"reclaim_block={int(bool(row.get('pullback_reclaim_blocked', False)))}; "
                f"reset_include={int(bool(row.get('pullback_reset_inclusion', False)))}; "
                f"reset_breadth_exp={_fmt_num(row.get('pullback_reset_theme_breadth_expansion'))}; "
                f"reset_breadth_gate={int(bool(row.get('pullback_reset_theme_breadth_expanding', False)))}; "
                f"reset_rs_accel={_fmt_num(row.get('pullback_reset_rs_acceleration'))}; "
                f"reset_rs_gate={int(bool(row.get('pullback_reset_rs_accelerating', False)))}; "
                f"reset_theme_strength={_fmt_num(row.get('pullback_reset_theme_strength_delta_score'))}; "
                f"reset_theme_strength_gate={int(bool(row.get('pullback_reset_theme_strength_supported', False)))}; "
                f"reset_rank_credit={_fmt_num(row.get('pullback_reset_theme_strength_score_rank_credit'))}; "
                f"reset_activation={_fmt_num(row.get('pullback_reset_activation_score'))}; "
                f"reset_activation_gate={int(bool(row.get('pullback_reset_activation_supported', False)))}; "
                f"reset_activation_credit={_fmt_num(row.get('pullback_reset_activation_score_rank_credit'))}; "
                f"reset_reclaim={_fmt_num(row.get('pullback_reset_reclaim_score'))}; "
                f"reset_reclaim_gate={int(bool(row.get('pullback_reset_reclaim_supported', False)))}; "
                f"reset_reclaim_credit={_fmt_num(row.get('pullback_reset_reclaim_score_rank_credit'))}; "
                f"reset_breadth_safe={int(bool(row.get('pullback_reset_theme_breadth_safe', True)))}; "
                f"reset_sub_candidate={int(bool(row.get('pullback_reset_same_theme_substitution_candidate', False)))}; "
                f"reset_sub={int(bool(row.get('pullback_reset_same_theme_substitution_promoted', False)))}; "
                f"reset_sub_demote={int(bool(row.get('pullback_reset_same_theme_substitution_demoted', False)))}; "
                f"reset_sub_support={_fmt_num(row.get('pullback_reset_same_theme_substitution_support_score'))}; "
                f"reentry_penalty=-{_fmt_num(row.get('reentry_discipline_penalty'))}; "
                f"breadth_active={_fmt_num(row.get('reentry_theme_breadth_active_share'))}; "
                f"breadth_shortfall={_fmt_num(row.get('reentry_theme_breadth_shortfall'))}; "
                f"breadth_gate={int(bool(row.get('reentry_theme_breadth_deteriorating', False)))}; "
                f"theme_breadth_accel={_fmt_num(row.get('theme_breadth_acceleration_score'))}; "
                f"theme_active_share={_fmt_num(row.get('theme_breadth_active_share_prior'))}; "
                f"theme_peer_rs_prior={_fmt_num(row.get('theme_breadth_peer_rs_mean_prior'))}; "
                f"theme_peer_rs_accel={_fmt_num(row.get('theme_breadth_peer_rs_acceleration'))}; "
                f"theme_strength_delta={_fmt_num(row.get('theme_strength_delta_score'))}; "
                f"theme_strength_prior={_fmt_num(row.get('theme_strength_score_prior'))}; "
                f"theme_strength_rs_delta={_fmt_num(row.get('theme_strength_peer_rs_delta'))}; "
                f"theme_leader_tilt={_fmt_num(row.get('theme_leader_tilt_score'))}; "
                f"theme_leader_rank={_fmt_num(row.get('theme_leader_tilt_theme_rank_pct'))}; "
                f"theme_peers={int(row.get('theme_leader_tilt_theme_peer_count', 0) or 0)}; "
                f"compound_leader_credit={_fmt_num(row.get('compound_leader_score_credit_score'))}; "
                f"compound_base_rank={_fmt_num(row.get('compound_leader_base_rank'))}; "
                f"compound_126_rank={_fmt_num(row.get('compound_leader_126d_voladj_rank'))}; "
                f"compound_252_rank={_fmt_num(row.get('compound_leader_252d_voladj_rank'))}; "
                f"compound_ret10={_fmt_pct(row.get('compound_leader_ret_10d'))}; "
                f"compound_boundary={int(bool(row.get('compound_leader_boundary_eligible', False)))}; "
                f"compound_block={int(bool(row.get('compound_leader_score_credit_blocked', False)))}; "
                f"medium_term_pullback={_fmt_num(row.get('medium_term_leader_pullback_score'))}; "
                f"medium_term_ret100_rank={_fmt_num(row.get('medium_term_leader_pullback_ret_100d_rank'))}; "
                f"medium_term_ret5={_fmt_pct(row.get('medium_term_leader_pullback_ret_5d'))}; "
                f"medium_term_ret10={_fmt_pct(row.get('medium_term_leader_pullback_ret_10d'))}; "
                f"medium_term_volume={_fmt_num(row.get('medium_term_leader_pullback_volume_expansion'))}; "
                f"medium_term_block={int(bool(row.get('medium_term_leader_pullback_blocked', False)))}; "
                f"post_earnings_drift={_fmt_num(row.get('post_earnings_drift_score'))}; "
                f"post_earnings_days={_fmt_num(row.get('post_earnings_drift_days_since_earnings'))}; "
                f"post_earnings_surprise={_fmt_num(row.get('post_earnings_drift_surprise_eps_pct'))}; "
                f"post_earnings_move={_fmt_pct(row.get('post_earnings_drift_expected_move'))}; "
                f"post_earnings_block={int(bool(row.get('post_earnings_drift_blocked', False)))}; "
                f"rs_score={_fmt_num(row.get('relative_strength_score'))}; "
                f"theme={_fmt_num(row.get('theme_score'))}; "
                f"short_weakness={_fmt_num(row.get('short_weakness_count'))}; "
                f"short_rebound={_fmt_num(row.get('short_rebound_avoidance_score'))}; "
                f"short_block={int(bool(row.get('short_rebound_avoidance_blocked', False)))}; "
                f"score_weights={row.get('momentum_score_weight_profile', 'default')}; "
                f"fundamental={row.get('fundamental_score', 50):.1f}; "
                f"event_risk={row.get('event_risk_score', 0):.1f}"
            ),
            axis=1,
        )
        valid["reason_for_entry"] = valid.apply(_entry_reason, axis=1)
        signals = valid[output_columns(valid)].sort_values(["date", "final_score"], ascending=[True, False])
        return signals.groupby("date", group_keys=False).head(max_positions * 2).reset_index(drop=True)


def _momentum_final_score(frame: pd.DataFrame, params: dict) -> pd.Series:
    """Return the momentum ranking score, optionally using a feature-flagged weight profile.

    Feature flag: ``momentum_custom_final_score_weights`` defaults to false.
    Reasonable ranges:
    - ``momentum_final_score_technical_weight``: 0.30 to 0.50.
    - ``momentum_final_score_relative_strength_weight``: 0.25 to 0.40.
    - ``momentum_final_score_theme_weight``: 0.10 to 0.25.
    - ``momentum_final_score_fundamental_weight``: 0.00 to 0.20.
    - ``momentum_final_score_event_risk_penalty``: 0.03 to 0.10.
    - ``momentum_custom_final_score_blend``: 0.0 to 1.0.
    Circuit breaker: invalid weights, or a non-positive custom weight sum, fall
    back to the legacy default score. The blend is clipped into ``[0, 1]`` so
    the custom profile cannot exceed a bounded interpolation versus legacy.
    """

    default_score = final_score(frame)
    frame["momentum_custom_score_effective_blend"] = 0.0
    frame["momentum_custom_score_controlled_entry_eligible"] = False
    if not bool(params.get("momentum_custom_final_score_weights", False)):
        frame["momentum_score_weight_profile"] = "default"
        return default_score

    weights = {
        "technical_score": float(params.get("momentum_final_score_technical_weight", 0.42)),
        "relative_strength_score": float(params.get("momentum_final_score_relative_strength_weight", 0.33)),
        "theme_score": float(params.get("momentum_final_score_theme_weight", 0.20)),
        "fundamental_score": float(params.get("momentum_final_score_fundamental_weight", 0.05)),
    }
    event_penalty = float(params.get("momentum_final_score_event_risk_penalty", 0.05))
    blend = float(np.clip(params.get("momentum_custom_final_score_blend", 1.0), 0.0, 1.0))
    positive_sum = sum(max(value, 0.0) for value in weights.values())
    if positive_sum <= 0:
        LOGGER.warning("momentum_custom_score_disabled_invalid_weights")
        frame["momentum_score_weight_profile"] = "default"
        return default_score

    normalized = {key: max(value, 0.0) / positive_sum for key, value in weights.items()}
    custom_score = pd.Series(0.0, index=frame.index, dtype=float)
    for column, weight in normalized.items():
        fill_default = 50.0 if column in {"theme_score", "fundamental_score"} else 0.0
        values = frame[column] if column in frame else pd.Series(fill_default, index=frame.index)
        custom_score += weight * pd.to_numeric(values, errors="coerce").fillna(fill_default)
    event_risk = frame["event_risk_score"] if "event_risk_score" in frame else pd.Series(0.0, index=frame.index)
    custom_score -= event_penalty * pd.to_numeric(event_risk, errors="coerce").fillna(0.0)
    score = ((1.0 - blend) * default_score) + (blend * custom_score)
    frame["momentum_score_weight_profile"] = (
        f"blend{blend:.2f}_tech{normalized['technical_score']:.2f}_rs{normalized['relative_strength_score']:.2f}_"
        f"theme{normalized['theme_score']:.2f}_fund{normalized['fundamental_score']:.2f}_"
        f"event{event_penalty:.2f}"
    )
    effective_blend = _momentum_custom_score_controlled_entry_blend(frame, params, blend)
    if not effective_blend.eq(blend).all():
        score = ((1.0 - effective_blend) * default_score) + (effective_blend * custom_score)
    LOGGER.info(
        "momentum_custom_score_enabled",
        extra={
            "blend": blend,
            "technical_weight": normalized["technical_score"],
            "relative_strength_weight": normalized["relative_strength_score"],
            "theme_weight": normalized["theme_score"],
            "fundamental_weight": normalized["fundamental_score"],
            "event_penalty": event_penalty,
        },
    )
    return score.clip(lower=0.0, upper=100.0)


def _momentum_custom_score_controlled_entry_blend(frame: pd.DataFrame, params: dict, base_blend: float) -> pd.Series:
    """Optionally restrict custom score weights to controlled long setups.

    Feature flag: ``momentum_custom_score_controlled_entry_overlay`` defaults to
    false. Reasonable ranges:
    - ``momentum_custom_score_min_relative_strength_score``: 55 to 90.
    - ``momentum_custom_score_min_theme_score``: 50 to 85.
    - ``momentum_custom_score_max_above_ma20_pct``: 0.03 to 0.15.
    - ``momentum_custom_score_max_event_risk_score_circuit_breaker``: 10 to 35.
    - ``momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker``: 15 to 45.

    Circuit breakers: when the benchmark is not risk-on, the theme is inactive,
    price is stretched above MA20, price is below MA50, or event/gap risk is
    elevated, the custom score falls back to the legacy default score for that
    row. This keeps the round-23 tech/RS tilt focused on controlled entries.
    """

    effective_blend = pd.Series(float(base_blend), index=frame.index, dtype=float)
    frame["momentum_custom_score_effective_blend"] = effective_blend
    frame["momentum_custom_score_controlled_entry_eligible"] = False
    if not _bool_param(params, "momentum_custom_score_controlled_entry_overlay", False):
        return effective_blend

    adj_close = pd.to_numeric(frame.get("adj_close", np.nan), errors="coerce")
    ma20 = pd.to_numeric(_column(frame, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(frame, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan).fillna(np.inf)
    rs_score = pd.to_numeric(frame.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(frame.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(frame.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(_column(frame, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(frame.get("benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = pd.Series(frame.get("benchmark_risk_on", False), index=frame.index).fillna(False).astype(bool)
    theme_active = pd.Series(frame.get("theme_active", True), index=frame.index).fillna(True).astype(bool)

    min_rs = float(params.get("momentum_custom_score_min_relative_strength_score", 62.0))
    min_theme = float(params.get("momentum_custom_score_min_theme_score", 58.0))
    max_above_ma20_pct = max(float(params.get("momentum_custom_score_max_above_ma20_pct", 0.10)), 0.0)
    max_event_risk = float(params.get("momentum_custom_score_max_event_risk_score_circuit_breaker", 20.0))
    max_gap_risk = float(params.get("momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker", 30.0))
    min_benchmark_ret63d = float(params.get("momentum_custom_score_min_benchmark_ret63d_circuit_breaker", 0.0))
    require_benchmark_risk_on = _bool_param(params, "momentum_custom_score_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "momentum_custom_score_require_theme_active", True)
    require_price_above_ma50 = _bool_param(params, "momentum_custom_score_require_price_above_ma50", True)

    eligible = (
        rs_score.ge(min_rs)
        & theme_score.ge(min_theme)
        & above_ma20.le(max_above_ma20_pct)
        & event_risk.le(max_event_risk)
        & gap_risk.le(max_gap_risk)
        & benchmark_ret_63d.ge(min_benchmark_ret63d)
    )
    if require_benchmark_risk_on:
        eligible &= benchmark_risk_on
    if require_theme_active:
        eligible &= theme_active
    if require_price_above_ma50:
        eligible &= adj_close.gt(ma50).fillna(False)

    effective_blend = effective_blend.where(eligible, 0.0)
    frame["momentum_custom_score_effective_blend"] = effective_blend
    frame["momentum_custom_score_controlled_entry_eligible"] = eligible.fillna(False)
    active_label = f"{frame['momentum_score_weight_profile'].iloc[0]}_controlled"
    frame["momentum_score_weight_profile"] = np.where(effective_blend.gt(0.0), active_label, "default_controlled_fallback")
    LOGGER.info(
        "momentum_custom_score_controlled_entry_overlay_applied",
        extra={
            "event": "momentum_custom_score_controlled_entry_overlay_applied",
            "rows": int(eligible.sum()),
            "dates": int(pd.to_datetime(frame["date"]).nunique()) if "date" in frame else 0,
            "base_blend": float(base_blend),
            "avg_effective_blend": float(effective_blend.mean()),
            "min_relative_strength_score": min_rs,
            "min_theme_score": min_theme,
            "max_above_ma20_pct": max_above_ma20_pct,
            "max_event_risk_score": max_event_risk,
            "max_overnight_gap_risk_score": max_gap_risk,
            "min_benchmark_ret63d": min_benchmark_ret63d,
            "require_benchmark_risk_on": require_benchmark_risk_on,
            "require_theme_active": require_theme_active,
            "require_price_above_ma50": require_price_above_ma50,
        },
    )
    return effective_blend


def _apply_theme_strength_delta_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Continuously boost names in point-in-time strengthening themes.

    Feature flag: ``theme_strength_delta_overlay`` defaults to false. Unlike
    the binary breadth-acceleration gate, this overlay uses prior-date rolling
    changes in same-theme average theme score and peer relative strength. Inputs
    are shifted by one trading date inside each theme before they can affect a
    signal, so a date-t signal never sees date-t peer outcomes.

    Reasonable ranges:
    - ``theme_strength_delta_lookback_days``: 5 to 21.
    - ``theme_strength_delta_max_score_boost``: 0.5 to 2.5.
    - ``theme_strength_delta_min_peer_count``: 2 to 8.
    Circuit breakers: event risk, overnight gap risk, benchmark trend, and
    optional risk-on/theme-active requirements all remain explicit gates.
    """

    out = frame.copy()
    out["theme_strength_delta_score"] = 0.0
    out["theme_strength_delta_boost"] = 0.0
    out["theme_strength_score_prior"] = 0.0
    out["theme_strength_delta"] = 0.0
    out["theme_strength_peer_rs_prior"] = 0.0
    out["theme_strength_peer_rs_delta"] = 0.0
    out["theme_strength_peer_count"] = 0
    out["theme_strength_delta_eligible"] = False
    if not _bool_param(params, "theme_strength_delta_overlay", False):
        return out

    date_key = pd.to_datetime(out["date"]).dt.normalize()
    if "primary_theme" in out:
        theme_key = out["primary_theme"]
    elif "theme_reason" in out:
        theme_key = out["theme_reason"]
    else:
        theme_key = pd.Series("unknown", index=out.index)
    theme_key = (
        pd.Series(theme_key, index=out.index)
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
    )
    peer_group = [date_key, theme_key]
    peer_count = out.groupby(peer_group)["symbol"].transform("nunique").fillna(0).astype(int)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(out.get("benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    if "benchmark_risk_on" in out:
        benchmark_risk_on = pd.Series(out["benchmark_risk_on"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        benchmark_risk_on = pd.Series(False, index=out.index, dtype=bool)
    if "theme_active" in out:
        theme_active = pd.Series(out["theme_active"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index, dtype=bool)

    lookback_days = max(int(params.get("theme_strength_delta_lookback_days", 10)), 1)
    min_peer_count = max(int(params.get("theme_strength_delta_min_peer_count", 3)), 1)
    min_theme_score = float(params.get("theme_strength_delta_min_theme_score", 50.0))
    min_relative_strength_score = float(params.get("theme_strength_delta_min_relative_strength_score", 56.0))
    min_theme_prior = float(params.get("theme_strength_delta_min_theme_prior", 48.0))
    min_peer_rs_prior = float(params.get("theme_strength_delta_min_peer_rs_prior", 52.0))
    min_combined_delta = float(params.get("theme_strength_delta_min_combined_delta", 0.0))
    max_score_boost = max(float(params.get("theme_strength_delta_max_score_boost", 1.5)), 0.0)
    max_event_risk = float(params.get("theme_strength_delta_max_event_risk_score_circuit_breaker", 20.0))
    max_gap_risk = float(params.get("theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker", 35.0))
    min_benchmark_ret63d = float(params.get("theme_strength_delta_min_benchmark_ret63d_circuit_breaker", 0.0))
    require_benchmark_risk_on = _bool_param(params, "theme_strength_delta_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "theme_strength_delta_require_theme_active", False)

    theme_daily = (
        pd.DataFrame(
            {
                "date": date_key,
                "theme": theme_key,
                "theme_score": theme_score,
                "relative_strength_score": rs_score,
                "peer_count": peer_count,
            }
        )
        .groupby(["date", "theme"], as_index=False)
        .agg(
            theme_score_mean=("theme_score", "mean"),
            peer_rs_mean=("relative_strength_score", "mean"),
            peer_count=("peer_count", "max"),
        )
        .sort_values(["theme", "date"])
    )
    theme_daily = theme_daily.rename(columns={"peer_count": "theme_strength_peer_count_daily"})
    theme_daily["theme_score_prior"] = theme_daily.groupby("theme")["theme_score_mean"].shift(1)
    theme_daily["theme_score_base"] = theme_daily.groupby("theme")["theme_score_mean"].shift(lookback_days + 1)
    theme_daily["theme_score_delta"] = theme_daily["theme_score_prior"] - theme_daily["theme_score_base"]
    theme_daily["peer_rs_prior"] = theme_daily.groupby("theme")["peer_rs_mean"].shift(1)
    theme_daily["peer_rs_base"] = theme_daily.groupby("theme")["peer_rs_mean"].shift(lookback_days + 1)
    theme_daily["peer_rs_delta"] = theme_daily["peer_rs_prior"] - theme_daily["peer_rs_base"]

    out["_theme_strength_date_key"] = date_key
    out["_theme_strength_theme_key"] = theme_key
    out = out.merge(
        theme_daily[
            [
                "date",
                "theme",
                "theme_score_prior",
                "theme_score_delta",
                "peer_rs_prior",
                "peer_rs_delta",
                "theme_strength_peer_count_daily",
            ]
        ],
        left_on=["_theme_strength_date_key", "_theme_strength_theme_key"],
        right_on=["date", "theme"],
        how="left",
        suffixes=("", "_theme_strength"),
    )
    out = out.drop(columns=["date_theme_strength", "theme", "_theme_strength_date_key", "_theme_strength_theme_key"], errors="ignore")
    theme_score_prior = pd.to_numeric(out.get("theme_score_prior", 0.0), errors="coerce").fillna(0.0)
    theme_delta = pd.to_numeric(out.get("theme_score_delta", 0.0), errors="coerce").fillna(0.0)
    peer_rs_prior = pd.to_numeric(out.get("peer_rs_prior", 0.0), errors="coerce").fillna(0.0)
    peer_rs_delta = pd.to_numeric(out.get("peer_rs_delta", 0.0), errors="coerce").fillna(0.0)
    strength_peer_count = pd.to_numeric(out.get("theme_strength_peer_count_daily", peer_count), errors="coerce").fillna(peer_count).astype(int)
    combined_delta = (0.45 * theme_delta) + (0.55 * peer_rs_delta)

    eligible = theme_key.ne("unknown")
    eligible &= strength_peer_count.ge(min_peer_count)
    eligible &= theme_score.ge(min_theme_score)
    eligible &= rs_score.ge(min_relative_strength_score)
    eligible &= theme_score_prior.ge(min_theme_prior)
    eligible &= peer_rs_prior.ge(min_peer_rs_prior)
    eligible &= combined_delta.ge(min_combined_delta)
    eligible &= event_risk.le(max_event_risk)
    eligible &= overnight_gap_risk.le(max_gap_risk)
    eligible &= benchmark_ret_63d.ge(min_benchmark_ret63d)
    if require_benchmark_risk_on:
        eligible &= benchmark_risk_on
    if require_theme_active:
        eligible &= theme_active

    theme_delta_component = (theme_delta / 8.0).clip(lower=0.0, upper=1.0)
    peer_rs_delta_component = (peer_rs_delta / 10.0).clip(lower=0.0, upper=1.0)
    theme_level_component = ((theme_score_prior - min_theme_prior) / max(100.0 - min_theme_prior, 1e-9)).clip(lower=0.0, upper=1.0)
    peer_rs_level_component = ((peer_rs_prior - min_peer_rs_prior) / max(100.0 - min_peer_rs_prior, 1e-9)).clip(lower=0.0, upper=1.0)
    strength_score = (
        25.0 * peer_rs_delta_component
        + 15.0 * theme_delta_component
        + 40.0 * peer_rs_level_component
        + 20.0 * theme_level_component
    ).clip(lower=0.0, upper=100.0)
    strength_score = strength_score.where(eligible, 0.0)
    boost = ((strength_score / 100.0) * max_score_boost).where(eligible, 0.0)

    out["theme_strength_delta_score"] = strength_score.fillna(0.0)
    out["theme_strength_delta_boost"] = boost.fillna(0.0)
    out["theme_strength_score_prior"] = theme_score_prior
    out["theme_strength_delta"] = theme_delta
    out["theme_strength_peer_rs_prior"] = peer_rs_prior
    out["theme_strength_peer_rs_delta"] = peer_rs_delta
    out["theme_strength_peer_count"] = strength_peer_count
    out["theme_strength_delta_eligible"] = eligible.fillna(False)
    out["final_score"] = (pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0) + boost.fillna(0.0)).clip(0.0, 100.0)
    out = out.drop(
        columns=[
            "theme_score_prior",
            "theme_score_delta",
            "peer_rs_prior",
            "peer_rs_delta",
            "theme_strength_peer_count_daily",
        ],
        errors="ignore",
    )
    if eligible.any():
        LOGGER.info(
            "momentum_theme_strength_delta_overlay_applied",
            extra={
                "event": "momentum_theme_strength_delta_overlay_applied",
                "rows": int(eligible.sum()),
                "avg_boost": float(boost.loc[eligible].mean()),
                "avg_theme_score_prior": float(theme_score_prior.loc[eligible].mean()),
                "avg_theme_delta": float(theme_delta.loc[eligible].mean()),
                "avg_peer_rs_prior": float(peer_rs_prior.loc[eligible].mean()),
                "avg_peer_rs_delta": float(peer_rs_delta.loc[eligible].mean()),
                "lookback_days": lookback_days,
                "max_score_boost": max_score_boost,
            },
        )
    return out


def _apply_theme_breadth_acceleration_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost members of themes whose point-in-time breadth is expanding.

    Feature flag: ``theme_breadth_acceleration_overlay`` defaults to false.
    The breadth inputs are shifted by one date inside each theme, so a signal
    formed after date t never sees same-day breadth from date t.

    Optional peer-strength refinement: ``theme_breadth_acceleration_peer_strength_overlay``
    defaults to false. When enabled, the overlay requires the theme's prior-date
    mean relative-strength score to be improving as well, which keeps the boost
    aimed at expanding leader groups rather than broad but weak participation.

    Reasonable ranges:
    - ``theme_breadth_acceleration_min_peer_rs_mean``: 55 to 75.
    - ``theme_breadth_acceleration_min_peer_rs_change``: 2 to 10.
    - ``theme_breadth_acceleration_peer_strength_weight``: 0.10 to 0.50.
    Circuit breaker: peer-strength inputs are optional and clipped. If the peer
    strength flag is off, the legacy breadth-only score is used unchanged.
    """

    out = frame.copy()
    out["theme_breadth_acceleration_score"] = 0.0
    out["theme_breadth_acceleration_boost"] = 0.0
    out["theme_breadth_active_share_prior"] = 0.0
    out["theme_breadth_acceleration"] = 0.0
    out["theme_breadth_peer_count"] = 0
    out["theme_breadth_acceleration_eligible"] = False
    out["theme_breadth_peer_rs_mean_prior"] = 0.0
    out["theme_breadth_peer_rs_acceleration"] = 0.0
    if not _bool_param(params, "theme_breadth_acceleration_overlay", False):
        return out

    date_key = pd.to_datetime(out["date"]).dt.normalize()
    if "primary_theme" in out:
        theme_key = out["primary_theme"]
    elif "theme_reason" in out:
        theme_key = out["theme_reason"]
    else:
        theme_key = pd.Series("unknown", index=out.index)
    theme_key = (
        pd.Series(theme_key, index=out.index)
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
    )
    peer_group = [date_key, theme_key]
    peer_count = out.groupby(peer_group)["symbol"].transform("nunique").fillna(0).astype(int)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(out.get("benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    if "benchmark_risk_on" in out:
        benchmark_risk_on = pd.Series(out["benchmark_risk_on"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        benchmark_risk_on = pd.Series(False, index=out.index, dtype=bool)
    if "theme_active" in out:
        theme_active = pd.Series(out["theme_active"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index, dtype=bool)
    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)

    min_rs = float(params.get("theme_breadth_acceleration_min_relative_strength_score", 58.0))
    min_theme = float(params.get("theme_breadth_acceleration_min_theme_score", 55.0))
    min_peer_count = max(int(params.get("theme_breadth_acceleration_min_theme_peer_count", 3)), 1)
    lookback_days = max(int(params.get("theme_breadth_acceleration_lookback_days", 10)), 1)
    min_active_share = float(np.clip(params.get("theme_breadth_acceleration_min_active_share", 0.45), 0.0, 1.0))
    min_acceleration = float(params.get("theme_breadth_acceleration_min_change", 0.08))
    max_score_boost = max(float(params.get("theme_breadth_acceleration_max_score_boost", 2.0)), 0.0)
    max_event_risk = float(params.get("theme_breadth_acceleration_max_event_risk_score_circuit_breaker", 20.0))
    max_gap_risk = float(params.get("theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker", 35.0))
    min_benchmark_ret63d = float(params.get("theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker", 0.0))
    require_benchmark_risk_on = _bool_param(params, "theme_breadth_acceleration_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "theme_breadth_acceleration_require_theme_active", True)
    require_peer_strength = _bool_param(params, "theme_breadth_acceleration_peer_strength_overlay", False)
    min_peer_rs_mean = float(params.get("theme_breadth_acceleration_min_peer_rs_mean", 62.0))
    min_peer_rs_change = float(params.get("theme_breadth_acceleration_min_peer_rs_change", 4.0))
    peer_strength_weight = float(np.clip(params.get("theme_breadth_acceleration_peer_strength_weight", 0.35), 0.0, 0.60))

    active_member = (adj_close > ma20) & rs_score.ge(min_rs)
    breadth_daily = (
        pd.DataFrame(
            {
                "date": date_key,
                "theme": theme_key,
                "active_member": active_member.fillna(False).astype(float),
                "peer_count": peer_count,
                "relative_strength_score": rs_score,
            }
        )
        .groupby(["date", "theme"], as_index=False)
        .agg(
            active_share=("active_member", "mean"),
            peer_count=("peer_count", "max"),
            peer_rs_mean=("relative_strength_score", "mean"),
        )
        .sort_values(["theme", "date"])
    )
    breadth_daily = breadth_daily.rename(columns={"peer_count": "breadth_peer_count"})
    breadth_daily["active_share_prior"] = breadth_daily.groupby("theme")["active_share"].shift(1)
    breadth_daily["active_share_prior_base"] = breadth_daily.groupby("theme")["active_share"].shift(lookback_days + 1)
    breadth_daily["breadth_acceleration"] = breadth_daily["active_share_prior"] - breadth_daily["active_share_prior_base"]
    breadth_daily["peer_rs_mean_prior"] = breadth_daily.groupby("theme")["peer_rs_mean"].shift(1)
    breadth_daily["peer_rs_mean_prior_base"] = breadth_daily.groupby("theme")["peer_rs_mean"].shift(lookback_days + 1)
    breadth_daily["peer_rs_acceleration"] = breadth_daily["peer_rs_mean_prior"] - breadth_daily["peer_rs_mean_prior_base"]
    out["_theme_breadth_date_key"] = date_key
    out["_theme_breadth_theme_key"] = theme_key
    out = out.merge(
        breadth_daily[
            [
                "date",
                "theme",
                "active_share_prior",
                "breadth_acceleration",
                "breadth_peer_count",
                "peer_rs_mean_prior",
                "peer_rs_acceleration",
            ]
        ],
        left_on=["_theme_breadth_date_key", "_theme_breadth_theme_key"],
        right_on=["date", "theme"],
        how="left",
        suffixes=("", "_breadth"),
    )
    out = out.drop(columns=["date_breadth", "theme", "_theme_breadth_date_key", "_theme_breadth_theme_key"], errors="ignore")
    active_share_prior = pd.to_numeric(out.get("active_share_prior", 0.0), errors="coerce").fillna(0.0)
    acceleration = pd.to_numeric(out.get("breadth_acceleration", 0.0), errors="coerce").fillna(0.0)
    breadth_peer_count = pd.to_numeric(out.get("breadth_peer_count", peer_count), errors="coerce").fillna(peer_count).astype(int)
    peer_rs_mean_prior = pd.to_numeric(out.get("peer_rs_mean_prior", 0.0), errors="coerce").fillna(0.0)
    peer_rs_acceleration = pd.to_numeric(out.get("peer_rs_acceleration", 0.0), errors="coerce").fillna(0.0)

    eligible = theme_key.ne("unknown")
    eligible &= breadth_peer_count.ge(min_peer_count)
    eligible &= active_share_prior.ge(min_active_share)
    eligible &= acceleration.ge(min_acceleration)
    eligible &= theme_score.ge(min_theme)
    eligible &= rs_score.ge(min_rs)
    eligible &= event_risk.le(max_event_risk)
    eligible &= overnight_gap_risk.le(max_gap_risk)
    eligible &= benchmark_ret_63d.ge(min_benchmark_ret63d)
    if require_benchmark_risk_on:
        eligible &= benchmark_risk_on
    if require_theme_active:
        eligible &= theme_active
    if require_peer_strength:
        eligible &= peer_rs_mean_prior.ge(min_peer_rs_mean)
        eligible &= peer_rs_acceleration.ge(min_peer_rs_change)

    accel_component = (acceleration / max(min_acceleration * 3.0, 1e-9)).clip(lower=0.0, upper=1.0)
    active_component = ((active_share_prior - min_active_share) / max(1.0 - min_active_share, 1e-9)).clip(lower=0.0, upper=1.0)
    rs_component = ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(lower=0.0, upper=1.0)
    breadth_score_base = ((55.0 * accel_component) + (25.0 * active_component) + (20.0 * rs_component)).clip(lower=0.0, upper=100.0)
    if require_peer_strength:
        peer_rs_mean_component = ((peer_rs_mean_prior - min_peer_rs_mean) / max(100.0 - min_peer_rs_mean, 1e-9)).clip(lower=0.0, upper=1.0)
        peer_rs_accel_component = (peer_rs_acceleration / max(min_peer_rs_change * 3.0, 1e-9)).clip(lower=0.0, upper=1.0)
        peer_strength_score = ((65.0 * peer_rs_accel_component) + (35.0 * peer_rs_mean_component)).clip(lower=0.0, upper=100.0)
        breadth_score = (((1.0 - peer_strength_weight) * breadth_score_base) + (peer_strength_weight * peer_strength_score)).where(
            eligible, 0.0
        )
    else:
        breadth_score = breadth_score_base.where(eligible, 0.0)
    boost = ((breadth_score / 100.0) * max_score_boost).where(eligible, 0.0)

    out["theme_breadth_acceleration_score"] = breadth_score.fillna(0.0)
    out["theme_breadth_acceleration_boost"] = boost.fillna(0.0)
    out["theme_breadth_active_share_prior"] = active_share_prior
    out["theme_breadth_acceleration"] = acceleration
    out["theme_breadth_peer_count"] = breadth_peer_count
    out["theme_breadth_acceleration_eligible"] = eligible.fillna(False)
    out["theme_breadth_peer_rs_mean_prior"] = peer_rs_mean_prior
    out["theme_breadth_peer_rs_acceleration"] = peer_rs_acceleration
    out["final_score"] = (pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0) + boost.fillna(0.0)).clip(0.0, 100.0)
    out = out.drop(
        columns=[
            "active_share_prior",
            "breadth_acceleration",
            "breadth_peer_count",
            "peer_rs_mean_prior",
            "peer_rs_acceleration",
        ],
        errors="ignore",
    )
    if eligible.any():
        LOGGER.info(
            "momentum_theme_breadth_acceleration_overlay_applied",
            extra={
                "event": "momentum_theme_breadth_acceleration_overlay_applied",
                "rows": int(eligible.sum()),
                "avg_boost": float(boost.loc[eligible].mean()),
                "avg_active_share_prior": float(active_share_prior.loc[eligible].mean()),
                "avg_breadth_acceleration": float(acceleration.loc[eligible].mean()),
                "avg_peer_rs_mean_prior": float(peer_rs_mean_prior.loc[eligible].mean()),
                "avg_peer_rs_acceleration": float(peer_rs_acceleration.loc[eligible].mean()),
                "lookback_days": lookback_days,
                "min_active_share": min_active_share,
                "min_acceleration": min_acceleration,
                "max_score_boost": max_score_boost,
                "peer_strength_overlay": require_peer_strength,
                "min_peer_rs_mean": min_peer_rs_mean,
                "min_peer_rs_change": min_peer_rs_change,
                "peer_strength_weight": peer_strength_weight,
            },
        )
    return out


def _apply_theme_leader_tilt_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Tilt toward the strongest same-theme leaders before portfolio construction.

    Feature flag: ``theme_leader_tilt_overlay`` defaults to false.
    Reasonable ranges:
    - ``theme_leader_tilt_min_theme_score``: 55 to 80.
    - ``theme_leader_tilt_min_relative_strength_score``: 70 to 92.
    - ``theme_leader_tilt_min_theme_peer_count``: 2 to 6.
    - ``theme_leader_tilt_min_within_theme_rank_pct``: 0.65 to 0.90.
    - ``theme_leader_tilt_max_score_boost``: 1.0 to 4.0.
    Circuit breakers: the tilt is capped by ``theme_leader_tilt_max_score_boost``
    and only applies when benchmark/theme state is healthy and event/gap risk
    stay below explicit thresholds.
    """

    out = frame.copy()
    out["theme_leader_tilt_score"] = 0.0
    out["theme_leader_tilt_boost"] = 0.0
    out["theme_leader_tilt_theme_rank_pct"] = 0.0
    out["theme_leader_tilt_theme_peer_count"] = 0
    out["theme_leader_tilt_eligible"] = False
    if not _bool_param(params, "theme_leader_tilt_overlay", False):
        return out

    date_key = pd.to_datetime(out["date"]).dt.normalize()
    if "primary_theme" in out:
        theme_key = out["primary_theme"]
    elif "theme_reason" in out:
        theme_key = out["theme_reason"]
    else:
        theme_key = pd.Series("unknown", index=out.index)
    theme_key = (
        pd.Series(theme_key, index=out.index)
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
    )
    peer_group = [date_key, theme_key]
    peer_count = out.groupby(peer_group)["symbol"].transform("nunique").fillna(0).astype(int)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    technical_score = pd.to_numeric(out.get("technical_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(out.get("benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    if "benchmark_risk_on" in out:
        benchmark_risk_on = pd.Series(out["benchmark_risk_on"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        benchmark_risk_on = pd.Series(False, index=out.index, dtype=bool)
    if "theme_active" in out:
        theme_active = pd.Series(out["theme_active"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index, dtype=bool)
    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)

    rs_rank = rs_score.groupby(peer_group).rank(pct=True, method="average").fillna(0.0)
    technical_rank = technical_score.groupby(peer_group).rank(pct=True, method="average").fillna(0.0)
    leader_rank = (0.65 * rs_rank + 0.35 * technical_rank).clip(lower=0.0, upper=1.0)

    min_theme = float(params.get("theme_leader_tilt_min_theme_score", 60.0))
    min_rs = float(params.get("theme_leader_tilt_min_relative_strength_score", 78.0))
    min_peer_count = max(int(params.get("theme_leader_tilt_min_theme_peer_count", 2)), 1)
    min_rank = float(np.clip(params.get("theme_leader_tilt_min_within_theme_rank_pct", 0.75), 0.0, 1.0))
    max_score_boost = max(float(params.get("theme_leader_tilt_max_score_boost", 2.5)), 0.0)
    max_event_risk = float(params.get("theme_leader_tilt_max_event_risk_score_circuit_breaker", 15.0))
    max_gap_risk = float(params.get("theme_leader_tilt_max_overnight_gap_risk_score_circuit_breaker", 30.0))
    min_benchmark_ret63d = float(params.get("theme_leader_tilt_min_benchmark_ret63d_circuit_breaker", 0.0))
    require_benchmark_risk_on = _bool_param(params, "theme_leader_tilt_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "theme_leader_tilt_require_theme_active", True)

    eligible = theme_key.ne("unknown")
    eligible &= peer_count.ge(min_peer_count)
    eligible &= leader_rank.ge(min_rank)
    eligible &= theme_score.ge(min_theme)
    eligible &= rs_score.ge(min_rs)
    eligible &= mom_return.gt(0.0)
    eligible &= adj_close.gt(ma50)
    eligible &= event_risk.le(max_event_risk)
    eligible &= overnight_gap_risk.le(max_gap_risk)
    eligible &= benchmark_ret_63d.ge(min_benchmark_ret63d)
    if require_benchmark_risk_on:
        eligible &= benchmark_risk_on
    if require_theme_active:
        eligible &= theme_active

    leader_component = ((leader_rank - min_rank) / max(1.0 - min_rank, 1e-9)).clip(lower=0.0, upper=1.0)
    peer_component = ((peer_count - min_peer_count + 1) / max(min_peer_count, 1)).clip(lower=0.0, upper=2.0) / 2.0
    tilt_score = ((leader_component * 85.0) + (peer_component * 15.0)).where(eligible, 0.0)
    boost = ((tilt_score / 100.0) * max_score_boost).where(eligible, 0.0)

    out["theme_leader_tilt_score"] = tilt_score.fillna(0.0)
    out["theme_leader_tilt_boost"] = boost.fillna(0.0)
    out["theme_leader_tilt_theme_rank_pct"] = leader_rank.fillna(0.0)
    out["theme_leader_tilt_theme_peer_count"] = peer_count
    out["theme_leader_tilt_eligible"] = eligible.fillna(False)
    out["final_score"] = (pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0) + boost.fillna(0.0)).clip(0.0, 100.0)
    if eligible.any():
        LOGGER.info(
            "momentum_theme_leader_tilt_overlay_applied",
            extra={
                "event": "momentum_theme_leader_tilt_overlay_applied",
                "rows": int(eligible.sum()),
                "avg_boost": float(boost.loc[eligible].mean()),
                "avg_rank_pct": float(leader_rank.loc[eligible].mean()),
                "avg_theme_peer_count": float(peer_count.loc[eligible].mean()),
                "min_theme_score": min_theme,
                "min_relative_strength_score": min_rs,
                "min_theme_peer_count": min_peer_count,
                "min_within_theme_rank_pct": min_rank,
                "max_score_boost": max_score_boost,
                "max_event_risk_score_circuit_breaker": max_event_risk,
                "max_overnight_gap_risk_score_circuit_breaker": max_gap_risk,
                "min_benchmark_ret63d_circuit_breaker": min_benchmark_ret63d,
                "require_benchmark_risk_on": require_benchmark_risk_on,
                "require_theme_active": require_theme_active,
            },
        )
    return out


def _apply_compound_leader_score_credit_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost long-horizon compound leaders that pause without breaking trend.

    Feature flag: ``compound_leader_score_credit_overlay`` defaults to false.
    Optional refinement flag: ``compound_leader_score_credit_boundary_overlay``
    defaults to false and limits the credit to a narrow score-rank window
    around the long cutoff.
    Reasonable ranges:
    - ``compound_leader_score_credit_min_126d_voladj_rank``: 0.65 to 0.90.
    - ``compound_leader_score_credit_min_252d_voladj_rank``: 0.65 to 0.92.
    - ``compound_leader_score_credit_min_relative_strength_score``: 75 to 92.
    - ``compound_leader_score_credit_min_theme_score``: 55 to 80.
    - ``compound_leader_score_credit_max_ret_10d``: -0.03 to 0.08.
    - ``compound_leader_score_credit_max_drawdown_from_high``: 0.04 to 0.18.
    - ``compound_leader_score_credit_max_above_ma20_pct``: 0.02 to 0.10.
    - ``compound_leader_score_credit_max_score_boost``: 0.5 to 3.0.
    - ``compound_leader_score_credit_rank_buffer_below``: 0.03 to 0.15.
    - ``compound_leader_score_credit_rank_buffer_above``: 0.00 to 0.05.
    - ``compound_leader_score_credit_min_ret_100d_rank``: 0.55 to 0.90.
    - ``compound_leader_score_credit_min_volume_expansion``: 0.80 to 1.40.

    Circuit breakers: benchmark risk-on, theme-active state, price above MA50,
    event/gap risk caps, minimum ADX, and a capped score boost keep this as a
    narrow entry-ranking credit rather than a broad chase overlay.
    """

    out = frame.copy()
    out["compound_leader_score_credit_score"] = 0.0
    out["compound_leader_score_credit_boost"] = 0.0
    out["compound_leader_126d_voladj_rank"] = 0.0
    out["compound_leader_252d_voladj_rank"] = 0.0
    out["compound_leader_ret_100d_rank"] = 0.0
    out["compound_leader_ret_10d"] = np.nan
    out["compound_leader_volume_expansion_signal"] = 0.0
    out["compound_leader_theme_peer_count"] = 0
    out["compound_leader_base_rank"] = 0.0
    out["compound_leader_boundary_eligible"] = False
    out["compound_leader_volume_confirmation_eligible"] = False
    out["compound_leader_score_credit_blocked"] = False
    out["compound_leader_score_credit_eligible"] = False
    if not _bool_param(params, "compound_leader_score_credit_overlay", False):
        return out

    max_score_boost = float(np.clip(params.get("compound_leader_score_credit_max_score_boost", 1.75), 0.0, 4.0))
    if max_score_boost <= 0.0:
        return out

    date_key = pd.to_datetime(out["date"]).dt.normalize()
    if "primary_theme" in out:
        theme_key = out["primary_theme"]
    elif "theme_reason" in out:
        theme_key = out["theme_reason"]
    else:
        theme_key = pd.Series("unknown", index=out.index)
    theme_key = (
        pd.Series(theme_key, index=out.index)
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
    )
    peer_group = [date_key, theme_key]
    peer_count = out.groupby(peer_group)["symbol"].transform("nunique").fillna(0).astype(int)
    grouped = out.groupby("symbol", group_keys=False)

    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan)
    ret_10d = pd.to_numeric(out.get("ret_10d"), errors="coerce") if "ret_10d" in out else grouped["adj_close"].pct_change(10)
    ret_100d = pd.to_numeric(out.get("ret_100d"), errors="coerce") if "ret_100d" in out else grouped["adj_close"].pct_change(100)
    volume_expansion = pd.to_numeric(_column(out, "volume_expansion", 1.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    distance_high = pd.to_numeric(out.get("mom_distance_high", out.get("distance_to_prior_high_252")), errors="coerce")
    drawdown_from_high = (-distance_high).clip(lower=0.0).fillna(np.inf)
    base_score = pd.to_numeric(out.get("final_score", 50.0), errors="coerce").fillna(50.0)
    base_rank = base_score.groupby(out["date"]).rank(pct=True, method="average").fillna(0.0)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(_column(out, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    trend_adx = pd.to_numeric(_column(out, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    if "benchmark_risk_on" in out:
        benchmark_risk_on = pd.Series(out["benchmark_risk_on"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        benchmark_risk_on = pd.Series(False, index=out.index, dtype=bool)
    if "theme_active" in out:
        theme_active = pd.Series(out["theme_active"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index, dtype=bool)

    delayed_126 = grouped["adj_close"].shift(5)
    base_126 = grouped["adj_close"].shift(131)
    mom_126 = delayed_126 / base_126 - 1.0
    vol_126 = grouped["return_1d"].transform(lambda s: s.shift(5).rolling(126, min_periods=max(20, 126 // 3)).std())
    mom_126_voladj = mom_126 / vol_126.replace(0, np.nan)
    delayed_252 = grouped["adj_close"].shift(21)
    base_252 = grouped["adj_close"].shift(273)
    mom_252 = delayed_252 / base_252 - 1.0
    vol_252 = grouped["return_1d"].transform(lambda s: s.shift(21).rolling(252, min_periods=max(20, 252 // 3)).std())
    mom_252_voladj = mom_252 / vol_252.replace(0, np.nan)
    rank_126 = mom_126_voladj.groupby(date_key).rank(pct=True, method="average").fillna(0.0)
    rank_252 = mom_252_voladj.groupby(date_key).rank(pct=True, method="average").fillna(0.0)
    ret_100d_rank = ret_100d.groupby(date_key).rank(pct=True, method="average").fillna(0.0)

    min_rank_126 = float(np.clip(params.get("compound_leader_score_credit_min_126d_voladj_rank", 0.78), 0.0, 1.0))
    min_rank_252 = float(np.clip(params.get("compound_leader_score_credit_min_252d_voladj_rank", 0.80), 0.0, 1.0))
    min_final = float(params.get("compound_leader_score_credit_min_final_score", 60.0))
    min_rs = float(params.get("compound_leader_score_credit_min_relative_strength_score", 80.0))
    min_theme = float(params.get("compound_leader_score_credit_min_theme_score", 60.0))
    min_mom_return = float(params.get("compound_leader_score_credit_min_mom_return", 0.05))
    min_peer_count = max(int(params.get("compound_leader_score_credit_min_theme_peer_count", 2)), 1)
    max_ret_10d = float(params.get("compound_leader_score_credit_max_ret_10d", 0.04))
    max_drawdown = max(float(params.get("compound_leader_score_credit_max_drawdown_from_high", 0.12)), 0.0)
    max_above_ma20 = max(float(params.get("compound_leader_score_credit_max_above_ma20_pct", 0.08)), 0.0)
    max_event_risk = float(params.get("compound_leader_score_credit_max_event_risk_score_circuit_breaker", 15.0))
    max_gap_risk = float(params.get("compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker", 25.0))
    min_benchmark_ret63d = float(params.get("compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_adx = float(params.get("compound_leader_score_credit_min_adx_circuit_breaker", 20.0))
    require_benchmark_risk_on = _bool_param(params, "compound_leader_score_credit_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "compound_leader_score_credit_require_theme_active", True)
    require_price_above_ma50 = _bool_param(params, "compound_leader_score_credit_require_price_above_ma50", True)
    boundary_overlay = _bool_param(params, "compound_leader_score_credit_boundary_overlay", False)
    volume_confirmation_overlay = _bool_param(params, "compound_leader_score_credit_volume_confirmation_overlay", False)
    min_ret_100d_rank = float(np.clip(params.get("compound_leader_score_credit_min_ret_100d_rank", 0.65), 0.0, 1.0))
    min_volume_expansion = max(float(params.get("compound_leader_score_credit_min_volume_expansion", 1.0)), 0.0)
    long_q = float(np.clip(params.get("long_quantile", 0.20), 0.0, 1.0))
    boundary_rank = float(np.clip(1.0 - long_q, 0.0, 1.0))
    rank_buffer_below = max(float(params.get("compound_leader_score_credit_rank_buffer_below", 0.08)), 0.0)
    rank_buffer_above = max(float(params.get("compound_leader_score_credit_rank_buffer_above", 0.02)), 0.0)
    min_rank = float(np.clip(boundary_rank - rank_buffer_below, 0.0, 1.0))
    max_rank = float(np.clip(boundary_rank + rank_buffer_above, min_rank, 1.0))

    candidate = theme_key.ne("unknown")
    candidate &= peer_count.ge(min_peer_count)
    candidate &= rank_126.ge(min_rank_126)
    candidate &= rank_252.ge(min_rank_252)
    candidate &= base_score.ge(min_final)
    candidate &= rs_score.ge(min_rs)
    candidate &= theme_score.ge(min_theme)
    candidate &= mom_return.ge(min_mom_return)
    candidate &= ret_10d.le(max_ret_10d).fillna(False)
    candidate &= drawdown_from_high.le(max_drawdown)
    candidate &= above_ma20.le(max_above_ma20).fillna(False)
    volume_confirmation_eligible = ret_100d_rank.ge(min_ret_100d_rank) & volume_expansion.ge(min_volume_expansion)
    if volume_confirmation_overlay:
        candidate &= volume_confirmation_eligible
    boundary_eligible = pd.Series(True, index=out.index, dtype=bool)
    if boundary_overlay:
        boundary_eligible = base_rank.between(min_rank, max_rank, inclusive="both")
        candidate &= boundary_eligible
    if require_price_above_ma50:
        candidate &= adj_close.gt(ma50).fillna(False)

    circuit_breaker = (
        event_risk.le(max_event_risk)
        & overnight_gap_risk.le(max_gap_risk)
        & benchmark_ret_63d.ge(min_benchmark_ret63d)
        & trend_adx.ge(min_adx)
    )
    if require_benchmark_risk_on:
        circuit_breaker &= benchmark_risk_on
    if require_theme_active:
        circuit_breaker &= theme_active

    active = candidate & circuit_breaker
    blocked = candidate & ~circuit_breaker

    rank_126_component = ((rank_126 - min_rank_126) / max(1.0 - min_rank_126, 1e-9)).clip(lower=0.0, upper=1.0)
    rank_252_component = ((rank_252 - min_rank_252) / max(1.0 - min_rank_252, 1e-9)).clip(lower=0.0, upper=1.0)
    ret10_component = ((max_ret_10d - ret_10d) / max(abs(max_ret_10d) + 0.10, 1e-9)).clip(lower=0.0, upper=1.0)
    near_high_component = (1.0 - (drawdown_from_high / max(max_drawdown, 1e-9))).clip(lower=0.0, upper=1.0)
    peer_component = ((peer_count - min_peer_count + 1) / max(min_peer_count, 1)).clip(lower=0.0, upper=2.0) / 2.0
    ret_100d_component = ((ret_100d_rank - min_ret_100d_rank) / max(1.0 - min_ret_100d_rank, 1e-9)).clip(lower=0.0, upper=1.0)
    volume_component = ((volume_expansion - min_volume_expansion) / max(max(min_volume_expansion, 1.0), 1e-9)).clip(lower=0.0, upper=1.0)
    score = (
        35.0 * rank_252_component
        + 30.0 * rank_126_component
        + 15.0 * ret10_component
        + 10.0 * near_high_component
        + 10.0 * peer_component
    ).where(active, 0.0)
    confirmation_score = (65.0 * ret_100d_component + 35.0 * volume_component).where(active, 0.0)
    if volume_confirmation_overlay:
        score = (score * (0.75 + 0.25 * (confirmation_score / 100.0))).where(active, 0.0)
    boost = ((score / 100.0) * max_score_boost).where(active, 0.0)

    out["compound_leader_score_credit_score"] = score.fillna(0.0)
    out["compound_leader_score_credit_boost"] = boost.fillna(0.0)
    out["compound_leader_126d_voladj_rank"] = rank_126.fillna(0.0)
    out["compound_leader_252d_voladj_rank"] = rank_252.fillna(0.0)
    out["compound_leader_ret_100d_rank"] = ret_100d_rank.fillna(0.0)
    out["compound_leader_ret_10d"] = ret_10d
    out["compound_leader_volume_expansion_signal"] = volume_expansion.fillna(1.0)
    out["compound_leader_theme_peer_count"] = peer_count
    out["compound_leader_base_rank"] = base_rank.fillna(0.0)
    out["compound_leader_boundary_eligible"] = boundary_eligible.fillna(False)
    out["compound_leader_volume_confirmation_eligible"] = volume_confirmation_eligible.fillna(False)
    out["compound_leader_score_credit_blocked"] = blocked.fillna(False)
    out["compound_leader_score_credit_eligible"] = active.fillna(False)
    out["final_score"] = (base_score + boost.fillna(0.0)).clip(0.0, 100.0)
    if active.any() or blocked.any():
        LOGGER.info(
            "momentum_compound_leader_score_credit_overlay_applied",
            extra={
                "event": "momentum_compound_leader_score_credit_overlay_applied",
                "rows": int(active.sum()),
                "blocked_rows": int(blocked.sum()),
                "avg_boost": float(boost.loc[active].mean()) if active.any() else 0.0,
                "avg_126d_rank": float(rank_126.loc[active].mean()) if active.any() else 0.0,
                "avg_252d_rank": float(rank_252.loc[active].mean()) if active.any() else 0.0,
                "avg_ret_100d_rank": float(ret_100d_rank.loc[active].mean()) if active.any() else 0.0,
                "avg_ret_10d": float(ret_10d.loc[active].mean()) if active.any() else 0.0,
                "avg_volume_expansion": float(volume_expansion.loc[active].mean()) if active.any() else 0.0,
                "avg_peer_count": float(peer_count.loc[active].mean()) if active.any() else 0.0,
                "avg_base_rank": float(base_rank.loc[active].mean()) if active.any() else 0.0,
                "min_126d_voladj_rank": min_rank_126,
                "min_252d_voladj_rank": min_rank_252,
                "volume_confirmation_overlay": volume_confirmation_overlay,
                "min_ret_100d_rank": min_ret_100d_rank,
                "min_volume_expansion": min_volume_expansion,
                "min_final_score": min_final,
                "min_relative_strength_score": min_rs,
                "min_theme_score": min_theme,
                "min_mom_return": min_mom_return,
                "max_ret_10d": max_ret_10d,
                "max_drawdown_from_high": max_drawdown,
                "max_above_ma20_pct": max_above_ma20,
                "max_event_risk_score_circuit_breaker": max_event_risk,
                "max_overnight_gap_risk_score_circuit_breaker": max_gap_risk,
                "min_benchmark_ret63d_circuit_breaker": min_benchmark_ret63d,
                "min_adx_circuit_breaker": min_adx,
                "require_benchmark_risk_on": require_benchmark_risk_on,
                "require_theme_active": require_theme_active,
                "require_price_above_ma50": require_price_above_ma50,
                "boundary_overlay": boundary_overlay,
                "rank_buffer_below": rank_buffer_below,
                "rank_buffer_above": rank_buffer_above,
                "max_score_boost": max_score_boost,
            },
        )
    return out


def _apply_medium_term_leader_pullback_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost medium-term leaders that pause without short-term chase behavior.

    Feature flag: ``medium_term_leader_pullback_overlay`` defaults to false.
    Reasonable ranges:
    - ``medium_term_leader_pullback_min_ret_100d_rank``: 0.55 to 0.85.
    - ``medium_term_leader_pullback_max_ret_5d``: -0.02 to 0.03.
    - ``medium_term_leader_pullback_max_ret_10d``: 0.00 to 0.06.
    - ``medium_term_leader_pullback_min_drawdown_from_high``: 0.01 to 0.08.
    - ``medium_term_leader_pullback_max_drawdown_from_high``: 0.06 to 0.20.
    - ``medium_term_leader_pullback_max_above_ma20_pct``: 0.00 to 0.08.
    - ``medium_term_leader_pullback_max_volume_expansion``: 0.80 to 1.40.
    - ``medium_term_leader_pullback_min_relative_strength_score``: 65 to 90.
    - ``medium_term_leader_pullback_min_theme_score``: 50 to 80.
    - ``medium_term_leader_pullback_max_score_boost``: 0.5 to 3.0.

    Circuit breakers: bounded event risk, bounded overnight gap risk, minimum
    benchmark trend, minimum ADX, optional benchmark risk-on and theme-active
    requirements, and price above MA50 keep this as a narrow entry-ranking
    overlay rather than a broad chase feature.
    """

    out = frame.copy()
    out["medium_term_leader_pullback_score"] = 0.0
    out["medium_term_leader_pullback_boost"] = 0.0
    out["medium_term_leader_pullback_ret_100d_rank"] = 0.0
    out["medium_term_leader_pullback_ret_5d"] = np.nan
    out["medium_term_leader_pullback_ret_10d"] = np.nan
    out["medium_term_leader_pullback_volume_expansion"] = 0.0
    out["medium_term_leader_pullback_blocked"] = False
    out["medium_term_leader_pullback_eligible"] = False
    if not _bool_param(params, "medium_term_leader_pullback_overlay", False):
        return out

    max_score_boost = float(np.clip(params.get("medium_term_leader_pullback_max_score_boost", 2.0), 0.0, 4.0))
    if max_score_boost <= 0.0:
        return out

    grouped = out.groupby("symbol", group_keys=False)
    date_key = pd.to_datetime(out["date"]).dt.normalize()
    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan)
    ret_5d = pd.to_numeric(out.get("ret_5d"), errors="coerce") if "ret_5d" in out else grouped["adj_close"].pct_change(5)
    ret_10d = pd.to_numeric(out.get("ret_10d"), errors="coerce") if "ret_10d" in out else grouped["adj_close"].pct_change(10)
    ret_100d = pd.to_numeric(out.get("ret_100d"), errors="coerce") if "ret_100d" in out else grouped["adj_close"].pct_change(100)
    ret_100d_rank = ret_100d.groupby(date_key).rank(pct=True, method="average").fillna(0.0)
    volume_expansion = pd.to_numeric(_column(out, "volume_expansion", 1.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    distance_high = pd.to_numeric(out.get("mom_distance_high", out.get("distance_to_prior_high_252")), errors="coerce")
    drawdown_from_high = (-distance_high).clip(lower=0.0).fillna(np.inf)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    base_score = pd.to_numeric(out.get("final_score", 50.0), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(_column(out, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    trend_adx = pd.to_numeric(_column(out, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(out, "benchmark_risk_on", False).fillna(False).astype(bool)
    if "theme_active" in out:
        theme_active = pd.Series(out["theme_active"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index, dtype=bool)

    min_ret_100d_rank = float(np.clip(params.get("medium_term_leader_pullback_min_ret_100d_rank", 0.60), 0.0, 1.0))
    max_ret_5d = float(params.get("medium_term_leader_pullback_max_ret_5d", 0.015))
    max_ret_10d = float(params.get("medium_term_leader_pullback_max_ret_10d", 0.03))
    min_drawdown = max(float(params.get("medium_term_leader_pullback_min_drawdown_from_high", 0.02)), 0.0)
    max_drawdown = max(float(params.get("medium_term_leader_pullback_max_drawdown_from_high", 0.16)), min_drawdown)
    max_above_ma20 = max(float(params.get("medium_term_leader_pullback_max_above_ma20_pct", 0.04)), 0.0)
    max_volume_expansion = max(float(params.get("medium_term_leader_pullback_max_volume_expansion", 1.15)), 0.0)
    min_rs = float(params.get("medium_term_leader_pullback_min_relative_strength_score", 68.0))
    min_theme = float(params.get("medium_term_leader_pullback_min_theme_score", 56.0))
    min_mom_return = float(params.get("medium_term_leader_pullback_min_mom_return", 0.05))
    max_event_risk = float(params.get("medium_term_leader_pullback_max_event_risk_score_circuit_breaker", 18.0))
    max_gap_risk = float(params.get("medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker", 28.0))
    min_benchmark_ret63d = float(params.get("medium_term_leader_pullback_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_adx = float(params.get("medium_term_leader_pullback_min_adx_circuit_breaker", 18.0))
    require_benchmark_risk_on = _bool_param(params, "medium_term_leader_pullback_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "medium_term_leader_pullback_require_theme_active", False)
    require_price_above_ma50 = _bool_param(params, "medium_term_leader_pullback_require_price_above_ma50", True)

    candidate = ret_100d_rank.ge(min_ret_100d_rank)
    candidate &= ret_5d.le(max_ret_5d).fillna(False)
    candidate &= ret_10d.le(max_ret_10d).fillna(False)
    candidate &= drawdown_from_high.between(min_drawdown, max_drawdown, inclusive="both")
    candidate &= above_ma20.le(max_above_ma20).fillna(False)
    candidate &= volume_expansion.le(max_volume_expansion)
    candidate &= rs_score.ge(min_rs)
    candidate &= theme_score.ge(min_theme)
    candidate &= mom_return.ge(min_mom_return)

    circuit_breaker = (
        event_risk.le(max_event_risk)
        & overnight_gap_risk.le(max_gap_risk)
        & benchmark_ret_63d.ge(min_benchmark_ret63d)
        & trend_adx.ge(min_adx)
    )
    if require_benchmark_risk_on:
        circuit_breaker &= benchmark_risk_on
    if require_theme_active:
        circuit_breaker &= theme_active
    if require_price_above_ma50:
        circuit_breaker &= adj_close.gt(ma50).fillna(False)

    active = candidate & circuit_breaker
    blocked = candidate & ~circuit_breaker

    ret_100d_component = ((ret_100d_rank - min_ret_100d_rank) / max(1.0 - min_ret_100d_rank, 1e-9)).clip(lower=0.0, upper=1.0)
    ret5_component = ((max_ret_5d - ret_5d) / max(abs(max_ret_5d) + 0.08, 1e-9)).clip(lower=0.0, upper=1.0)
    ret10_component = ((max_ret_10d - ret_10d) / max(abs(max_ret_10d) + 0.12, 1e-9)).clip(lower=0.0, upper=1.0)
    drawdown_component = ((drawdown_from_high - min_drawdown) / max(max_drawdown - min_drawdown, 1e-9)).clip(lower=0.0, upper=1.0)
    rs_component = ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(lower=0.0, upper=1.0)
    theme_component = ((theme_score - min_theme) / max(100.0 - min_theme, 1e-9)).clip(lower=0.0, upper=1.0)
    score = (
        40.0 * ret_100d_component
        + 25.0 * ret5_component
        + 15.0 * ret10_component
        + 10.0 * drawdown_component
        + 5.0 * rs_component
        + 5.0 * theme_component
    ).where(active, 0.0)
    boost = ((score / 100.0) * max_score_boost).where(active, 0.0)

    out["medium_term_leader_pullback_score"] = score.fillna(0.0)
    out["medium_term_leader_pullback_boost"] = boost.fillna(0.0)
    out["medium_term_leader_pullback_ret_100d_rank"] = ret_100d_rank.fillna(0.0)
    out["medium_term_leader_pullback_ret_5d"] = ret_5d
    out["medium_term_leader_pullback_ret_10d"] = ret_10d
    out["medium_term_leader_pullback_volume_expansion"] = volume_expansion.fillna(1.0)
    out["medium_term_leader_pullback_blocked"] = blocked.fillna(False)
    out["medium_term_leader_pullback_eligible"] = active.fillna(False)
    out["final_score"] = (base_score + boost.fillna(0.0)).clip(0.0, 100.0)
    if active.any() or blocked.any():
        LOGGER.info(
            "momentum_medium_term_leader_pullback_overlay_applied",
            extra={
                "event": "momentum_medium_term_leader_pullback_overlay_applied",
                "rows": int(active.sum()),
                "blocked_rows": int(blocked.sum()),
                "avg_boost": float(boost.loc[active].mean()) if active.any() else 0.0,
                "avg_ret_100d_rank": float(ret_100d_rank.loc[active].mean()) if active.any() else 0.0,
                "avg_ret_5d": float(ret_5d.loc[active].mean()) if active.any() else 0.0,
                "avg_ret_10d": float(ret_10d.loc[active].mean()) if active.any() else 0.0,
                "avg_drawdown_from_high": float(drawdown_from_high.loc[active].mean()) if active.any() else 0.0,
                "avg_volume_expansion": float(volume_expansion.loc[active].mean()) if active.any() else 0.0,
                "min_ret_100d_rank": min_ret_100d_rank,
                "max_ret_5d": max_ret_5d,
                "max_ret_10d": max_ret_10d,
                "min_drawdown_from_high": min_drawdown,
                "max_drawdown_from_high": max_drawdown,
                "max_above_ma20_pct": max_above_ma20,
                "max_volume_expansion": max_volume_expansion,
                "min_relative_strength_score": min_rs,
                "min_theme_score": min_theme,
                "min_mom_return": min_mom_return,
                "max_event_risk_score_circuit_breaker": max_event_risk,
                "max_overnight_gap_risk_score_circuit_breaker": max_gap_risk,
                "min_benchmark_ret63d_circuit_breaker": min_benchmark_ret63d,
                "min_adx_circuit_breaker": min_adx,
                "require_benchmark_risk_on": require_benchmark_risk_on,
                "require_theme_active": require_theme_active,
                "require_price_above_ma50": require_price_above_ma50,
                "max_score_boost": max_score_boost,
            },
        )
    return out


def _apply_post_earnings_drift_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost recent positive earnings surprises without using same-day reports.

    Feature flag: ``post_earnings_drift_overlay`` defaults to false.
    Reasonable ranges:
    - ``post_earnings_drift_min_surprise_eps_pct``: 5 to 30.
    - ``post_earnings_drift_min_days_since_earnings``: 1 to 3.
    - ``post_earnings_drift_max_days_since_earnings``: 3 to 12.
    - ``post_earnings_drift_min_relative_strength_score``: 60 to 90.
    - ``post_earnings_drift_min_theme_score``: 50 to 80.
    - ``post_earnings_drift_min_mom_return``: 0.00 to 0.20.
    - ``post_earnings_drift_max_above_ma20_pct``: 0.03 to 0.12.
    - ``post_earnings_drift_max_overnight_gap_risk_score_circuit_breaker``:
      10 to 40.
    - ``post_earnings_drift_max_expected_move_pct_circuit_breaker``:
      0.05 to 0.25.
    - ``post_earnings_drift_max_score_boost``: 0.5 to 3.0.

    Circuit breakers: the overlay only looks at rows at least one trading day
    after earnings, requires a positive surprise, enforces bounded
    extension/gap/expected-move conditions, and keeps optional benchmark,
    theme, and MA50 gates intact. The boost is capped and cannot bypass the
    legacy event-risk penalty or portfolio sizing path.
    """

    out = frame.copy()
    out["post_earnings_drift_score"] = 0.0
    out["post_earnings_drift_boost"] = 0.0
    out["post_earnings_drift_days_since_earnings"] = np.nan
    out["post_earnings_drift_surprise_eps_pct"] = 0.0
    out["post_earnings_drift_expected_move"] = 0.0
    out["post_earnings_drift_eligible"] = False
    out["post_earnings_drift_blocked"] = False
    if not _bool_param(params, "post_earnings_drift_overlay", False):
        return out

    max_boost = float(np.clip(params.get("post_earnings_drift_max_score_boost", 1.75), 0.0, 4.0))
    if max_boost <= 0.0:
        return out

    base_score = pd.to_numeric(out.get("final_score", 50.0), errors="coerce").fillna(50.0)
    days_to_earnings = pd.to_numeric(_column(out, "days_to_earnings", np.nan), errors="coerce")
    days_since_earnings = (-days_to_earnings).replace([np.inf, -np.inf], np.nan)
    surprise_eps_pct = pd.to_numeric(_column(out, "surprise_eps_pct", 0.0), errors="coerce").fillna(0.0)
    expected_move = pd.to_numeric(_column(out, "expected_move", 0.0), errors="coerce").abs().fillna(0.0)
    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(_column(out, "mom_return", 0.0), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(_column(out, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(out, "benchmark_risk_on", False).fillna(False).astype(bool)
    theme_active = _column(out, "theme_active", True).fillna(True).astype(bool)

    min_surprise_eps_pct = float(params.get("post_earnings_drift_min_surprise_eps_pct", 8.0))
    min_days_since = max(int(params.get("post_earnings_drift_min_days_since_earnings", 1)), 1)
    max_days_since = max(int(params.get("post_earnings_drift_max_days_since_earnings", 6)), min_days_since)
    min_rs = float(params.get("post_earnings_drift_min_relative_strength_score", 68.0))
    min_theme = float(params.get("post_earnings_drift_min_theme_score", 58.0))
    min_mom_return = float(params.get("post_earnings_drift_min_mom_return", 0.05))
    max_above_ma20 = max(float(params.get("post_earnings_drift_max_above_ma20_pct", 0.06)), 0.0)
    max_gap_risk = float(params.get("post_earnings_drift_max_overnight_gap_risk_score_circuit_breaker", 30.0))
    max_expected_move = max(float(params.get("post_earnings_drift_max_expected_move_pct_circuit_breaker", 0.18)), 0.0)
    min_benchmark_ret63d = float(params.get("post_earnings_drift_min_benchmark_ret63d_circuit_breaker", 0.0))
    require_benchmark_risk_on = _bool_param(params, "post_earnings_drift_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "post_earnings_drift_require_theme_active", False)
    require_price_above_ma50 = _bool_param(params, "post_earnings_drift_require_price_above_ma50", True)

    candidate = days_since_earnings.between(min_days_since, max_days_since, inclusive="both")
    candidate &= surprise_eps_pct.ge(min_surprise_eps_pct)
    candidate &= rs_score.ge(min_rs)
    candidate &= theme_score.ge(min_theme)
    candidate &= mom_return.ge(min_mom_return)

    circuit_breaker = above_ma20.le(max_above_ma20).fillna(False)
    circuit_breaker &= gap_risk.le(max_gap_risk)
    circuit_breaker &= benchmark_ret_63d.ge(min_benchmark_ret63d)
    circuit_breaker &= expected_move.le(max_expected_move) | expected_move.le(0.0)
    if require_benchmark_risk_on:
        circuit_breaker &= benchmark_risk_on
    if require_theme_active:
        circuit_breaker &= theme_active
    if require_price_above_ma50:
        circuit_breaker &= adj_close.gt(ma50).fillna(False)

    active = candidate & circuit_breaker
    blocked = candidate & ~circuit_breaker

    surprise_component = ((surprise_eps_pct - min_surprise_eps_pct) / max(25.0, abs(min_surprise_eps_pct))).clip(lower=0.0, upper=1.0)
    recency_component = (
        (max_days_since - days_since_earnings) / max(float(max_days_since - min_days_since + 1), 1.0)
    ).clip(lower=0.0, upper=1.0)
    rs_component = ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(lower=0.0, upper=1.0)
    theme_component = ((theme_score - min_theme) / max(100.0 - min_theme, 1e-9)).clip(lower=0.0, upper=1.0)
    extension_component = ((max_above_ma20 - above_ma20.fillna(max_above_ma20)) / max(max_above_ma20 + 0.05, 1e-9)).clip(
        lower=0.0,
        upper=1.0,
    )
    score = (
        35.0 * surprise_component
        + 25.0 * recency_component
        + 20.0 * rs_component
        + 10.0 * theme_component
        + 10.0 * extension_component
    ).where(active, 0.0)
    boost = ((score / 100.0) * max_boost).where(active, 0.0)

    out["post_earnings_drift_score"] = score.fillna(0.0)
    out["post_earnings_drift_boost"] = boost.fillna(0.0)
    out["post_earnings_drift_days_since_earnings"] = days_since_earnings
    out["post_earnings_drift_surprise_eps_pct"] = surprise_eps_pct
    out["post_earnings_drift_expected_move"] = expected_move
    out["post_earnings_drift_eligible"] = active.fillna(False)
    out["post_earnings_drift_blocked"] = blocked.fillna(False)
    out["final_score"] = (base_score + boost.fillna(0.0)).clip(0.0, 100.0)
    if active.any() or blocked.any():
        LOGGER.info(
            "momentum_post_earnings_drift_overlay_applied",
            extra={
                "event": "momentum_post_earnings_drift_overlay_applied",
                "rows": int(active.sum()),
                "blocked_rows": int(blocked.sum()),
                "avg_boost": float(boost.loc[active].mean()) if active.any() else 0.0,
                "avg_days_since_earnings": float(days_since_earnings.loc[active].mean()) if active.any() else 0.0,
                "avg_surprise_eps_pct": float(surprise_eps_pct.loc[active].mean()) if active.any() else 0.0,
                "avg_expected_move": float(expected_move.loc[active].mean()) if active.any() else 0.0,
                "min_surprise_eps_pct": min_surprise_eps_pct,
                "min_days_since_earnings": min_days_since,
                "max_days_since_earnings": max_days_since,
                "min_relative_strength_score": min_rs,
                "min_theme_score": min_theme,
                "min_mom_return": min_mom_return,
                "max_above_ma20_pct": max_above_ma20,
                "max_overnight_gap_risk_score_circuit_breaker": max_gap_risk,
                "max_expected_move_pct_circuit_breaker": max_expected_move,
                "min_benchmark_ret63d_circuit_breaker": min_benchmark_ret63d,
                "require_benchmark_risk_on": require_benchmark_risk_on,
                "require_theme_active": require_theme_active,
                "require_price_above_ma50": require_price_above_ma50,
                "max_score_boost": max_boost,
            },
        )
    return out


def _apply_boundary_rs_theme_credit_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost high-RS/theme names just below the long cutoff with hard breakers.

    Feature flag: ``boundary_rs_theme_credit_overlay`` defaults to false.
    Reasonable ranges:
    - ``boundary_rs_theme_credit_rank_buffer_below``: 0.05 to 0.18.
    - ``boundary_rs_theme_credit_rank_buffer_above``: 0.00 to 0.05.
    - ``boundary_rs_theme_credit_min_relative_strength_score``: 80 to 95.
    - ``boundary_rs_theme_credit_min_theme_score``: 60 to 85.
    - ``boundary_rs_theme_credit_min_technical_score``: 50 to 70.
    - ``boundary_rs_theme_credit_min_drawdown_from_high``: 0.00 to 0.08.
    - ``boundary_rs_theme_credit_max_drawdown_from_high``: 0.05 to 0.18.
    - ``boundary_rs_theme_credit_max_above_ma20_pct``: 0.02 to 0.10.
    - ``boundary_rs_theme_credit_min_adx_circuit_breaker``: 15 to 35.
    Circuit breakers: risk-on/theme-active state, price above MA50, bounded
    event and overnight-gap risk, minimum ADX, and a capped score boost keep
    this as a narrow boundary credit rather than a broad chase overlay.
    """

    out = frame.copy()
    out["boundary_rs_theme_credit_score"] = 0.0
    out["boundary_rs_theme_credit_boost"] = 0.0
    out["boundary_rs_theme_credit_base_rank"] = 0.0
    out["boundary_rs_theme_credit_blocked"] = False
    out["boundary_rs_theme_credit_eligible"] = False
    if not _bool_param(params, "boundary_rs_theme_credit_overlay", False):
        return out

    max_boost = float(np.clip(params.get("boundary_rs_theme_credit_max_score_boost", 1.75), 0.0, 5.0))
    if max_boost <= 0.0:
        return out

    base_score = pd.to_numeric(out.get("final_score", 50.0), errors="coerce").fillna(50.0)
    base_rank = base_score.groupby(out["date"]).rank(pct=True, method="average").fillna(0.0)
    long_q = float(np.clip(params.get("long_quantile", 0.20), 0.0, 1.0))
    boundary_rank = float(np.clip(1.0 - long_q, 0.0, 1.0))
    buffer_below = max(float(params.get("boundary_rs_theme_credit_rank_buffer_below", 0.10)), 0.0)
    buffer_above = max(float(params.get("boundary_rs_theme_credit_rank_buffer_above", 0.03)), 0.0)
    min_rank = float(np.clip(boundary_rank - buffer_below, 0.0, 1.0))
    max_rank = float(np.clip(boundary_rank + buffer_above, min_rank, 1.0))

    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan)
    distance_high = pd.to_numeric(out.get("mom_distance_high", out.get("distance_to_prior_high_252")), errors="coerce")
    drawdown_from_high = (-distance_high).clip(lower=0.0)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    technical_score = pd.to_numeric(out.get("technical_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(_column(out, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(out, "benchmark_risk_on", False).fillna(False).astype(bool)
    trend_adx = pd.to_numeric(_column(out, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    if "theme_active" in out:
        theme_active = pd.Series(out["theme_active"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index)

    min_rs = float(params.get("boundary_rs_theme_credit_min_relative_strength_score", 88.0))
    min_theme = float(params.get("boundary_rs_theme_credit_min_theme_score", 70.0))
    min_technical = float(params.get("boundary_rs_theme_credit_min_technical_score", 54.0))
    min_mom_return = float(params.get("boundary_rs_theme_credit_min_mom_return", 0.0))
    min_drawdown = max(float(params.get("boundary_rs_theme_credit_min_drawdown_from_high", 0.0)), 0.0)
    max_drawdown = max(float(params.get("boundary_rs_theme_credit_max_drawdown_from_high", 0.12)), min_drawdown)
    max_above_ma20 = max(float(params.get("boundary_rs_theme_credit_max_above_ma20_pct", 0.08)), 0.0)
    max_event_risk = float(params.get("boundary_rs_theme_credit_max_event_risk_score_circuit_breaker", 15.0))
    max_gap_risk = float(params.get("boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker", 25.0))
    min_benchmark_ret63d = float(params.get("boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_adx = float(params.get("boundary_rs_theme_credit_min_adx_circuit_breaker", 18.0))
    require_benchmark_risk_on = _bool_param(params, "boundary_rs_theme_credit_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "boundary_rs_theme_credit_require_theme_active", True)
    require_price_above_ma50 = _bool_param(params, "boundary_rs_theme_credit_require_price_above_ma50", True)

    boundary_window = base_rank.between(min_rank, max_rank, inclusive="both")
    leadership = rs_score.ge(min_rs) & theme_score.ge(min_theme) & technical_score.ge(min_technical) & mom_return.ge(min_mom_return)
    mild_reset = drawdown_from_high.between(min_drawdown, max_drawdown, inclusive="both") & above_ma20.le(max_above_ma20).fillna(False)
    if min_drawdown <= 0.0:
        mild_reset |= drawdown_from_high.le(max_drawdown) & above_ma20.le(max_above_ma20).fillna(False)
    if require_price_above_ma50:
        leadership &= adj_close.gt(ma50).fillna(False)

    candidate = boundary_window & leadership & mild_reset
    circuit_breaker = (
        event_risk.le(max_event_risk)
        & overnight_gap_risk.le(max_gap_risk)
        & benchmark_ret_63d.ge(min_benchmark_ret63d)
        & trend_adx.ge(min_adx)
    )
    if require_benchmark_risk_on:
        circuit_breaker &= benchmark_risk_on
    if require_theme_active:
        circuit_breaker &= theme_active

    active = candidate & circuit_breaker
    blocked = candidate & ~circuit_breaker

    rank_component = ((base_rank - min_rank) / max(max_rank - min_rank, 1e-9)).clip(lower=0.0, upper=1.0)
    rs_component = ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(lower=0.0, upper=1.0)
    theme_component = ((theme_score - min_theme) / max(100.0 - min_theme, 1e-9)).clip(lower=0.0, upper=1.0)
    pullback_component = ((drawdown_from_high - min_drawdown) / max(max_drawdown - min_drawdown, 1e-9)).clip(lower=0.0, upper=1.0)
    if min_drawdown <= 0.0:
        pullback_component = (drawdown_from_high / max(max_drawdown, 1e-9)).clip(lower=0.0, upper=1.0)
    score = (
        40.0 * rank_component
        + 25.0 * rs_component
        + 20.0 * theme_component
        + 15.0 * pullback_component
    ).where(active, 0.0)
    boost = (score / 100.0 * max_boost).clip(lower=0.0, upper=max_boost)

    out["boundary_rs_theme_credit_score"] = score.fillna(0.0)
    out["boundary_rs_theme_credit_boost"] = boost.fillna(0.0)
    out["boundary_rs_theme_credit_base_rank"] = base_rank.fillna(0.0)
    out["boundary_rs_theme_credit_blocked"] = blocked.fillna(False)
    out["boundary_rs_theme_credit_eligible"] = active.fillna(False)
    out["final_score"] = (base_score + boost.fillna(0.0)).clip(0.0, 100.0)
    if active.any() or blocked.any():
        LOGGER.info(
            "momentum_boundary_rs_theme_credit_overlay_applied",
            extra={
                "event": "momentum_boundary_rs_theme_credit_overlay_applied",
                "rows": int(active.sum()),
                "blocked_rows": int(blocked.sum()),
                "avg_boost": float(boost.loc[active].mean()) if active.any() else 0.0,
                "avg_base_rank": float(base_rank.loc[active].mean()) if active.any() else 0.0,
                "avg_drawdown_from_high": float(drawdown_from_high.loc[active].mean()) if active.any() else 0.0,
                "rank_buffer_below": buffer_below,
                "rank_buffer_above": buffer_above,
                "min_relative_strength_score": min_rs,
                "min_theme_score": min_theme,
                "min_technical_score": min_technical,
                "min_mom_return": min_mom_return,
                "min_drawdown_from_high": min_drawdown,
                "max_drawdown_from_high": max_drawdown,
                "max_above_ma20_pct": max_above_ma20,
                "max_event_risk_score_circuit_breaker": max_event_risk,
                "max_overnight_gap_risk_score_circuit_breaker": max_gap_risk,
                "min_benchmark_ret63d_circuit_breaker": min_benchmark_ret63d,
                "min_adx_circuit_breaker": min_adx,
                "require_benchmark_risk_on": require_benchmark_risk_on,
                "require_theme_active": require_theme_active,
                "require_price_above_ma50": require_price_above_ma50,
            },
        )
    return out


def _apply_boundary_rank_promotion_overlay(frame: pd.DataFrame, params: dict, long_quantile: float) -> pd.DataFrame:
    """Promote a few near-cutoff long candidates by rewriting boundary scores.

    Feature flag: ``boundary_rank_promotion_overlay`` defaults to false.
    Optional refinement flag: ``boundary_rank_promotion_compound_pullback_overlay``
    defaults to false and restricts promotions to long-horizon leaders that are
    pausing on a controlled short-term reset.
    Reasonable ranges:
    - ``boundary_rank_promotion_rank_buffer_below``: 0.03 to 0.12.
    - ``boundary_rank_promotion_min_relative_strength_score``: 85 to 96.
    - ``boundary_rank_promotion_min_theme_score``: 60 to 85.
    - ``boundary_rank_promotion_min_technical_score``: 55 to 75.
    - ``boundary_rank_promotion_min_mom_return``: 0.00 to 0.20.
    - ``boundary_rank_promotion_min_drawdown_from_high``: 0.00 to 0.08.
    - ``boundary_rank_promotion_max_drawdown_from_high``: 0.04 to 0.18.
    - ``boundary_rank_promotion_max_above_ma20_pct``: 0.02 to 0.10.
    - ``boundary_rank_promotion_max_final_score_deficit``: 0.5 to 4.0.
    - ``boundary_rank_promotion_max_promotions_per_date``: 1 to 3.
    - ``boundary_rank_promotion_promoted_score_step``: 0.05 to 0.50.
    - ``boundary_rank_promotion_compound_pullback_min_ret_100d_rank``:
      0.50 to 0.85.
    - ``boundary_rank_promotion_compound_pullback_min_252d_voladj_rank``:
      0.50 to 0.88.
    - ``boundary_rank_promotion_compound_pullback_max_ret_5d``:
      -0.02 to 0.03.
    - ``boundary_rank_promotion_compound_pullback_max_ret_10d``:
      0.00 to 0.06.
    - ``boundary_rank_promotion_compound_pullback_max_volume_expansion``:
      0.75 to 1.25.

    Circuit breakers: the overlay only considers names just below the long
    cutoff, caps the permitted score deficit versus the weakest currently
    admitted long, limits promotions per date, and keeps benchmark/theme/MA50,
    event-risk, overnight-gap-risk, and ADX breakers intact. This makes the
    mutation directly observable in realized ranks without turning it into a
    broad score overlay.
    """

    out = frame.copy()
    out["boundary_rank_promotion_score"] = 0.0
    out["boundary_rank_promotion_base_rank"] = 0.0
    out["boundary_rank_promotion_score_gap"] = np.nan
    out["boundary_rank_promotion_promoted"] = False
    out["boundary_rank_promotion_blocked"] = False
    out["boundary_rank_promotion_compound_pullback_eligible"] = False
    out["boundary_rank_promotion_compound_pullback_ret_100d_rank"] = 0.0
    out["boundary_rank_promotion_compound_pullback_252d_voladj_rank"] = 0.0
    out["boundary_rank_promotion_compound_pullback_ret_5d"] = np.nan
    out["boundary_rank_promotion_compound_pullback_ret_10d"] = np.nan
    out["boundary_rank_promotion_compound_pullback_volume_expansion"] = 0.0
    if not _bool_param(params, "boundary_rank_promotion_overlay", False):
        return out

    max_promotions_per_date = max(int(params.get("boundary_rank_promotion_max_promotions_per_date", 1)), 0)
    if max_promotions_per_date <= 0:
        return out

    base_score = pd.to_numeric(out.get("final_score", 50.0), errors="coerce").fillna(50.0)
    base_rank = base_score.groupby(out["date"]).rank(pct=True, method="average").fillna(0.0)
    out["boundary_rank_promotion_base_rank"] = base_rank
    boundary_rank = float(np.clip(1.0 - float(np.clip(long_quantile, 0.0, 1.0)), 0.0, 1.0))
    buffer_below = max(float(params.get("boundary_rank_promotion_rank_buffer_below", 0.08)), 0.0)
    min_rank = float(np.clip(boundary_rank - buffer_below, 0.0, boundary_rank))

    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    grouped = out.groupby("symbol", group_keys=False)
    date_key = pd.to_datetime(out["date"]).dt.normalize()
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan)
    distance_high = pd.to_numeric(out.get("mom_distance_high", out.get("distance_to_prior_high_252")), errors="coerce")
    drawdown_from_high = (-distance_high).clip(lower=0.0)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    technical_score = pd.to_numeric(out.get("technical_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    ret_5d = pd.to_numeric(out.get("ret_5d"), errors="coerce") if "ret_5d" in out else grouped["adj_close"].pct_change(5)
    ret_10d = pd.to_numeric(out.get("ret_10d"), errors="coerce") if "ret_10d" in out else grouped["adj_close"].pct_change(10)
    ret_100d = pd.to_numeric(out.get("ret_100d"), errors="coerce") if "ret_100d" in out else grouped["adj_close"].pct_change(100)
    ret_100d_rank = ret_100d.groupby(date_key).rank(pct=True, method="average").fillna(0.0)
    delayed_252 = grouped["adj_close"].shift(21)
    base_252 = grouped["adj_close"].shift(273)
    mom_252 = delayed_252 / base_252 - 1.0
    vol_252 = grouped["return_1d"].transform(lambda s: s.shift(21).rolling(252, min_periods=max(20, 252 // 3)).std())
    mom_252_voladj = mom_252 / vol_252.replace(0, np.nan)
    rank_252 = mom_252_voladj.groupby(date_key).rank(pct=True, method="average").fillna(0.0)
    volume_expansion = pd.to_numeric(_column(out, "volume_expansion", 1.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    event_risk = pd.to_numeric(_column(out, "event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(_column(out, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(out, "benchmark_risk_on", False).fillna(False).astype(bool)
    trend_adx = pd.to_numeric(_column(out, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    if "theme_active" in out:
        theme_active = pd.Series(out["theme_active"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index)

    min_rs = float(params.get("boundary_rank_promotion_min_relative_strength_score", 88.0))
    min_theme = float(params.get("boundary_rank_promotion_min_theme_score", 68.0))
    min_technical = float(params.get("boundary_rank_promotion_min_technical_score", 58.0))
    min_mom_return = float(params.get("boundary_rank_promotion_min_mom_return", 0.05))
    min_drawdown = max(float(params.get("boundary_rank_promotion_min_drawdown_from_high", 0.01)), 0.0)
    max_drawdown = max(float(params.get("boundary_rank_promotion_max_drawdown_from_high", 0.12)), min_drawdown)
    max_above_ma20 = max(float(params.get("boundary_rank_promotion_max_above_ma20_pct", 0.05)), 0.0)
    max_final_score_deficit = max(float(params.get("boundary_rank_promotion_max_final_score_deficit", 3.0)), 0.0)
    promoted_score_step = max(float(params.get("boundary_rank_promotion_promoted_score_step", 0.10)), 0.0)
    max_event_risk = float(params.get("boundary_rank_promotion_max_event_risk_score_circuit_breaker", 15.0))
    max_gap_risk = float(params.get("boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker", 25.0))
    min_benchmark_ret63d = float(params.get("boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_adx = float(params.get("boundary_rank_promotion_min_adx_circuit_breaker", 18.0))
    require_benchmark_risk_on = _bool_param(params, "boundary_rank_promotion_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "boundary_rank_promotion_require_theme_active", True)
    require_price_above_ma50 = _bool_param(params, "boundary_rank_promotion_require_price_above_ma50", True)
    compound_pullback_overlay = _bool_param(params, "boundary_rank_promotion_compound_pullback_overlay", False)
    compound_min_ret_100d_rank = float(np.clip(params.get("boundary_rank_promotion_compound_pullback_min_ret_100d_rank", 0.55), 0.0, 1.0))
    compound_min_252d_voladj_rank = float(
        np.clip(params.get("boundary_rank_promotion_compound_pullback_min_252d_voladj_rank", 0.55), 0.0, 1.0)
    )
    compound_max_ret_5d = float(params.get("boundary_rank_promotion_compound_pullback_max_ret_5d", 0.015))
    compound_max_ret_10d = float(params.get("boundary_rank_promotion_compound_pullback_max_ret_10d", 0.03))
    compound_max_volume_expansion = max(
        float(params.get("boundary_rank_promotion_compound_pullback_max_volume_expansion", 1.10)),
        0.0,
    )

    near_cutoff = base_rank.ge(min_rank) & base_rank.lt(boundary_rank)
    quality = (
        rs_score.ge(min_rs)
        & theme_score.ge(min_theme)
        & technical_score.ge(min_technical)
        & mom_return.ge(min_mom_return)
        & drawdown_from_high.between(min_drawdown, max_drawdown, inclusive="both")
        & above_ma20.le(max_above_ma20).fillna(False)
    )
    if require_price_above_ma50:
        quality &= adj_close.gt(ma50).fillna(False)
    compound_pullback_eligible = (
        ret_100d_rank.ge(compound_min_ret_100d_rank)
        & rank_252.ge(compound_min_252d_voladj_rank)
        & ret_5d.le(compound_max_ret_5d).fillna(False)
        & ret_10d.le(compound_max_ret_10d).fillna(False)
        & volume_expansion.le(compound_max_volume_expansion)
    )
    if compound_pullback_overlay:
        quality &= compound_pullback_eligible

    circuit_breaker = (
        event_risk.le(max_event_risk)
        & overnight_gap_risk.le(max_gap_risk)
        & benchmark_ret_63d.ge(min_benchmark_ret63d)
        & trend_adx.ge(min_adx)
    )
    if require_benchmark_risk_on:
        circuit_breaker &= benchmark_risk_on
    if require_theme_active:
        circuit_breaker &= theme_active

    candidate = near_cutoff & quality
    blocked = candidate & ~circuit_breaker
    out["boundary_rank_promotion_compound_pullback_eligible"] = (
        compound_pullback_eligible if compound_pullback_overlay else pd.Series(False, index=out.index, dtype=bool)
    ).fillna(False)
    out["boundary_rank_promotion_compound_pullback_ret_100d_rank"] = ret_100d_rank.fillna(0.0)
    out["boundary_rank_promotion_compound_pullback_252d_voladj_rank"] = rank_252.fillna(0.0)
    out["boundary_rank_promotion_compound_pullback_ret_5d"] = ret_5d
    out["boundary_rank_promotion_compound_pullback_ret_10d"] = ret_10d
    out["boundary_rank_promotion_compound_pullback_volume_expansion"] = volume_expansion.fillna(1.0)

    rank_component = ((base_rank - min_rank) / max(boundary_rank - min_rank, 1e-9)).clip(lower=0.0, upper=1.0)
    rs_component = ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(lower=0.0, upper=1.0)
    theme_component = ((theme_score - min_theme) / max(100.0 - min_theme, 1e-9)).clip(lower=0.0, upper=1.0)
    pullback_component = ((drawdown_from_high - min_drawdown) / max(max_drawdown - min_drawdown, 1e-9)).clip(lower=0.0, upper=1.0)
    promotion_score = (35.0 * rank_component + 25.0 * rs_component + 20.0 * theme_component + 20.0 * pullback_component).where(
        candidate & circuit_breaker,
        0.0,
    )
    if compound_pullback_overlay:
        ret_100d_component = ((ret_100d_rank - compound_min_ret_100d_rank) / max(1.0 - compound_min_ret_100d_rank, 1e-9)).clip(
            lower=0.0,
            upper=1.0,
        )
        rank_252_component = ((rank_252 - compound_min_252d_voladj_rank) / max(1.0 - compound_min_252d_voladj_rank, 1e-9)).clip(
            lower=0.0,
            upper=1.0,
        )
        ret5_component = ((compound_max_ret_5d - ret_5d) / max(abs(compound_max_ret_5d) + 0.08, 1e-9)).clip(lower=0.0, upper=1.0)
        ret10_component = ((compound_max_ret_10d - ret_10d) / max(abs(compound_max_ret_10d) + 0.12, 1e-9)).clip(lower=0.0, upper=1.0)
        volume_component = ((compound_max_volume_expansion - volume_expansion) / max(compound_max_volume_expansion + 0.25, 1e-9)).clip(
            lower=0.0,
            upper=1.0,
        )
        promotion_score = (
            20.0 * rank_component
            + 20.0 * rs_component
            + 15.0 * theme_component
            + 15.0 * pullback_component
            + 15.0 * ret_100d_component
            + 10.0 * rank_252_component
            + 3.0 * ret5_component
            + 1.0 * ret10_component
            + 1.0 * volume_component
        ).where(candidate & circuit_breaker, 0.0)
    out["boundary_rank_promotion_score"] = promotion_score.fillna(0.0)

    promoted_mask = pd.Series(False, index=out.index, dtype=bool)
    score_gap = pd.Series(np.nan, index=out.index, dtype=float)
    for date, date_rows in out.groupby("date"):
        base_selected = base_rank.loc[date_rows.index].ge(boundary_rank)
        if not base_selected.any():
            continue
        reference_score = float(base_score.loc[date_rows.index[base_selected]].min())
        active_rows = date_rows.loc[candidate.loc[date_rows.index] & circuit_breaker.loc[date_rows.index]].copy()
        if active_rows.empty:
            continue
        active_rows["boundary_rank_promotion_score_gap"] = reference_score - base_score.loc[active_rows.index]
        active_rows = active_rows[
            active_rows["boundary_rank_promotion_score_gap"].le(max_final_score_deficit)
        ].sort_values(
            ["boundary_rank_promotion_score", "boundary_rank_promotion_base_rank", "final_score"],
            ascending=[False, False, False],
        )
        if active_rows.empty:
            continue
        chosen_rows = active_rows.head(max_promotions_per_date)
        for offset, (idx, row) in enumerate(chosen_rows.iterrows(), start=1):
            promoted_target = max(float(base_score.loc[idx]), reference_score + promoted_score_step * offset)
            out.at[idx, "final_score"] = promoted_target
            promoted_mask.loc[idx] = True
            score_gap.loc[idx] = float(row["boundary_rank_promotion_score_gap"])

    out["boundary_rank_promotion_score_gap"] = score_gap
    out["boundary_rank_promotion_promoted"] = promoted_mask
    out["boundary_rank_promotion_blocked"] = blocked.fillna(False)
    if promoted_mask.any() or blocked.any():
        LOGGER.info(
            "momentum_boundary_rank_promotion_overlay_applied",
            extra={
                "event": "momentum_boundary_rank_promotion_overlay_applied",
                "rows": int(promoted_mask.sum()),
                "blocked_rows": int(blocked.sum()),
                "dates": int(pd.to_datetime(out.loc[promoted_mask, "date"]).nunique()) if promoted_mask.any() else 0,
                "avg_base_rank": float(base_rank.loc[promoted_mask].mean()) if promoted_mask.any() else 0.0,
                "avg_score_gap": float(score_gap.loc[promoted_mask].mean()) if promoted_mask.any() else 0.0,
                "rank_buffer_below": buffer_below,
                "max_final_score_deficit": max_final_score_deficit,
                "max_promotions_per_date": max_promotions_per_date,
                "promoted_score_step": promoted_score_step,
                "min_relative_strength_score": min_rs,
                "min_theme_score": min_theme,
                "min_technical_score": min_technical,
                "min_mom_return": min_mom_return,
                "min_drawdown_from_high": min_drawdown,
                "max_drawdown_from_high": max_drawdown,
                "max_above_ma20_pct": max_above_ma20,
                "max_event_risk_score_circuit_breaker": max_event_risk,
                "max_overnight_gap_risk_score_circuit_breaker": max_gap_risk,
                "min_benchmark_ret63d_circuit_breaker": min_benchmark_ret63d,
                "min_adx_circuit_breaker": min_adx,
                "require_benchmark_risk_on": require_benchmark_risk_on,
                "require_theme_active": require_theme_active,
                "require_price_above_ma50": require_price_above_ma50,
                "compound_pullback_overlay": compound_pullback_overlay,
                "compound_pullback_rows": int((candidate & circuit_breaker & compound_pullback_eligible).sum())
                if compound_pullback_overlay
                else 0,
                "compound_pullback_min_ret_100d_rank": compound_min_ret_100d_rank,
                "compound_pullback_min_252d_voladj_rank": compound_min_252d_voladj_rank,
                "compound_pullback_max_ret_5d": compound_max_ret_5d,
                "compound_pullback_max_ret_10d": compound_max_ret_10d,
                "compound_pullback_max_volume_expansion": compound_max_volume_expansion,
            },
        )
    return out


def _apply_exit_quality_rank_credit_overlay(frame: pd.DataFrame, params: dict, long_quantile: float) -> pd.DataFrame:
    """Credit near-cutoff leaders without changing their raw score.

    Feature flag: ``exit_quality_rank_credit_overlay`` defaults to false.
    Optional refinement flag: ``exit_quality_rank_credit_theme_support_overlay``
    defaults to false and only preserves the rank credit when same-theme peer
    support is still improving point-in-time. Optional refinement flag:
    ``exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay``
    defaults to false and swaps the reset-support breadth check from broad
    date-level activity to same-theme peer activity. Optional refinement flag:
    ``exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay``
    defaults to false and further requires those same-theme peers to be
    showing valid short-term reset participation, not just generic theme
    activity.
    Reasonable ranges:
    - ``exit_quality_rank_credit_rank_buffer_below``: 0.03 to 0.10.
    - ``exit_quality_rank_credit_rank_buffer_above``: 0.00 to 0.03.
    - ``exit_quality_rank_credit_min_final_score``: 60 to 80.
    - ``exit_quality_rank_credit_min_relative_strength_score``: 80 to 95.
    - ``exit_quality_rank_credit_min_mom_return``: 0.08 to 0.30.
    - ``exit_quality_rank_credit_max_score_rank_credit``: 0.01 to 0.06.
    - ``exit_quality_rank_credit_theme_support_lookback_days``: 3 to 15.
    - ``exit_quality_rank_credit_theme_support_min_peer_count``: 2 to 5.
    - ``exit_quality_rank_credit_theme_support_min_theme_score_prior``: 55 to 75.
    - ``exit_quality_rank_credit_theme_support_min_peer_rs_prior``: 60 to 85.
    - ``exit_quality_rank_credit_theme_support_min_combined_delta``: 0 to 8.
    - ``exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay``:
      true/false.
    - ``exit_quality_rank_credit_reset_support_same_theme_peer_min_count``:
      2 to 5.
    - ``exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay``:
      true/false.
    - ``exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share``:
      0.25 to 0.75.
    - ``exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count``:
      2 to 5.
    - ``exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay``:
      true/false.
    - ``exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs``:
      70 to 95.
    - ``exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay``:
      true/false.
    - ``exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct``:
      0.65 to 0.95.
    - ``exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap``:
      2 to 12.
    - ``exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count``:
      2 to 5.

    Circuit breakers keep this as a narrow selection tiebreaker: event/gap
    risk caps, benchmark risk-on, price above MA50, bounded drawdown from the
    prior high, and a stretch cap above MA20. This does not alter
    ``final_score``; it only creates an effective rank for long selection.
    """

    out = frame.copy()
    base_rank = pd.to_numeric(out.get("score_rank", 0.0), errors="coerce").fillna(0.0)
    out["exit_quality_rank_credit_score"] = 0.0
    out["exit_quality_rank_credit"] = 0.0
    out["exit_quality_base_rank"] = base_rank
    out["exit_quality_effective_score_rank"] = base_rank
    out["exit_quality_rank_credit_blocked"] = False
    out["exit_quality_rank_credit_eligible"] = False
    out["exit_quality_theme_support_score"] = 0.0
    out["exit_quality_theme_support_eligible"] = False
    out["exit_quality_reset_breadth_same_theme_peer_count"] = 0
    out["exit_quality_reset_breadth_same_theme_peer_reset_share"] = 0.0
    out["exit_quality_reset_breadth_same_theme_peer_reset_count"] = 0
    out["exit_quality_reset_breadth_same_theme_peer_reset_rs_mean"] = 0.0
    out["exit_quality_reset_same_theme_leader_rank_pct"] = 0.0
    out["exit_quality_reset_same_theme_leader_score_gap"] = 0.0
    out["exit_quality_reset_same_theme_leader_guard_eligible"] = False
    if not _bool_param(params, "exit_quality_rank_credit_overlay", False):
        return out

    max_rank_credit = float(np.clip(params.get("exit_quality_rank_credit_max_score_rank_credit", 0.04), 0.0, 0.08))
    if max_rank_credit <= 0.0:
        return out

    boundary_rank = float(np.clip(1.0 - float(np.clip(long_quantile, 0.0, 1.0)), 0.0, 1.0))
    buffer_below = max(float(params.get("exit_quality_rank_credit_rank_buffer_below", 0.07)), 0.0)
    buffer_above = max(float(params.get("exit_quality_rank_credit_rank_buffer_above", 0.01)), 0.0)
    min_rank = float(np.clip(boundary_rank - buffer_below, 0.0, 1.0))
    max_rank = float(np.clip(boundary_rank + buffer_above, min_rank, 1.0))

    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan)
    distance_high = pd.to_numeric(out.get("mom_distance_high", out.get("distance_to_prior_high_252")), errors="coerce")
    drawdown_from_high = (-distance_high).clip(lower=0.0).fillna(np.inf)
    final_score_value = pd.to_numeric(out.get("final_score", 50.0), errors="coerce").fillna(50.0)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    technical_score = pd.to_numeric(out.get("technical_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(_column(out, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(out, "benchmark_risk_on", False).fillna(False).astype(bool)
    trend_adx = pd.to_numeric(_column(out, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    if "theme_active" in out:
        theme_active = pd.Series(out["theme_active"], index=out.index).astype("boolean").fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index)

    min_final = float(params.get("exit_quality_rank_credit_min_final_score", 68.0))
    min_rs = float(params.get("exit_quality_rank_credit_min_relative_strength_score", 88.0))
    min_theme = float(params.get("exit_quality_rank_credit_min_theme_score", 62.0))
    min_technical = float(params.get("exit_quality_rank_credit_min_technical_score", 60.0))
    min_mom_return = float(params.get("exit_quality_rank_credit_min_mom_return", 0.12))
    max_drawdown = max(float(params.get("exit_quality_rank_credit_max_drawdown_from_high", 0.18)), 0.0)
    max_above_ma20 = max(float(params.get("exit_quality_rank_credit_max_above_ma20_pct", 0.08)), 0.0)
    max_event_risk = float(params.get("exit_quality_rank_credit_max_event_risk_score_circuit_breaker", 20.0))
    max_gap_risk = float(params.get("exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker", 30.0))
    min_benchmark_ret63d = float(params.get("exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_adx = float(params.get("exit_quality_rank_credit_min_adx_circuit_breaker", 18.0))
    require_benchmark_risk_on = _bool_param(params, "exit_quality_rank_credit_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "exit_quality_rank_credit_require_theme_active", False)
    require_price_above_ma50 = _bool_param(params, "exit_quality_rank_credit_require_price_above_ma50", True)
    theme_support_overlay = _bool_param(params, "exit_quality_rank_credit_theme_support_overlay", False)
    reset_breadth_overlay = _bool_param(params, "exit_quality_rank_credit_reset_support_overlay", False)

    boundary_window = base_rank.between(min_rank, max_rank, inclusive="both")
    quality = (
        final_score_value.ge(min_final)
        & rs_score.ge(min_rs)
        & theme_score.ge(min_theme)
        & technical_score.ge(min_technical)
        & mom_return.ge(min_mom_return)
        & drawdown_from_high.le(max_drawdown)
        & above_ma20.le(max_above_ma20).fillna(False)
    )
    if require_price_above_ma50:
        quality &= adj_close.gt(ma50).fillna(False)

    candidate = boundary_window & quality
    circuit_breaker = (
        event_risk.le(max_event_risk)
        & overnight_gap_risk.le(max_gap_risk)
        & benchmark_ret_63d.ge(min_benchmark_ret63d)
        & trend_adx.ge(min_adx)
    )
    if require_benchmark_risk_on:
        circuit_breaker &= benchmark_risk_on
    if require_theme_active:
        circuit_breaker &= theme_active

    theme_support_eligible = pd.Series(True, index=out.index, dtype=bool)
    theme_support_score = pd.Series(0.0, index=out.index, dtype=float)
    if theme_support_overlay:
        date_key = pd.to_datetime(out["date"]).dt.normalize()
        if "primary_theme" in out:
            theme_key = out["primary_theme"]
        elif "theme_reason" in out:
            theme_key = out["theme_reason"]
        else:
            theme_key = pd.Series("unknown", index=out.index)
        theme_key = (
            pd.Series(theme_key, index=out.index)
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
        )
        peer_count = out.groupby([date_key, theme_key])["symbol"].transform("nunique").fillna(0).astype(int)
        lookback_days = max(int(params.get("exit_quality_rank_credit_theme_support_lookback_days", 5)), 1)
        min_peer_count = max(int(params.get("exit_quality_rank_credit_theme_support_min_peer_count", 2)), 1)
        min_theme_score_prior = float(params.get("exit_quality_rank_credit_theme_support_min_theme_score_prior", 60.0))
        min_peer_rs_prior = float(params.get("exit_quality_rank_credit_theme_support_min_peer_rs_prior", 68.0))
        min_combined_delta = float(params.get("exit_quality_rank_credit_theme_support_min_combined_delta", 1.0))
        theme_daily = (
            pd.DataFrame(
                {
                    "date": date_key,
                    "theme": theme_key,
                    "theme_score": theme_score,
                    "relative_strength_score": rs_score,
                    "peer_count": peer_count,
                }
            )
            .groupby(["date", "theme"], as_index=False)
            .agg(
                theme_score_mean=("theme_score", "mean"),
                peer_rs_mean=("relative_strength_score", "mean"),
                peer_count=("peer_count", "max"),
            )
            .sort_values(["theme", "date"])
        )
        theme_daily["theme_score_prior"] = theme_daily.groupby("theme")["theme_score_mean"].shift(1)
        theme_daily["theme_score_base"] = theme_daily.groupby("theme")["theme_score_mean"].shift(lookback_days + 1)
        theme_daily["theme_score_delta"] = theme_daily["theme_score_prior"] - theme_daily["theme_score_base"]
        theme_daily["peer_rs_prior"] = theme_daily.groupby("theme")["peer_rs_mean"].shift(1)
        theme_daily["peer_rs_base"] = theme_daily.groupby("theme")["peer_rs_mean"].shift(lookback_days + 1)
        theme_daily["peer_rs_delta"] = theme_daily["peer_rs_prior"] - theme_daily["peer_rs_base"]
        out["_exit_quality_theme_date_key"] = date_key
        out["_exit_quality_theme_key"] = theme_key
        out = out.merge(
            theme_daily[
                [
                    "date",
                    "theme",
                    "peer_count",
                    "theme_score_prior",
                    "theme_score_delta",
                    "peer_rs_prior",
                    "peer_rs_delta",
                ]
            ],
            left_on=["_exit_quality_theme_date_key", "_exit_quality_theme_key"],
            right_on=["date", "theme"],
            how="left",
            suffixes=("", "_exit_quality_theme"),
        )
        out = out.drop(
            columns=["date_exit_quality_theme", "theme", "_exit_quality_theme_date_key", "_exit_quality_theme_key"],
            errors="ignore",
        )
        theme_peer_count = pd.to_numeric(out.get("peer_count", 0), errors="coerce").fillna(0).astype(int)
        theme_score_prior = pd.to_numeric(out.get("theme_score_prior", 0.0), errors="coerce").fillna(0.0)
        theme_score_delta = pd.to_numeric(out.get("theme_score_delta", 0.0), errors="coerce").fillna(0.0)
        peer_rs_prior = pd.to_numeric(out.get("peer_rs_prior", 0.0), errors="coerce").fillna(0.0)
        peer_rs_delta = pd.to_numeric(out.get("peer_rs_delta", 0.0), errors="coerce").fillna(0.0)
        combined_delta = (0.45 * theme_score_delta) + (0.55 * peer_rs_delta)
        theme_support_eligible = theme_key.ne("unknown")
        theme_support_eligible &= theme_peer_count.ge(min_peer_count)
        theme_support_eligible &= theme_score_prior.ge(min_theme_score_prior)
        theme_support_eligible &= peer_rs_prior.ge(min_peer_rs_prior)
        theme_support_eligible &= combined_delta.ge(min_combined_delta)
        theme_prior_component = ((theme_score_prior - min_theme_score_prior) / max(100.0 - min_theme_score_prior, 1e-9)).clip(lower=0.0, upper=1.0)
        peer_rs_component = ((peer_rs_prior - min_peer_rs_prior) / max(100.0 - min_peer_rs_prior, 1e-9)).clip(lower=0.0, upper=1.0)
        delta_component = (combined_delta / max(max(min_combined_delta, 1.0) * 4.0, 1e-9)).clip(lower=0.0, upper=1.0)
        theme_support_score = (35.0 * theme_prior_component + 40.0 * peer_rs_component + 25.0 * delta_component).where(theme_support_eligible, 0.0)
        out["exit_quality_theme_support_score"] = theme_support_score.fillna(0.0)
        out["exit_quality_theme_support_eligible"] = theme_support_eligible.fillna(False)

    reset_support_eligible = pd.Series(True, index=out.index, dtype=bool)
    reset_support_score = pd.Series(0.0, index=out.index, dtype=float)
    reset_breadth_active_share = pd.Series(1.0, index=out.index, dtype=float)
    reset_breadth_shortfall = pd.Series(0.0, index=out.index, dtype=float)
    reset_breadth_safe = pd.Series(True, index=out.index, dtype=bool)
    reset_same_theme_peer_count = pd.Series(0, index=out.index, dtype=int)
    reset_same_theme_peer_reset_share = pd.Series(0.0, index=out.index, dtype=float)
    reset_same_theme_peer_reset_count = pd.Series(0, index=out.index, dtype=int)
    reset_same_theme_peer_reset_rs_mean = pd.Series(0.0, index=out.index, dtype=float)
    reset_same_theme_leader_rank = pd.Series(0.0, index=out.index, dtype=float)
    reset_same_theme_leader_score_gap = pd.Series(0.0, index=out.index, dtype=float)
    reset_same_theme_leader_guard_eligible = pd.Series(True, index=out.index, dtype=bool)
    if reset_breadth_overlay:
        min_reset_score = float(params.get("exit_quality_rank_credit_reset_support_min_reset_score", 20.0))
        short_term_reset_score = pd.to_numeric(out.get("pullback_short_term_reset_score", 0.0), errors="coerce").fillna(0.0)
        short_term_reset_blocked = out.get("pullback_short_term_reset_blocked", False).fillna(False).astype(bool)
        lookback_days = max(int(params.get("exit_quality_rank_credit_reset_support_theme_breadth_lookback_days", 10)), 2)
        active_share_threshold = float(
            params.get("exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold", 0.42)
        )
        shortfall_threshold = max(
            float(params.get("exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold", 0.08)),
            0.0,
        )
        min_active_share = float(params.get("exit_quality_rank_credit_reset_support_theme_breadth_min_active_share", 0.40))
        require_expansion = _bool_param(
            params,
            "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion",
            False,
        )
        use_same_theme_peer_breadth = _bool_param(
            params,
            "exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay",
            False,
        )
        use_same_theme_peer_reset_breadth = _bool_param(
            params,
            "exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay",
            False,
        )
        same_theme_peer_min_count = max(
            int(params.get("exit_quality_rank_credit_reset_support_same_theme_peer_min_count", 2)),
            1,
        )
        same_theme_peer_reset_min_share = float(
            np.clip(
                params.get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share", 0.34),
                0.0,
                1.0,
            )
        )
        same_theme_peer_reset_min_count = max(
            int(
                params.get(
                    "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count",
                    same_theme_peer_min_count,
                )
            ),
            1,
        )
        use_same_theme_peer_quality = _bool_param(
            params,
            "exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay",
            False,
        )
        use_same_theme_leader_guard = _bool_param(
            params,
            "exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay",
            False,
        )
        same_theme_peer_reset_min_avg_rs = float(
            params.get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs", 84.0)
        )
        same_theme_leader_min_rank = float(
            np.clip(
                params.get("exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct", 0.78),
                0.0,
                1.0,
            )
        )
        same_theme_leader_max_score_gap = max(
            float(params.get("exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap", 6.0)),
            0.0,
        )
        same_theme_leader_min_peer_count = max(
            int(
                params.get(
                    "exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count",
                    same_theme_peer_min_count,
                )
            ),
            1,
        )
        expansion_threshold = max(
            float(params.get("exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold", 0.02)),
            0.0,
        )
        use_same_theme_peer_reset_breadth = use_same_theme_peer_breadth and use_same_theme_peer_reset_breadth
        need_same_theme_context = use_same_theme_peer_breadth or use_same_theme_leader_guard
        if need_same_theme_context:
            date_key = pd.to_datetime(out["date"]).dt.normalize()
            if "primary_theme" in out:
                theme_key = out["primary_theme"]
            elif "theme_reason" in out:
                theme_key = out["theme_reason"]
            else:
                theme_key = pd.Series("unknown", index=out.index)
            theme_key = (
                pd.Series(theme_key, index=out.index)
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
            )
            theme_group = [date_key, theme_key]
            same_theme_peer_count = out.groupby(theme_group)["symbol"].transform("nunique").fillna(0).astype(int)
            same_theme_final_rank = final_score_value.groupby(theme_group).rank(pct=True, method="average").fillna(0.0)
            same_theme_rs_rank = rs_score.groupby(theme_group).rank(pct=True, method="average").fillna(0.0)
            same_theme_technical_rank = technical_score.groupby(theme_group).rank(pct=True, method="average").fillna(0.0)
            reset_same_theme_leader_rank = (
                0.50 * same_theme_final_rank + 0.30 * same_theme_rs_rank + 0.20 * same_theme_technical_rank
            ).clip(lower=0.0, upper=1.0)
            same_theme_best_final_score = final_score_value.groupby(theme_group).transform("max").fillna(final_score_value)
            reset_same_theme_leader_score_gap = (same_theme_best_final_score - final_score_value).clip(lower=0.0).fillna(0.0)
            reset_same_theme_leader_guard_eligible = (
                theme_key.ne("unknown")
                & same_theme_peer_count.ge(same_theme_leader_min_peer_count)
                & reset_same_theme_leader_rank.ge(same_theme_leader_min_rank)
                & reset_same_theme_leader_score_gap.le(same_theme_leader_max_score_gap)
            )
        if use_same_theme_peer_breadth:
            theme_daily = (
                pd.DataFrame(
                    {
                        "date": date_key,
                        "theme": theme_key,
                        "symbol": out["symbol"].astype(str),
                        "theme_active_flag": theme_active.astype(float),
                        "reset_peer_rs_score": rs_score.where(
                            short_term_reset_score.ge(min_reset_score) & ~short_term_reset_blocked & theme_active,
                            np.nan,
                        ),
                        "reset_peer_active_flag": (
                            short_term_reset_score.ge(min_reset_score) & ~short_term_reset_blocked & theme_active
                        ).astype(float),
                    }
                )
                .groupby(["date", "theme"], as_index=False)
                .agg(
                    pullback_reset_theme_breadth_active_share=("theme_active_flag", "mean"),
                    exit_quality_reset_breadth_same_theme_peer_count=("symbol", "nunique"),
                    exit_quality_reset_breadth_same_theme_peer_reset_share=("reset_peer_active_flag", "mean"),
                    exit_quality_reset_breadth_same_theme_peer_reset_count=("reset_peer_active_flag", "sum"),
                    exit_quality_reset_breadth_same_theme_peer_reset_rs_mean=("reset_peer_rs_score", "mean"),
                )
                .sort_values(["theme", "date"])
            )
            min_periods = min(lookback_days, max(3, lookback_days // 2))
            theme_daily["pullback_reset_theme_breadth_baseline_share"] = (
                theme_daily.groupby("theme")["pullback_reset_theme_breadth_active_share"]
                .shift(1)
                .rolling(lookback_days, min_periods=min_periods)
                .mean()
            )
            theme_daily["pullback_reset_theme_breadth_shortfall"] = (
                theme_daily["pullback_reset_theme_breadth_baseline_share"]
                - theme_daily["pullback_reset_theme_breadth_active_share"]
            ).clip(lower=0.0)
            theme_daily["pullback_reset_theme_breadth_safe"] = (
                theme_daily["pullback_reset_theme_breadth_baseline_share"].isna()
                | ~(
                    theme_daily["pullback_reset_theme_breadth_active_share"].le(active_share_threshold)
                    & theme_daily["pullback_reset_theme_breadth_shortfall"].ge(shortfall_threshold)
                )
            )
            out["_exit_quality_reset_theme_date_key"] = date_key
            out["_exit_quality_reset_theme_key"] = theme_key
            out = out.merge(
                theme_daily,
                left_on=["_exit_quality_reset_theme_date_key", "_exit_quality_reset_theme_key"],
                right_on=["date", "theme"],
                how="left",
                suffixes=("", "_exit_quality_reset_theme"),
            )
            for column in (
                "pullback_reset_theme_breadth_active_share",
                "pullback_reset_theme_breadth_baseline_share",
                "pullback_reset_theme_breadth_shortfall",
                "pullback_reset_theme_breadth_safe",
                "exit_quality_reset_breadth_same_theme_peer_count",
                "exit_quality_reset_breadth_same_theme_peer_reset_share",
                "exit_quality_reset_breadth_same_theme_peer_reset_count",
                "exit_quality_reset_breadth_same_theme_peer_reset_rs_mean",
            ):
                ctx = f"{column}_exit_quality_reset_theme"
                if ctx in out:
                    out[column] = out[ctx].fillna(out[column]) if column in out else out[ctx]
                    out = out.drop(columns=[ctx])
            out = out.drop(
                columns=["date_exit_quality_reset_theme", "theme", "_exit_quality_reset_theme_date_key", "_exit_quality_reset_theme_key"],
                errors="ignore",
            )
            reset_breadth_active_share = pd.to_numeric(
                out.get("pullback_reset_theme_breadth_active_share", 1.0), errors="coerce"
            ).fillna(1.0)
            reset_breadth_shortfall = pd.to_numeric(
                out.get("pullback_reset_theme_breadth_shortfall", 0.0), errors="coerce"
            ).fillna(0.0)
            reset_breadth_safe = out.get("pullback_reset_theme_breadth_safe", True).fillna(True).astype(bool)
            reset_same_theme_peer_count = pd.to_numeric(
                out.get("exit_quality_reset_breadth_same_theme_peer_count", 0), errors="coerce"
            ).fillna(0).astype(int)
            reset_same_theme_peer_reset_share = pd.to_numeric(
                out.get("exit_quality_reset_breadth_same_theme_peer_reset_share", 0.0), errors="coerce"
            ).fillna(0.0)
            reset_same_theme_peer_reset_count = pd.to_numeric(
                out.get("exit_quality_reset_breadth_same_theme_peer_reset_count", 0), errors="coerce"
            ).fillna(0).astype(int)
            reset_same_theme_peer_reset_rs_mean = pd.to_numeric(
                out.get("exit_quality_reset_breadth_same_theme_peer_reset_rs_mean", 0.0), errors="coerce"
            ).fillna(0.0)
            breadth_confirmed = (
                reset_breadth_safe
                & reset_breadth_active_share.ge(min_active_share)
                & reset_same_theme_peer_count.ge(same_theme_peer_min_count)
            )
            if use_same_theme_peer_reset_breadth:
                breadth_confirmed &= reset_same_theme_peer_reset_share.ge(same_theme_peer_reset_min_share)
                breadth_confirmed &= reset_same_theme_peer_reset_count.ge(same_theme_peer_reset_min_count)
            if use_same_theme_peer_quality:
                breadth_confirmed &= reset_same_theme_peer_reset_rs_mean.ge(same_theme_peer_reset_min_avg_rs)
            if use_same_theme_leader_guard:
                breadth_confirmed &= reset_same_theme_leader_guard_eligible
            if require_expansion:
                expansion = (
                    reset_breadth_active_share
                    - pd.to_numeric(out.get("pullback_reset_theme_breadth_baseline_share", 1.0), errors="coerce").fillna(1.0)
                ).fillna(0.0)
                breadth_confirmed &= expansion.ge(expansion_threshold)
        else:
            out = _attach_pullback_reset_theme_breadth_safety_context(
                out,
                lookback_days=lookback_days,
                active_share_threshold=active_share_threshold,
                shortfall_threshold=shortfall_threshold,
            )
            reset_breadth_active_share = pd.to_numeric(
                out.get("pullback_reset_theme_breadth_active_share", 1.0), errors="coerce"
            ).fillna(1.0)
            reset_breadth_shortfall = pd.to_numeric(
                out.get("pullback_reset_theme_breadth_shortfall", 0.0), errors="coerce"
            ).fillna(0.0)
            reset_breadth_safe = out.get("pullback_reset_theme_breadth_safe", True).fillna(True).astype(bool)
            breadth_confirmed = reset_breadth_safe & reset_breadth_active_share.ge(min_active_share)
            if use_same_theme_leader_guard:
                breadth_confirmed &= reset_same_theme_leader_guard_eligible
            if require_expansion:
                expansion = (
                    reset_breadth_active_share
                    - pd.to_numeric(out.get("pullback_reset_theme_breadth_baseline_share", 1.0), errors="coerce").fillna(1.0)
                ).fillna(0.0)
                breadth_confirmed &= expansion.ge(expansion_threshold)
        reset_support_eligible = short_term_reset_score.ge(min_reset_score) & ~short_term_reset_blocked & breadth_confirmed
        breadth_component_source = reset_breadth_active_share
        breadth_component_floor = min_active_share
        if use_same_theme_peer_reset_breadth:
            breadth_component_source = reset_same_theme_peer_reset_share
            breadth_component_floor = same_theme_peer_reset_min_share
        reset_component = ((short_term_reset_score - min_reset_score) / max(100.0 - min_reset_score, 1e-9)).clip(lower=0.0, upper=1.0)
        breadth_component = (
            (breadth_component_source - breadth_component_floor) / max(1.0 - breadth_component_floor, 1e-9)
        ).clip(lower=0.0, upper=1.0)
        safety_component = (1.0 - (reset_breadth_shortfall / max(shortfall_threshold + 0.10, 1e-9))).clip(lower=0.0, upper=1.0)
        peer_reset_quality_component = (
            (reset_same_theme_peer_reset_rs_mean - same_theme_peer_reset_min_avg_rs)
            / max(100.0 - same_theme_peer_reset_min_avg_rs, 1e-9)
        ).clip(lower=0.0, upper=1.0)
        reset_support_score = 60.0 * reset_component + 25.0 * breadth_component + 15.0 * safety_component
        if use_same_theme_peer_quality:
            reset_support_score = (
                50.0 * reset_component
                + 20.0 * breadth_component
                + 15.0 * safety_component
                + 15.0 * peer_reset_quality_component
            )
        reset_support_score = reset_support_score.where(reset_support_eligible, 0.0)
        out["exit_quality_reset_support_score"] = reset_support_score.fillna(0.0)
        out["exit_quality_reset_support_eligible"] = reset_support_eligible.fillna(False)
        out["exit_quality_reset_breadth_active_share"] = reset_breadth_active_share.fillna(1.0)
        out["exit_quality_reset_breadth_shortfall"] = reset_breadth_shortfall.fillna(0.0)
        out["exit_quality_reset_breadth_safe"] = reset_breadth_safe.fillna(True)
        out["exit_quality_reset_breadth_same_theme_peer_count"] = reset_same_theme_peer_count.fillna(0).astype(int)
        out["exit_quality_reset_breadth_same_theme_peer_reset_share"] = reset_same_theme_peer_reset_share.fillna(0.0)
        out["exit_quality_reset_breadth_same_theme_peer_reset_count"] = reset_same_theme_peer_reset_count.fillna(0).astype(int)
        out["exit_quality_reset_breadth_same_theme_peer_reset_rs_mean"] = reset_same_theme_peer_reset_rs_mean.fillna(0.0)
        out["exit_quality_reset_same_theme_leader_rank_pct"] = reset_same_theme_leader_rank.fillna(0.0)
        out["exit_quality_reset_same_theme_leader_score_gap"] = reset_same_theme_leader_score_gap.fillna(0.0)
        out["exit_quality_reset_same_theme_leader_guard_eligible"] = reset_same_theme_leader_guard_eligible.fillna(False)

    support_eligible = theme_support_eligible & reset_support_eligible
    active = candidate & circuit_breaker & support_eligible
    blocked = candidate & ~(circuit_breaker & support_eligible)

    rank_component = ((base_rank - min_rank) / max(max_rank - min_rank, 1e-9)).clip(lower=0.0, upper=1.0)
    final_component = ((final_score_value - min_final) / max(100.0 - min_final, 1e-9)).clip(lower=0.0, upper=1.0)
    rs_component = ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(lower=0.0, upper=1.0)
    mom_component = ((mom_return - min_mom_return) / max(abs(min_mom_return), 0.10)).clip(lower=0.0, upper=1.0)
    score = (35.0 * rank_component + 25.0 * final_component + 25.0 * rs_component + 15.0 * mom_component).where(active, 0.0)
    if theme_support_overlay:
        score = (score * (0.75 + 0.25 * (theme_support_score / 100.0))).where(active, 0.0)
    if reset_breadth_overlay:
        score = (score * (0.70 + 0.30 * (reset_support_score / 100.0))).where(active, 0.0)
    credit = (score / 100.0 * max_rank_credit).clip(lower=0.0, upper=max_rank_credit)

    out["exit_quality_rank_credit_score"] = score.fillna(0.0)
    out["exit_quality_rank_credit"] = credit.fillna(0.0)
    out["exit_quality_effective_score_rank"] = (base_rank + credit.fillna(0.0)).clip(0.0, 1.0)
    out["exit_quality_rank_credit_blocked"] = blocked.fillna(False)
    out["exit_quality_rank_credit_eligible"] = active.fillna(False)
    if active.any() or blocked.any():
        LOGGER.info(
            "momentum_exit_quality_rank_credit_overlay_applied",
            extra={
                "event": "momentum_exit_quality_rank_credit_overlay_applied",
                "rows": int(active.sum()),
                "blocked_rows": int(blocked.sum()),
                "avg_rank_credit": float(credit.loc[active].mean()) if active.any() else 0.0,
                "avg_base_rank": float(base_rank.loc[active].mean()) if active.any() else 0.0,
                "rank_buffer_below": buffer_below,
                "rank_buffer_above": buffer_above,
                "min_final_score": min_final,
                "min_relative_strength_score": min_rs,
                "min_theme_score": min_theme,
                "min_technical_score": min_technical,
                "min_mom_return": min_mom_return,
                "max_drawdown_from_high": max_drawdown,
                "max_above_ma20_pct": max_above_ma20,
                "max_score_rank_credit": max_rank_credit,
                "require_benchmark_risk_on": require_benchmark_risk_on,
                "require_theme_active": require_theme_active,
                "require_price_above_ma50": require_price_above_ma50,
                "theme_support_overlay": theme_support_overlay,
                "avg_theme_support_score": float(theme_support_score.loc[active].mean()) if active.any() else 0.0,
                "theme_support_rows": int((candidate & circuit_breaker & theme_support_eligible).sum()) if theme_support_overlay else 0,
                "reset_breadth_overlay": reset_breadth_overlay,
                "avg_reset_support_score": float(reset_support_score.loc[active].mean()) if active.any() else 0.0,
                "reset_support_rows": int((candidate & circuit_breaker & reset_support_eligible).sum()) if reset_breadth_overlay else 0,
                "avg_reset_breadth_active_share": float(reset_breadth_active_share.loc[active].mean()) if active.any() else 0.0,
                "avg_reset_breadth_shortfall": float(reset_breadth_shortfall.loc[active].mean()) if active.any() else 0.0,
                "reset_same_theme_peer_breadth_overlay": _bool_param(
                    params,
                    "exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay",
                    False,
                ),
                "reset_same_theme_peer_reset_breadth_overlay": _bool_param(
                    params,
                    "exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay",
                    False,
                ),
                "reset_same_theme_peer_quality_overlay": _bool_param(
                    params,
                    "exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay",
                    False,
                ),
                "reset_same_theme_leader_guard_overlay": _bool_param(
                    params,
                    "exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay",
                    False,
                ),
                "avg_reset_same_theme_peer_count": float(reset_same_theme_peer_count.loc[active].mean()) if active.any() else 0.0,
                "avg_reset_same_theme_peer_reset_share": float(reset_same_theme_peer_reset_share.loc[active].mean()) if active.any() else 0.0,
                "avg_reset_same_theme_peer_reset_count": float(reset_same_theme_peer_reset_count.loc[active].mean()) if active.any() else 0.0,
                "avg_reset_same_theme_peer_reset_rs_mean": (
                    float(reset_same_theme_peer_reset_rs_mean.loc[active].mean()) if active.any() else 0.0
                ),
                "avg_reset_same_theme_leader_rank_pct": float(reset_same_theme_leader_rank.loc[active].mean()) if active.any() else 0.0,
                "avg_reset_same_theme_leader_score_gap": float(reset_same_theme_leader_score_gap.loc[active].mean()) if active.any() else 0.0,
                "reset_same_theme_peer_reset_min_avg_rs": same_theme_peer_reset_min_avg_rs if reset_breadth_overlay else 0.0,
                "reset_same_theme_leader_min_rank_pct": same_theme_leader_min_rank if reset_breadth_overlay else 0.0,
                "reset_same_theme_leader_max_final_score_gap": same_theme_leader_max_score_gap if reset_breadth_overlay else 0.0,
                "reset_same_theme_leader_min_peer_count": same_theme_leader_min_peer_count if reset_breadth_overlay else 0,
            },
        )
    return out


def _attach_benchmark_regime(frame: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Attach a date-level benchmark risk-on flag using only same-day closes."""

    out = frame.copy()
    benchmark_rows = out[out["symbol"].astype(str).str.upper() == benchmark.upper()].copy()
    if benchmark_rows.empty:
        out["benchmark_risk_on"] = True
        out["benchmark_ret_63d"] = 0.0
        return out
    ma_50 = _column(benchmark_rows, "ma_50", np.nan)
    ma_200 = _column(benchmark_rows, "ma_200", np.nan)
    ret_63d = _column(benchmark_rows, "ret_63d", 0.0)
    benchmark_rows["benchmark_risk_on"] = (
        (benchmark_rows["adj_close"] > ma_50)
        & (ma_50 > ma_200)
        & (ret_63d.fillna(0.0) > 0.0)
    )
    benchmark_rows["benchmark_ret_63d"] = ret_63d.fillna(0.0)
    by_date = benchmark_rows[["date", "benchmark_risk_on", "benchmark_ret_63d"]].drop_duplicates("date")
    out = out.merge(by_date, on="date", how="left")
    out["benchmark_risk_on"] = out["benchmark_risk_on"].map(lambda value: bool(value) if pd.notna(value) else False)
    out["benchmark_ret_63d"] = pd.to_numeric(out["benchmark_ret_63d"], errors="coerce").fillna(0.0)
    return out


def _long_quality_gate(frame: pd.DataFrame, params: dict) -> pd.Series:
    """Optional long-side gates for AI-cycle style quality control."""

    gate = pd.Series(True, index=frame.index)
    gate &= frame["final_score"].fillna(0.0) >= float(params.get("long_min_final_score", 0.0))
    gate &= frame["technical_score"].fillna(0.0) >= float(params.get("long_min_technical_score", 0.0))
    gate &= frame["relative_strength_score"].fillna(0.0) >= float(params.get("long_min_relative_strength_score", 0.0))
    gate &= frame["theme_score"].fillna(50.0) >= float(params.get("long_min_theme_score", 0.0))
    if _bool_param(params, "require_long_theme_active", False):
        gate &= frame.get("theme_active", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    if _bool_param(params, "long_require_price_above_ma50", False):
        gate &= frame["adj_close"] > frame.get("ma_50", np.inf)
    if _bool_param(params, "long_require_price_above_ma200", False):
        gate &= frame["adj_close"] > frame.get("ma_200", np.inf)
    max_above_ma20 = params.get("long_max_above_ma20_pct")
    if max_above_ma20 is not None:
        above_ma20 = frame["adj_close"] / _column(frame, "ma_20", np.nan).replace(0, np.nan) - 1.0
        gate &= above_ma20 <= float(max_above_ma20)
    return gate


def _apply_pullback_entry_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost strong theme leaders during controlled pullbacks.

    Feature flag: ``long_pullback_entry_overlay`` defaults to false.
    Reasonable ranges:
    - ``pullback_min_relative_strength_score``: 60 to 90.
    - ``pullback_min_theme_score``: 50 to 80.
    - ``pullback_min_drawdown_from_high``: 0.02 to 0.08.
    - ``pullback_max_drawdown_from_high``: 0.10 to 0.30.
    - ``pullback_max_above_ma20_pct``: 0.02 to 0.12.
    Circuit breaker: ``pullback_max_score_boost`` caps the single-row score
    change and the overlay never boosts shorts or names below the 50-day MA.
    """

    out = frame.copy()
    out["pullback_entry_score"] = 0.0
    out["pullback_entry_boost"] = 0.0
    if not _bool_param(params, "long_pullback_entry_overlay", False):
        return out

    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    ma200 = pd.to_numeric(_column(out, "ma_200", np.nan), errors="coerce").replace(0, np.nan)
    distance_high = pd.to_numeric(out.get("mom_distance_high", out.get("distance_to_prior_high_252")), errors="coerce")
    drawdown_from_high = (-distance_high).clip(lower=0.0)
    above_ma20 = adj_close / ma20 - 1.0
    ma20_vs_50 = ma20 / ma50 - 1.0
    ma50_vs_200 = ma50 / ma200 - 1.0
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)

    min_drawdown = float(params.get("pullback_min_drawdown_from_high", 0.03))
    max_drawdown = float(params.get("pullback_max_drawdown_from_high", 0.18))
    max_above_ma20 = float(params.get("pullback_max_above_ma20_pct", 0.06))
    min_rs = float(params.get("pullback_min_relative_strength_score", 70.0))
    min_theme = float(params.get("pullback_min_theme_score", 60.0))

    trend_intact = (adj_close > ma50) & (ma20_vs_50 > 0.0) & ((adj_close > ma200) | ma50_vs_200.fillna(0.0).gt(-0.05))
    controlled_pullback = drawdown_from_high.between(min_drawdown, max_drawdown, inclusive="both") & above_ma20.le(max_above_ma20)
    leadership = rs_score.ge(min_rs) & theme_score.ge(min_theme) & mom_return.gt(0.0)
    eligible = trend_intact & controlled_pullback & leadership

    score = (
        ((drawdown_from_high - min_drawdown) / max(max_drawdown - min_drawdown, 1e-9)).clip(0.0, 1.0) * 35
        + (1.0 - (above_ma20.clip(lower=0.0) / max(max_above_ma20, 1e-9))).clip(0.0, 1.0) * 25
        + ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(0.0, 1.0) * 25
        + ((theme_score - min_theme) / max(100.0 - min_theme, 1e-9)).clip(0.0, 1.0) * 15
    ).where(eligible, 0.0)
    max_boost = float(np.clip(params.get("pullback_max_score_boost", 6.0), 0.0, 10.0))
    boost = (score / 100.0 * max_boost).clip(lower=0.0, upper=max_boost)
    out["pullback_entry_score"] = score.fillna(0.0)
    out["pullback_entry_boost"] = boost.fillna(0.0)
    out["final_score"] = (pd.to_numeric(out["final_score"], errors="coerce").fillna(50.0) + out["pullback_entry_boost"]).clip(0.0, 100.0)
    if eligible.any():
        LOGGER.info(
            "momentum_pullback_entry_overlay_applied",
            extra={
                "event": "momentum_pullback_entry_overlay_applied",
                "rows": int(eligible.sum()),
                "avg_boost": float(out.loc[eligible, "pullback_entry_boost"].mean()),
                "max_boost": max_boost,
                "min_rs": min_rs,
                "min_theme": min_theme,
            },
        )
    return out


def _apply_pullback_volume_contraction_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost controlled leader pullbacks that reset on contracting volume.

    Feature flag: ``long_pullback_volume_contraction_overlay`` defaults to false.
    Reasonable ranges:
    - ``pullback_volume_contraction_max_volume_expansion``: 0.55 to 1.00.
    - ``pullback_volume_contraction_min_drawdown_from_high``: 0.02 to 0.08.
    - ``pullback_volume_contraction_max_drawdown_from_high``: 0.08 to 0.25.
    - ``pullback_volume_contraction_max_above_ma20_pct``: 0.00 to 0.06.
    - ``pullback_volume_contraction_min_relative_strength_score``: 60 to 90.
    - ``pullback_volume_contraction_min_theme_score``: 50 to 80.
    - ``pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker``: 10 to 50.
    - ``pullback_volume_contraction_require_benchmark_risk_on``: true/false.
    - ``pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker``: -0.02 to 0.08.
    - ``pullback_volume_contraction_min_adx_circuit_breaker``: 18 to 35.
    Circuit breakers: the boost is blocked unless the benchmark regime,
    overnight gap risk, and the name's trend ADX all clear their thresholds, and
    ``pullback_volume_contraction_max_score_boost`` caps the single-row impact.
    """

    out = frame.copy()
    out["pullback_volume_contraction_score"] = 0.0
    out["pullback_volume_contraction_boost"] = 0.0
    out["pullback_volume_contraction_blocked"] = False
    if not _bool_param(params, "long_pullback_volume_contraction_overlay", False):
        return out

    max_boost = float(np.clip(params.get("pullback_volume_contraction_max_score_boost", 3.0), 0.0, 10.0))
    if max_boost <= 0.0:
        return out

    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    ma200 = pd.to_numeric(_column(out, "ma_200", np.nan), errors="coerce").replace(0, np.nan)
    distance_high = pd.to_numeric(out.get("mom_distance_high", out.get("distance_to_prior_high_252")), errors="coerce")
    volume_expansion = pd.to_numeric(out.get("mom_volume_expansion", out.get("volume_expansion", 1.0)), errors="coerce")
    volume_expansion = volume_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    trend_adx = pd.to_numeric(_column(out, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(out, "benchmark_risk_on", False).fillna(False).astype(bool)
    benchmark_ret_63d = pd.to_numeric(_column(out, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)

    min_drawdown = float(params.get("pullback_volume_contraction_min_drawdown_from_high", 0.03))
    max_drawdown = float(params.get("pullback_volume_contraction_max_drawdown_from_high", 0.16))
    max_above_ma20 = float(params.get("pullback_volume_contraction_max_above_ma20_pct", 0.03))
    max_volume_expansion = float(params.get("pullback_volume_contraction_max_volume_expansion", 0.90))
    min_rs = float(params.get("pullback_volume_contraction_min_relative_strength_score", 68.0))
    min_theme = float(params.get("pullback_volume_contraction_min_theme_score", 58.0))
    min_benchmark_ret63d = float(params.get("pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_adx = float(params.get("pullback_volume_contraction_min_adx_circuit_breaker", 22.0))
    max_overnight_gap_risk = float(params.get("pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker", 35.0))
    require_benchmark_risk_on = _bool_param(params, "pullback_volume_contraction_require_benchmark_risk_on", True)

    drawdown_from_high = (-distance_high).clip(lower=0.0)
    above_ma20 = adj_close / ma20 - 1.0
    ma20_vs_50 = ma20 / ma50 - 1.0
    ma50_vs_200 = ma50 / ma200 - 1.0

    trend_intact = (adj_close > ma50) & (ma20_vs_50 > 0.0) & ((adj_close > ma200) | ma50_vs_200.fillna(0.0).gt(-0.05))
    controlled_pullback = drawdown_from_high.between(min_drawdown, max_drawdown, inclusive="both") & above_ma20.le(max_above_ma20)
    leadership = rs_score.ge(min_rs) & theme_score.ge(min_theme) & mom_return.gt(0.0)
    volume_reset = volume_expansion.le(max_volume_expansion)
    benchmark_ok = benchmark_risk_on if require_benchmark_risk_on else benchmark_ret_63d.ge(min_benchmark_ret63d)
    circuit_breaker = benchmark_ok & overnight_gap_risk.le(max_overnight_gap_risk) & trend_adx.ge(min_adx)
    candidate = trend_intact & controlled_pullback & leadership & volume_reset
    eligible = candidate & circuit_breaker
    blocked = candidate & ~circuit_breaker

    drawdown_component = ((drawdown_from_high - min_drawdown) / max(max_drawdown - min_drawdown, 1e-9)).clip(0.0, 1.0)
    volume_component = ((max_volume_expansion - volume_expansion) / max(max_volume_expansion, 1e-9)).clip(0.0, 1.0)
    rs_component = ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(0.0, 1.0)
    theme_component = ((theme_score - min_theme) / max(100.0 - min_theme, 1e-9)).clip(0.0, 1.0)
    score = (
        drawdown_component * 35.0
        + volume_component * 35.0
        + rs_component * 20.0
        + theme_component * 10.0
    ).where(eligible, 0.0)
    boost = (score / 100.0 * max_boost).clip(lower=0.0, upper=max_boost)

    out["pullback_volume_contraction_score"] = score.fillna(0.0)
    out["pullback_volume_contraction_boost"] = boost.fillna(0.0)
    out["pullback_volume_contraction_blocked"] = blocked.fillna(False)
    out["final_score"] = (pd.to_numeric(out["final_score"], errors="coerce").fillna(50.0) + out["pullback_volume_contraction_boost"]).clip(0.0, 100.0)
    if candidate.any():
        LOGGER.info(
            "momentum_pullback_volume_contraction_overlay_applied",
            extra={
                "event": "momentum_pullback_volume_contraction_overlay_applied",
                "rows": int(eligible.sum()),
                "blocked_rows": int(blocked.sum()),
                "avg_boost": float(out.loc[eligible, "pullback_volume_contraction_boost"].mean()) if eligible.any() else 0.0,
                "max_boost": max_boost,
                "max_volume_expansion": max_volume_expansion,
                "min_drawdown": min_drawdown,
                "max_drawdown": max_drawdown,
                "min_benchmark_ret63d": min_benchmark_ret63d,
                "min_adx": min_adx,
                "max_overnight_gap_risk": max_overnight_gap_risk,
                "require_benchmark_risk_on": require_benchmark_risk_on,
            },
        )
    return out


def _apply_gap_adjusted_continuation_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost controlled-pullback leaders only when overnight gap risk is benign.

    Feature flag: ``gap_adjusted_continuation_overlay`` defaults to false.
    Optional refinement flag:
    ``gap_adjusted_continuation_same_theme_peer_quality_overlay`` defaults to
    false and requires the candidate's theme to show enough same-theme peers
    with benign or improving gap risk before the boost can fire.
    Reasonable ranges:
    - ``gap_adjusted_continuation_min_drawdown_from_high``: 0.02 to 0.08.
    - ``gap_adjusted_continuation_max_drawdown_from_high``: 0.08 to 0.22.
    - ``gap_adjusted_continuation_max_above_ma20_pct``: 0.00 to 0.07.
    - ``gap_adjusted_continuation_max_gap_risk_score``: 15 to 40.
    - ``gap_adjusted_continuation_benign_gap_risk_score``: 5 to 25.
    - ``gap_adjusted_continuation_min_gap_risk_improvement``: 1 to 10.
    - ``gap_adjusted_continuation_same_theme_peer_min_count``: 2 to 6.
    - ``gap_adjusted_continuation_same_theme_peer_min_share``: 0.25 to 0.80.
    - ``gap_adjusted_continuation_same_theme_peer_min_avg_rs``: 70 to 95.
    The rule is deliberately a small score tilt, not an entry rule. It keeps
    benchmark, trend, event-risk, and gap-risk circuit breakers intact.
    """

    out = frame.copy()
    out["gap_adjusted_continuation_score"] = 0.0
    out["gap_adjusted_continuation_boost"] = 0.0
    out["gap_adjusted_continuation_gap_risk"] = 0.0
    out["gap_adjusted_continuation_gap_improvement"] = 0.0
    out["gap_adjusted_continuation_eligible"] = False
    out["gap_adjusted_continuation_blocked"] = False
    out["gap_adjusted_continuation_same_theme_peer_count"] = 0
    out["gap_adjusted_continuation_same_theme_peer_share"] = 0.0
    out["gap_adjusted_continuation_same_theme_peer_avg_rs"] = 0.0
    out["gap_adjusted_continuation_same_theme_peer_quality_ok"] = False
    if not _bool_param(params, "gap_adjusted_continuation_overlay", False):
        return out

    max_boost = float(np.clip(params.get("gap_adjusted_continuation_max_score_boost", 2.5), 0.0, 8.0))
    if max_boost <= 0.0:
        return out

    ordered = out.sort_values(["symbol", "date"]).copy()
    original_order = ordered.index.to_numpy(copy=True)
    adj_close = pd.to_numeric(ordered["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(ordered, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(ordered, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    ma200 = pd.to_numeric(_column(ordered, "ma_200", np.nan), errors="coerce").replace(0, np.nan)
    distance_high = pd.to_numeric(ordered.get("mom_distance_high", ordered.get("distance_to_prior_high_252")), errors="coerce")
    rs_score = pd.to_numeric(ordered.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(ordered.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(ordered.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    volume_expansion = pd.to_numeric(ordered.get("mom_volume_expansion", ordered.get("volume_expansion", 1.0)), errors="coerce")
    volume_expansion = volume_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    trend_adx = pd.to_numeric(_column(ordered, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(_column(ordered, "event_risk_score", 0.0), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(_column(ordered, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(ordered, "benchmark_risk_on", False).fillna(False).astype(bool)
    benchmark_ret_63d = pd.to_numeric(_column(ordered, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    theme_active = _column(ordered, "theme_active", True).fillna(False).astype(bool)

    min_drawdown = float(params.get("gap_adjusted_continuation_min_drawdown_from_high", 0.03))
    max_drawdown = float(params.get("gap_adjusted_continuation_max_drawdown_from_high", 0.16))
    max_above_ma20 = float(params.get("gap_adjusted_continuation_max_above_ma20_pct", 0.05))
    min_rs = float(params.get("gap_adjusted_continuation_min_relative_strength_score", 72.0))
    min_theme = float(params.get("gap_adjusted_continuation_min_theme_score", 58.0))
    min_adx = float(params.get("gap_adjusted_continuation_min_adx_circuit_breaker", 18.0))
    max_event_risk = float(params.get("gap_adjusted_continuation_max_event_risk_score_circuit_breaker", 18.0))
    max_gap_risk = float(params.get("gap_adjusted_continuation_max_gap_risk_score_circuit_breaker", 28.0))
    benign_gap_risk = float(params.get("gap_adjusted_continuation_benign_gap_risk_score", 14.0))
    gap_lookback = max(int(params.get("gap_adjusted_continuation_gap_risk_lookback_days", 10)), 2)
    min_gap_improvement = float(params.get("gap_adjusted_continuation_min_gap_risk_improvement", 3.0))
    min_benchmark_ret63d = float(params.get("gap_adjusted_continuation_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_volume_expansion = float(params.get("gap_adjusted_continuation_min_volume_expansion", 0.60))
    max_volume_expansion = float(params.get("gap_adjusted_continuation_max_volume_expansion", 1.35))
    require_benchmark_risk_on = _bool_param(params, "gap_adjusted_continuation_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "gap_adjusted_continuation_require_theme_active", False)
    use_same_theme_peer_quality = _bool_param(
        params,
        "gap_adjusted_continuation_same_theme_peer_quality_overlay",
        False,
    )
    same_theme_peer_min_count = max(int(params.get("gap_adjusted_continuation_same_theme_peer_min_count", 3)), 1)
    same_theme_peer_min_share = float(
        np.clip(params.get("gap_adjusted_continuation_same_theme_peer_min_share", 0.45), 0.0, 1.0)
    )
    same_theme_peer_min_avg_rs = float(params.get("gap_adjusted_continuation_same_theme_peer_min_avg_rs", 82.0))

    drawdown_from_high = (-distance_high).clip(lower=0.0)
    above_ma20 = adj_close / ma20 - 1.0
    ma20_vs_50 = ma20 / ma50 - 1.0
    ma50_vs_200 = ma50 / ma200 - 1.0
    ordered["_gap_adjusted_continuation_gap_risk_input"] = gap_risk
    prior_gap_risk = ordered.groupby("symbol", group_keys=False)["_gap_adjusted_continuation_gap_risk_input"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(gap_lookback, min_periods=max(2, gap_lookback // 2)).mean()
    )
    gap_improvement = (prior_gap_risk - gap_risk).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gap_ok = gap_risk.le(benign_gap_risk) | ((gap_risk.le(max_gap_risk)) & gap_improvement.ge(min_gap_improvement))

    trend_intact = (adj_close > ma50) & (ma20_vs_50 > 0.0) & ((adj_close > ma200) | ma50_vs_200.fillna(0.0).gt(-0.05))
    controlled_pullback = drawdown_from_high.between(min_drawdown, max_drawdown, inclusive="both") & above_ma20.le(max_above_ma20)
    leadership = rs_score.ge(min_rs) & theme_score.ge(min_theme) & mom_return.gt(0.0)
    volume_ok = volume_expansion.between(min_volume_expansion, max_volume_expansion, inclusive="both")
    benchmark_ok = benchmark_risk_on if require_benchmark_risk_on else benchmark_ret_63d.ge(min_benchmark_ret63d)
    circuit_breaker = benchmark_ok & event_risk.le(max_event_risk) & gap_risk.le(max_gap_risk) & trend_adx.ge(min_adx)
    if require_theme_active:
        circuit_breaker &= theme_active
    candidate = trend_intact & controlled_pullback & leadership & volume_ok
    same_theme_peer_count = pd.Series(0, index=ordered.index, dtype=int)
    same_theme_peer_share = pd.Series(0.0, index=ordered.index, dtype=float)
    same_theme_peer_avg_rs = pd.Series(0.0, index=ordered.index, dtype=float)
    same_theme_peer_quality_ok = pd.Series(True, index=ordered.index, dtype=bool)
    if use_same_theme_peer_quality:
        theme_key = (
            pd.Series(
                ordered["primary_theme"] if "primary_theme" in ordered.columns else _column(ordered, "theme_reason", "unknown"),
                index=ordered.index,
            )
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown", "unclassified": "unknown"})
        )
        date_key = pd.to_datetime(ordered["date"]).dt.normalize()
        peer_member = trend_intact & leadership & gap_ok & event_risk.le(max_event_risk) & gap_risk.le(max_gap_risk)
        if require_theme_active:
            peer_member &= theme_active
        if require_benchmark_risk_on:
            peer_member &= benchmark_risk_on
        else:
            peer_member &= benchmark_ret_63d.ge(min_benchmark_ret63d)
        theme_daily = (
            pd.DataFrame(
                {
                    "date": date_key,
                    "theme": theme_key,
                    "symbol": ordered["symbol"].astype(str),
                    "peer_member": peer_member.astype(float),
                    "peer_rs": rs_score.where(peer_member),
                }
            )
            .groupby(["date", "theme"], as_index=False)
            .agg(
                gap_adjusted_continuation_same_theme_peer_count=("peer_member", "sum"),
                gap_adjusted_continuation_same_theme_peer_share=("peer_member", "mean"),
                gap_adjusted_continuation_same_theme_peer_avg_rs=("peer_rs", "mean"),
            )
        )
        ordered["_gap_adjusted_continuation_theme_date_key"] = date_key
        ordered["_gap_adjusted_continuation_theme_key"] = theme_key
        ordered = ordered.merge(
            theme_daily,
            left_on=["_gap_adjusted_continuation_theme_date_key", "_gap_adjusted_continuation_theme_key"],
            right_on=["date", "theme"],
            how="left",
            suffixes=("", "_gap_adjusted_continuation_theme"),
        )
        for column in (
            "gap_adjusted_continuation_same_theme_peer_count",
            "gap_adjusted_continuation_same_theme_peer_share",
            "gap_adjusted_continuation_same_theme_peer_avg_rs",
        ):
            context_column = f"{column}_gap_adjusted_continuation_theme"
            if context_column in ordered.columns:
                ordered[column] = ordered[context_column].fillna(ordered[column])
                ordered = ordered.drop(columns=[context_column])
        ordered = ordered.drop(
            columns=["date_gap_adjusted_continuation_theme", "theme", "_gap_adjusted_continuation_theme_date_key", "_gap_adjusted_continuation_theme_key"],
            errors="ignore",
        )
        same_theme_peer_count = pd.to_numeric(
            ordered.get("gap_adjusted_continuation_same_theme_peer_count", 0),
            errors="coerce",
        ).fillna(0).astype(int)
        same_theme_peer_share = pd.to_numeric(
            ordered.get("gap_adjusted_continuation_same_theme_peer_share", 0.0),
            errors="coerce",
        ).fillna(0.0)
        same_theme_peer_avg_rs = pd.to_numeric(
            ordered.get("gap_adjusted_continuation_same_theme_peer_avg_rs", 0.0),
            errors="coerce",
        ).fillna(0.0)
        same_theme_peer_quality_ok = theme_key.ne("unknown")
        same_theme_peer_quality_ok &= same_theme_peer_count.ge(same_theme_peer_min_count)
        same_theme_peer_quality_ok &= same_theme_peer_share.ge(same_theme_peer_min_share)
        same_theme_peer_quality_ok &= same_theme_peer_avg_rs.ge(same_theme_peer_min_avg_rs)

    eligible = candidate & circuit_breaker & gap_ok & same_theme_peer_quality_ok
    blocked = candidate & ~(circuit_breaker & gap_ok & same_theme_peer_quality_ok)

    pullback_component = ((drawdown_from_high - min_drawdown) / max(max_drawdown - min_drawdown, 1e-9)).clip(0.0, 1.0)
    extension_component = (1.0 - (above_ma20.clip(lower=0.0) / max(max_above_ma20, 1e-9))).clip(0.0, 1.0)
    gap_level_component = (1.0 - (gap_risk / max(max_gap_risk, 1e-9))).clip(0.0, 1.0)
    gap_improvement_component = (gap_improvement / max(min_gap_improvement * 2.0, 1e-9)).clip(0.0, 1.0)
    rs_component = ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(0.0, 1.0)
    theme_component = ((theme_score - min_theme) / max(100.0 - min_theme, 1e-9)).clip(0.0, 1.0)
    if use_same_theme_peer_quality:
        peer_share_component = (
            (same_theme_peer_share - same_theme_peer_min_share) / max(1.0 - same_theme_peer_min_share, 1e-9)
        ).clip(0.0, 1.0)
        peer_rs_component = (
            (same_theme_peer_avg_rs - same_theme_peer_min_avg_rs) / max(100.0 - same_theme_peer_min_avg_rs, 1e-9)
        ).clip(0.0, 1.0)
        score = (
            pullback_component * 18.0
            + extension_component * 14.0
            + gap_level_component * 22.0
            + gap_improvement_component * 16.0
            + rs_component * 10.0
            + theme_component * 8.0
            + peer_share_component * 7.0
            + peer_rs_component * 5.0
        ).where(eligible, 0.0)
    else:
        score = (
            pullback_component * 20.0
            + extension_component * 15.0
            + gap_level_component * 25.0
            + gap_improvement_component * 20.0
            + rs_component * 12.0
            + theme_component * 8.0
        ).where(eligible, 0.0)
    boost = (score / 100.0 * max_boost).clip(lower=0.0, upper=max_boost)

    ordered["gap_adjusted_continuation_score"] = score.fillna(0.0)
    ordered["gap_adjusted_continuation_boost"] = boost.fillna(0.0)
    ordered["gap_adjusted_continuation_gap_risk"] = gap_risk.fillna(0.0)
    ordered["gap_adjusted_continuation_gap_improvement"] = gap_improvement.fillna(0.0)
    ordered["gap_adjusted_continuation_eligible"] = eligible.fillna(False)
    ordered["gap_adjusted_continuation_blocked"] = blocked.fillna(False)
    ordered["gap_adjusted_continuation_same_theme_peer_count"] = same_theme_peer_count.to_numpy()
    ordered["gap_adjusted_continuation_same_theme_peer_share"] = same_theme_peer_share.to_numpy()
    ordered["gap_adjusted_continuation_same_theme_peer_avg_rs"] = same_theme_peer_avg_rs.to_numpy()
    ordered["gap_adjusted_continuation_same_theme_peer_quality_ok"] = same_theme_peer_quality_ok.to_numpy()
    ordered["final_score"] = (pd.to_numeric(ordered["final_score"], errors="coerce").fillna(50.0) + boost).clip(0.0, 100.0)
    if candidate.any():
        LOGGER.info(
            "momentum_gap_adjusted_continuation_overlay_applied",
            extra={
                "event": "momentum_gap_adjusted_continuation_overlay_applied",
                "rows": int(eligible.sum()),
                "blocked_rows": int(blocked.sum()),
                "avg_boost": float(ordered.loc[eligible, "gap_adjusted_continuation_boost"].mean()) if eligible.any() else 0.0,
                "gap_lookback": gap_lookback,
                "max_gap_risk": max_gap_risk,
                "benign_gap_risk": benign_gap_risk,
                "min_gap_improvement": min_gap_improvement,
                "same_theme_peer_quality_enabled": use_same_theme_peer_quality,
                "same_theme_peer_min_count": same_theme_peer_min_count,
                "same_theme_peer_min_share": same_theme_peer_min_share,
                "same_theme_peer_min_avg_rs": same_theme_peer_min_avg_rs,
                "avg_same_theme_peer_count": float(same_theme_peer_count.loc[eligible].mean()) if eligible.any() else 0.0,
                "avg_same_theme_peer_share": float(same_theme_peer_share.loc[eligible].mean()) if eligible.any() else 0.0,
                "avg_same_theme_peer_avg_rs": float(same_theme_peer_avg_rs.loc[eligible].mean()) if eligible.any() else 0.0,
                "max_boost": max_boost,
            },
        )
    return ordered.sort_index()


def _apply_pullback_short_term_reset_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost leader pullbacks that reset on weak 5/10-day tape without chasing.

    Feature flag: ``pullback_short_term_reset_overlay`` defaults to false.
    Reasonable ranges:
    - ``pullback_short_term_reset_min_drawdown_from_high``: 0.02 to 0.08.
    - ``pullback_short_term_reset_max_drawdown_from_high``: 0.08 to 0.22.
    - ``pullback_short_term_reset_max_above_ma20_pct``: 0.00 to 0.05.
    - ``pullback_short_term_reset_max_ret_5d``: -0.03 to 0.03.
    - ``pullback_short_term_reset_max_ret_10d``: -0.05 to 0.05.
    - ``pullback_short_term_reset_min_relative_strength_score``: 55 to 85.
    - ``pullback_short_term_reset_min_theme_score``: 50 to 80.
    - ``pullback_short_term_reset_max_event_risk_score_circuit_breaker``: 10 to 35.
    - ``pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker``: 10 to 40.
    - ``pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker``: -0.02 to 0.08.
    - ``pullback_short_term_reset_min_adx_circuit_breaker``: 18 to 35.
    Circuit breakers: the boost is blocked unless the benchmark regime,
    event/gap risk, and trend ADX all clear their thresholds, and
    ``pullback_short_term_reset_max_score_boost`` caps the single-row impact.
    """

    out = frame.copy()
    out["pullback_short_term_reset_score"] = 0.0
    out["pullback_short_term_reset_boost"] = 0.0
    out["pullback_short_term_reset_blocked"] = False
    out["pullback_short_term_reset_ret_5d"] = np.nan
    out["pullback_short_term_reset_ret_10d"] = np.nan
    if not _bool_param(params, "pullback_short_term_reset_overlay", False):
        return out

    max_boost = float(np.clip(params.get("pullback_short_term_reset_max_score_boost", 2.5), 0.0, 10.0))
    if max_boost <= 0.0:
        return out

    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    distance_high = pd.to_numeric(out.get("mom_distance_high", out.get("distance_to_prior_high_252")), errors="coerce")
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(out, "benchmark_risk_on", False).fillna(False).astype(bool)
    benchmark_ret_63d = pd.to_numeric(_column(out, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    trend_adx = pd.to_numeric(_column(out, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    if "theme_active" in out:
        theme_active = out["theme_active"].fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index)

    grouped = out.sort_values(["symbol", "date"]).groupby("symbol", group_keys=False)
    ret_5d = pd.to_numeric(out.get("ret_5d"), errors="coerce") if "ret_5d" in out else grouped["adj_close"].pct_change(5)
    ret_10d = pd.to_numeric(out.get("ret_10d"), errors="coerce") if "ret_10d" in out else grouped["adj_close"].pct_change(10)
    out["pullback_short_term_reset_ret_5d"] = ret_5d
    out["pullback_short_term_reset_ret_10d"] = ret_10d

    min_drawdown = float(params.get("pullback_short_term_reset_min_drawdown_from_high", 0.03))
    max_drawdown = float(params.get("pullback_short_term_reset_max_drawdown_from_high", 0.16))
    max_above_ma20 = float(params.get("pullback_short_term_reset_max_above_ma20_pct", 0.03))
    max_ret_5d = float(params.get("pullback_short_term_reset_max_ret_5d", 0.01))
    max_ret_10d = float(params.get("pullback_short_term_reset_max_ret_10d", 0.02))
    min_rs = float(params.get("pullback_short_term_reset_min_relative_strength_score", 58.0))
    min_theme = float(params.get("pullback_short_term_reset_min_theme_score", 56.0))
    max_event_risk = float(params.get("pullback_short_term_reset_max_event_risk_score_circuit_breaker", 20.0))
    max_gap_risk = float(params.get("pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker", 30.0))
    min_benchmark_ret63d = float(params.get("pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_adx = float(params.get("pullback_short_term_reset_min_adx_circuit_breaker", 20.0))
    require_benchmark_risk_on = _bool_param(params, "pullback_short_term_reset_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "pullback_short_term_reset_require_theme_active", True)

    drawdown_from_high = (-distance_high).clip(lower=0.0)
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan)
    trend_intact = adj_close.gt(ma50).fillna(False) & mom_return.gt(0.0)
    controlled_pullback = drawdown_from_high.between(min_drawdown, max_drawdown, inclusive="both") & above_ma20.le(max_above_ma20)
    leadership = rs_score.ge(min_rs) & theme_score.ge(min_theme)
    recent_reset = ret_5d.le(max_ret_5d).fillna(False) & ret_10d.le(max_ret_10d).fillna(False)
    circuit_breaker = event_risk.le(max_event_risk) & overnight_gap_risk.le(max_gap_risk) & benchmark_ret_63d.ge(min_benchmark_ret63d) & trend_adx.ge(min_adx)
    if require_benchmark_risk_on:
        circuit_breaker &= benchmark_risk_on
    if require_theme_active:
        circuit_breaker &= theme_active

    eligible = trend_intact & controlled_pullback & leadership & recent_reset
    blocked = eligible & ~circuit_breaker
    active = eligible & circuit_breaker

    ret5_span = max(abs(max_ret_5d) + 0.08, 1e-9)
    ret10_span = max(abs(max_ret_10d) + 0.12, 1e-9)
    ret5_component = ((max_ret_5d - ret_5d) / ret5_span).clip(lower=0.0, upper=1.0)
    ret10_component = ((max_ret_10d - ret_10d) / ret10_span).clip(lower=0.0, upper=1.0)
    drawdown_component = ((drawdown_from_high - min_drawdown) / max(max_drawdown - min_drawdown, 1e-9)).clip(lower=0.0, upper=1.0)
    above_ma20_component = (1.0 - (above_ma20.clip(lower=0.0) / max(max_above_ma20, 1e-9))).clip(lower=0.0, upper=1.0)
    score = (
        ret5_component * 35.0
        + ret10_component * 30.0
        + drawdown_component * 20.0
        + above_ma20_component * 15.0
    ).where(active, 0.0)
    boost = (score / 100.0 * max_boost).clip(lower=0.0, upper=max_boost)

    out["pullback_short_term_reset_score"] = score.fillna(0.0)
    out["pullback_short_term_reset_boost"] = boost.fillna(0.0)
    out["pullback_short_term_reset_blocked"] = blocked.fillna(False)
    out["final_score"] = (pd.to_numeric(out["final_score"], errors="coerce").fillna(50.0) + boost.fillna(0.0)).clip(0.0, 100.0)
    if active.any() or blocked.any():
        LOGGER.info(
            "momentum_pullback_short_term_reset_overlay_applied",
            extra={
                "event": "momentum_pullback_short_term_reset_overlay_applied",
                "rows": int(active.sum()),
                "blocked_rows": int(blocked.sum()),
                "avg_boost": float(boost.loc[active].mean()) if active.any() else 0.0,
                "avg_ret_5d": float(ret_5d.loc[active].mean()) if active.any() else 0.0,
                "avg_ret_10d": float(ret_10d.loc[active].mean()) if active.any() else 0.0,
                "min_drawdown_from_high": min_drawdown,
                "max_drawdown_from_high": max_drawdown,
                "max_above_ma20_pct": max_above_ma20,
                "max_ret_5d": max_ret_5d,
                "max_ret_10d": max_ret_10d,
                "min_relative_strength_score": min_rs,
                "min_theme_score": min_theme,
                "max_event_risk_score_circuit_breaker": max_event_risk,
                "max_overnight_gap_risk_score_circuit_breaker": max_gap_risk,
                "min_benchmark_ret63d_circuit_breaker": min_benchmark_ret63d,
                "min_adx_circuit_breaker": min_adx,
                "require_benchmark_risk_on": require_benchmark_risk_on,
                "require_theme_active": require_theme_active,
            },
        )
    return out


def _apply_short_term_volume_tilt_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Lightly reward short-term reset plus volume confirmation near the rank boundary.

    Feature flag: ``short_term_volume_tilt_overlay`` defaults to false.
    Optional refinement flags:
    ``short_term_volume_tilt_same_theme_peer_confirmation_overlay`` defaults to
    false and only allows the lift when same-theme peers confirm the reset-plus-
    volume impulse locally, which keeps isolated volume spikes from ranking up
    weak themes on their own.
    ``short_term_volume_tilt_same_theme_peer_volume_substitution_overlay``
    defaults to false and allows same-theme peer confirmation to stand in for
    weak single-name volume only when the symbol still clears a tighter volume
    floor and receives a scaled-down boost.
    Reasonable ranges:
    - ``short_term_volume_tilt_max_ret_5d``: -0.06 to 0.03.
    - ``short_term_volume_tilt_min_volume_expansion``: 0.90 to 1.60.
    - ``short_term_volume_tilt_min_relative_strength_score``: 50 to 80.
    - ``short_term_volume_tilt_min_theme_score``: 50 to 75.
    - ``short_term_volume_tilt_max_above_ma20_pct``: 0.00 to 0.12.
    - ``short_term_volume_tilt_max_event_risk_score_circuit_breaker``: 10 to 35.
    - ``short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker``: 10 to 45.
    - ``short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker``: -0.03 to 0.05.
    - ``short_term_volume_tilt_min_adx_circuit_breaker``: 15 to 35.
    - ``short_term_volume_tilt_same_theme_peer_min_count``: 2 to 5.
    - ``short_term_volume_tilt_same_theme_peer_min_share``: 0.25 to 0.80.
    - ``short_term_volume_tilt_same_theme_peer_min_avg_mom_return``: 0.05 to 0.30.
    - ``short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion``:
      0.40 to 1.00.
    - ``short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall``:
      0.05 to 0.80.
    - ``short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale``:
      0.20 to 0.90.
    - ``short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count``:
      2 to 6.
    - ``short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share``:
      0.35 to 0.90.
    - ``short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return``:
      0.08 to 0.35.
    - ``short_term_volume_tilt_non_risk_on_exception_score_boost_scale``:
      0.20 to 1.00.
    Circuit breakers keep this as a small ranking tilt, not a standalone entry
    rule. The boost is capped by ``short_term_volume_tilt_max_score_boost`` and
    can be scaled down further for the non-risk-on exception path.
    """

    out = frame.copy()
    out["short_term_volume_tilt_score"] = 0.0
    out["short_term_volume_tilt_boost"] = 0.0
    out["short_term_volume_tilt_ret_5d"] = np.nan
    out["short_term_volume_tilt_volume_expansion"] = np.nan
    out["short_term_volume_tilt_candidate"] = False
    out["short_term_volume_tilt_circuit_ok"] = False
    out["short_term_volume_tilt_peer_confirmed"] = False
    out["short_term_volume_tilt_blocked"] = False
    out["short_term_volume_tilt_block_reason"] = ""
    out["short_term_volume_tilt_same_theme_peer_count"] = 0
    out["short_term_volume_tilt_same_theme_peer_confirm_share"] = 0.0
    out["short_term_volume_tilt_same_theme_peer_avg_mom_return"] = 0.0
    out["short_term_volume_tilt_volume_shortfall"] = 0.0
    out["short_term_volume_tilt_peer_volume_substitution_candidate"] = False
    out["short_term_volume_tilt_activation_mode"] = "off"
    if not _bool_param(params, "short_term_volume_tilt_overlay", False):
        return out

    max_boost = float(np.clip(params.get("short_term_volume_tilt_max_score_boost", 1.75), 0.0, 5.0))
    if max_boost <= 0.0:
        return out

    grouped = out.sort_values(["symbol", "date"]).groupby("symbol", group_keys=False)
    ret_5d = pd.to_numeric(out.get("ret_5d"), errors="coerce") if "ret_5d" in out else grouped["adj_close"].pct_change(5)
    volume_expansion = pd.to_numeric(out.get("mom_volume_expansion", out.get("volume_expansion", 1.0)), errors="coerce")
    volume_expansion = volume_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(out, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(out, "benchmark_risk_on", False).fillna(False).astype(bool)
    benchmark_ret_63d = pd.to_numeric(_column(out, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    trend_adx = pd.to_numeric(_column(out, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    if "theme_active" in out:
        theme_active = out["theme_active"].fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index)

    max_ret_5d = float(params.get("short_term_volume_tilt_max_ret_5d", 0.00))
    min_volume_expansion = float(params.get("short_term_volume_tilt_min_volume_expansion", 1.05))
    min_rs = float(params.get("short_term_volume_tilt_min_relative_strength_score", 58.0))
    min_theme = float(params.get("short_term_volume_tilt_min_theme_score", 55.0))
    max_above_ma20 = float(params.get("short_term_volume_tilt_max_above_ma20_pct", 0.08))
    max_event_risk = float(params.get("short_term_volume_tilt_max_event_risk_score_circuit_breaker", 20.0))
    max_gap_risk = float(params.get("short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker", 32.0))
    min_benchmark_ret63d = float(params.get("short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_adx = float(params.get("short_term_volume_tilt_min_adx_circuit_breaker", 18.0))
    require_benchmark_risk_on = _bool_param(params, "short_term_volume_tilt_require_benchmark_risk_on", True)
    require_theme_active = _bool_param(params, "short_term_volume_tilt_require_theme_active", False)
    require_price_above_ma50 = _bool_param(params, "short_term_volume_tilt_require_price_above_ma50", True)
    use_same_theme_peer_confirmation = _bool_param(
        params,
        "short_term_volume_tilt_same_theme_peer_confirmation_overlay",
        False,
    )
    allow_same_theme_peer_volume_substitution = _bool_param(
        params,
        "short_term_volume_tilt_same_theme_peer_volume_substitution_overlay",
        False,
    )
    same_theme_peer_min_count = max(int(params.get("short_term_volume_tilt_same_theme_peer_min_count", 2)), 1)
    same_theme_peer_min_share = float(
        np.clip(params.get("short_term_volume_tilt_same_theme_peer_min_share", 0.50), 0.0, 1.0)
    )
    same_theme_peer_min_avg_mom_return = float(
        params.get("short_term_volume_tilt_same_theme_peer_min_avg_mom_return", 0.10)
    )
    same_theme_peer_substitute_min_volume_expansion = float(
        params.get("short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion", 0.70)
    )
    same_theme_peer_substitute_max_volume_shortfall = float(
        max(params.get("short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall", 0.35), 0.0)
    )
    same_theme_peer_substitute_score_boost_scale = float(
        np.clip(
            params.get("short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale", 0.60),
            0.0,
            1.0,
        )
    )
    allow_non_risk_on_exception = _bool_param(
        params,
        "short_term_volume_tilt_non_risk_on_exception_overlay",
        False,
    )
    non_risk_on_exception_min_count = max(
        int(params.get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count", 3)),
        1,
    )
    non_risk_on_exception_min_share = float(
        np.clip(
            params.get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share", 0.60),
            0.0,
            1.0,
        )
    )
    non_risk_on_exception_min_avg_mom_return = float(
        params.get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return", 0.14)
    )
    non_risk_on_exception_score_boost_scale = float(
        np.clip(
            params.get("short_term_volume_tilt_non_risk_on_exception_score_boost_scale", 0.50),
            0.0,
            1.0,
        )
    )

    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan)
    reset = ret_5d.le(max_ret_5d).fillna(False)
    volume_confirmation = volume_expansion.ge(min_volume_expansion)
    volume_shortfall = (min_volume_expansion - volume_expansion).clip(lower=0.0)
    leadership = rs_score.ge(min_rs) & theme_score.ge(min_theme) & mom_return.gt(0.0)
    if require_price_above_ma50:
        leadership &= adj_close.gt(ma50).fillna(False)
    extension_ok = above_ma20.le(max_above_ma20).fillna(False)
    base_circuit_breaker = (
        event_risk.le(max_event_risk)
        & overnight_gap_risk.le(max_gap_risk)
        & benchmark_ret_63d.ge(min_benchmark_ret63d)
        & trend_adx.ge(min_adx)
    )
    if require_theme_active:
        base_circuit_breaker &= theme_active
    circuit_breaker = base_circuit_breaker.copy()
    if require_benchmark_risk_on:
        circuit_breaker &= benchmark_risk_on

    base_candidate = reset & leadership & extension_ok
    candidate = base_candidate & volume_confirmation
    peer_volume_substitution_candidate = base_candidate & volume_confirmation.eq(False)
    same_theme_peer_count = pd.Series(0, index=out.index, dtype=int)
    same_theme_peer_confirm_share = pd.Series(0.0, index=out.index, dtype=float)
    same_theme_peer_avg_mom_return = pd.Series(0.0, index=out.index, dtype=float)
    same_theme_peer_confirmed = pd.Series(True, index=out.index, dtype=bool)
    same_theme_peer_exception_ok = pd.Series(False, index=out.index, dtype=bool)
    same_theme_peer_exception_count = pd.Series(0, index=out.index, dtype=int)
    same_theme_peer_exception_share = pd.Series(0.0, index=out.index, dtype=float)
    same_theme_peer_exception_avg_mom_return = pd.Series(0.0, index=out.index, dtype=float)
    if use_same_theme_peer_confirmation or allow_non_risk_on_exception or allow_same_theme_peer_volume_substitution:
        date_key = pd.to_datetime(out["date"]).dt.normalize()
        if "primary_theme" in out:
            theme_key = out["primary_theme"]
        elif "theme_reason" in out:
            theme_key = out["theme_reason"]
        else:
            theme_key = pd.Series("unknown", index=out.index)
        theme_key = (
            pd.Series(theme_key, index=out.index)
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
        )
        peer_confirm_flag = candidate & circuit_breaker
        exception_peer_confirm_flag = candidate & base_circuit_breaker
        theme_daily = (
            pd.DataFrame(
                {
                    "date": date_key,
                    "theme": theme_key,
                    "symbol": out["symbol"].astype(str),
                    "peer_confirm_flag": peer_confirm_flag.astype(float),
                    "peer_confirm_mom_return": mom_return.where(peer_confirm_flag),
                    "exception_peer_confirm_flag": exception_peer_confirm_flag.astype(float),
                    "exception_peer_confirm_mom_return": mom_return.where(exception_peer_confirm_flag),
                }
            )
            .groupby(["date", "theme"], as_index=False)
            .agg(
                short_term_volume_tilt_same_theme_peer_count=("symbol", "nunique"),
                short_term_volume_tilt_same_theme_peer_confirm_share=("peer_confirm_flag", "mean"),
                short_term_volume_tilt_same_theme_peer_avg_mom_return=("peer_confirm_mom_return", "mean"),
                short_term_volume_tilt_exception_same_theme_peer_count=("symbol", "nunique"),
                short_term_volume_tilt_exception_same_theme_peer_confirm_share=("exception_peer_confirm_flag", "mean"),
                short_term_volume_tilt_exception_same_theme_peer_avg_mom_return=("exception_peer_confirm_mom_return", "mean"),
            )
        )
        out["_short_term_volume_theme_date_key"] = date_key
        out["_short_term_volume_theme_key"] = theme_key
        out = out.merge(
            theme_daily,
            left_on=["_short_term_volume_theme_date_key", "_short_term_volume_theme_key"],
            right_on=["date", "theme"],
            how="left",
            suffixes=("", "_short_term_volume_theme"),
        )
        for column in (
            "short_term_volume_tilt_same_theme_peer_count",
            "short_term_volume_tilt_same_theme_peer_confirm_share",
            "short_term_volume_tilt_same_theme_peer_avg_mom_return",
        ):
            context_column = f"{column}_short_term_volume_theme"
            if context_column in out:
                out[column] = out[context_column].fillna(out[column])
                out = out.drop(columns=[context_column])
        out = out.drop(
            columns=["date_short_term_volume_theme", "theme", "_short_term_volume_theme_date_key", "_short_term_volume_theme_key"],
            errors="ignore",
        )
        same_theme_peer_count = pd.to_numeric(
            out.get("short_term_volume_tilt_same_theme_peer_count", 0),
            errors="coerce",
        ).fillna(0).astype(int)
        same_theme_peer_confirm_share = pd.to_numeric(
            out.get("short_term_volume_tilt_same_theme_peer_confirm_share", 0.0),
            errors="coerce",
        ).fillna(0.0)
        same_theme_peer_avg_mom_return = pd.to_numeric(
            out.get("short_term_volume_tilt_same_theme_peer_avg_mom_return", 0.0),
            errors="coerce",
        ).fillna(0.0)
        same_theme_peer_exception_count = pd.to_numeric(
            out.get("short_term_volume_tilt_exception_same_theme_peer_count", 0),
            errors="coerce",
        ).fillna(0).astype(int)
        same_theme_peer_exception_share = pd.to_numeric(
            out.get("short_term_volume_tilt_exception_same_theme_peer_confirm_share", 0.0),
            errors="coerce",
        ).fillna(0.0)
        same_theme_peer_exception_avg_mom_return = pd.to_numeric(
            out.get("short_term_volume_tilt_exception_same_theme_peer_avg_mom_return", 0.0),
            errors="coerce",
        ).fillna(0.0)
        same_theme_peer_confirmed = theme_key.ne("unknown")
        same_theme_peer_confirmed &= same_theme_peer_count.ge(same_theme_peer_min_count)
        same_theme_peer_confirmed &= same_theme_peer_confirm_share.ge(same_theme_peer_min_share)
        same_theme_peer_confirmed &= same_theme_peer_avg_mom_return.ge(same_theme_peer_min_avg_mom_return)
        same_theme_peer_exception_ok = theme_key.ne("unknown")
        same_theme_peer_exception_ok &= same_theme_peer_exception_count.ge(non_risk_on_exception_min_count)
        same_theme_peer_exception_ok &= same_theme_peer_exception_share.ge(non_risk_on_exception_min_share)
        same_theme_peer_exception_ok &= same_theme_peer_exception_avg_mom_return.ge(non_risk_on_exception_min_avg_mom_return)

    risk_on_active = candidate & circuit_breaker & same_theme_peer_confirmed
    non_risk_on_exception_active = pd.Series(False, index=out.index, dtype=bool)
    if require_benchmark_risk_on and allow_non_risk_on_exception:
        non_risk_on_exception_active = (
            candidate
            & base_circuit_breaker
            & benchmark_risk_on.eq(False)
            & same_theme_peer_exception_ok
        )
    same_theme_peer_volume_substitution_active = pd.Series(False, index=out.index, dtype=bool)
    if allow_same_theme_peer_volume_substitution:
        same_theme_peer_volume_substitution_active = (
            peer_volume_substitution_candidate
            & circuit_breaker
            & same_theme_peer_confirmed
            & volume_expansion.ge(same_theme_peer_substitute_min_volume_expansion)
            & volume_shortfall.le(same_theme_peer_substitute_max_volume_shortfall)
        )
    active = risk_on_active | non_risk_on_exception_active | same_theme_peer_volume_substitution_active
    blocked = (candidate | peer_volume_substitution_candidate) & ~active
    block_reason = pd.Series("", index=out.index, dtype=object)
    if require_benchmark_risk_on:
        block_reason = np.where(
            (candidate | peer_volume_substitution_candidate)
            & ~benchmark_risk_on
            & ~non_risk_on_exception_active,
            "benchmark_risk_off",
            block_reason,
        )
    if require_theme_active:
        block_reason = np.where((candidate | peer_volume_substitution_candidate) & theme_active.eq(False) & pd.Series(block_reason, index=out.index).eq(""), "theme_inactive", block_reason)
    block_reason = np.where((candidate | peer_volume_substitution_candidate) & event_risk.gt(max_event_risk) & pd.Series(block_reason, index=out.index).eq(""), "event_risk", block_reason)
    block_reason = np.where((candidate | peer_volume_substitution_candidate) & overnight_gap_risk.gt(max_gap_risk) & pd.Series(block_reason, index=out.index).eq(""), "overnight_gap_risk", block_reason)
    block_reason = np.where((candidate | peer_volume_substitution_candidate) & benchmark_ret_63d.lt(min_benchmark_ret63d) & pd.Series(block_reason, index=out.index).eq(""), "benchmark_ret63d", block_reason)
    block_reason = np.where((candidate | peer_volume_substitution_candidate) & trend_adx.lt(min_adx) & pd.Series(block_reason, index=out.index).eq(""), "adx", block_reason)
    block_reason = np.where(
        peer_volume_substitution_candidate
        & allow_same_theme_peer_volume_substitution
        & (
            volume_expansion.lt(same_theme_peer_substitute_min_volume_expansion)
            | volume_shortfall.gt(same_theme_peer_substitute_max_volume_shortfall)
        )
        & pd.Series(block_reason, index=out.index).eq(""),
        "volume_confirmation",
        block_reason,
    )
    block_reason = np.where(
        peer_volume_substitution_candidate
        & allow_same_theme_peer_volume_substitution
        & ~same_theme_peer_confirmed
        & pd.Series(block_reason, index=out.index).eq(""),
        "same_theme_peer_substitution",
        block_reason,
    )
    block_reason = np.where(candidate & ~same_theme_peer_confirmed & pd.Series(block_reason, index=out.index).eq(""), "same_theme_peer", block_reason)
    block_reason = pd.Series(block_reason, index=out.index).where(blocked, "")

    ret_span = max(abs(max_ret_5d) + 0.08, 1e-9)
    ret_component = ((max_ret_5d - ret_5d) / ret_span).clip(lower=0.0, upper=1.0)
    volume_component = ((volume_expansion - min_volume_expansion) / max(min_volume_expansion, 1e-9)).clip(lower=0.0, upper=1.0)
    extension_component = (1.0 - (above_ma20.clip(lower=0.0) / max(max_above_ma20, 1e-9))).clip(lower=0.0, upper=1.0)
    score = (ret_component * 45.0 + volume_component * 35.0 + extension_component * 20.0).where(active, 0.0)
    activation_mode = pd.Series("off", index=out.index, dtype=object)
    activation_mode = activation_mode.mask(risk_on_active, "risk_on")
    activation_mode = activation_mode.mask(non_risk_on_exception_active, "same_theme_peer_exception")
    activation_mode = activation_mode.mask(
        same_theme_peer_volume_substitution_active,
        "same_theme_peer_volume_substitution",
    )
    display_same_theme_peer_count = same_theme_peer_count.where(
        ~non_risk_on_exception_active,
        same_theme_peer_exception_count,
    )
    display_same_theme_peer_confirm_share = same_theme_peer_confirm_share.where(
        ~non_risk_on_exception_active,
        same_theme_peer_exception_share,
    )
    display_same_theme_peer_avg_mom_return = same_theme_peer_avg_mom_return.where(
        ~non_risk_on_exception_active,
        same_theme_peer_exception_avg_mom_return,
    )
    boost_scale = pd.Series(1.0, index=out.index, dtype=float)
    boost_scale = boost_scale.mask(
        non_risk_on_exception_active,
        non_risk_on_exception_score_boost_scale,
    )
    boost_scale = boost_scale.mask(
        same_theme_peer_volume_substitution_active,
        same_theme_peer_substitute_score_boost_scale,
    )
    boost = (score / 100.0 * max_boost * boost_scale).clip(lower=0.0, upper=max_boost)

    out["short_term_volume_tilt_score"] = score.fillna(0.0)
    out["short_term_volume_tilt_boost"] = boost.fillna(0.0)
    out["short_term_volume_tilt_ret_5d"] = ret_5d
    out["short_term_volume_tilt_volume_expansion"] = volume_expansion
    out["short_term_volume_tilt_candidate"] = (candidate | peer_volume_substitution_candidate).fillna(False)
    out["short_term_volume_tilt_circuit_ok"] = circuit_breaker.fillna(False)
    out["short_term_volume_tilt_peer_confirmed"] = same_theme_peer_confirmed.fillna(False)
    out["short_term_volume_tilt_blocked"] = blocked.fillna(False)
    out["short_term_volume_tilt_block_reason"] = block_reason.fillna("")
    out["short_term_volume_tilt_same_theme_peer_count"] = display_same_theme_peer_count.fillna(0).astype(int)
    out["short_term_volume_tilt_same_theme_peer_confirm_share"] = display_same_theme_peer_confirm_share.fillna(0.0)
    out["short_term_volume_tilt_same_theme_peer_avg_mom_return"] = display_same_theme_peer_avg_mom_return.fillna(0.0)
    out["short_term_volume_tilt_volume_shortfall"] = volume_shortfall.fillna(0.0)
    out["short_term_volume_tilt_peer_volume_substitution_candidate"] = peer_volume_substitution_candidate.fillna(False)
    out["short_term_volume_tilt_activation_mode"] = activation_mode
    out["final_score"] = (pd.to_numeric(out["final_score"], errors="coerce").fillna(50.0) + boost.fillna(0.0)).clip(0.0, 100.0)
    if active.any() or blocked.any():
        block_reason_counts = (
            pd.Series(block_reason, index=out.index)
            .where(blocked, "")
            .loc[lambda values: values.notna() & values.ne("")]
            .value_counts()
            .to_dict()
        )
        LOGGER.info(
            "momentum_short_term_volume_tilt_overlay_applied",
            extra={
                "event": "momentum_short_term_volume_tilt_overlay_applied",
                "rows": int(active.sum()),
                "risk_on_rows": int(risk_on_active.sum()),
                "non_risk_on_exception_rows": int(non_risk_on_exception_active.sum()),
                "same_theme_peer_volume_substitution_rows": int(same_theme_peer_volume_substitution_active.sum()),
                "blocked_rows": int(blocked.sum()),
                "blocked_reason_counts": block_reason_counts,
                "avg_boost": float(boost.loc[active].mean()) if active.any() else 0.0,
                "avg_ret_5d": float(ret_5d.loc[active].mean()) if active.any() else 0.0,
                "avg_volume_expansion": float(volume_expansion.loc[active].mean()) if active.any() else 0.0,
                "max_score_boost": max_boost,
                "max_ret_5d": max_ret_5d,
                "min_volume_expansion": min_volume_expansion,
                "max_above_ma20_pct": max_above_ma20,
                "min_relative_strength_score": min_rs,
                "min_theme_score": min_theme,
                "max_event_risk_score_circuit_breaker": max_event_risk,
                "max_overnight_gap_risk_score_circuit_breaker": max_gap_risk,
                "min_benchmark_ret63d_circuit_breaker": min_benchmark_ret63d,
                "min_adx_circuit_breaker": min_adx,
                "same_theme_peer_confirmation_overlay": use_same_theme_peer_confirmation,
                "same_theme_peer_min_count": same_theme_peer_min_count,
                "same_theme_peer_min_share": same_theme_peer_min_share,
                "same_theme_peer_min_avg_mom_return": same_theme_peer_min_avg_mom_return,
                "same_theme_peer_volume_substitution_overlay": allow_same_theme_peer_volume_substitution,
                "same_theme_peer_substitute_min_volume_expansion": same_theme_peer_substitute_min_volume_expansion,
                "same_theme_peer_substitute_max_volume_shortfall": same_theme_peer_substitute_max_volume_shortfall,
                "same_theme_peer_substitute_score_boost_scale": same_theme_peer_substitute_score_boost_scale,
                "non_risk_on_exception_overlay": allow_non_risk_on_exception,
                "non_risk_on_exception_min_same_theme_peer_count": non_risk_on_exception_min_count,
                "non_risk_on_exception_min_same_theme_peer_share": non_risk_on_exception_min_share,
                "non_risk_on_exception_min_same_theme_peer_avg_mom_return": non_risk_on_exception_min_avg_mom_return,
                "non_risk_on_exception_score_boost_scale": non_risk_on_exception_score_boost_scale,
                "avg_same_theme_peer_count": float(display_same_theme_peer_count.loc[active].mean()) if active.any() else 0.0,
                "avg_same_theme_peer_confirm_share": float(display_same_theme_peer_confirm_share.loc[active].mean()) if active.any() else 0.0,
                "avg_same_theme_peer_avg_mom_return": float(display_same_theme_peer_avg_mom_return.loc[active].mean()) if active.any() else 0.0,
            },
        )
    return out


def _apply_pullback_reclaim_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boost leaders that reclaim the 20-day trend after a controlled reset.

    Feature flag: ``long_pullback_reclaim_overlay`` defaults to false.
    Reasonable ranges:
    - ``pullback_reclaim_min_relative_strength_score``: 65 to 90.
    - ``pullback_reclaim_min_theme_score``: 55 to 80.
    - ``pullback_reclaim_min_drawdown_from_high``: 0.03 to 0.12.
    - ``pullback_reclaim_max_drawdown_from_high``: 0.08 to 0.25.
    - ``pullback_reclaim_recent_below_ma20_lookback_days``: 3 to 10.
    - ``pullback_reclaim_min_recent_below_ma20_pct``: 0.005 to 0.04.
    - ``pullback_reclaim_max_above_ma20_pct``: 0.0 to 0.04.
    - ``pullback_reclaim_min_volume_expansion``: 0.90 to 1.50.
    - ``pullback_reclaim_min_adx_circuit_breaker``: 18 to 35.
    - ``pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker``: 10 to 50.
    - ``pullback_reclaim_same_theme_gap_confirmation_min_peer_count``: 2 to 5.
    - ``pullback_reclaim_same_theme_gap_confirmation_min_active_share``: 0.35 to 0.85.
    - ``pullback_reclaim_same_theme_gap_confirmation_max_avg_gap_risk_score``: 10 to 30.
    - ``pullback_reclaim_same_theme_gap_confirmation_min_avg_relative_strength_score``: 70 to 95.
    Circuit breakers: the reclaim must occur in a benchmark-risk-on tape (or
    clear the configured benchmark return floor), with bounded overnight gap
    risk, sufficient ADX, a capped ``pullback_reclaim_max_score_boost``, and,
    when enabled, same-theme breadth plus gap-quality confirmation.
    """

    out = frame.copy()
    out["pullback_reclaim_score"] = 0.0
    out["pullback_reclaim_boost"] = 0.0
    out["pullback_reclaim_recent_below_ma20_pct"] = 0.0
    out["pullback_reclaim_same_theme_gap_confirmation_ok"] = False
    out["pullback_reclaim_same_theme_peer_count"] = 0
    out["pullback_reclaim_same_theme_active_share"] = 0.0
    out["pullback_reclaim_same_theme_avg_gap_risk"] = 0.0
    out["pullback_reclaim_same_theme_avg_rs"] = 0.0
    out["pullback_reclaim_blocked"] = False
    if not _bool_param(params, "long_pullback_reclaim_overlay", False):
        return out

    max_boost = float(np.clip(params.get("pullback_reclaim_max_score_boost", 3.0), 0.0, 10.0))
    if max_boost <= 0.0:
        return out

    ordered = out.sort_values(["symbol", "date"]).copy()
    original_order = ordered.index.to_numpy(copy=True)
    ordered = ordered.reset_index(drop=True)
    adj_close = pd.to_numeric(ordered["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(ordered, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(ordered, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    ma200 = pd.to_numeric(_column(ordered, "ma_200", np.nan), errors="coerce").replace(0, np.nan)
    distance_high = pd.to_numeric(ordered.get("mom_distance_high", ordered.get("distance_to_prior_high_252")), errors="coerce")
    volume_expansion = pd.to_numeric(ordered.get("mom_volume_expansion", ordered.get("volume_expansion", 1.0)), errors="coerce")
    volume_expansion = volume_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    rs_score = pd.to_numeric(ordered.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(ordered.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    trend_adx = pd.to_numeric(_column(ordered, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(ordered, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = _column(ordered, "benchmark_risk_on", False).fillna(False).astype(bool)
    benchmark_ret_63d = pd.to_numeric(_column(ordered, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    mom_return = pd.to_numeric(ordered.get("mom_return", 0.0), errors="coerce").fillna(0.0)

    min_drawdown = float(params.get("pullback_reclaim_min_drawdown_from_high", 0.04))
    max_drawdown = float(params.get("pullback_reclaim_max_drawdown_from_high", 0.18))
    lookback_days = max(int(params.get("pullback_reclaim_recent_below_ma20_lookback_days", 5)), 2)
    min_recent_below_ma20 = max(float(params.get("pullback_reclaim_min_recent_below_ma20_pct", 0.01)), 0.0)
    max_above_ma20 = float(params.get("pullback_reclaim_max_above_ma20_pct", 0.025))
    min_volume_expansion = float(params.get("pullback_reclaim_min_volume_expansion", 1.0))
    min_rs = float(params.get("pullback_reclaim_min_relative_strength_score", 72.0))
    min_theme = float(params.get("pullback_reclaim_min_theme_score", 60.0))
    min_benchmark_ret63d = float(params.get("pullback_reclaim_min_benchmark_ret63d_circuit_breaker", 0.0))
    min_adx = float(params.get("pullback_reclaim_min_adx_circuit_breaker", 22.0))
    max_overnight_gap_risk = float(params.get("pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker", 35.0))
    require_benchmark_risk_on = _bool_param(params, "pullback_reclaim_require_benchmark_risk_on", True)
    use_same_theme_gap_confirmation = _bool_param(params, "pullback_reclaim_same_theme_gap_confirmation_overlay", False)

    drawdown_from_high = (-distance_high).clip(lower=0.0)
    above_ma20 = adj_close / ma20 - 1.0
    ma20_vs_50 = ma20 / ma50 - 1.0
    ma50_vs_200 = ma50 / ma200 - 1.0
    min_periods = max(2, lookback_days // 2)
    recent_below_ma20 = (
        -ordered.groupby("symbol", group_keys=False)["adj_close"].transform(
            lambda _: pd.Series(above_ma20, index=ordered.index).loc[_.index].shift(1).rolling(lookback_days, min_periods=min_periods).min()
        )
    ).clip(lower=0.0)
    ordered["pullback_reclaim_recent_below_ma20_pct"] = recent_below_ma20.fillna(0.0)

    trend_intact = (adj_close > ma50) & (ma20_vs_50 > 0.0) & ((adj_close > ma200) | ma50_vs_200.fillna(0.0).gt(-0.05))
    controlled_pullback = drawdown_from_high.between(min_drawdown, max_drawdown, inclusive="both")
    leadership = rs_score.ge(min_rs) & theme_score.ge(min_theme) & mom_return.gt(0.0)
    reclaiming = recent_below_ma20.ge(min_recent_below_ma20) & above_ma20.between(0.0, max_above_ma20, inclusive="both")
    volume_confirmation = volume_expansion.ge(min_volume_expansion)
    benchmark_ok = benchmark_risk_on if require_benchmark_risk_on else benchmark_ret_63d.ge(min_benchmark_ret63d)
    circuit_breaker = benchmark_ok & overnight_gap_risk.le(max_overnight_gap_risk) & trend_adx.ge(min_adx)
    candidate = trend_intact & controlled_pullback & leadership & reclaiming & volume_confirmation

    same_theme_gap_confirmation_ok = pd.Series(True, index=ordered.index, dtype=bool)
    same_theme_peer_count = pd.Series(0, index=ordered.index, dtype=int)
    same_theme_active_share = pd.Series(0.0, index=ordered.index, dtype=float)
    same_theme_avg_gap_risk = pd.Series(0.0, index=ordered.index, dtype=float)
    same_theme_avg_rs = pd.Series(0.0, index=ordered.index, dtype=float)
    if use_same_theme_gap_confirmation:
        min_peer_count = max(int(params.get("pullback_reclaim_same_theme_gap_confirmation_min_peer_count", 3)), 2)
        min_active_share = float(params.get("pullback_reclaim_same_theme_gap_confirmation_min_active_share", 0.50))
        max_avg_gap_risk = float(params.get("pullback_reclaim_same_theme_gap_confirmation_max_avg_gap_risk_score", 22.0))
        min_avg_rs = float(params.get("pullback_reclaim_same_theme_gap_confirmation_min_avg_relative_strength_score", 80.0))
        if "primary_theme" in ordered:
            theme_key = ordered["primary_theme"]
        elif "theme_reason" in ordered:
            theme_key = ordered["theme_reason"]
        else:
            theme_key = pd.Series("unknown", index=ordered.index)
        theme_key = (
            pd.Series(theme_key, index=ordered.index)
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
        )
        theme_active = _column(ordered, "theme_active", True).fillna(False).astype(bool)
        date_key = pd.to_datetime(ordered["date"]).dt.normalize()
        same_theme_context = (
            pd.DataFrame(
                {
                    "_pullback_reclaim_theme_date_key": date_key,
                    "_pullback_reclaim_theme_key": theme_key,
                    "symbol": ordered["symbol"],
                    "same_theme_active": (adj_close.gt(ma20).fillna(False) & theme_active).astype(float),
                    "gap_risk": overnight_gap_risk,
                    "rs_score": rs_score,
                }
            )
            .groupby(["_pullback_reclaim_theme_date_key", "_pullback_reclaim_theme_key"], as_index=False)
            .agg(
                pullback_reclaim_same_theme_peer_count=("symbol", "nunique"),
                pullback_reclaim_same_theme_active_share=("same_theme_active", "mean"),
                pullback_reclaim_same_theme_avg_gap_risk=("gap_risk", "mean"),
                pullback_reclaim_same_theme_avg_rs=("rs_score", "mean"),
            )
        )
        ordered = ordered.drop(
            columns=[
                "pullback_reclaim_same_theme_peer_count",
                "pullback_reclaim_same_theme_active_share",
                "pullback_reclaim_same_theme_avg_gap_risk",
                "pullback_reclaim_same_theme_avg_rs",
            ],
            errors="ignore",
        )
        ordered["_pullback_reclaim_theme_date_key"] = date_key
        ordered["_pullback_reclaim_theme_key"] = theme_key
        ordered = ordered.merge(
            same_theme_context,
            on=["_pullback_reclaim_theme_date_key", "_pullback_reclaim_theme_key"],
            how="left",
        )
        same_theme_peer_count = pd.to_numeric(ordered.get("pullback_reclaim_same_theme_peer_count", 0), errors="coerce").fillna(0).astype(int)
        same_theme_active_share = pd.to_numeric(ordered.get("pullback_reclaim_same_theme_active_share", 0.0), errors="coerce").fillna(0.0)
        same_theme_avg_gap_risk = pd.to_numeric(ordered.get("pullback_reclaim_same_theme_avg_gap_risk", 0.0), errors="coerce").fillna(0.0)
        same_theme_avg_rs = pd.to_numeric(ordered.get("pullback_reclaim_same_theme_avg_rs", 0.0), errors="coerce").fillna(0.0)
        same_theme_gap_confirmation_ok = theme_key.ne("unknown")
        same_theme_gap_confirmation_ok &= same_theme_peer_count.ge(min_peer_count)
        same_theme_gap_confirmation_ok &= same_theme_active_share.ge(min_active_share)
        same_theme_gap_confirmation_ok &= same_theme_avg_gap_risk.le(max_avg_gap_risk)
        same_theme_gap_confirmation_ok &= same_theme_avg_rs.ge(min_avg_rs)

    eligible = candidate & circuit_breaker & same_theme_gap_confirmation_ok
    blocked = candidate & ~circuit_breaker
    if use_same_theme_gap_confirmation:
        blocked |= candidate & circuit_breaker & ~same_theme_gap_confirmation_ok

    reclaim_depth_component = (recent_below_ma20 / max(min_recent_below_ma20 * 2.0, 1e-9)).clip(0.0, 1.0)
    reclaim_extension_component = (1.0 - (above_ma20.clip(lower=0.0) / max(max_above_ma20, 1e-9))).clip(0.0, 1.0)
    volume_component = ((volume_expansion - min_volume_expansion) / max(min_volume_expansion, 1e-9)).clip(0.0, 2.0) / 2.0
    rs_component = ((rs_score - min_rs) / max(100.0 - min_rs, 1e-9)).clip(0.0, 1.0)
    theme_component = ((theme_score - min_theme) / max(100.0 - min_theme, 1e-9)).clip(0.0, 1.0)
    score = (
        reclaim_depth_component * 30.0
        + reclaim_extension_component * 30.0
        + volume_component * 20.0
        + rs_component * 10.0
        + theme_component * 10.0
    ).where(eligible, 0.0)
    boost = (score / 100.0 * max_boost).clip(lower=0.0, upper=max_boost)

    ordered["pullback_reclaim_score"] = score.fillna(0.0)
    ordered["pullback_reclaim_boost"] = boost.fillna(0.0)
    ordered["pullback_reclaim_same_theme_gap_confirmation_ok"] = same_theme_gap_confirmation_ok.fillna(False)
    ordered["pullback_reclaim_same_theme_peer_count"] = same_theme_peer_count.fillna(0).astype(int)
    ordered["pullback_reclaim_same_theme_active_share"] = same_theme_active_share.fillna(0.0)
    ordered["pullback_reclaim_same_theme_avg_gap_risk"] = same_theme_avg_gap_risk.fillna(0.0)
    ordered["pullback_reclaim_same_theme_avg_rs"] = same_theme_avg_rs.fillna(0.0)
    ordered["pullback_reclaim_blocked"] = blocked.fillna(False)
    ordered["final_score"] = (pd.to_numeric(ordered["final_score"], errors="coerce").fillna(50.0) + ordered["pullback_reclaim_boost"]).clip(0.0, 100.0)
    if candidate.any():
        same_theme_confirmation_blocked = candidate & circuit_breaker & ~same_theme_gap_confirmation_ok
        LOGGER.info(
            "momentum_pullback_reclaim_overlay_applied",
            extra={
                "event": "momentum_pullback_reclaim_overlay_applied",
                "rows": int(eligible.sum()),
                "blocked_rows": int(blocked.sum()),
                "same_theme_gap_confirmation_rows": int((eligible & same_theme_gap_confirmation_ok).sum()),
                "same_theme_gap_confirmation_blocked_rows": int(same_theme_confirmation_blocked.sum()),
                "avg_boost": float(ordered.loc[eligible, "pullback_reclaim_boost"].mean()) if eligible.any() else 0.0,
                "max_boost": max_boost,
                "lookback_days": lookback_days,
                "min_recent_below_ma20": min_recent_below_ma20,
                "max_above_ma20": max_above_ma20,
                "min_volume_expansion": min_volume_expansion,
                "min_rs": min_rs,
                "min_theme": min_theme,
                "min_benchmark_ret63d": min_benchmark_ret63d,
                "min_adx": min_adx,
                "max_overnight_gap_risk": max_overnight_gap_risk,
                "require_benchmark_risk_on": require_benchmark_risk_on,
                "same_theme_gap_confirmation_overlay": use_same_theme_gap_confirmation,
                "same_theme_gap_confirmation_min_peer_count": int(params.get("pullback_reclaim_same_theme_gap_confirmation_min_peer_count", 3)),
                "same_theme_gap_confirmation_min_active_share": float(params.get("pullback_reclaim_same_theme_gap_confirmation_min_active_share", 0.50)),
                "same_theme_gap_confirmation_max_avg_gap_risk_score": float(params.get("pullback_reclaim_same_theme_gap_confirmation_max_avg_gap_risk_score", 22.0)),
                "same_theme_gap_confirmation_min_avg_relative_strength_score": float(params.get("pullback_reclaim_same_theme_gap_confirmation_min_avg_relative_strength_score", 80.0)),
                "avg_same_theme_peer_count": float(same_theme_peer_count.loc[eligible].mean()) if eligible.any() else 0.0,
                "avg_same_theme_active_share": float(same_theme_active_share.loc[eligible].mean()) if eligible.any() else 0.0,
                "avg_same_theme_avg_gap_risk": float(same_theme_avg_gap_risk.loc[eligible].mean()) if eligible.any() else 0.0,
                "avg_same_theme_avg_rs": float(same_theme_avg_rs.loc[eligible].mean()) if eligible.any() else 0.0,
            },
        )
    ordered = ordered.drop(
        columns=[
            "_pullback_reclaim_theme_date_key",
            "_pullback_reclaim_theme_key",
        ],
        errors="ignore",
    )
    ordered["_pullback_reclaim_restore_order"] = original_order
    ordered = ordered.sort_values("_pullback_reclaim_restore_order").drop(columns=["_pullback_reclaim_restore_order"], errors="ignore")
    return ordered


def _long_pullback_reset_inclusion_gate(frame: pd.DataFrame, params: dict) -> pd.Series:
    """Allow a small number of high-quality volume-reset leaders into longs.

    Feature flag: ``long_pullback_reset_inclusion_overlay`` defaults to false.
    Reasonable ranges:
    - ``pullback_reset_inclusion_min_score``: 40 to 80.
    - ``pullback_reset_inclusion_max_names_per_date``: 1 to 5.
    - ``pullback_reset_inclusion_min_final_score``: 45 to 65.
    - ``pullback_reset_inclusion_min_score_rank``: 0.40 to 0.80.
    - ``pullback_reset_inclusion_require_theme_breadth_expansion``: true/false.
    - ``pullback_reset_inclusion_theme_breadth_lookback_days``: 5 to 20.
    - ``pullback_reset_inclusion_theme_breadth_min_active_share``: 0.25 to 0.80.
    - ``pullback_reset_inclusion_theme_breadth_expansion_threshold``: 0.02 to 0.20.
    - ``pullback_reset_inclusion_require_rs_acceleration``: true/false.
    - ``pullback_reset_inclusion_rs_acceleration_lookback_days``: 5 to 20.
    - ``pullback_reset_inclusion_min_current_rs_score``: 65 to 95.
    - ``pullback_reset_inclusion_min_rs_acceleration``: 2 to 15 score points.
    - ``pullback_reset_inclusion_theme_strength_support_overlay``: true/false.
    - ``pullback_reset_inclusion_min_theme_strength_delta_score``: 20 to 80.
    - ``pullback_reset_inclusion_theme_strength_score_rank_credit``: 0.02 to 0.15.
    - ``pullback_reset_inclusion_activation_support_overlay``: true/false.
    - ``pullback_reset_inclusion_activation_min_score``: 20 to 70.
    - ``pullback_reset_inclusion_activation_score_rank_credit``: 0.02 to 0.15.
    - ``pullback_reset_inclusion_reclaim_support_overlay``: true/false.
    - ``pullback_reset_inclusion_min_reclaim_score``: 15 to 60.
    - ``pullback_reset_inclusion_reclaim_score_rank_credit``: 0.02 to 0.15.
    - ``pullback_reset_inclusion_require_theme_breadth_not_deteriorating``:
      true/false.
    - ``pullback_reset_inclusion_theme_breadth_active_share_threshold``: 0.20
      to 0.70.
    - ``pullback_reset_inclusion_theme_breadth_shortfall_threshold``: 0.02 to
      0.20.

    Circuit breakers: keeps the existing volume-contraction benchmark/gap/ADX
    gates for ordinary reset entries, disables the breadth/RS-acceleration gates
    when there is not enough prior-date history, only uses prior-date
    theme-strength rows that already cleared their own benchmark/event/gap
    gates, only widens reset admissions with reclaim support after a real MA20
    reclaim, blocks that reclaim support when theme breadth is deteriorating,
    and caps the number of extra candidates per date.
    """

    frame["pullback_reset_inclusion"] = False
    frame["pullback_reset_theme_breadth_active_share"] = 1.0
    frame["pullback_reset_theme_breadth_baseline_share"] = 1.0
    frame["pullback_reset_theme_breadth_expansion"] = 0.0
    frame["pullback_reset_theme_breadth_expanding"] = False
    frame["pullback_reset_rs_score"] = pd.to_numeric(frame.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    frame["pullback_reset_rs_baseline"] = np.nan
    frame["pullback_reset_rs_acceleration"] = 0.0
    frame["pullback_reset_rs_accelerating"] = False
    frame["pullback_reset_theme_strength_delta_score"] = pd.to_numeric(
        frame.get("theme_strength_delta_score", 0.0), errors="coerce"
    ).fillna(0.0)
    frame["pullback_reset_theme_strength_supported"] = False
    frame["pullback_reset_theme_strength_score_rank_credit"] = 0.0
    frame["pullback_reset_activation_score"] = 0.0
    frame["pullback_reset_activation_supported"] = False
    frame["pullback_reset_activation_score_rank_credit"] = 0.0
    frame["pullback_reset_reclaim_score"] = pd.to_numeric(frame.get("pullback_reclaim_score", 0.0), errors="coerce").fillna(0.0)
    frame["pullback_reset_reclaim_supported"] = False
    frame["pullback_reset_reclaim_score_rank_credit"] = 0.0
    frame["pullback_reset_theme_breadth_safe"] = True
    frame["pullback_reset_theme_breadth_shortfall"] = 0.0
    if not _bool_param(params, "long_pullback_reset_inclusion_overlay", False):
        return pd.Series(False, index=frame.index)

    if _bool_param(params, "pullback_reset_inclusion_require_theme_breadth_expansion", False):
        breadth_context = _attach_pullback_reset_theme_breadth_context(frame, params)
        for column in (
            "pullback_reset_theme_breadth_active_share",
            "pullback_reset_theme_breadth_baseline_share",
            "pullback_reset_theme_breadth_expansion",
            "pullback_reset_theme_breadth_expanding",
        ):
            frame[column] = breadth_context[column]
    if _bool_param(params, "pullback_reset_inclusion_require_rs_acceleration", False):
        rs_context = _attach_pullback_reset_rs_acceleration_context(frame, params)
        for column in (
            "pullback_reset_rs_score",
            "pullback_reset_rs_baseline",
            "pullback_reset_rs_acceleration",
            "pullback_reset_rs_accelerating",
        ):
            frame[column] = rs_context[column]

    reset_score = pd.to_numeric(frame.get("pullback_volume_contraction_score", 0.0), errors="coerce").fillna(0.0)
    blocked = frame.get("pullback_volume_contraction_blocked", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    final = pd.to_numeric(frame.get("final_score", 0.0), errors="coerce").fillna(0.0)
    score_rank = pd.to_numeric(frame.get("score_rank", 0.0), errors="coerce").fillna(0.0)
    fundamental = pd.to_numeric(frame.get("fundamental_score", 50.0), errors="coerce").fillna(50.0)
    adj_close = pd.to_numeric(frame.get("adj_close", np.nan), errors="coerce")
    ma20 = pd.to_numeric(_column(frame, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(frame, "ma_50", np.nan), errors="coerce")
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan)
    rs_score = pd.to_numeric(frame.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(frame.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(frame.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    overnight_gap_risk = pd.to_numeric(_column(frame, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(_column(frame, "benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)
    trend_adx = pd.to_numeric(_column(frame, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    benchmark_risk_on = pd.Series(_column(frame, "benchmark_risk_on", False), index=frame.index)
    benchmark_risk_on = benchmark_risk_on.where(benchmark_risk_on.notna(), False).astype(bool)
    theme_active = pd.Series(_column(frame, "theme_active", True), index=frame.index)
    theme_active = theme_active.where(theme_active.notna(), True).astype(bool)
    mom_return = pd.to_numeric(frame.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    breadth_expanding = frame.get("pullback_reset_theme_breadth_expanding", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    require_breadth_expansion = _bool_param(params, "pullback_reset_inclusion_require_theme_breadth_expansion", False)
    rs_accelerating = frame.get("pullback_reset_rs_accelerating", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    require_rs_acceleration = _bool_param(params, "pullback_reset_inclusion_require_rs_acceleration", False)
    theme_strength_support = _bool_param(params, "pullback_reset_inclusion_theme_strength_support_overlay", False)
    theme_strength_score = pd.to_numeric(frame.get("theme_strength_delta_score", 0.0), errors="coerce").fillna(0.0)
    theme_strength_eligible = frame.get("theme_strength_delta_eligible", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    activation_support = _bool_param(params, "pullback_reset_inclusion_activation_support_overlay", False)
    reclaim_support = _bool_param(params, "pullback_reset_inclusion_reclaim_support_overlay", False)
    reclaim_score = pd.to_numeric(frame.get("pullback_reclaim_score", 0.0), errors="coerce").fillna(0.0)
    reclaim_blocked = frame.get("pullback_reclaim_blocked", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    require_theme_breadth_not_deteriorating = _bool_param(
        params,
        "pullback_reset_inclusion_require_theme_breadth_not_deteriorating",
        False,
    )

    min_reset_score = float(params.get("pullback_reset_inclusion_min_score", 50.0))
    min_final_score = float(params.get("pullback_reset_inclusion_min_final_score", 52.0))
    min_score_rank = float(params.get("pullback_reset_inclusion_min_score_rank", 0.55))
    min_theme_strength_score = float(params.get("pullback_reset_inclusion_min_theme_strength_delta_score", 35.0))
    max_theme_strength_rank_credit = float(
        np.clip(params.get("pullback_reset_inclusion_theme_strength_score_rank_credit", 0.08), 0.0, 0.15)
    )
    min_activation_score = float(params.get("pullback_reset_inclusion_activation_min_score", 35.0))
    max_activation_rank_credit = float(
        np.clip(params.get("pullback_reset_inclusion_activation_score_rank_credit", 0.08), 0.0, 0.15)
    )
    min_reclaim_score = float(params.get("pullback_reset_inclusion_min_reclaim_score", 25.0))
    max_reclaim_rank_credit = float(
        np.clip(params.get("pullback_reset_inclusion_reclaim_score_rank_credit", 0.08), 0.0, 0.15)
    )
    max_names = max(0, int(params.get("pullback_reset_inclusion_max_names_per_date", 2)))
    if max_names <= 0:
        return pd.Series(False, index=frame.index)

    if reclaim_support and require_theme_breadth_not_deteriorating:
        lookback_days = max(int(params.get("pullback_reset_inclusion_theme_breadth_lookback_days", 10)), 2)
        active_share_threshold = float(params.get("pullback_reset_inclusion_theme_breadth_active_share_threshold", 0.38))
        shortfall_threshold = max(float(params.get("pullback_reset_inclusion_theme_breadth_shortfall_threshold", 0.05)), 0.0)
        breadth = _compute_theme_breadth_frame(frame, lookback_days).rename(
            columns={
                "theme_breadth_active_share": "pullback_reset_theme_breadth_active_share",
                "theme_breadth_baseline_share": "pullback_reset_theme_breadth_baseline_share",
            }
        )
        breadth["pullback_reset_theme_breadth_shortfall"] = (
            breadth["pullback_reset_theme_breadth_baseline_share"] - breadth["pullback_reset_theme_breadth_active_share"]
        ).clip(lower=0.0)
        breadth_deteriorating = (
            breadth["pullback_reset_theme_breadth_baseline_share"].notna()
            & breadth["pullback_reset_theme_breadth_active_share"].le(active_share_threshold)
            & breadth["pullback_reset_theme_breadth_shortfall"].ge(shortfall_threshold)
        )
        breadth["pullback_reset_theme_breadth_safe"] = breadth["pullback_reset_theme_breadth_baseline_share"].isna() | ~breadth_deteriorating
        breadth = breadth.set_index("date")
        normalized_dates = pd.to_datetime(frame["date"]).dt.normalize()
        frame["pullback_reset_theme_breadth_active_share"] = pd.to_numeric(
            normalized_dates.map(breadth["pullback_reset_theme_breadth_active_share"]), errors="coerce"
        ).fillna(1.0)
        frame["pullback_reset_theme_breadth_baseline_share"] = pd.to_numeric(
            normalized_dates.map(breadth["pullback_reset_theme_breadth_baseline_share"]), errors="coerce"
        ).fillna(1.0)
        frame["pullback_reset_theme_breadth_shortfall"] = pd.to_numeric(
            normalized_dates.map(breadth["pullback_reset_theme_breadth_shortfall"]), errors="coerce"
        ).fillna(0.0)
        frame["pullback_reset_theme_breadth_safe"] = normalized_dates.map(
            breadth["pullback_reset_theme_breadth_safe"]
        ).fillna(True).astype(bool)

    if theme_strength_support:
        theme_strength_supported = theme_strength_eligible & theme_strength_score.ge(min_theme_strength_score)
    else:
        theme_strength_supported = pd.Series(False, index=frame.index)
    theme_strength_rank_credit = (
        ((theme_strength_score - min_theme_strength_score) / max(100.0 - min_theme_strength_score, 1e-9)).clip(0.0, 1.0)
        * max_theme_strength_rank_credit
    ).where(theme_strength_supported, 0.0)
    grouped = frame.sort_values(["symbol", "date"]).groupby("symbol", group_keys=False)
    ret_5d = pd.to_numeric(frame.get("ret_5d"), errors="coerce") if "ret_5d" in frame else grouped["adj_close"].pct_change(5)
    volume_expansion = pd.to_numeric(frame.get("mom_volume_expansion", frame.get("volume_expansion", 1.0)), errors="coerce")
    volume_expansion = volume_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    activation_max_ret_5d = float(params.get("short_term_volume_tilt_max_ret_5d", 0.00))
    activation_min_volume_expansion = float(params.get("short_term_volume_tilt_min_volume_expansion", 1.05))
    activation_min_rs = float(params.get("short_term_volume_tilt_min_relative_strength_score", 58.0))
    activation_min_theme = float(params.get("short_term_volume_tilt_min_theme_score", 55.0))
    activation_max_above_ma20 = float(params.get("short_term_volume_tilt_max_above_ma20_pct", 0.08))
    activation_max_event_risk = float(params.get("short_term_volume_tilt_max_event_risk_score_circuit_breaker", 20.0))
    activation_max_gap_risk = float(params.get("short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker", 32.0))
    activation_min_benchmark_ret63d = float(params.get("short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker", 0.0))
    activation_min_adx = float(params.get("short_term_volume_tilt_min_adx_circuit_breaker", 18.0))
    activation_require_benchmark_risk_on = _bool_param(params, "short_term_volume_tilt_require_benchmark_risk_on", True)
    activation_require_theme_active = _bool_param(params, "short_term_volume_tilt_require_theme_active", False)
    activation_require_price_above_ma50 = _bool_param(params, "short_term_volume_tilt_require_price_above_ma50", True)
    if activation_support:
        activation_candidate = (
            ret_5d.le(activation_max_ret_5d).fillna(False)
            & volume_expansion.ge(activation_min_volume_expansion)
            & rs_score.ge(activation_min_rs)
            & theme_score.ge(activation_min_theme)
            & above_ma20.le(activation_max_above_ma20).fillna(False)
            & mom_return.gt(0.0)
        )
        activation_circuit_breaker = (
            event_risk.le(activation_max_event_risk)
            & overnight_gap_risk.le(activation_max_gap_risk)
            & benchmark_ret_63d.ge(activation_min_benchmark_ret63d)
            & trend_adx.ge(activation_min_adx)
        )
        if activation_require_benchmark_risk_on:
            activation_circuit_breaker &= benchmark_risk_on
        if activation_require_theme_active:
            activation_circuit_breaker &= theme_active
        if activation_require_price_above_ma50:
            activation_circuit_breaker &= adj_close.gt(ma50).fillna(False)
        ret_span = max(abs(activation_max_ret_5d) + 0.08, 1e-9)
        activation_ret_component = ((activation_max_ret_5d - ret_5d) / ret_span).clip(lower=0.0, upper=1.0)
        activation_volume_component = (
            (volume_expansion - activation_min_volume_expansion) / max(activation_min_volume_expansion, 1e-9)
        ).clip(lower=0.0, upper=1.0)
        activation_extension_component = (
            1.0 - (above_ma20.clip(lower=0.0) / max(activation_max_above_ma20, 1e-9))
        ).clip(lower=0.0, upper=1.0)
        activation_rs_component = ((rs_score - activation_min_rs) / max(100.0 - activation_min_rs, 1e-9)).clip(lower=0.0, upper=1.0)
        activation_theme_component = ((theme_score - activation_min_theme) / max(100.0 - activation_min_theme, 1e-9)).clip(lower=0.0, upper=1.0)
        activation_score = (
            activation_ret_component * 35.0
            + activation_volume_component * 30.0
            + activation_extension_component * 15.0
            + activation_rs_component * 10.0
            + activation_theme_component * 10.0
        ).where(activation_candidate & activation_circuit_breaker, 0.0)
        activation_supported = activation_score.ge(min_activation_score)
        activation_rank_credit = (
            ((activation_score - min_activation_score) / max(100.0 - min_activation_score, 1e-9)).clip(0.0, 1.0)
            * max_activation_rank_credit
        ).where(activation_supported, 0.0)
    else:
        activation_score = pd.Series(0.0, index=frame.index, dtype=float)
        activation_supported = pd.Series(False, index=frame.index)
        activation_rank_credit = pd.Series(0.0, index=frame.index, dtype=float)
    if reclaim_support:
        reclaim_supported = reclaim_score.ge(min_reclaim_score) & ~reclaim_blocked
        if require_theme_breadth_not_deteriorating:
            reclaim_supported &= frame["pullback_reset_theme_breadth_safe"]
    else:
        reclaim_supported = pd.Series(False, index=frame.index)
    reclaim_rank_credit = (
        ((reclaim_score - min_reclaim_score) / max(100.0 - min_reclaim_score, 1e-9)).clip(0.0, 1.0)
        * max_reclaim_rank_credit
    ).where(reclaim_supported, 0.0)
    effective_score_rank = (score_rank + theme_strength_rank_credit + activation_rank_credit + reclaim_rank_credit).clip(upper=1.0)
    frame["pullback_reset_theme_strength_delta_score"] = theme_strength_score
    frame["pullback_reset_theme_strength_supported"] = theme_strength_supported
    frame["pullback_reset_theme_strength_score_rank_credit"] = theme_strength_rank_credit
    frame["pullback_reset_activation_score"] = activation_score
    frame["pullback_reset_activation_supported"] = activation_supported
    frame["pullback_reset_activation_score_rank_credit"] = activation_rank_credit
    frame["pullback_reset_reclaim_score"] = reclaim_score
    frame["pullback_reset_reclaim_supported"] = reclaim_supported
    frame["pullback_reset_reclaim_score_rank_credit"] = reclaim_rank_credit

    supported_union = theme_strength_supported | activation_supported | reclaim_supported
    reset_entry_ok = reset_score.ge(min_reset_score)
    if activation_support:
        reset_entry_ok |= activation_supported
    if reclaim_support:
        reset_entry_ok |= reclaim_supported

    reset_selection_score = reset_score.where(~activation_supported, np.maximum(reset_score, activation_score))
    raw = (
        reset_entry_ok
        & ~blocked
        & final.ge(min_final_score)
        & effective_score_rank.ge(min_score_rank)
        & fundamental.ge(float(params.get("long_min_fundamental_score", 40.0)))
        & (adj_close > ma50)
    )
    if require_breadth_expansion:
        raw &= breadth_expanding
    if require_rs_acceleration:
        raw &= rs_accelerating
    reset_rank = reset_selection_score.where(raw, -np.inf).groupby(frame["date"]).rank(method="first", ascending=False)
    include = raw & reset_rank.le(max_names)
    frame.loc[include, "pullback_reset_inclusion"] = True
    if include.any() or theme_strength_supported.any() or activation_supported.any():
        LOGGER.info(
            "momentum_pullback_reset_inclusion_applied",
            extra={
                "event": "momentum_pullback_reset_inclusion_applied",
                "rows": int(include.sum()),
                "dates": int(frame.loc[include, "date"].nunique()),
                "max_names_per_date": max_names,
                "min_reset_score": min_reset_score,
                "min_final_score": min_final_score,
                "min_score_rank": min_score_rank,
                "require_theme_breadth_expansion": require_breadth_expansion,
                "theme_breadth_expanding_rows": int(breadth_expanding.sum()),
                "require_rs_acceleration": require_rs_acceleration,
                "rs_accelerating_rows": int(rs_accelerating.sum()),
                "require_theme_strength_support": theme_strength_support,
                "theme_strength_supported_rows": int(theme_strength_supported.sum()),
                "min_theme_strength_score": min_theme_strength_score,
                "max_theme_strength_rank_credit": max_theme_strength_rank_credit,
                "avg_theme_strength_score_supported": float(theme_strength_score.loc[theme_strength_supported].mean())
                if theme_strength_supported.any()
                else 0.0,
                "avg_theme_strength_rank_credit": float(theme_strength_rank_credit.loc[theme_strength_supported].mean())
                if theme_strength_supported.any()
                else 0.0,
                "require_activation_support": activation_support,
                "activation_supported_rows": int(activation_supported.sum()),
                "activation_only_rows": int((activation_supported & reset_score.lt(min_reset_score)).sum()),
                "activation_min_score": min_activation_score,
                "activation_max_score_rank_credit": max_activation_rank_credit,
                "activation_max_ret_5d": activation_max_ret_5d,
                "activation_min_volume_expansion": activation_min_volume_expansion,
                "activation_max_above_ma20_pct": activation_max_above_ma20,
                "activation_min_relative_strength_score": activation_min_rs,
                "activation_min_theme_score": activation_min_theme,
                "activation_max_event_risk_score": activation_max_event_risk,
                "activation_max_overnight_gap_risk_score": activation_max_gap_risk,
                "activation_min_benchmark_ret63d": activation_min_benchmark_ret63d,
                "activation_min_adx": activation_min_adx,
                "activation_require_benchmark_risk_on": activation_require_benchmark_risk_on,
                "activation_require_theme_active": activation_require_theme_active,
                "activation_require_price_above_ma50": activation_require_price_above_ma50,
                "avg_activation_score_supported": float(activation_score.loc[activation_supported].mean())
                if activation_supported.any()
                else 0.0,
                "avg_activation_rank_credit": float(activation_rank_credit.loc[activation_supported].mean())
                if activation_supported.any()
                else 0.0,
                "require_reclaim_support": reclaim_support,
                "reclaim_supported_rows": int(reclaim_supported.sum()),
                "reclaim_only_rows": int((reclaim_supported & reset_score.lt(min_reset_score)).sum()),
                "min_reclaim_score": min_reclaim_score,
                "max_reclaim_rank_credit": max_reclaim_rank_credit,
                "avg_reclaim_score_supported": float(reclaim_score.loc[reclaim_supported].mean()) if reclaim_supported.any() else 0.0,
                "avg_reclaim_rank_credit": float(reclaim_rank_credit.loc[reclaim_supported].mean()) if reclaim_supported.any() else 0.0,
                "require_theme_breadth_not_deteriorating": require_theme_breadth_not_deteriorating,
                "theme_breadth_safe_rows": int(frame["pullback_reset_theme_breadth_safe"].sum()),
                "avg_theme_breadth_shortfall": float(frame["pullback_reset_theme_breadth_shortfall"].mean()),
                "avg_effective_score_rank_supported": float(effective_score_rank.loc[supported_union].mean())
                if supported_union.any()
                else 0.0,
                "avg_reset_selection_score_supported": float(reset_selection_score.loc[supported_union].mean())
                if supported_union.any()
                else 0.0,
                "avg_rs_acceleration": float(frame["pullback_reset_rs_acceleration"].mean()),
                "avg_theme_breadth_active_share": float(frame["pullback_reset_theme_breadth_active_share"].mean()),
                "avg_theme_breadth_baseline_share": float(frame["pullback_reset_theme_breadth_baseline_share"].mean()),
                "avg_theme_breadth_expansion": float(frame["pullback_reset_theme_breadth_expansion"].mean()),
            },
        )
    return include


def _apply_pullback_reset_same_theme_substitution_overlay(
    frame: pd.DataFrame,
    params: dict,
    ordinary_long_mask: pd.Series,
    reset_include_mask: pd.Series,
    event_allowed: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Swap in a guarded same-theme reset leader for the weakest admitted peer.

    Feature flag: ``pullback_reset_same_theme_substitution_overlay`` defaults to
    false.
    Reasonable ranges:
    - ``pullback_reset_same_theme_substitution_min_relative_strength_edge``: 2
      to 15 score points.
    - ``pullback_reset_same_theme_substitution_min_theme_score_edge``: 0 to 10
      score points.
    - ``pullback_reset_same_theme_substitution_min_reset_score``: 15 to 70.
    - ``pullback_reset_same_theme_substitution_min_reclaim_score``: 0 to 60.
    - ``pullback_reset_same_theme_substitution_min_theme_strength_delta_score``:
      0 to 40.
    - ``pullback_reset_same_theme_substitution_max_final_score_deficit``: 0 to
      8 score points.
    - ``pullback_reset_same_theme_substitution_max_score_rank_gap``: 0.02 to
      0.20.
    - ``pullback_reset_same_theme_substitution_max_event_risk_score_circuit_breaker``:
      10 to 35.
    - ``pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker``:
      10 to 40.
    - ``pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker``:
      0.02 to 0.10.
    - ``pullback_reset_same_theme_substitution_require_activation_support``:
      true/false.
    - ``pullback_reset_same_theme_substitution_require_theme_breadth_not_deteriorating``:
      true/false.
    - ``pullback_reset_same_theme_substitution_theme_breadth_lookback_days``:
      5 to 20.
    - ``pullback_reset_same_theme_substitution_theme_breadth_active_share_threshold``:
      0.20 to 0.70.
    - ``pullback_reset_same_theme_substitution_theme_breadth_shortfall_threshold``:
      0.02 to 0.20.
    - ``pullback_reset_same_theme_substitution_max_promotions_per_date``: 1 to
      3.

    Circuit breakers: the overlay is off by default; it only operates inside a
    known theme or industry peer bucket; it never increases the net long count;
    and the promoted candidate must improve on the weakest admitted peer's
    relative strength and theme score while also carrying no more event, gap,
    volatility, or theme-breadth deterioration risk than the explicit caps
    allow. When explicitly enabled, substitutions also require the candidate to
    clear the reset activation support circuit breaker before it can replace an
    incumbent.
    """

    false_mask = pd.Series(False, index=frame.index, dtype=bool)
    frame["pullback_reset_same_theme_substitution_candidate"] = False
    frame["pullback_reset_same_theme_substitution_promoted"] = False
    frame["pullback_reset_same_theme_substitution_demoted"] = False
    frame["pullback_reset_same_theme_substitution_support_score"] = 0.0
    frame["pullback_reset_same_theme_substitution_breadth_safe"] = True
    frame["pullback_reset_same_theme_substitution_breadth_shortfall"] = 0.0
    if not _bool_param(params, "pullback_reset_same_theme_substitution_overlay", False):
        return false_mask, false_mask

    theme_key = pd.Series(frame.get("primary_theme", "unknown"), index=frame.index).fillna("unknown").astype(str).str.strip()
    theme_key = theme_key.replace({"": "unknown", "nan": "unknown", "None": "unknown"})
    industry_key = pd.Series(frame.get("industry", "unknown"), index=frame.index).fillna("unknown").astype(str).str.strip()
    industry_key = industry_key.replace({"": "unknown", "nan": "unknown", "None": "unknown"})
    peer_key = np.where(
        theme_key.isin({"unknown", "unclassified"}),
        np.where(industry_key.eq("unknown"), "unknown", "industry:" + industry_key),
        "theme:" + theme_key,
    )
    peer_key = pd.Series(peer_key, index=frame.index, dtype="object")

    long_q = float(params.get("long_quantile", 0.2))
    score_rank = pd.to_numeric(frame.get("score_rank", 0.0), errors="coerce").fillna(0.0)
    cutoff = max(0.0, 1.0 - long_q)
    max_score_rank_gap = max(float(params.get("pullback_reset_same_theme_substitution_max_score_rank_gap", 0.08)), 0.0)
    min_rs_edge = float(params.get("pullback_reset_same_theme_substitution_min_relative_strength_edge", 4.0))
    min_theme_edge = float(params.get("pullback_reset_same_theme_substitution_min_theme_score_edge", 2.0))
    min_reset_score = float(params.get("pullback_reset_same_theme_substitution_min_reset_score", 25.0))
    min_reclaim_score = float(params.get("pullback_reset_same_theme_substitution_min_reclaim_score", 25.0))
    min_theme_strength_score = float(
        params.get("pullback_reset_same_theme_substitution_min_theme_strength_delta_score", 20.0)
    )
    max_final_score_deficit = max(float(params.get("pullback_reset_same_theme_substitution_max_final_score_deficit", 4.0)), 0.0)
    max_event_risk = float(
        params.get("pullback_reset_same_theme_substitution_max_event_risk_score_circuit_breaker", 20.0)
    )
    max_gap_risk = float(
        params.get("pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker", 25.0)
    )
    max_vol_20d = max(float(params.get("pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker", 0.045)), 0.0)
    max_promotions_per_date = max(int(params.get("pullback_reset_same_theme_substitution_max_promotions_per_date", 1)), 0)
    if max_promotions_per_date <= 0:
        return false_mask, false_mask

    require_theme_active = _bool_param(params, "pullback_reset_same_theme_substitution_require_theme_active", True)
    require_activation_support = _bool_param(
        params,
        "pullback_reset_same_theme_substitution_require_activation_support",
        False,
    )
    require_breadth_safety = _bool_param(
        params,
        "pullback_reset_same_theme_substitution_require_theme_breadth_not_deteriorating",
        False,
    )
    theme_active = pd.Series(frame.get("theme_active", True), index=frame.index).fillna(True).astype(bool)
    rs_score = pd.to_numeric(frame.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(frame.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    final_score = pd.to_numeric(frame.get("final_score", 50.0), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(frame.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(_column(frame, "overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    vol_20d = pd.to_numeric(frame.get("vol_20d", np.nan), errors="coerce").fillna(np.inf)
    reset_score = pd.to_numeric(frame.get("pullback_volume_contraction_score", 0.0), errors="coerce").fillna(0.0)
    reclaim_score = pd.to_numeric(frame.get("pullback_reclaim_score", 0.0), errors="coerce").fillna(0.0)
    activation_score = pd.to_numeric(frame.get("pullback_reset_activation_score", 0.0), errors="coerce").fillna(0.0)
    activation_supported = (
        frame.get("pullback_reset_activation_supported", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    theme_strength_score = pd.to_numeric(frame.get("theme_strength_delta_score", 0.0), errors="coerce").fillna(0.0)
    if require_breadth_safety:
        breadth_context = _attach_pullback_reset_theme_breadth_safety_context(
            frame,
            lookback_days=max(int(params.get("pullback_reset_same_theme_substitution_theme_breadth_lookback_days", 10)), 2),
            active_share_threshold=float(
                params.get("pullback_reset_same_theme_substitution_theme_breadth_active_share_threshold", 0.38)
            ),
            shortfall_threshold=max(
                float(params.get("pullback_reset_same_theme_substitution_theme_breadth_shortfall_threshold", 0.05)),
                0.0,
            ),
        )
        frame["pullback_reset_theme_breadth_active_share"] = breadth_context["pullback_reset_theme_breadth_active_share"]
        frame["pullback_reset_theme_breadth_baseline_share"] = breadth_context["pullback_reset_theme_breadth_baseline_share"]
        frame["pullback_reset_theme_breadth_shortfall"] = breadth_context["pullback_reset_theme_breadth_shortfall"]
        frame["pullback_reset_theme_breadth_safe"] = breadth_context["pullback_reset_theme_breadth_safe"]
        frame["pullback_reset_same_theme_substitution_breadth_safe"] = breadth_context["pullback_reset_theme_breadth_safe"]
        frame["pullback_reset_same_theme_substitution_breadth_shortfall"] = breadth_context["pullback_reset_theme_breadth_shortfall"]
    breadth_safe = frame["pullback_reset_same_theme_substitution_breadth_safe"].fillna(True).astype(bool)

    support_score = reset_score + reclaim_score * 0.35 + theme_strength_score * 0.20
    reset_support = reset_include_mask | reset_score.ge(min_reset_score) | reclaim_score.ge(min_reclaim_score)
    candidate_mask = reset_support & ~ordinary_long_mask & event_allowed & peer_key.ne("unknown")
    candidate_mask &= score_rank.ge(cutoff - max_score_rank_gap)
    candidate_mask &= event_risk.le(max_event_risk)
    candidate_mask &= gap_risk.le(max_gap_risk)
    candidate_mask &= vol_20d.le(max_vol_20d)
    candidate_mask &= theme_strength_score.ge(min_theme_strength_score)
    if require_theme_active:
        candidate_mask &= theme_active
    if require_activation_support:
        candidate_mask &= activation_supported
    if require_breadth_safety:
        candidate_mask &= breadth_safe
    frame.loc[candidate_mask, "pullback_reset_same_theme_substitution_candidate"] = True
    frame.loc[candidate_mask, "pullback_reset_same_theme_substitution_support_score"] = support_score.loc[candidate_mask]

    base_selected = ordinary_long_mask & event_allowed & peer_key.ne("unknown")
    proposals: list[dict[str, object]] = []
    grouped = frame.groupby([pd.to_datetime(frame["date"]).dt.normalize(), peer_key], sort=False)
    for (date, bucket), group in grouped:
        if bucket == "unknown":
            continue
        selected = group[base_selected.loc[group.index]]
        candidates = group[candidate_mask.loc[group.index]]
        if selected.empty or candidates.empty:
            continue
        weakest = selected.sort_values(
            ["relative_strength_score", "theme_score", "final_score", "score_rank"],
            ascending=[True, True, True, True],
        ).iloc[0]
        eligible_candidates = candidates[
            candidates["relative_strength_score"].ge(float(weakest.get("relative_strength_score", 50.0)) + min_rs_edge)
            & candidates["theme_score"].ge(float(weakest.get("theme_score", 50.0)) + min_theme_edge)
            & candidates["final_score"].ge(float(weakest.get("final_score", 50.0)) - max_final_score_deficit)
            & candidates["event_risk_score"].le(min(max_event_risk, float(weakest.get("event_risk_score", 0.0))))
            & pd.to_numeric(candidates.get("overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0).le(
                min(max_gap_risk, float(weakest.get("overnight_gap_risk_score", 0.0)))
            )
        ]
        if eligible_candidates.empty:
            continue
        chosen = eligible_candidates.sort_values(
            [
                "pullback_reset_same_theme_substitution_support_score",
                "relative_strength_score",
                "theme_score",
                "final_score",
            ],
            ascending=[False, False, False, False],
        ).iloc[0]
        proposals.append(
            {
                "date": pd.Timestamp(date),
                "candidate_idx": chosen.name,
                "victim_idx": weakest.name,
                "promotion_score": float(chosen.get("pullback_reset_same_theme_substitution_support_score", 0.0))
                + float(chosen.get("relative_strength_score", 50.0) - weakest.get("relative_strength_score", 50.0))
                + 0.5 * float(chosen.get("theme_score", 50.0) - weakest.get("theme_score", 50.0)),
                "peer_bucket": str(bucket),
            }
        )

    if not proposals:
        return false_mask, false_mask

    promote_mask = pd.Series(False, index=frame.index, dtype=bool)
    demote_mask = pd.Series(False, index=frame.index, dtype=bool)
    proposals_frame = pd.DataFrame(proposals).sort_values(["date", "promotion_score"], ascending=[True, False])
    for _, date_rows in proposals_frame.groupby("date", sort=False):
        candidate_limit = date_rows.head(max_promotions_per_date)
        for _, row in candidate_limit.iterrows():
            promote_mask.loc[int(row["candidate_idx"])] = True
            demote_mask.loc[int(row["victim_idx"])] = True

    frame.loc[promote_mask, "pullback_reset_same_theme_substitution_promoted"] = True
    frame.loc[demote_mask, "pullback_reset_same_theme_substitution_demoted"] = True
    if promote_mask.any():
        LOGGER.info(
            "momentum_pullback_reset_same_theme_substitution_applied",
            extra={
                "event": "momentum_pullback_reset_same_theme_substitution_applied",
                "rows": int(promote_mask.sum()),
                "demoted_rows": int(demote_mask.sum()),
                "dates": int(frame.loc[promote_mask, "date"].nunique()),
                "min_relative_strength_edge": min_rs_edge,
                "min_theme_score_edge": min_theme_edge,
                "min_reset_score": min_reset_score,
                "min_reclaim_score": min_reclaim_score,
                "min_theme_strength_delta_score": min_theme_strength_score,
                "max_final_score_deficit": max_final_score_deficit,
                "max_score_rank_gap": max_score_rank_gap,
                "max_event_risk_score": max_event_risk,
                "max_overnight_gap_risk_score": max_gap_risk,
                "max_vol_20d": max_vol_20d,
                "max_promotions_per_date": max_promotions_per_date,
                "require_theme_active": require_theme_active,
                "require_activation_support": require_activation_support,
                "activation_supported_rows": int(activation_supported.sum()),
                "avg_activation_score": float(activation_score.loc[promote_mask].mean()) if promote_mask.any() else 0.0,
                "require_theme_breadth_not_deteriorating": require_breadth_safety,
                "breadth_safe_rows": int(frame["pullback_reset_same_theme_substitution_breadth_safe"].sum()),
                "avg_breadth_shortfall": float(frame["pullback_reset_same_theme_substitution_breadth_shortfall"].mean()),
                "avg_support_score": float(frame.loc[promote_mask, "pullback_reset_same_theme_substitution_support_score"].mean()),
            },
        )
    return promote_mask, demote_mask


def _attach_pullback_reset_rs_acceleration_context(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Attach point-in-time relative-strength acceleration for reset entries."""

    out = frame.copy()
    out["pullback_reset_rs_score"] = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    out["pullback_reset_rs_baseline"] = np.nan
    out["pullback_reset_rs_acceleration"] = 0.0
    out["pullback_reset_rs_accelerating"] = False
    lookback_days = max(int(params.get("pullback_reset_inclusion_rs_acceleration_lookback_days", 10)), 2)
    min_current_rs = float(params.get("pullback_reset_inclusion_min_current_rs_score", 78.0))
    min_acceleration = float(params.get("pullback_reset_inclusion_min_rs_acceleration", 5.0))
    ordered = out.sort_values(["symbol", "date"]).copy()
    min_periods = max(2, lookback_days // 2)
    ordered["pullback_reset_rs_baseline"] = ordered.groupby("symbol")["pullback_reset_rs_score"].transform(
        lambda s: s.shift(1).rolling(lookback_days, min_periods=min_periods).mean()
    )
    ordered["pullback_reset_rs_acceleration"] = (
        ordered["pullback_reset_rs_score"] - ordered["pullback_reset_rs_baseline"]
    ).fillna(0.0)
    ordered["pullback_reset_rs_accelerating"] = (
        ordered["pullback_reset_rs_baseline"].notna()
        & ordered["pullback_reset_rs_score"].ge(min_current_rs)
        & ordered["pullback_reset_rs_acceleration"].ge(min_acceleration)
    )
    return ordered.sort_index()


def _apply_reentry_discipline_overlay(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Penalize late, weakly confirmed long chases that are not true pullbacks.

    Feature flag: ``long_reentry_discipline_overlay`` defaults to false.
    Reasonable ranges:
    - ``reentry_discipline_near_high_threshold``: 0.01 to 0.06.
    - ``reentry_discipline_min_above_ma20_pct``: 0.02 to 0.12.
    - ``reentry_discipline_max_volume_expansion``: 0.70 to 1.10.
    - ``reentry_discipline_min_adx_circuit_breaker``: 20 to 40.
    - ``reentry_discipline_min_pullback_volume_reset_score_exemption``: 10 to 60.
    - ``reentry_discipline_min_relative_strength_score_exemption``: 65 to 95.
    - ``reentry_discipline_min_theme_score_exemption``: 55 to 90.
    - ``reentry_discipline_require_theme_deterioration``: true/false.
    - ``reentry_discipline_theme_score_deterioration_threshold``: 50 to 75.
    - ``reentry_discipline_require_theme_breadth_deterioration``: true/false.
    - ``reentry_discipline_theme_breadth_lookback_days``: 5 to 20.
    - ``reentry_discipline_theme_breadth_active_share_threshold``: 0.20 to 0.70.
    - ``reentry_discipline_theme_breadth_shortfall_threshold``: 0.02 to 0.20.
    Circuit breakers: true pullback leaders are exempt, strong-trend names above
    the ADX threshold are exempt, qualifying volume-reset pullbacks are exempt,
    theme-healthy names can be exempted, insufficient breadth history disables
    the breadth gate, and ``reentry_discipline_max_score_penalty`` caps the
    single-row score deduction.
    """

    out = frame.copy()
    out["reentry_discipline_score"] = 0.0
    out["reentry_discipline_penalty"] = 0.0
    if not _bool_param(params, "long_reentry_discipline_overlay", False):
        return out

    max_penalty = float(np.clip(params.get("reentry_discipline_max_score_penalty", 4.0), 0.0, 10.0))
    if max_penalty <= 0.0:
        return out

    out = _attach_reentry_theme_breadth_context(out, params)
    adj_close = pd.to_numeric(out["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(out, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    ma50 = pd.to_numeric(_column(out, "ma_50", np.nan), errors="coerce").replace(0, np.nan)
    distance_high = pd.to_numeric(out.get("mom_distance_high", out.get("distance_to_prior_high_252")), errors="coerce").fillna(0.0)
    above_ma20 = adj_close / ma20 - 1.0
    volume_expansion = pd.to_numeric(out.get("mom_volume_expansion", out.get("volume_expansion", 1.0)), errors="coerce")
    volume_expansion = volume_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    trend_adx = pd.to_numeric(_column(out, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    pullback_score = pd.to_numeric(out.get("pullback_entry_score", 0.0), errors="coerce").fillna(0.0)
    pullback_reset_score = pd.to_numeric(out.get("pullback_volume_contraction_score", 0.0), errors="coerce").fillna(0.0)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    if "theme_active" in out:
        theme_active = out["theme_active"].fillna(False).astype(bool)
    else:
        theme_active = pd.Series(True, index=out.index)

    near_high_threshold = max(float(params.get("reentry_discipline_near_high_threshold", 0.03)), 0.0)
    min_above_ma20 = float(params.get("reentry_discipline_min_above_ma20_pct", 0.04))
    max_volume_expansion = float(params.get("reentry_discipline_max_volume_expansion", 0.95))
    min_adx_circuit_breaker = float(params.get("reentry_discipline_min_adx_circuit_breaker", 25.0))
    max_extension = max(float(params.get("reentry_discipline_max_extension_pct", 0.12)), min_above_ma20 + 1e-6)
    min_pullback_reset_score_exemption = float(params.get("reentry_discipline_min_pullback_volume_reset_score_exemption", 25.0))
    min_rs_exemption = float(params.get("reentry_discipline_min_relative_strength_score_exemption", 78.0))
    min_theme_exemption = float(params.get("reentry_discipline_min_theme_score_exemption", 70.0))
    require_theme_deterioration = _bool_param(params, "reentry_discipline_require_theme_deterioration", False)
    theme_deterioration_threshold = float(params.get("reentry_discipline_theme_score_deterioration_threshold", 64.0))
    penalize_theme_inactive = _bool_param(params, "reentry_discipline_penalize_theme_inactive", True)
    require_theme_breadth_deterioration = _bool_param(params, "reentry_discipline_require_theme_breadth_deterioration", False)
    theme_breadth_deteriorating = out.get(
        "reentry_theme_breadth_deteriorating",
        pd.Series(False, index=out.index),
    ).fillna(False).astype(bool)

    trend_intact = (adj_close > ma50).fillna(False)
    near_high = distance_high.ge(-near_high_threshold) if near_high_threshold > 0 else pd.Series(False, index=out.index)
    extended_from_ma20 = above_ma20.ge(min_above_ma20)
    late_entry = near_high | extended_from_ma20
    weak_confirmation = volume_expansion.le(max_volume_expansion) & trend_adx.lt(min_adx_circuit_breaker)
    reset_exempt = (
        pullback_reset_score.ge(min_pullback_reset_score_exemption)
        & rs_score.ge(min_rs_exemption)
        & theme_score.ge(min_theme_exemption)
    )
    theme_deteriorating = theme_score.lt(theme_deterioration_threshold)
    if penalize_theme_inactive:
        theme_deteriorating |= ~theme_active
    eligible = trend_intact & mom_return.gt(0.0) & late_entry & weak_confirmation & pullback_score.le(0.0) & ~reset_exempt
    if require_theme_deterioration:
        eligible &= theme_deteriorating
    if require_theme_breadth_deterioration:
        eligible &= theme_breadth_deteriorating

    high_component = pd.Series(0.0, index=out.index, dtype=float)
    if near_high_threshold > 0:
        high_component = (1.0 - ((-distance_high).clip(lower=0.0) / max(near_high_threshold, 1e-9))).clip(0.0, 1.0)
    extension_component = ((above_ma20 - min_above_ma20) / max(max_extension - min_above_ma20, 1e-9)).clip(0.0, 1.0)
    volume_component = ((max_volume_expansion - volume_expansion) / max(max_volume_expansion, 1e-9)).clip(0.0, 1.0)
    adx_component = ((min_adx_circuit_breaker - trend_adx) / max(min_adx_circuit_breaker, 1e-9)).clip(0.0, 1.0)
    score = (
        high_component * 35.0
        + extension_component * 35.0
        + volume_component * 15.0
        + adx_component * 15.0
    ).where(eligible, 0.0)
    penalty = (score / 100.0 * max_penalty).clip(lower=0.0, upper=max_penalty)

    out["reentry_discipline_score"] = score.fillna(0.0)
    out["reentry_discipline_penalty"] = penalty.fillna(0.0)
    out["final_score"] = (pd.to_numeric(out["final_score"], errors="coerce").fillna(50.0) - out["reentry_discipline_penalty"]).clip(0.0, 100.0)
    if eligible.any() or reset_exempt.any():
        LOGGER.info(
            "momentum_reentry_discipline_overlay_applied",
            extra={
                "event": "momentum_reentry_discipline_overlay_applied",
                "rows": int(eligible.sum()),
                "exempt_reset_rows": int(reset_exempt.sum()),
                "avg_penalty": float(out.loc[eligible, "reentry_discipline_penalty"].mean()),
                "max_penalty": max_penalty,
                "near_high_threshold": near_high_threshold,
                "min_above_ma20": min_above_ma20,
                "max_volume_expansion": max_volume_expansion,
                "min_adx_circuit_breaker": min_adx_circuit_breaker,
                "min_pullback_reset_score_exemption": min_pullback_reset_score_exemption,
                "min_relative_strength_score_exemption": min_rs_exemption,
                "min_theme_score_exemption": min_theme_exemption,
                "require_theme_deterioration": require_theme_deterioration,
                "theme_deterioration_threshold": theme_deterioration_threshold,
                "theme_deteriorating_rows": int(theme_deteriorating.sum()),
                "require_theme_breadth_deterioration": require_theme_breadth_deterioration,
                "theme_breadth_deteriorating_rows": int(theme_breadth_deteriorating.sum()),
                "avg_theme_breadth_active_share": float(out["reentry_theme_breadth_active_share"].mean()),
                "avg_theme_breadth_shortfall": float(out["reentry_theme_breadth_shortfall"].mean()),
            },
        )
    return out


def _attach_reentry_theme_breadth_context(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Attach date-level theme breadth diagnostics for reentry discipline.

    The breadth gate uses same-date theme activity and only prior-date rolling
    breadth history. If there is not enough history, the gate stays disabled.
    """

    out = frame.copy()
    out["reentry_theme_breadth_active_share"] = 1.0
    out["reentry_theme_breadth_baseline_share"] = 1.0
    out["reentry_theme_breadth_shortfall"] = 0.0
    out["reentry_theme_breadth_deteriorating"] = False
    if not _bool_param(params, "reentry_discipline_require_theme_breadth_deterioration", False):
        return out

    lookback_days = max(int(params.get("reentry_discipline_theme_breadth_lookback_days", 10)), 2)
    active_share_threshold = float(params.get("reentry_discipline_theme_breadth_active_share_threshold", 0.38))
    shortfall_threshold = max(float(params.get("reentry_discipline_theme_breadth_shortfall_threshold", 0.05)), 0.0)
    breadth = _compute_theme_breadth_frame(out, lookback_days).rename(
        columns={
            "theme_breadth_active_share": "reentry_theme_breadth_active_share",
            "theme_breadth_baseline_share": "reentry_theme_breadth_baseline_share",
        }
    )
    breadth["reentry_theme_breadth_shortfall"] = (
        breadth["reentry_theme_breadth_baseline_share"] - breadth["reentry_theme_breadth_active_share"]
    ).clip(lower=0.0)
    breadth["reentry_theme_breadth_deteriorating"] = (
        breadth["reentry_theme_breadth_baseline_share"].notna()
        & breadth["reentry_theme_breadth_active_share"].le(active_share_threshold)
        & breadth["reentry_theme_breadth_shortfall"].ge(shortfall_threshold)
    )
    out = out.merge(breadth, on="date", how="left", suffixes=("", "_ctx"))
    for column in (
        "reentry_theme_breadth_active_share",
        "reentry_theme_breadth_baseline_share",
        "reentry_theme_breadth_shortfall",
        "reentry_theme_breadth_deteriorating",
    ):
        ctx = f"{column}_ctx"
        if ctx in out:
            out[column] = out[ctx].fillna(out[column]) if column in out else out[ctx]
            out = out.drop(columns=[ctx])
    out["reentry_theme_breadth_active_share"] = pd.to_numeric(
        out["reentry_theme_breadth_active_share"], errors="coerce"
    ).fillna(1.0)
    out["reentry_theme_breadth_baseline_share"] = pd.to_numeric(
        out["reentry_theme_breadth_baseline_share"], errors="coerce"
    ).fillna(1.0)
    out["reentry_theme_breadth_shortfall"] = pd.to_numeric(
        out["reentry_theme_breadth_shortfall"], errors="coerce"
    ).fillna(0.0)
    out["reentry_theme_breadth_deteriorating"] = out["reentry_theme_breadth_deteriorating"].fillna(False).astype(bool)
    return out


def _attach_pullback_reset_theme_breadth_context(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Attach date-level theme breadth expansion diagnostics for reset entries.

    This gate uses only same-date theme activity and prior-date rolling
    breadth history. If there is not enough history, the gate stays disabled.
    """

    out = frame.copy()
    out["pullback_reset_theme_breadth_active_share"] = 1.0
    out["pullback_reset_theme_breadth_baseline_share"] = 1.0
    out["pullback_reset_theme_breadth_expansion"] = 0.0
    out["pullback_reset_theme_breadth_expanding"] = False
    if not _bool_param(params, "pullback_reset_inclusion_require_theme_breadth_expansion", False):
        return out

    lookback_days = max(int(params.get("pullback_reset_inclusion_theme_breadth_lookback_days", 10)), 2)
    min_active_share = float(params.get("pullback_reset_inclusion_theme_breadth_min_active_share", 0.45))
    expansion_threshold = max(float(params.get("pullback_reset_inclusion_theme_breadth_expansion_threshold", 0.03)), 0.0)
    breadth = _compute_theme_breadth_frame(out, lookback_days).rename(
        columns={
            "theme_breadth_active_share": "pullback_reset_theme_breadth_active_share",
            "theme_breadth_baseline_share": "pullback_reset_theme_breadth_baseline_share",
        }
    )
    breadth["pullback_reset_theme_breadth_expansion"] = (
        breadth["pullback_reset_theme_breadth_active_share"] - breadth["pullback_reset_theme_breadth_baseline_share"]
    ).clip(lower=0.0)
    breadth["pullback_reset_theme_breadth_expanding"] = (
        breadth["pullback_reset_theme_breadth_baseline_share"].notna()
        & breadth["pullback_reset_theme_breadth_active_share"].ge(min_active_share)
        & breadth["pullback_reset_theme_breadth_expansion"].ge(expansion_threshold)
    )
    out = out.merge(breadth, on="date", how="left", suffixes=("", "_ctx"))
    for column in (
        "pullback_reset_theme_breadth_active_share",
        "pullback_reset_theme_breadth_baseline_share",
        "pullback_reset_theme_breadth_expansion",
        "pullback_reset_theme_breadth_expanding",
    ):
        ctx = f"{column}_ctx"
        if ctx in out:
            out[column] = out[ctx].fillna(out[column]) if column in out else out[ctx]
            out = out.drop(columns=[ctx])
    out["pullback_reset_theme_breadth_active_share"] = pd.to_numeric(
        out["pullback_reset_theme_breadth_active_share"], errors="coerce"
    ).fillna(1.0)
    out["pullback_reset_theme_breadth_baseline_share"] = pd.to_numeric(
        out["pullback_reset_theme_breadth_baseline_share"], errors="coerce"
    ).fillna(1.0)
    out["pullback_reset_theme_breadth_expansion"] = pd.to_numeric(
        out["pullback_reset_theme_breadth_expansion"], errors="coerce"
    ).fillna(0.0)
    out["pullback_reset_theme_breadth_expanding"] = out["pullback_reset_theme_breadth_expanding"].fillna(False).astype(bool)
    return out


def _attach_pullback_reset_theme_breadth_safety_context(
    frame: pd.DataFrame,
    *,
    lookback_days: int,
    active_share_threshold: float,
    shortfall_threshold: float,
) -> pd.DataFrame:
    """Attach theme breadth deterioration diagnostics for reset-entry guards."""

    out = frame.copy()
    out["pullback_reset_theme_breadth_active_share"] = 1.0
    out["pullback_reset_theme_breadth_baseline_share"] = 1.0
    out["pullback_reset_theme_breadth_shortfall"] = 0.0
    out["pullback_reset_theme_breadth_safe"] = True
    breadth = _compute_theme_breadth_frame(out, lookback_days).rename(
        columns={
            "theme_breadth_active_share": "pullback_reset_theme_breadth_active_share",
            "theme_breadth_baseline_share": "pullback_reset_theme_breadth_baseline_share",
        }
    )
    breadth["pullback_reset_theme_breadth_shortfall"] = (
        breadth["pullback_reset_theme_breadth_baseline_share"] - breadth["pullback_reset_theme_breadth_active_share"]
    ).clip(lower=0.0)
    breadth_deteriorating = (
        breadth["pullback_reset_theme_breadth_baseline_share"].notna()
        & breadth["pullback_reset_theme_breadth_active_share"].le(active_share_threshold)
        & breadth["pullback_reset_theme_breadth_shortfall"].ge(shortfall_threshold)
    )
    breadth["pullback_reset_theme_breadth_safe"] = breadth["pullback_reset_theme_breadth_baseline_share"].isna() | ~breadth_deteriorating
    out = out.merge(breadth, on="date", how="left", suffixes=("", "_ctx"))
    for column in (
        "pullback_reset_theme_breadth_active_share",
        "pullback_reset_theme_breadth_baseline_share",
        "pullback_reset_theme_breadth_shortfall",
        "pullback_reset_theme_breadth_safe",
    ):
        ctx = f"{column}_ctx"
        if ctx in out:
            out[column] = out[ctx].fillna(out[column]) if column in out else out[ctx]
            out = out.drop(columns=[ctx])
    out["pullback_reset_theme_breadth_active_share"] = pd.to_numeric(
        out["pullback_reset_theme_breadth_active_share"], errors="coerce"
    ).fillna(1.0)
    out["pullback_reset_theme_breadth_baseline_share"] = pd.to_numeric(
        out["pullback_reset_theme_breadth_baseline_share"], errors="coerce"
    ).fillna(1.0)
    out["pullback_reset_theme_breadth_shortfall"] = pd.to_numeric(
        out["pullback_reset_theme_breadth_shortfall"], errors="coerce"
    ).fillna(0.0)
    out["pullback_reset_theme_breadth_safe"] = out["pullback_reset_theme_breadth_safe"].fillna(True).astype(bool)
    return out


def _compute_theme_breadth_frame(frame: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Summarize date-level theme activity using only same-date rows."""

    theme_score = pd.to_numeric(frame.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    rs_score = pd.to_numeric(frame.get("relative_strength_score", 50.0), errors="coerce").fillna(50.0)
    if "theme_active" in frame:
        theme_active = frame["theme_active"].fillna(False).astype(bool)
    else:
        theme_active = theme_score.ge(60.0) & rs_score.ge(55.0)
    breadth = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(frame["date"]).dt.normalize(),
                "theme_active_flag": theme_active.astype(float),
            }
        )
        .groupby("date", as_index=False)["theme_active_flag"]
        .mean()
        .sort_values("date")
        .rename(columns={"theme_active_flag": "theme_breadth_active_share"})
    )
    min_periods = min(lookback_days, max(3, lookback_days // 2))
    breadth["theme_breadth_baseline_share"] = (
        breadth["theme_breadth_active_share"].shift(1).rolling(lookback_days, min_periods=min_periods).mean()
    )
    return breadth


def _short_quality_gate(frame: pd.DataFrame, params: dict) -> pd.Series:
    """Optional short-side gates so shorts require genuine weakness."""

    conditions = pd.DataFrame(index=frame.index)
    conditions["negative_momentum"] = frame["mom_return"].fillna(0.0) < 0.0
    conditions["negative_rs"] = frame.get("rs_63d", pd.Series(0.0, index=frame.index)).fillna(0.0) < 0.0
    conditions["below_ma50"] = frame["adj_close"] < frame.get("ma_50", -np.inf)
    conditions["below_ma200"] = frame["adj_close"] < frame.get("ma_200", -np.inf)
    conditions["weak_technical"] = frame["technical_score"].fillna(50.0) <= float(params.get("short_max_technical_score", 45.0))
    conditions["weak_fundamental"] = frame["fundamental_score"].fillna(50.0) <= float(params.get("short_max_fundamental_score", 60.0))
    weakness_count = conditions.sum(axis=1)
    frame["short_weakness_count"] = weakness_count
    gate = weakness_count >= int(params.get("short_min_weakness_conditions", 0))
    if _bool_param(params, "short_only_when_benchmark_risk_off", False):
        gate &= ~frame.get("benchmark_risk_on", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    gate &= frame["final_score"].fillna(100.0) <= float(params.get("short_max_final_score", 100.0))
    gate &= _short_rebound_avoidance_gate(frame, params)
    return gate


def _short_rebound_avoidance_gate(frame: pd.DataFrame, params: dict) -> pd.Series:
    """Block exhausted short entries that are prone to reflexive rebound.

    Feature flag: ``short_rebound_avoidance_overlay`` defaults to false.
    Reasonable ranges:
    - ``short_rebound_avoidance_min_below_ma20_pct``: 0.04 to 0.15.
    - ``short_rebound_avoidance_min_distance_from_high``: 0.12 to 0.35.
    - ``short_rebound_avoidance_min_volume_expansion``: 1.0 to 2.0.
    - ``short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker``: -0.20 to -0.02.
    - ``short_rebound_avoidance_min_adx_circuit_breaker``: 25 to 45.
    Circuit breakers: allow the short when the benchmark is already in a deep
    downtrend or the individual name remains in a strong-trend breakdown.
    """

    allow = pd.Series(True, index=frame.index)
    frame["short_rebound_avoidance_score"] = 0.0
    frame["short_rebound_avoidance_blocked"] = False
    if not _bool_param(params, "short_rebound_avoidance_overlay", False):
        return allow

    adj_close = pd.to_numeric(frame["adj_close"], errors="coerce")
    ma20 = pd.to_numeric(_column(frame, "ma_20", np.nan), errors="coerce").replace(0, np.nan)
    distance_high = pd.to_numeric(frame.get("mom_distance_high", frame.get("distance_to_prior_high_252")), errors="coerce").fillna(0.0)
    volume_expansion = pd.to_numeric(frame.get("mom_volume_expansion", frame.get("volume_expansion", 1.0)), errors="coerce")
    volume_expansion = volume_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    trend_adx = pd.to_numeric(_column(frame, "trend_adx", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_63d = pd.to_numeric(frame.get("benchmark_ret_63d", 0.0), errors="coerce").fillna(0.0)

    min_below_ma20 = max(float(params.get("short_rebound_avoidance_min_below_ma20_pct", 0.08)), 0.0)
    min_distance_from_high = max(float(params.get("short_rebound_avoidance_min_distance_from_high", 0.20)), 0.0)
    min_volume_expansion = max(float(params.get("short_rebound_avoidance_min_volume_expansion", 1.10)), 0.0)
    min_benchmark_ret63d_circuit_breaker = float(params.get("short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker", -0.12))
    min_adx_circuit_breaker = float(params.get("short_rebound_avoidance_min_adx_circuit_breaker", 32.0))

    below_ma20_pct = (1.0 - (adj_close / ma20)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    drawdown_from_high = (-distance_high).clip(lower=0.0)
    oversold = below_ma20_pct.ge(min_below_ma20)
    washed_out = drawdown_from_high.ge(min_distance_from_high)
    crowded_breakdown = volume_expansion.ge(min_volume_expansion)
    exhausted = oversold & washed_out & crowded_breakdown
    circuit_breaker = trend_adx.ge(min_adx_circuit_breaker) | benchmark_ret_63d.le(min_benchmark_ret63d_circuit_breaker)

    below_component = (below_ma20_pct / max(min_below_ma20, 1e-9)).clip(0.0, 2.0) / 2.0
    drawdown_component = (drawdown_from_high / max(min_distance_from_high, 1e-9)).clip(0.0, 2.0) / 2.0
    volume_component = (volume_expansion / max(min_volume_expansion, 1e-9)).clip(0.0, 2.0) / 2.0
    score = (below_component * 40.0 + drawdown_component * 40.0 + volume_component * 20.0).where(exhausted, 0.0)
    blocked = exhausted & ~circuit_breaker

    frame["short_rebound_avoidance_score"] = score.fillna(0.0)
    frame["short_rebound_avoidance_blocked"] = blocked.fillna(False)
    if blocked.any():
        LOGGER.info(
            "momentum_short_rebound_avoidance_overlay_applied",
            extra={
                "event": "momentum_short_rebound_avoidance_overlay_applied",
                "rows": int(blocked.sum()),
                "avg_score": float(frame.loc[blocked, "short_rebound_avoidance_score"].mean()),
                "min_below_ma20": min_below_ma20,
                "min_distance_from_high": min_distance_from_high,
                "min_volume_expansion": min_volume_expansion,
                "min_benchmark_ret63d_circuit_breaker": min_benchmark_ret63d_circuit_breaker,
                "min_adx_circuit_breaker": min_adx_circuit_breaker,
            },
        )
    return ~blocked


def _entry_reason(row: pd.Series) -> str:
    evidence = (
        f"momentum={_fmt_pct(row.get('mom_return'))}, "
        f"risk_adj={_fmt_num(row.get('mom_risk_adjusted'))}, "
        f"prior_high_gap={_fmt_pct(row.get('mom_distance_high'))}, "
        f"volume={_fmt_num(row.get('mom_volume_expansion'))}x, "
        f"pullback_volume_reset={_fmt_num(row.get('pullback_volume_contraction_score'))}, "
        f"pullback_short_term_reset={_fmt_num(row.get('pullback_short_term_reset_score'))}, "
        f"pullback_ret5={_fmt_pct(row.get('pullback_short_term_reset_ret_5d'))}, "
        f"pullback_ret10={_fmt_pct(row.get('pullback_short_term_reset_ret_10d'))}, "
        f"short_term_volume_tilt={_fmt_num(row.get('short_term_volume_tilt_score'))}, "
        f"short_term_volume_ret5={_fmt_pct(row.get('short_term_volume_tilt_ret_5d'))}, "
        f"short_term_volume={_fmt_num(row.get('short_term_volume_tilt_volume_expansion'))}, "
        f"short_term_volume_block={int(bool(row.get('short_term_volume_tilt_blocked', False)))}, "
        f"pullback_reclaim={_fmt_num(row.get('pullback_reclaim_score'))}, "
        f"reclaim_recent_below={_fmt_pct(row.get('pullback_reclaim_recent_below_ma20_pct'))}, "
        f"reclaim_theme_gap_ok={int(bool(row.get('pullback_reclaim_same_theme_gap_confirmation_ok', False)))}, "
        f"reclaim_theme_peers={int(row.get('pullback_reclaim_same_theme_peer_count', 0) or 0)}, "
        f"reclaim_theme_share={_fmt_num(row.get('pullback_reclaim_same_theme_active_share'))}, "
        f"reclaim_theme_gap={_fmt_num(row.get('pullback_reclaim_same_theme_avg_gap_risk'))}, "
        f"reclaim_block={int(bool(row.get('pullback_reclaim_blocked', False)))}, "
        f"reset_breadth_expansion={_fmt_num(row.get('pullback_reset_theme_breadth_expansion'))}, "
        f"reentry_penalty={_fmt_num(row.get('reentry_discipline_penalty'))}, "
        f"theme_leader_tilt={_fmt_num(row.get('theme_leader_tilt_score'))}, "
        f"theme_leader_rank={_fmt_num(row.get('theme_leader_tilt_theme_rank_pct'))}, "
        f"theme_peers={int(row.get('theme_leader_tilt_theme_peer_count', 0) or 0)}, "
        f"compound_leader_credit={_fmt_num(row.get('compound_leader_score_credit_score'))}, "
        f"compound_126_rank={_fmt_num(row.get('compound_leader_126d_voladj_rank'))}, "
        f"compound_252_rank={_fmt_num(row.get('compound_leader_252d_voladj_rank'))}, "
        f"compound_ret10={_fmt_pct(row.get('compound_leader_ret_10d'))}, "
        f"compound_block={int(bool(row.get('compound_leader_score_credit_blocked', False)))}, "
        f"short_rebound={_fmt_num(row.get('short_rebound_avoidance_score'))}, "
        f"short_block={int(bool(row.get('short_rebound_avoidance_blocked', False)))}, "
        f"rs={_fmt_num(row.get('relative_strength_score'))}, "
        f"theme={_fmt_num(row.get('theme_score'))}, "
        f"gate={row.get('momentum_gate_reason', 'n/a')}, "
        f"fundamental={row.get('fundamental_score', 50):.1f}, "
        f"event={row.get('event_risk_score', 0):.1f}"
    )
    if row["signal"] > 0:
        return f"Long momentum: top-ranked delayed return with positive tape evidence; {evidence}."
    if row["signal"] < 0:
        return f"Short momentum: bottom-ranked delayed return and weak final score; {evidence}."
    return f"No momentum position: rank or risk gate did not clear; {evidence}."


def _gate_reason(row: pd.Series, params: dict) -> str:
    """Human-readable explanation of the optional quality gates."""

    long_gates = []
    if "long_min_final_score" in params:
        long_gates.append(f"final>={float(params['long_min_final_score']):.0f}")
    if "long_min_relative_strength_score" in params:
        long_gates.append(f"RS>={float(params['long_min_relative_strength_score']):.0f}")
    if "long_min_theme_score" in params:
        long_gates.append(f"theme>={float(params['long_min_theme_score']):.0f}")
    if _bool_param(params, "long_require_price_above_ma50", False):
        long_gates.append("price>MA50")
    if _bool_param(params, "long_require_price_above_ma200", False):
        long_gates.append("price>MA200")
    short_rules = []
    if params.get("short_min_weakness_conditions", 0):
        short_rules.append(f"weakness_count>={int(params['short_min_weakness_conditions'])}")
    if _bool_param(params, "short_only_when_benchmark_risk_off", False):
        short_rules.append("benchmark risk-off only")
    side = "long" if row.get("signal", 0.0) > 0 else "short" if row.get("signal", 0.0) < 0 else "flat"
    return f"{side}; long gates: {', '.join(long_gates) or 'rank only'}; short gates: {', '.join(short_rules) or 'rank only'}"


def _bool_param(params: dict, key: str, default: bool) -> bool:
    value = params.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _column(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index, dtype=float)


def _fmt_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.1%}"


def _fmt_num(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.2f}"
