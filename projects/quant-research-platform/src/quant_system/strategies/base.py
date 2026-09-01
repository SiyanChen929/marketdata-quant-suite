"""Strategy interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategyContext:
    """Data passed into strategy signal generation."""

    benchmark: str = "SPY"
    fundamentals: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    theme_scores: pd.DataFrame | None = None


class Strategy(ABC):
    """Base class for signal generators."""

    name: str

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame, context: StrategyContext) -> pd.DataFrame:
        """Return date/symbol signal rows with scores and explanations."""


def attach_context_scores(frame: pd.DataFrame, context: StrategyContext) -> pd.DataFrame:
    """Attach fundamental, event, and theme scores to a signal frame."""

    out = frame.copy()
    out["theme_score"] = out.get("theme_score", 50.0)
    out["fundamental_score"] = out.get("fundamental_score", 50.0)
    out["event_risk_score"] = out.get("event_risk_score", 0.0)
    if context.theme_scores is not None and not context.theme_scores.empty:
        cols = [
            col
            for col in ["date", "symbol", "theme_score", "theme_active", "theme_reason", "theme_memberships", "primary_theme"]
            if col in context.theme_scores.columns
        ]
        if {"date", "symbol", "theme_score"}.issubset(cols):
            out = out.merge(context.theme_scores[cols], on=["date", "symbol"], how="left", suffixes=("", "_ctx"))
            out["theme_score"] = out["theme_score_ctx"].fillna(out["theme_score"])
            out = out.drop(columns=["theme_score_ctx"])
            for column in ("theme_active", "theme_reason", "theme_memberships", "primary_theme"):
                ctx = f"{column}_ctx"
                if ctx in out:
                    if column in out:
                        out[column] = out[ctx].fillna(out[column])
                    else:
                        out[column] = out[ctx]
                    out = out.drop(columns=[ctx])
    if context.fundamentals is not None and not context.fundamentals.empty:
        fundamental_cols = [
            col
            for col in [
                "date",
                "symbol",
                "sector",
                "industry",
                "fundamental_known_date",
                "fundamental_data_age_days",
                "fundamental_score",
                "fundamental_factor_coverage",
                "fundamental_missing_group_count",
                "fundamental_data_quality",
                "moat_score",
                "growth_score",
                "quality_score",
                "balance_sheet_score",
                "valuation_score",
                "revision_score",
            ]
            if col in context.fundamentals.columns
        ]
        out = out.merge(context.fundamentals[fundamental_cols], on=["date", "symbol"], how="left", suffixes=("", "_ctx"))
        out["fundamental_score"] = out["fundamental_score_ctx"].fillna(out["fundamental_score"])
        out = out.drop(columns=["fundamental_score_ctx"])
        for column in (
            "sector",
            "industry",
            "fundamental_known_date",
            "fundamental_data_age_days",
            "fundamental_factor_coverage",
            "fundamental_missing_group_count",
            "fundamental_data_quality",
            "moat_score",
            "growth_score",
            "quality_score",
            "balance_sheet_score",
            "valuation_score",
            "revision_score",
        ):
            ctx = f"{column}_ctx"
            if ctx in out.columns:
                out[column] = out[ctx].fillna(out[column]) if column in out.columns else out[ctx]
                out = out.drop(columns=[ctx])
    if context.events is not None and not context.events.empty:
        event_cols = [
            col
            for col in [
                "event_risk_score",
                "days_to_earnings",
                "next_earnings_date",
                "event_known_date",
                "expected_move",
                "report_time",
                "surprise_eps_pct",
            ]
            if col in context.events.columns
        ]
        out = out.merge(context.events[["date", "symbol", *event_cols]], on=["date", "symbol"], how="left", suffixes=("", "_ctx"))
        if "event_risk_score_ctx" in out:
            out["event_risk_score"] = out["event_risk_score_ctx"].fillna(out["event_risk_score"])
            out = out.drop(columns=["event_risk_score_ctx"])
        for column in ("days_to_earnings", "next_earnings_date", "event_known_date", "expected_move", "report_time", "surprise_eps_pct"):
            ctx = f"{column}_ctx"
            if ctx in out.columns:
                out[column] = out[ctx].fillna(out[column]) if column in out.columns else out[ctx]
                out = out.drop(columns=[ctx])
    return out


def final_score(frame: pd.DataFrame) -> pd.Series:
    """Default ensemble-compatible score."""

    return (
        0.30 * frame["technical_score"].fillna(50.0)
        + 0.25 * frame["relative_strength_score"].fillna(50.0)
        + 0.20 * frame["theme_score"].fillna(50.0)
        + 0.15 * frame["fundamental_score"].fillna(50.0)
        - 0.10 * frame["event_risk_score"].fillna(0.0)
    )


def output_columns(frame: pd.DataFrame) -> list[str]:
    """Common signal output columns that exist in frame."""

    columns = [
        "date",
        "symbol",
        "signal",
        "technical_score",
        "relative_strength_score",
        "theme_score",
        "theme_active",
        "theme_reason",
        "limited_history_flag",
        "filter_reason",
        "candidate_reason",
        "fundamental_score",
        "sector",
        "industry",
        "fundamental_known_date",
        "fundamental_data_age_days",
        "fundamental_factor_coverage",
        "fundamental_missing_group_count",
        "fundamental_data_quality",
        "moat_score",
        "growth_score",
        "quality_score",
        "balance_sheet_score",
        "valuation_score",
        "revision_score",
        "event_risk_score",
        "final_score",
        "score_decomposition",
        "reason_for_entry",
        "stop_loss_atr",
        "trailing_stop_atr",
        "take_profit_r_multiple",
        "max_holding_days",
        "momentum_gate_reason",
        "short_weakness_count",
        "benchmark_risk_on",
        "crowding_risk_score",
        "crowding_multiplier",
        "theme_memberships",
        "primary_theme",
        "theme_gross_exposure",
        "sector_gross_exposure",
        "industry_gross_exposure",
        "overnight_gap_risk_score",
        "overnight_gap_risk_status",
        "overnight_gap_multiplier",
        "overnight_gap_vol_20d",
        "overnight_gap_abs_p95_252d",
        "minute_pretrade_risk_score",
        "minute_pretrade_risk_status",
        "minute_pretrade_multiplier",
    ]
    if "days_to_earnings" in frame.columns:
        columns.append("days_to_earnings")
    for column in ("next_earnings_date", "event_known_date", "expected_move", "report_time"):
        if column in frame.columns:
            columns.append(column)
    optional_factor_columns = [
        "mom_return",
        "mom_risk_adjusted",
        "mom_distance_high",
        "mom_volume_expansion",
        "distance_to_high_252",
        "distance_to_prior_high_252",
        "breakout_component",
        "volume_component",
        "rs_component",
        "retest_component",
        "retest_distance_ma_pct",
        "trend_price_vs_slow_pct",
        "trend_adx",
        "mean_reversion_pressure",
        "rsi_14",
        "zscore_20",
        "vol_20d",
        "vol_63d",
        "atr_14",
        "pullback_entry_score",
        "pullback_volume_contraction_score",
        "pullback_volume_contraction_blocked",
        "gap_adjusted_continuation_score",
        "gap_adjusted_continuation_boost",
        "gap_adjusted_continuation_gap_risk",
        "gap_adjusted_continuation_gap_improvement",
        "gap_adjusted_continuation_eligible",
        "gap_adjusted_continuation_blocked",
        "pullback_short_term_reset_score",
        "pullback_short_term_reset_boost",
        "pullback_short_term_reset_ret_5d",
        "pullback_short_term_reset_ret_10d",
        "pullback_short_term_reset_blocked",
        "short_term_volume_tilt_score",
        "short_term_volume_tilt_boost",
        "short_term_volume_tilt_ret_5d",
        "short_term_volume_tilt_volume_expansion",
        "short_term_volume_tilt_candidate",
        "short_term_volume_tilt_circuit_ok",
        "short_term_volume_tilt_peer_confirmed",
        "short_term_volume_tilt_blocked",
        "short_term_volume_tilt_block_reason",
        "short_term_volume_tilt_same_theme_peer_count",
        "short_term_volume_tilt_same_theme_peer_confirm_share",
        "short_term_volume_tilt_same_theme_peer_avg_mom_return",
        "boundary_rs_theme_credit_score",
        "boundary_rs_theme_credit_boost",
        "boundary_rs_theme_credit_base_rank",
        "boundary_rs_theme_credit_blocked",
        "boundary_rs_theme_credit_eligible",
        "exit_quality_rank_credit_score",
        "exit_quality_rank_credit",
        "exit_quality_base_rank",
        "exit_quality_effective_score_rank",
        "exit_quality_rank_credit_blocked",
        "exit_quality_rank_credit_eligible",
        "exit_quality_theme_support_score",
        "exit_quality_theme_support_eligible",
        "pullback_reclaim_score",
        "pullback_reclaim_boost",
        "pullback_reclaim_recent_below_ma20_pct",
        "pullback_reclaim_blocked",
        "pullback_reset_inclusion",
        "pullback_reset_theme_breadth_active_share",
        "pullback_reset_theme_breadth_baseline_share",
        "pullback_reset_theme_breadth_expansion",
        "pullback_reset_theme_breadth_expanding",
        "pullback_reset_theme_breadth_safe",
        "pullback_reset_theme_breadth_shortfall",
        "pullback_reset_rs_score",
        "pullback_reset_rs_baseline",
        "pullback_reset_rs_acceleration",
        "pullback_reset_rs_accelerating",
        "pullback_reset_theme_strength_delta_score",
        "pullback_reset_theme_strength_supported",
        "pullback_reset_theme_strength_score_rank_credit",
        "pullback_reset_reclaim_score",
        "pullback_reset_reclaim_supported",
        "pullback_reset_reclaim_score_rank_credit",
        "theme_strength_delta_score",
        "theme_strength_delta_boost",
        "theme_strength_score_prior",
        "theme_strength_delta",
        "theme_strength_peer_rs_prior",
        "theme_strength_peer_rs_delta",
        "theme_strength_peer_count",
        "theme_strength_delta_eligible",
        "theme_breadth_acceleration_score",
        "theme_breadth_acceleration_boost",
        "theme_breadth_active_share_prior",
        "theme_breadth_acceleration",
        "theme_breadth_peer_count",
        "theme_breadth_acceleration_eligible",
        "theme_leader_tilt_score",
        "theme_leader_tilt_boost",
        "theme_leader_tilt_theme_rank_pct",
        "theme_leader_tilt_theme_peer_count",
        "theme_leader_tilt_eligible",
    ]
    columns.extend(optional_factor_columns)
    return list(dict.fromkeys(column for column in columns if column in frame.columns))


def event_entries_allowed(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Return whether ordinary strategy entries are allowed around events."""

    if not bool(params.get("block_high_event_risk", True)):
        return pd.Series(True, index=frame.index)
    max_score = float(params.get("max_event_risk_score", 70))
    return frame["event_risk_score"].fillna(0.0) <= max_score
