"""Portfolio construction from signals."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_system.config import PortfolioConfig
from quant_system.trade_location import attach_trade_location_scores


def rebalance_dates(dates: pd.Series, frequency: str) -> set[pd.Timestamp]:
    """Return dates that should generate target portfolios."""

    unique = pd.Series(pd.to_datetime(dates).sort_values().unique())
    if unique.empty:
        return set()
    if frequency in {"daily", "daily_throttled"}:
        return set(pd.to_datetime(unique))
    period = unique.dt.to_period("W-FRI" if frequency == "weekly" else "M")
    return set(unique.groupby(period).max().dt.normalize())


def construct_target_weights(signals: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Convert signal scores into target weights per rebalance date."""

    if signals.empty:
        return pd.DataFrame(columns=["date", "symbol", "target_weight"])
    signals = signals.copy()
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
    out_rows: list[pd.DataFrame] = []
    for date, day in signals.groupby("date"):
        selected = day[day["signal"].abs() > 0].copy()
        if selected.empty:
            continue
        selected["side"] = np.sign(selected["signal"])
        long = selected[selected["side"] > 0].sort_values("final_score", ascending=False)
        short = selected[selected["side"] < 0].sort_values("final_score", ascending=True)
        if config.max_total_positions > 0:
            long_slots = max(1, int(np.ceil(config.max_total_positions * 0.6)))
            short_slots = max(0, config.max_total_positions - long_slots)
            long = long.head(long_slots)
            short = short.head(short_slots)
        gross = float(config.target_gross_exposure)
        net = float(config.target_net_exposure)
        long_gross = max(0.0, min(gross, (gross + net) / 2.0))
        short_gross = max(0.0, min(gross, (gross - net) / 2.0))
        parts = []
        if not long.empty and long_gross > 0:
            parts.append(
                _side_weights(
                    long,
                    long_gross,
                    config.max_position_weight,
                    config.min_target_weight,
                    config.construction,
                    positive=True,
                    config=config,
                )
            )
        if not short.empty and short_gross > 0:
            parts.append(
                _side_weights(
                    short,
                    short_gross,
                    config.max_position_weight,
                    config.min_target_weight,
                    config.construction,
                    positive=False,
                    config=config,
                )
            )
        if parts:
            day_weights = pd.concat(parts, ignore_index=True)
            day_weights["date"] = date
            if config.rebalance == "daily_throttled" and (
                bool(getattr(config, "leader_hold_buffer_overlay", False))
                or bool(getattr(config, "leader_delayed_exit_overlay", False))
            ):
                inactive = day.loc[~day["symbol"].isin(day_weights["symbol"])].copy()
                if not inactive.empty:
                    inactive["target_weight"] = 0.0
                    day_weights = pd.concat([day_weights, inactive], ignore_index=True, sort=False)
            out_rows.append(day_weights)
    targets = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["date", "symbol", "target_weight"])
    if config.rebalance == "daily_throttled":
        return _throttle_daily_targets(targets, config)
    return targets


def _side_weights(
    frame: pd.DataFrame,
    side_gross: float,
    max_weight: float,
    min_weight: float,
    construction: str,
    positive: bool,
    config: PortfolioConfig | None = None,
) -> pd.DataFrame:
    frame = attach_trade_location_scores(frame, config, positive=positive) if config is not None else frame.copy()
    score = frame["final_score"].astype(float)
    if positive:
        score_edge = (score - score.min()).clip(lower=0) + 1e-6
    else:
        score_edge = (score.max() - score).clip(lower=0) + 1e-6
    mode = str(construction or "score_weighted").lower()
    if mode in {"equal_weight", "equal"}:
        raw = pd.Series(1.0, index=frame.index)
    elif mode in {"volatility_parity", "risk_parity"}:
        raw = 1.0 / _risk_unit(frame)
    elif mode in {"score_volatility", "score_volatility_weighted", "score_vol"}:
        score_strength = (score / 100.0).clip(lower=0.05, upper=1.0) if positive else ((100.0 - score) / 100.0).clip(lower=0.05, upper=1.0)
        raw = score_strength / _risk_unit(frame)
    else:
        raw = score_edge
    leader_eligible = _leader_addon_eligible(frame, config) if positive and config is not None else pd.Series(False, index=frame.index)
    leader_multiplier = float(getattr(config, "leader_addon_multiplier", 1.0) if config is not None else 1.0)
    leader_multiplier = float(np.clip(leader_multiplier, 1.0, 2.0))
    if positive and leader_eligible.any() and leader_multiplier > 1.0:
        raw = raw * np.where(leader_eligible, leader_multiplier, 1.0)
    conviction_context = _conviction_sizing_context(frame, config, positive) if config is not None else pd.DataFrame(index=frame.index)
    conviction_multiplier = (
        pd.to_numeric(conviction_context.get("conviction_sizing_multiplier", pd.Series(1.0, index=frame.index)), errors="coerce")
        .reindex(frame.index)
        .fillna(1.0)
        .astype(float)
    )
    if bool(getattr(config, "conviction_sizing_overlay", False) if config is not None else False):
        raw = raw * conviction_multiplier
    persistence_context = _leader_persistence_context(frame, config) if positive and config is not None else pd.DataFrame(index=frame.index)
    addon_theme_context = _leader_addon_theme_peer_context(frame, config) if positive and config is not None else pd.DataFrame(index=frame.index)
    if float(raw.sum()) <= 0.0:
        return _zero_weight_location_rows(frame)
    weights = raw / raw.sum() * side_gross
    weights = weights.clip(upper=max_weight)
    if not positive:
        weights = -weights
    keep = [
        "symbol",
        "signal",
        "final_score",
    ]
    for column in (
        "technical_score",
        "relative_strength_score",
        "theme_score",
        "theme_active",
        "theme_reason",
        "limited_history_flag",
        "filter_reason",
        "candidate_reason",
        "fundamental_score",
        "fundamental_factor_coverage",
        "fundamental_missing_group_count",
        "fundamental_data_quality",
        "moat_score",
        "sector",
        "industry",
        "fundamental_known_date",
        "fundamental_data_age_days",
        "growth_score",
        "quality_score",
        "balance_sheet_score",
        "valuation_score",
        "revision_score",
        "event_risk_score",
        "days_to_earnings",
        "next_earnings_date",
        "event_known_date",
        "expected_move",
        "score_decomposition",
        "reason_for_entry",
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
        "gap_adjusted_continuation_score",
        "gap_adjusted_continuation_boost",
        "gap_adjusted_continuation_gap_risk",
        "gap_adjusted_continuation_gap_improvement",
        "gap_adjusted_continuation_eligible",
        "gap_adjusted_continuation_blocked",
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
        "leader_addon_eligible",
        "leader_addon_multiplier",
        "leader_addon_reason",
        "leader_addon_circuit_breaker",
        "leader_addon_log",
        "leader_addon_theme_peer_count",
        "leader_addon_theme_peer_share",
        "leader_addon_same_theme_peer_support",
        "leader_addon_gap_quality_peer_count",
        "leader_addon_gap_quality_peer_share",
        "leader_addon_gap_quality_support",
        "leader_persistence_theme_peer_count",
        "leader_persistence_theme_peer_share",
        "leader_persistence_eligible",
        "conviction_sizing_multiplier",
        "conviction_sizing_eligible",
        "conviction_sizing_rank",
        "entry_quality_score",
        "reward_risk_estimate",
        "trade_location_type",
        "trade_location_multiplier",
        "trade_location_note",
    ):
        if column in frame.columns:
            keep.append(column)
    if "quality_position_multiplier" in frame.columns:
        keep.append("quality_position_multiplier")
    if "quality_adjusted" in frame.columns:
        keep.append("quality_adjusted")
    if "quality_adjustment_reason" in frame.columns:
        keep.append("quality_adjustment_reason")
    for column in ("stop_loss_atr", "trailing_stop_atr", "take_profit_r_multiple", "max_holding_days"):
        if column in frame.columns:
            keep.append(column)
    out = frame[keep].copy()
    out["leader_addon_eligible"] = leader_eligible.reindex(frame.index).fillna(False).astype(bool).to_numpy()
    out["leader_addon_multiplier"] = np.where(out["leader_addon_eligible"], leader_multiplier, 1.0)
    addon_meta = _leader_addon_metadata(frame, config) if positive and config is not None else pd.DataFrame(index=frame.index)
    if not addon_meta.empty:
        out["leader_addon_circuit_breaker"] = addon_meta["leader_addon_circuit_breaker"].reindex(frame.index).fillna("feature_off")
        out["leader_addon_log"] = addon_meta["leader_addon_log"].reindex(frame.index).fillna("")
        out["leader_addon_reason"] = np.where(
            out["leader_addon_eligible"],
            addon_meta["leader_addon_reason"].reindex(frame.index).fillna("").to_numpy(),
            "",
        )
    else:
        out["leader_addon_circuit_breaker"] = "feature_off"
        out["leader_addon_log"] = ""
        out["leader_addon_reason"] = np.where(
            out["leader_addon_eligible"],
            (
                "Leader add-on: high final/RS/theme score, positive momentum, "
                "and benchmark risk-on gate passed."
            ),
            "",
        )
    if not addon_theme_context.empty:
        for column in (
            "leader_addon_theme_peer_count",
            "leader_addon_theme_peer_share",
            "leader_addon_same_theme_peer_support",
            "leader_addon_gap_quality_peer_count",
            "leader_addon_gap_quality_peer_share",
            "leader_addon_gap_quality_support",
        ):
            out[column] = addon_theme_context[column].reindex(frame.index)
    else:
        out["leader_addon_theme_peer_count"] = 0
        out["leader_addon_theme_peer_share"] = 0.0
        out["leader_addon_same_theme_peer_support"] = False
        out["leader_addon_gap_quality_peer_count"] = 0
        out["leader_addon_gap_quality_peer_share"] = 0.0
        out["leader_addon_gap_quality_support"] = False
    if not persistence_context.empty:
        for column in ("leader_persistence_theme_peer_count", "leader_persistence_theme_peer_share", "leader_persistence_eligible"):
            out[column] = persistence_context[column].reindex(frame.index)
    else:
        out["leader_persistence_theme_peer_count"] = 0
        out["leader_persistence_theme_peer_share"] = 0.0
        out["leader_persistence_eligible"] = False
    out["conviction_sizing_multiplier"] = conviction_multiplier.to_numpy()
    conviction_eligible = conviction_context.get("conviction_sizing_eligible", pd.Series(False, index=frame.index)).reindex(frame.index).fillna(False).astype(bool)
    out["conviction_sizing_eligible"] = conviction_eligible.to_numpy()
    score_for_rank = pd.to_numeric(frame.get("final_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    out["conviction_sizing_rank"] = score_for_rank.rank(method="first", ascending=positive, pct=True).reindex(frame.index).to_numpy()
    out["conviction_sizing_circuit_breaker"] = (
        conviction_context.get("conviction_sizing_circuit_breaker", pd.Series("feature_off", index=frame.index)).reindex(frame.index).fillna("feature_off")
    )
    out["conviction_sizing_log"] = conviction_context.get("conviction_sizing_log", pd.Series("", index=frame.index)).reindex(frame.index).fillna("")
    out["conviction_sizing_theme_peer_count"] = (
        pd.to_numeric(conviction_context.get("conviction_sizing_theme_peer_count", pd.Series(0, index=frame.index)), errors="coerce").reindex(frame.index).fillna(0).astype(int)
    )
    out["conviction_sizing_theme_peer_share"] = (
        pd.to_numeric(conviction_context.get("conviction_sizing_theme_peer_share", pd.Series(0.0, index=frame.index)), errors="coerce").reindex(frame.index).fillna(0.0).astype(float)
    )
    out["conviction_sizing_same_theme_peer_support"] = (
        conviction_context.get("conviction_sizing_same_theme_peer_support", pd.Series(False, index=frame.index)).reindex(frame.index).fillna(False).astype(bool)
    )
    if "quality_position_multiplier" in frame.columns:
        weights *= frame["quality_position_multiplier"].fillna(1.0).astype(float).clip(lower=0.0, upper=1.0)
    out["target_weight"] = weights.to_numpy()
    out = out[out["target_weight"].abs() >= float(min_weight)]
    return out


def _zero_weight_location_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return blocked candidates for daily-throttled exit/hold visibility."""

    keep = [
        col
        for col in [
            "symbol",
            "signal",
            "final_score",
            "technical_score",
            "relative_strength_score",
            "theme_score",
            "fundamental_score",
            "event_risk_score",
            "reason_for_entry",
            "mom_return",
            "mom_volume_expansion",
            "distance_to_prior_high_252",
            "retest_distance_ma_pct",
            "rsi_14",
            "zscore_20",
            "primary_theme",
            "overnight_gap_risk_score",
            "entry_quality_score",
            "reward_risk_estimate",
            "trade_location_type",
            "trade_location_multiplier",
            "trade_location_note",
        ]
        if col in frame.columns
    ]
    out = frame[keep].copy()
    out["target_weight"] = 0.0
    if "reason_for_entry" in out.columns:
        out["reason_for_entry"] = (
            out["reason_for_entry"].fillna("").astype(str)
            + " | Trade location blocked fresh sizing: "
            + out.get("trade_location_note", pd.Series("", index=out.index)).fillna("").astype(str)
        )
    return out.reset_index(drop=True)


def _conviction_sizing_context(frame: pd.DataFrame, config: PortfolioConfig | None, positive: bool) -> pd.DataFrame:
    """Return conviction-sizing multipliers plus circuit-breaker diagnostics.

    Feature flag: ``conviction_sizing_overlay`` defaults to false.
    Optional theme-support gate: ``conviction_sizing_require_same_theme_peer_support``
    defaults to false and only boosts names whose theme still has enough strong
    local participation. Reasonable ranges:
    - ``conviction_sizing_same_theme_peer_min_count``: 2 to 5 names.
    - ``conviction_sizing_same_theme_peer_min_share``: 0.30 to 0.65.
    - ``conviction_sizing_same_theme_peer_min_avg_relative_strength_score``: 70 to 90.

    Circuit breakers: benchmark/event/gap guards remain in force, and the
    optional peer-support gate blocks isolated names from receiving extra size.
    Gross exposure is unchanged because the overlay only reweights names already
    present on the same side before normalization.
    """

    out = pd.DataFrame(index=frame.index)
    out["conviction_sizing_multiplier"] = 1.0
    out["conviction_sizing_eligible"] = False
    out["conviction_sizing_circuit_breaker"] = "feature_off"
    out["conviction_sizing_log"] = ""
    out["conviction_sizing_theme_peer_count"] = 0
    out["conviction_sizing_theme_peer_share"] = 0.0
    out["conviction_sizing_same_theme_peer_support"] = False
    if config is None or not bool(getattr(config, "conviction_sizing_overlay", False)) or frame.empty:
        return out

    min_names = max(2, int(getattr(config, "conviction_sizing_min_names", 8)))
    if len(frame) < min_names:
        out["conviction_sizing_circuit_breaker"] = "insufficient_names"
        return out

    def _series(column: str, default: float | bool) -> pd.Series:
        if column in frame.columns:
            return frame[column]
        return pd.Series(default, index=frame.index)

    final_score = pd.to_numeric(_series("final_score", 0.0), errors="coerce").fillna(0.0)
    rs_score = pd.to_numeric(_series("relative_strength_score", 0.0), errors="coerce").fillna(0.0)
    theme_score = pd.to_numeric(_series("theme_score", 50.0), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(_series("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(_series("overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ok = pd.Series(_series("benchmark_risk_on", True), index=frame.index, dtype="boolean").fillna(False).astype(bool)
    primary_theme = _series("primary_theme", "unknown").fillna("unknown").astype(str)
    mom_return = pd.to_numeric(_series("mom_return", 0.0), errors="coerce").fillna(0.0)

    min_final = float(getattr(config, "conviction_sizing_min_final_score", 60.0))
    min_rs = float(getattr(config, "conviction_sizing_min_relative_strength_score", 55.0))
    min_theme = float(getattr(config, "conviction_sizing_min_theme_score", 50.0))
    max_event = float(getattr(config, "conviction_sizing_max_event_risk_score_circuit_breaker", 25.0))
    max_gap = float(getattr(config, "conviction_sizing_max_overnight_gap_risk_score_circuit_breaker", 35.0))
    require_benchmark = bool(getattr(config, "conviction_sizing_require_benchmark_risk_on", True))
    require_peer_support = positive and bool(getattr(config, "conviction_sizing_require_same_theme_peer_support", False))
    allowed_themes = _normalise_theme_allowlist(getattr(config, "conviction_sizing_allowed_themes", ()))
    theme_allowed = (
        primary_theme.str.lower().isin(allowed_themes)
        if allowed_themes
        else pd.Series(True, index=frame.index)
    )
    base_eligible = (
        final_score.ge(min_final)
        & rs_score.ge(min_rs)
        & theme_score.ge(min_theme)
        & event_risk.le(max_event)
        & gap_risk.le(max_gap)
        & theme_allowed
    )
    if require_benchmark:
        base_eligible &= benchmark_ok

    peer_context = _conviction_sizing_theme_peer_context(frame, config, base_eligible, positive)
    out["conviction_sizing_theme_peer_count"] = (
        pd.to_numeric(peer_context["conviction_sizing_theme_peer_count"], errors="coerce").fillna(0).astype(int)
    )
    out["conviction_sizing_theme_peer_share"] = (
        pd.to_numeric(peer_context["conviction_sizing_theme_peer_share"], errors="coerce").fillna(0.0).astype(float)
    )
    out["conviction_sizing_same_theme_peer_support"] = peer_context["conviction_sizing_same_theme_peer_support"].fillna(False).astype(bool)

    eligible = base_eligible.copy()
    if require_peer_support:
        eligible &= out["conviction_sizing_same_theme_peer_support"].fillna(False).astype(bool)

    rank = final_score.rank(method="first", ascending=positive, pct=True).clip(lower=0.0, upper=1.0)
    power = float(np.clip(float(getattr(config, "conviction_sizing_power", 1.25)), 1.0, 3.0))
    max_multiplier = float(np.clip(float(getattr(config, "conviction_sizing_max_multiplier", 1.35)), 1.0, 2.0))
    scaled = np.power(rank, power)
    out.loc[eligible, "conviction_sizing_multiplier"] = 1.0 + (max_multiplier - 1.0) * scaled.loc[eligible]
    out["conviction_sizing_multiplier"] = pd.to_numeric(out["conviction_sizing_multiplier"], errors="coerce").fillna(1.0).clip(lower=1.0, upper=max_multiplier)
    out["conviction_sizing_eligible"] = eligible.fillna(False).astype(bool)

    min_peer_count = max(1, int(getattr(config, "conviction_sizing_same_theme_peer_min_count", 3)))
    min_peer_share = float(getattr(config, "conviction_sizing_same_theme_peer_min_share", 0.45))
    min_peer_avg_rs = float(getattr(config, "conviction_sizing_same_theme_peer_min_avg_relative_strength_score", 80.0))
    min_peer_avg_mom = float(getattr(config, "conviction_sizing_same_theme_peer_min_avg_mom_return", 0.10))
    peer_avg_rs = pd.to_numeric(peer_context["conviction_sizing_theme_peer_avg_relative_strength_score"], errors="coerce").fillna(0.0)
    peer_avg_mom = pd.to_numeric(peer_context["conviction_sizing_theme_peer_avg_mom_return"], errors="coerce").fillna(0.0)
    theme_known = primary_theme.ne("").fillna(False) & primary_theme.ne("unknown") & primary_theme.ne("unclassified")

    for idx in frame.index:
        breaker = "applied"
        if require_benchmark and not bool(benchmark_ok.loc[idx]):
            breaker = "benchmark_not_risk_on"
        elif float(event_risk.loc[idx]) > max_event:
            breaker = "event_risk_high"
        elif float(gap_risk.loc[idx]) > max_gap:
            breaker = "gap_risk_high"
        elif float(final_score.loc[idx]) < min_final:
            breaker = "final_score_low"
        elif float(rs_score.loc[idx]) < min_rs:
            breaker = "relative_strength_low"
        elif float(theme_score.loc[idx]) < min_theme:
            breaker = "theme_score_low"
        elif allowed_themes and not bool(theme_allowed.loc[idx]):
            breaker = "theme_not_allowed"
        elif require_peer_support and not bool(theme_known.loc[idx]):
            breaker = "theme_unknown"
        elif require_peer_support and int(out.at[idx, "conviction_sizing_theme_peer_count"]) < min_peer_count:
            breaker = "same_theme_peer_count_low"
        elif require_peer_support and float(out.at[idx, "conviction_sizing_theme_peer_share"]) < min_peer_share:
            breaker = "same_theme_peer_share_low"
        elif require_peer_support and float(peer_avg_rs.loc[idx]) < min_peer_avg_rs:
            breaker = "same_theme_peer_avg_relative_strength_low"
        elif require_peer_support and float(peer_avg_mom.loc[idx]) < min_peer_avg_mom:
            breaker = "same_theme_peer_avg_momentum_low"
        out.at[idx, "conviction_sizing_circuit_breaker"] = breaker
        out.at[idx, "conviction_sizing_log"] = json.dumps(
            {
                "benchmark_risk_on": bool(benchmark_ok.loc[idx]),
                "circuit_breaker": breaker,
                "conviction_sizing_eligible": bool(out.at[idx, "conviction_sizing_eligible"]),
                "conviction_sizing_multiplier": float(out.at[idx, "conviction_sizing_multiplier"]),
                "event_risk_score": float(event_risk.loc[idx]),
                "final_score": float(final_score.loc[idx]),
                "mom_return": float(mom_return.loc[idx]),
                "overnight_gap_risk_score": float(gap_risk.loc[idx]),
                "allowed_themes": sorted(allowed_themes),
                "require_same_theme_peer_support": require_peer_support,
                "relative_strength_score": float(rs_score.loc[idx]),
                "same_theme_peer_count": int(out.at[idx, "conviction_sizing_theme_peer_count"]),
                "same_theme_peer_share": float(out.at[idx, "conviction_sizing_theme_peer_share"]),
                "same_theme_peer_support": bool(out.at[idx, "conviction_sizing_same_theme_peer_support"]),
                "same_theme_peer_min_avg_mom_return": float(min_peer_avg_mom),
                "same_theme_peer_min_avg_relative_strength_score": float(min_peer_avg_rs),
                "same_theme_peer_min_count": int(min_peer_count),
                "same_theme_peer_min_share": float(min_peer_share),
                "theme_score": float(theme_score.loc[idx]),
            },
            sort_keys=True,
        )
    return out


def _normalise_theme_allowlist(raw: object) -> set[str]:
    """Return lower-case theme names for optional conviction-sizing allowlists."""

    if raw is None:
        return set()
    if isinstance(raw, str):
        values = [raw]
    else:
        try:
            values = list(raw)  # type: ignore[arg-type]
        except TypeError:
            values = [raw]
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _conviction_sizing_theme_peer_context(
    frame: pd.DataFrame,
    config: PortfolioConfig | None,
    base_eligible: pd.Series,
    positive: bool,
) -> pd.DataFrame:
    """Return same-theme breadth context for conviction sizing."""

    out = pd.DataFrame(index=frame.index)
    out["conviction_sizing_theme_peer_count"] = 0
    out["conviction_sizing_theme_peer_share"] = 0.0
    out["conviction_sizing_same_theme_peer_support"] = False
    out["conviction_sizing_theme_peer_avg_relative_strength_score"] = 0.0
    out["conviction_sizing_theme_peer_avg_mom_return"] = 0.0
    if config is None or frame.empty or not positive:
        return out

    primary_theme = frame.get("primary_theme", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    theme_name_known = primary_theme.ne("").fillna(False) & primary_theme.ne("unknown") & primary_theme.ne("unclassified")
    if not bool(theme_name_known.any()):
        return out

    theme_active = frame.get("theme_active", pd.Series(True, index=frame.index)).fillna(True).astype(bool)
    strong_peer = base_eligible.fillna(False).astype(bool) & theme_name_known
    if "theme_active" in frame.columns:
        strong_peer &= theme_active

    peer_count = strong_peer.groupby(primary_theme).transform("sum").fillna(0).astype(int)
    theme_population = theme_name_known.groupby(primary_theme).transform("sum").replace(0, np.nan)
    peer_share = (peer_count / theme_population).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    rs_score = pd.to_numeric(frame.get("relative_strength_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    mom_return = pd.to_numeric(frame.get("mom_return", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    peer_avg_rs = rs_score.where(strong_peer).groupby(primary_theme).transform("mean").fillna(0.0)
    peer_avg_mom = mom_return.where(strong_peer).groupby(primary_theme).transform("mean").fillna(0.0)

    min_count = max(1, int(getattr(config, "conviction_sizing_same_theme_peer_min_count", 3)))
    min_share = float(getattr(config, "conviction_sizing_same_theme_peer_min_share", 0.45))
    min_avg_rs = float(getattr(config, "conviction_sizing_same_theme_peer_min_avg_relative_strength_score", 80.0))
    min_avg_mom = float(getattr(config, "conviction_sizing_same_theme_peer_min_avg_mom_return", 0.10))
    out["conviction_sizing_theme_peer_count"] = peer_count
    out["conviction_sizing_theme_peer_share"] = peer_share
    out["conviction_sizing_theme_peer_avg_relative_strength_score"] = peer_avg_rs
    out["conviction_sizing_theme_peer_avg_mom_return"] = peer_avg_mom
    out["conviction_sizing_same_theme_peer_support"] = (
        strong_peer
        & peer_count.ge(min_count)
        & peer_share.ge(min_share)
        & peer_avg_rs.ge(min_avg_rs)
        & peer_avg_mom.ge(min_avg_mom)
    )
    return out


def _leader_addon_eligible(frame: pd.DataFrame, config: PortfolioConfig | None) -> pd.Series:
    """Return rows eligible for a small raw-weight tilt among existing longs.

    Feature flag: ``leader_addon_overlay`` defaults to false.
    Optional gap-quality flag: ``leader_addon_gap_quality_support_overlay``
    defaults to false and requires both the candidate symbol and enough
    same-theme peers to carry benign overnight-gap risk. Reasonable ranges:
    - ``leader_addon_gap_quality_max_symbol_gap_risk_score``: 12 to 25.
    - ``leader_addon_gap_quality_same_theme_max_gap_risk_score``: 10 to 22.
    - ``leader_addon_gap_quality_same_theme_min_count``: 2 to 4.
    - ``leader_addon_gap_quality_same_theme_min_share``: 0.20 to 0.50.

    Circuit breakers: the add-on still only applies to already-selected long
    rows, stays inside side-gross normalization and max-position caps, and the
    gap-quality overlay cannot bypass benchmark, event, or extension gates.
    """

    if config is None or not bool(getattr(config, "leader_addon_overlay", False)):
        return pd.Series(False, index=frame.index)
    max_names = max(0, int(getattr(config, "leader_addon_max_names_per_date", 5)))
    if max_names <= 0:
        return pd.Series(False, index=frame.index)
    final_score = pd.to_numeric(frame.get("final_score", 0.0), errors="coerce").fillna(0.0)
    rs_score = pd.to_numeric(frame.get("relative_strength_score", 0.0), errors="coerce").fillna(0.0)
    theme_score = pd.to_numeric(frame.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(frame.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    benchmark_ok = frame.get("benchmark_risk_on", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    drawdown_from_high = pd.to_numeric(frame.get("distance_to_high_252", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).abs()
    above_ma20 = pd.to_numeric(frame.get("retest_distance_ma_pct", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(frame.get("event_risk_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(frame.get("overnight_gap_risk_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    eligible = (
        final_score.ge(float(getattr(config, "leader_addon_min_final_score", 70.0)))
        & rs_score.ge(float(getattr(config, "leader_addon_min_relative_strength_score", 80.0)))
        & theme_score.ge(float(getattr(config, "leader_addon_min_theme_score", 65.0)))
        & mom_return.ge(float(getattr(config, "leader_addon_min_mom_return", 0.10)))
        & event_risk.le(float(getattr(config, "leader_addon_max_event_risk_score_circuit_breaker", 20.0)))
        & gap_risk.le(float(getattr(config, "leader_addon_max_overnight_gap_risk_score_circuit_breaker", 30.0)))
        & above_ma20.le(float(getattr(config, "leader_addon_max_above_ma20_pct_circuit_breaker", 0.08)))
    )
    if bool(getattr(config, "leader_addon_require_benchmark_risk_on", True)):
        eligible &= benchmark_ok
    if bool(getattr(config, "leader_addon_require_controlled_pullback", False)):
        eligible &= drawdown_from_high.ge(float(getattr(config, "leader_addon_min_drawdown_from_high", 0.02)))
        eligible &= drawdown_from_high.le(float(getattr(config, "leader_addon_max_drawdown_from_high", 0.15)))
    if bool(getattr(config, "leader_addon_require_same_theme_peer_support", False)):
        peer_context = _leader_addon_theme_peer_context(frame, config)
        eligible &= peer_context["leader_addon_same_theme_peer_support"].reindex(frame.index).fillna(False).astype(bool)
    if bool(getattr(config, "leader_addon_gap_quality_support_overlay", False)):
        peer_context = _leader_addon_theme_peer_context(frame, config)
        eligible &= gap_risk.le(float(getattr(config, "leader_addon_gap_quality_max_symbol_gap_risk_score", 20.0)))
        eligible &= peer_context["leader_addon_gap_quality_support"].reindex(frame.index).fillna(False).astype(bool)
    rank = final_score.where(eligible, -np.inf).groupby(frame["date"]).rank(method="first", ascending=False)
    return eligible & rank.le(max_names)


def _leader_addon_theme_peer_context(frame: pd.DataFrame, config: PortfolioConfig | None) -> pd.DataFrame:
    """Return same-theme peer breadth used by the leader add-on overlay."""

    out = pd.DataFrame(index=frame.index)
    out["leader_addon_theme_peer_count"] = 0
    out["leader_addon_theme_peer_share"] = 0.0
    out["leader_addon_same_theme_peer_support"] = False
    out["leader_addon_gap_quality_peer_count"] = 0
    out["leader_addon_gap_quality_peer_share"] = 0.0
    out["leader_addon_gap_quality_support"] = False
    if config is None or frame.empty:
        return out
    def _series(column: str, default: float | str | bool) -> pd.Series:
        if column in frame.columns:
            return frame[column]
        return pd.Series(default, index=frame.index)

    final_score = pd.to_numeric(frame.get("final_score", 0.0), errors="coerce").fillna(0.0)
    rs_score = pd.to_numeric(_series("relative_strength_score", 0.0), errors="coerce").fillna(0.0)
    theme_score = pd.to_numeric(_series("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(_series("mom_return", 0.0), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(_series("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(_series("overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ok = pd.Series(_series("benchmark_risk_on", True), index=frame.index, dtype="boolean").fillna(False).astype(bool)
    primary_theme = _series("primary_theme", "unknown").fillna("unknown").astype(str)
    theme_active = _series("theme_active", True).fillna(True).astype(bool)
    strong_peer = (
        final_score.ge(float(getattr(config, "leader_addon_min_final_score", 70.0)))
        & rs_score.ge(float(getattr(config, "leader_addon_min_relative_strength_score", 80.0)))
        & theme_score.ge(float(getattr(config, "leader_addon_min_theme_score", 65.0)))
        & mom_return.ge(float(getattr(config, "leader_addon_min_mom_return", 0.10)))
        & event_risk.le(float(getattr(config, "leader_addon_max_event_risk_score_circuit_breaker", 20.0)))
        & gap_risk.le(float(getattr(config, "leader_addon_max_overnight_gap_risk_score_circuit_breaker", 30.0)))
    )
    if bool(getattr(config, "leader_addon_require_benchmark_risk_on", True)):
        strong_peer &= benchmark_ok
    if "theme_active" in frame.columns:
        strong_peer &= theme_active
    theme_name_known = primary_theme.ne("").fillna(False) & primary_theme.ne("unknown") & primary_theme.ne("unclassified")
    strong_peer &= theme_name_known
    total_theme_names = theme_name_known.groupby(primary_theme).transform("sum").replace(0, np.nan)
    peer_count = strong_peer.groupby(primary_theme).transform("sum").fillna(0).astype(int)
    peer_share = (peer_count / total_theme_names).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["leader_addon_theme_peer_count"] = peer_count
    out["leader_addon_theme_peer_share"] = peer_share
    out["leader_addon_same_theme_peer_support"] = (
        peer_count.ge(max(1, int(getattr(config, "leader_addon_same_theme_peer_min_count", 2))))
        & peer_share.ge(float(getattr(config, "leader_addon_same_theme_peer_min_share", 0.20)))
    )
    gap_quality_peer = strong_peer & gap_risk.le(float(getattr(config, "leader_addon_gap_quality_same_theme_max_gap_risk_score", 18.0)))
    gap_quality_peer_count = gap_quality_peer.groupby(primary_theme).transform("sum").fillna(0).astype(int)
    gap_quality_peer_share = (gap_quality_peer_count / total_theme_names).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["leader_addon_gap_quality_peer_count"] = gap_quality_peer_count
    out["leader_addon_gap_quality_peer_share"] = gap_quality_peer_share
    out["leader_addon_gap_quality_support"] = (
        gap_quality_peer_count.ge(max(1, int(getattr(config, "leader_addon_gap_quality_same_theme_min_count", 2))))
        & gap_quality_peer_share.ge(float(getattr(config, "leader_addon_gap_quality_same_theme_min_share", 0.25)))
    )
    return out


def _leader_addon_metadata(frame: pd.DataFrame, config: PortfolioConfig | None) -> pd.DataFrame:
    """Return structured logging context for leader add-on activation."""

    out = pd.DataFrame(index=frame.index)
    out["leader_addon_circuit_breaker"] = "feature_off"
    out["leader_addon_reason"] = ""
    out["leader_addon_log"] = ""
    if config is None or not bool(getattr(config, "leader_addon_overlay", False)) or frame.empty:
        return out
    eligible = _leader_addon_eligible(frame, config)
    final_score = pd.to_numeric(frame.get("final_score", 0.0), errors="coerce").fillna(0.0)
    rs_score = pd.to_numeric(frame.get("relative_strength_score", 0.0), errors="coerce").fillna(0.0)
    theme_score = pd.to_numeric(frame.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(frame.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    benchmark_ok = frame.get("benchmark_risk_on", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    drawdown_from_high = pd.to_numeric(frame.get("distance_to_high_252", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).abs()
    above_ma20 = pd.to_numeric(frame.get("retest_distance_ma_pct", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(frame.get("event_risk_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(frame.get("overnight_gap_risk_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    peer_context = _leader_addon_theme_peer_context(frame, config)
    peer_count = pd.to_numeric(peer_context["leader_addon_theme_peer_count"], errors="coerce").fillna(0).astype(int)
    peer_share = pd.to_numeric(peer_context["leader_addon_theme_peer_share"], errors="coerce").fillna(0.0)
    same_theme_peer_support = peer_context["leader_addon_same_theme_peer_support"].fillna(False).astype(bool)
    gap_quality_peer_count = pd.to_numeric(peer_context["leader_addon_gap_quality_peer_count"], errors="coerce").fillna(0).astype(int)
    gap_quality_peer_share = pd.to_numeric(peer_context["leader_addon_gap_quality_peer_share"], errors="coerce").fillna(0.0)
    gap_quality_support = peer_context["leader_addon_gap_quality_support"].fillna(False).astype(bool)
    controlled_pullback_on = bool(getattr(config, "leader_addon_require_controlled_pullback", False))
    peer_support_required = bool(getattr(config, "leader_addon_require_same_theme_peer_support", False))
    gap_quality_required = bool(getattr(config, "leader_addon_gap_quality_support_overlay", False))
    min_final = float(getattr(config, "leader_addon_min_final_score", 70.0))
    min_rs = float(getattr(config, "leader_addon_min_relative_strength_score", 80.0))
    min_theme = float(getattr(config, "leader_addon_min_theme_score", 65.0))
    min_mom = float(getattr(config, "leader_addon_min_mom_return", 0.10))
    max_event = float(getattr(config, "leader_addon_max_event_risk_score_circuit_breaker", 20.0))
    max_gap = float(getattr(config, "leader_addon_max_overnight_gap_risk_score_circuit_breaker", 30.0))
    max_symbol_gap_quality = float(getattr(config, "leader_addon_gap_quality_max_symbol_gap_risk_score", 20.0))
    max_above_ma20 = float(getattr(config, "leader_addon_max_above_ma20_pct_circuit_breaker", 0.08))
    min_drawdown = float(getattr(config, "leader_addon_min_drawdown_from_high", 0.02))
    max_drawdown = float(getattr(config, "leader_addon_max_drawdown_from_high", 0.15))
    min_peer_count = max(1, int(getattr(config, "leader_addon_same_theme_peer_min_count", 2)))
    min_peer_share = float(getattr(config, "leader_addon_same_theme_peer_min_share", 0.20))
    min_gap_quality_peer_count = max(1, int(getattr(config, "leader_addon_gap_quality_same_theme_min_count", 2)))
    min_gap_quality_peer_share = float(getattr(config, "leader_addon_gap_quality_same_theme_min_share", 0.25))
    max_gap_quality_peer_gap = float(getattr(config, "leader_addon_gap_quality_same_theme_max_gap_risk_score", 18.0))
    for idx in frame.index:
        breaker = "applied"
        if bool(getattr(config, "leader_addon_require_benchmark_risk_on", True)) and not bool(benchmark_ok.loc[idx]):
            breaker = "benchmark_not_risk_on"
        elif float(event_risk.loc[idx]) > max_event:
            breaker = "event_risk_high"
        elif float(gap_risk.loc[idx]) > max_gap:
            breaker = "gap_risk_high"
        elif gap_quality_required and float(gap_risk.loc[idx]) > max_symbol_gap_quality:
            breaker = "gap_quality_symbol_gap_high"
        elif float(above_ma20.loc[idx]) > max_above_ma20:
            breaker = "above_ma20_too_extended"
        elif float(final_score.loc[idx]) < min_final:
            breaker = "final_score_low"
        elif float(rs_score.loc[idx]) < min_rs:
            breaker = "relative_strength_low"
        elif float(theme_score.loc[idx]) < min_theme:
            breaker = "theme_score_low"
        elif float(mom_return.loc[idx]) < min_mom:
            breaker = "momentum_return_low"
        elif controlled_pullback_on and float(drawdown_from_high.loc[idx]) < min_drawdown:
            breaker = "pullback_too_shallow"
        elif controlled_pullback_on and float(drawdown_from_high.loc[idx]) > max_drawdown:
            breaker = "pullback_too_deep"
        elif peer_support_required and not bool(same_theme_peer_support.loc[idx]):
            breaker = "same_theme_peer_support_low"
        elif gap_quality_required and not bool(gap_quality_support.loc[idx]):
            breaker = "gap_quality_same_theme_support_low"
        elif not bool(eligible.loc[idx]):
            breaker = "rank_not_selected"
        out.at[idx, "leader_addon_circuit_breaker"] = breaker
        if bool(eligible.loc[idx]):
            if controlled_pullback_on and peer_support_required and gap_quality_required:
                reason = (
                    "Leader add-on: controlled-pullback leader passed benchmark and risk breakers "
                    "with benign own gap risk and enough same-theme benign-gap peers to confirm safer leadership."
                )
            elif controlled_pullback_on and peer_support_required:
                reason = (
                    "Leader add-on: controlled-pullback leader passed benchmark and risk breakers "
                    "with enough strong same-theme peers to confirm local leadership."
                )
            elif controlled_pullback_on:
                reason = (
                    "Leader add-on: high-conviction leader in a controlled pullback passed "
                    "benchmark, event, gap, and extension circuit breakers."
                )
            elif peer_support_required:
                reason = (
                    "Leader add-on: high final/RS/theme leader passed risk gates with enough "
                    "strong same-theme peers to confirm active theme leadership."
                )
            else:
                reason = "Leader add-on: high final/RS/theme score, positive momentum, and benchmark risk-on gate passed."
            out.at[idx, "leader_addon_reason"] = reason
        out.at[idx, "leader_addon_log"] = json.dumps(
            {
                "benchmark_risk_on": bool(benchmark_ok.loc[idx]),
                "circuit_breaker": breaker,
                "controlled_pullback_required": controlled_pullback_on,
                "distance_to_high_252_abs": float(drawdown_from_high.loc[idx]),
                "event_risk_score": float(event_risk.loc[idx]),
                "final_score": float(final_score.loc[idx]),
                "same_theme_peer_count": int(peer_count.loc[idx]),
                "same_theme_peer_share": float(peer_share.loc[idx]),
                "same_theme_peer_support": bool(same_theme_peer_support.loc[idx]),
                "same_theme_peer_support_required": peer_support_required,
                "same_theme_peer_support_min_count": int(min_peer_count),
                "same_theme_peer_support_min_share": float(min_peer_share),
                "gap_quality_required": gap_quality_required,
                "gap_quality_same_theme_peer_count": int(gap_quality_peer_count.loc[idx]),
                "gap_quality_same_theme_peer_share": float(gap_quality_peer_share.loc[idx]),
                "gap_quality_same_theme_support": bool(gap_quality_support.loc[idx]),
                "gap_quality_same_theme_min_count": int(min_gap_quality_peer_count),
                "gap_quality_same_theme_min_share": float(min_gap_quality_peer_share),
                "gap_quality_same_theme_max_gap_risk_score": float(max_gap_quality_peer_gap),
                "gap_quality_max_symbol_gap_risk_score": float(max_symbol_gap_quality),
                "mom_return": float(mom_return.loc[idx]),
                "overnight_gap_risk_score": float(gap_risk.loc[idx]),
                "relative_strength_score": float(rs_score.loc[idx]),
                "retest_distance_ma_pct": float(above_ma20.loc[idx]),
                "theme_score": float(theme_score.loc[idx]),
            },
            sort_keys=True,
        )
    return out


def _leader_persistence_context(frame: pd.DataFrame, config: PortfolioConfig | None) -> pd.DataFrame:
    """Return same-theme leadership context for the persistence overlay."""

    out = pd.DataFrame(index=frame.index)
    out["leader_persistence_theme_peer_count"] = 0
    out["leader_persistence_theme_peer_share"] = 0.0
    out["leader_persistence_eligible"] = False
    if config is None or not bool(getattr(config, "leader_persistence_overlay", False)) or frame.empty:
        return out
    rs_score = pd.to_numeric(frame.get("relative_strength_score", 0.0), errors="coerce").fillna(0.0)
    theme_score = pd.to_numeric(frame.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    final_score = pd.to_numeric(frame.get("final_score", 0.0), errors="coerce").fillna(0.0)
    mom_return = pd.to_numeric(frame.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    theme_active = frame.get("theme_active", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    primary_theme = frame.get("primary_theme", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    strong = (
        final_score.ge(float(getattr(config, "leader_persistence_min_final_score", 68.0)))
        & rs_score.ge(float(getattr(config, "leader_persistence_min_relative_strength_score", 88.0)))
        & theme_score.ge(float(getattr(config, "leader_persistence_min_theme_score", 62.0)))
        & mom_return.ge(float(getattr(config, "leader_persistence_min_mom_return", 0.15)))
    )
    if bool(getattr(config, "leader_persistence_require_theme_active", True)):
        strong &= theme_active
    total_theme_names = primary_theme.groupby(primary_theme).transform("size").replace(0, np.nan)
    peer_count = strong.groupby(primary_theme).transform("sum").fillna(0).astype(int)
    peer_share = (peer_count / total_theme_names).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["leader_persistence_theme_peer_count"] = peer_count
    out["leader_persistence_theme_peer_share"] = peer_share
    out["leader_persistence_eligible"] = strong
    return out


def _risk_unit(frame: pd.DataFrame) -> pd.Series:
    """Return a positive per-symbol risk estimate known at signal close."""

    if "vol_20d" in frame.columns:
        risk = frame["vol_20d"].astype(float)
    elif {"atr_14", "adj_close"}.issubset(frame.columns):
        risk = frame["atr_14"].astype(float) / frame["adj_close"].replace(0, np.nan).astype(float)
    else:
        risk = pd.Series(0.03, index=frame.index)
    fallback = risk.replace([np.inf, -np.inf], np.nan).median()
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 0.03
    return risk.replace([np.inf, -np.inf], np.nan).fillna(fallback).clip(lower=0.01, upper=0.20)


def _throttle_daily_targets(targets: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Reduce daily churn while preserving full target portfolios when trading."""

    if targets.empty:
        return targets
    out: list[pd.DataFrame] = []
    active_rows: dict[str, pd.Series] = {}
    active_weights: dict[str, float] = {}
    active_scores: dict[str, float] = {}
    min_weight_change = float(config.throttle_min_weight_change)
    min_score_change = float(config.throttle_min_score_change)
    min_new_weight = float(config.throttle_min_new_weight)
    for date, day in targets.sort_values(["date", "symbol"]).groupby("date", sort=True):
        desired_rows = {str(row["symbol"]): row for _, row in day.iterrows()}
        desired_weights = {symbol: float(row["target_weight"]) for symbol, row in desired_rows.items()}
        desired_scores = {symbol: float(row.get("final_score", 50.0) or 50.0) for symbol, row in desired_rows.items()}
        changed = False
        removed_rows: list[pd.Series] = []
        for symbol in sorted(set(active_weights) | set(desired_weights)):
            current_weight = active_weights.get(symbol, 0.0)
            desired_weight = desired_weights.get(symbol, 0.0)
            current_score = active_scores.get(symbol, 50.0)
            desired_score = desired_scores.get(symbol, current_score)
            if symbol in active_rows and symbol in desired_rows:
                delayed_exit_decision = _leader_delayed_exit_decision(
                    current=active_rows[symbol],
                    desired=desired_rows[symbol],
                    current_weight=current_weight,
                    config=config,
                )
                desired_rows[symbol] = delayed_exit_decision["row"]
                desired_weights[symbol] = float(delayed_exit_decision["row"].get("target_weight", desired_weight) or 0.0)
                desired_weight = desired_weights[symbol]
                desired_scores[symbol] = _safe_float(delayed_exit_decision["row"].get("final_score"), desired_score)
                desired_score = desired_scores[symbol]
                hold_decision = _leader_hold_buffer_decision(
                    current=active_rows[symbol],
                    desired=desired_rows[symbol],
                    current_weight=current_weight,
                    config=config,
                )
                desired_rows[symbol] = hold_decision["row"]
                desired_weights[symbol] = float(hold_decision["row"].get("target_weight", desired_weight) or 0.0)
                desired_weight = desired_weights[symbol]
                desired_scores[symbol] = _safe_float(hold_decision["row"].get("final_score"), desired_score)
                desired_score = desired_scores[symbol]
                persistence_decision = _leader_persistence_decision(
                    current=active_rows[symbol],
                    desired=desired_rows[symbol],
                    current_weight=current_weight,
                    config=config,
                )
                desired_rows[symbol] = persistence_decision["row"]
                desired_weights[symbol] = float(persistence_decision["row"].get("target_weight", desired_weight) or 0.0)
                desired_weight = desired_weights[symbol]
            should_trade = False
            if symbol not in active_weights:
                should_trade = abs(desired_weight) > 0 and abs(desired_weight) >= min_new_weight
            elif symbol not in desired_weights:
                should_trade = True
            elif current_weight * desired_weight < 0:
                should_trade = True
            elif abs(desired_weight - current_weight) >= min_weight_change:
                should_trade = True
            elif abs(desired_score - current_score) >= min_score_change and abs(desired_weight - current_weight) >= min_new_weight:
                should_trade = True
            elif symbol in active_rows and symbol in desired_rows and _event_context_changed(active_rows[symbol], desired_rows[symbol]):
                should_trade = True
            if not should_trade:
                if symbol in active_rows and symbol in desired_rows:
                    row = desired_rows[symbol].copy()
                    row["date"] = date
                    row["target_weight"] = current_weight
                    active_rows[symbol] = row
                continue
            changed = True
            if symbol in desired_rows and abs(desired_weight) > 0 and abs(desired_weight) >= min_new_weight:
                row = desired_rows[symbol].copy()
                row["date"] = date
                if symbol in active_weights:
                    row["reason_for_entry"] = f"{row.get('reason_for_entry', '')} Daily-throttled rebalance threshold met.".strip()
                active_rows[symbol] = row
                active_weights[symbol] = desired_weight
                active_scores[symbol] = desired_score
            else:
                removed = pd.Series({"date": date, "symbol": symbol, "target_weight": 0.0, "reason_for_exit": "Dropped from daily-throttled target set."})
                if symbol in desired_rows:
                    removed = desired_rows[symbol].copy()
                    removed["date"] = date
                    removed["target_weight"] = 0.0
                    removed["reason_for_exit"] = "Dropped from daily-throttled target set."
                removed_rows.append(removed)
                active_rows.pop(symbol, None)
                active_weights.pop(symbol, None)
                active_scores.pop(symbol, None)
        if not changed:
            continue
        rows = []
        for row in active_rows.values():
            updated = row.copy()
            updated["date"] = date
            rows.append(updated)
        rows.extend(removed_rows)
        if rows:
            out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True) if out else targets.iloc[0:0].copy()


def _leader_delayed_exit_decision(
    current: pd.Series,
    desired: pd.Series,
    current_weight: float,
    config: PortfolioConfig,
) -> dict[str, pd.Series]:
    """Delay a rank-only long exit when the incumbent still has leader quality.

    Feature flag: ``leader_delayed_exit_overlay`` defaults to false.
    Optional stability gate: ``leader_delayed_exit_stability_overlay`` defaults
    to false and requires shallow score decay versus the incumbent before the
    delay can fire. Reasonable ranges:
    - ``leader_delayed_exit_max_final_score_drop``: 3 to 10 points.
    - ``leader_delayed_exit_max_relative_strength_drop``: 2 to 8 points.

    Circuit breakers: the delayed exit is blocked when event or overnight-gap
    risk is elevated, momentum weakens, drawdown from the high grows too large,
    or the stability overlay detects meaningful score deterioration versus the
    incumbent. This keeps the rule focused on rank-noise exits rather than
    fighting genuine trend decay.
    """

    row = desired.copy()
    defaults = {
        "leader_delayed_exit_applied": False,
        "leader_delayed_exit_days": 0,
        "leader_delayed_exit_target_weight": float(row.get("target_weight", 0.0) or 0.0),
        "leader_delayed_exit_circuit_breaker": "feature_off",
        "leader_delayed_exit_log": "",
    }
    for column, value in defaults.items():
        if column not in row.index:
            row[column] = value
    if not bool(getattr(config, "leader_delayed_exit_overlay", False)):
        return {"row": row}
    row["leader_delayed_exit_circuit_breaker"] = "not_rank_exit"
    desired_weight = _safe_float(row.get("target_weight"), 0.0)
    if current_weight <= 0 or desired_weight > 0:
        row["leader_delayed_exit_target_weight"] = desired_weight
        return {"row": row}
    if _safe_float(row.get("signal"), 0.0) < 0:
        row["leader_delayed_exit_circuit_breaker"] = "negative_signal"
        return {"row": row}
    benchmark_risk_on = bool(row.get("benchmark_risk_on", True))
    if bool(getattr(config, "leader_delayed_exit_require_benchmark_risk_on", True)) and not benchmark_risk_on:
        row["leader_delayed_exit_circuit_breaker"] = "benchmark_not_risk_on"
        return {"row": row}
    theme_active = bool(row.get("theme_active", False))
    if bool(getattr(config, "leader_delayed_exit_require_theme_active", False)) and not theme_active:
        row["leader_delayed_exit_circuit_breaker"] = "theme_inactive"
        return {"row": row}
    event_risk = _safe_float(row.get("event_risk_score"), 0.0)
    if event_risk > float(getattr(config, "leader_delayed_exit_max_event_risk_score_circuit_breaker", 20.0)):
        row["leader_delayed_exit_circuit_breaker"] = "event_risk_high"
        return {"row": row}
    gap_risk = _safe_float(row.get("overnight_gap_risk_score"), 0.0)
    if gap_risk > float(getattr(config, "leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker", 30.0)):
        row["leader_delayed_exit_circuit_breaker"] = "gap_risk_high"
        return {"row": row}
    final_score = _safe_float(row.get("final_score"), 0.0)
    if final_score < float(getattr(config, "leader_delayed_exit_min_final_score", 70.0)):
        row["leader_delayed_exit_circuit_breaker"] = "final_score_low"
        return {"row": row}
    rs_score = _safe_float(row.get("relative_strength_score"), 0.0)
    if rs_score < float(getattr(config, "leader_delayed_exit_min_relative_strength_score", 90.0)):
        row["leader_delayed_exit_circuit_breaker"] = "relative_strength_low"
        return {"row": row}
    theme_score = _safe_float(row.get("theme_score"), 50.0)
    if theme_score < float(getattr(config, "leader_delayed_exit_min_theme_score", 65.0)):
        row["leader_delayed_exit_circuit_breaker"] = "theme_score_low"
        return {"row": row}
    mom_return = _safe_float(row.get("mom_return"), 0.0)
    if mom_return < float(getattr(config, "leader_delayed_exit_min_mom_return", 0.15)):
        row["leader_delayed_exit_circuit_breaker"] = "momentum_return_low"
        return {"row": row}
    distance_to_high = _safe_float(row.get("distance_to_high_252"), 0.0)
    drawdown_from_high = abs(min(distance_to_high, 0.0))
    if drawdown_from_high > float(getattr(config, "leader_delayed_exit_max_drawdown_from_high", 0.18)):
        row["leader_delayed_exit_circuit_breaker"] = "drawdown_from_high_high"
        return {"row": row}
    current_final_score = _safe_float(current.get("final_score"), final_score)
    current_rs_score = _safe_float(current.get("relative_strength_score"), rs_score)
    final_score_drop = max(0.0, current_final_score - final_score)
    rs_score_drop = max(0.0, current_rs_score - rs_score)
    if bool(getattr(config, "leader_delayed_exit_stability_overlay", False)):
        if final_score_drop > float(getattr(config, "leader_delayed_exit_max_final_score_drop", 6.0)):
            row["leader_delayed_exit_circuit_breaker"] = "final_score_drop_high"
            return {"row": row}
        if rs_score_drop > float(getattr(config, "leader_delayed_exit_max_relative_strength_drop", 5.0)):
            row["leader_delayed_exit_circuit_breaker"] = "relative_strength_drop_high"
            return {"row": row}
    prior_days = max(0, int(_safe_float(current.get("leader_delayed_exit_days"), 0.0)))
    max_days = max(0, int(getattr(config, "leader_delayed_exit_max_days", 3)))
    if prior_days >= max_days:
        row["leader_delayed_exit_circuit_breaker"] = "max_days_reached"
        return {"row": row}
    retain_fraction = float(np.clip(getattr(config, "leader_delayed_exit_retain_fraction", 0.85), 0.0, 1.0))
    retained_weight = abs(current_weight) * retain_fraction
    retained_weight = min(abs(current_weight), retained_weight, float(getattr(config, "leader_delayed_exit_max_weight", 0.08)))
    retained_weight = max(float(config.min_target_weight), retained_weight)
    if retained_weight <= 0:
        row["leader_delayed_exit_circuit_breaker"] = "retained_weight_zero"
        return {"row": row}
    row["target_weight"] = retained_weight
    row["leader_delayed_exit_applied"] = True
    row["leader_delayed_exit_days"] = prior_days + 1
    row["leader_delayed_exit_target_weight"] = retained_weight
    row["leader_delayed_exit_circuit_breaker"] = "applied"
    payload = {
        "benchmark_risk_on": benchmark_risk_on,
        "current_weight": float(current_weight),
        "drawdown_from_high": float(drawdown_from_high),
        "event_risk_score": event_risk,
        "final_score": final_score,
        "final_score_drop": float(final_score_drop),
        "gap_risk_score": gap_risk,
        "hold_days": int(row["leader_delayed_exit_days"]),
        "mom_return": mom_return,
        "relative_strength_score": rs_score,
        "relative_strength_drop": float(rs_score_drop),
        "retained_weight": float(retained_weight),
        "theme_active": theme_active,
        "theme_score": theme_score,
    }
    row["leader_delayed_exit_log"] = json.dumps(payload, sort_keys=True)
    row["reason_for_entry"] = (
        f"{str(row.get('reason_for_entry', '')).strip()} "
        f"Leader delayed-exit kept {retained_weight:.2%} after a rank-only exit while trend evidence stayed strong."
    ).strip()
    return {"row": row}


def _leader_hold_buffer_decision(
    current: pd.Series,
    desired: pd.Series,
    current_weight: float,
    config: PortfolioConfig,
) -> dict[str, pd.Series]:
    """Optionally convert a full long exit into a small residual hold."""

    row = desired.copy()
    defaults = {
        "leader_hold_buffer_applied": False,
        "leader_hold_buffer_days": 0,
        "leader_hold_buffer_target_weight": float(row.get("target_weight", 0.0) or 0.0),
        "leader_hold_buffer_circuit_breaker": "feature_off",
        "leader_hold_buffer_log": "",
    }
    for column, value in defaults.items():
        if column not in row.index:
            row[column] = value
    if not bool(getattr(config, "leader_hold_buffer_overlay", False)):
        row["leader_hold_buffer_circuit_breaker"] = "feature_off"
        return {"row": row}
    row["leader_hold_buffer_circuit_breaker"] = "not_long_exit"
    desired_weight = _safe_float(row.get("target_weight"), 0.0)
    if current_weight <= 0 or desired_weight > 0:
        row["leader_hold_buffer_target_weight"] = desired_weight
        return {"row": row}
    if _safe_float(row.get("signal"), 0.0) < 0:
        row["leader_hold_buffer_circuit_breaker"] = "negative_signal"
        return {"row": row}
    final_score = _safe_float(row.get("final_score"), 0.0)
    rs_score = _safe_float(row.get("relative_strength_score"), 0.0)
    theme_score = _safe_float(row.get("theme_score"), 50.0)
    mom_return = _safe_float(row.get("mom_return"), 0.0)
    distance_to_high = _safe_float(row.get("distance_to_high_252"), 0.0)
    drawdown_from_high = abs(min(distance_to_high, 0.0))
    event_risk = _safe_float(row.get("event_risk_score"), 0.0)
    gap_risk = _safe_float(row.get("overnight_gap_risk_score"), 0.0)
    benchmark_risk_on = bool(row.get("benchmark_risk_on", True))
    theme_active = bool(row.get("theme_active", False))
    if bool(getattr(config, "leader_hold_buffer_require_benchmark_risk_on", True)) and not benchmark_risk_on:
        row["leader_hold_buffer_circuit_breaker"] = "benchmark_not_risk_on"
        return {"row": row}
    if bool(getattr(config, "leader_hold_buffer_require_theme_active", False)) and not theme_active:
        row["leader_hold_buffer_circuit_breaker"] = "theme_inactive"
        return {"row": row}
    if event_risk > float(getattr(config, "leader_hold_buffer_max_event_risk_score_circuit_breaker", 25.0)):
        row["leader_hold_buffer_circuit_breaker"] = "event_risk_high"
        return {"row": row}
    if gap_risk > float(getattr(config, "leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker", 35.0)):
        row["leader_hold_buffer_circuit_breaker"] = "gap_risk_high"
        return {"row": row}
    if final_score < float(getattr(config, "leader_hold_buffer_min_final_score", 66.0)):
        row["leader_hold_buffer_circuit_breaker"] = "final_score_low"
        return {"row": row}
    if rs_score < float(getattr(config, "leader_hold_buffer_min_relative_strength_score", 82.0)):
        row["leader_hold_buffer_circuit_breaker"] = "relative_strength_low"
        return {"row": row}
    if theme_score < float(getattr(config, "leader_hold_buffer_min_theme_score", 60.0)):
        row["leader_hold_buffer_circuit_breaker"] = "theme_score_low"
        return {"row": row}
    if mom_return < float(getattr(config, "leader_hold_buffer_min_mom_return", 0.0)):
        row["leader_hold_buffer_circuit_breaker"] = "momentum_return_low"
        return {"row": row}
    if drawdown_from_high > float(getattr(config, "leader_hold_buffer_max_drawdown_from_high", 1.0)):
        row["leader_hold_buffer_circuit_breaker"] = "drawdown_from_high_high"
        return {"row": row}
    prior_days = max(0, int(_safe_float(current.get("leader_hold_buffer_days"), 0.0)))
    max_days = max(0, int(getattr(config, "leader_hold_buffer_max_days", 3)))
    if prior_days >= max_days:
        row["leader_hold_buffer_circuit_breaker"] = "max_days_reached"
        return {"row": row}
    if prior_days > 0:
        hold_weight = min(abs(current_weight), float(getattr(config, "leader_hold_buffer_max_weight", 0.02)))
    else:
        fraction = float(np.clip(getattr(config, "leader_hold_buffer_weight_fraction", 0.5), 0.0, 1.0))
        scaled_weight = abs(current_weight) * fraction if fraction > 0 else 0.0
        raw_hold = min(abs(current_weight), float(getattr(config, "leader_hold_buffer_max_weight", 0.02)), scaled_weight)
        hold_weight = min(abs(current_weight), max(float(config.min_target_weight), raw_hold))
    if hold_weight <= 0:
        row["leader_hold_buffer_circuit_breaker"] = "hold_weight_zero"
        return {"row": row}
    row["target_weight"] = hold_weight
    row["leader_hold_buffer_applied"] = True
    row["leader_hold_buffer_days"] = prior_days + 1
    row["leader_hold_buffer_target_weight"] = hold_weight
    row["leader_hold_buffer_circuit_breaker"] = "applied"
    payload = {
        "benchmark_risk_on": benchmark_risk_on,
        "current_weight": float(current_weight),
        "drawdown_from_high": float(drawdown_from_high),
        "desired_weight": float(desired_weight),
        "event_risk_score": event_risk,
        "final_score": final_score,
        "hold_days": int(row["leader_hold_buffer_days"]),
        "hold_weight": float(hold_weight),
        "mom_return": mom_return,
        "overnight_gap_risk_score": gap_risk,
        "relative_strength_score": rs_score,
        "theme_active": theme_active,
        "theme_score": theme_score,
    }
    row["leader_hold_buffer_log"] = json.dumps(payload, sort_keys=True)
    row["reason_for_entry"] = (
        f"{str(row.get('reason_for_entry', '')).strip()} "
        f"Leader hold buffer retained a {hold_weight:.2%} residual position after rank-only exit."
    ).strip()
    return {"row": row}


def _leader_persistence_decision(
    current: pd.Series,
    desired: pd.Series,
    current_weight: float,
    config: PortfolioConfig,
) -> dict[str, pd.Series]:
    """Retain part of a planned long reduction for still-strong same-theme leaders.

    Optional peer-strength reward: ``leader_persistence_peer_strength_reward_overlay``
    defaults to false.
    Reasonable ranges:
    - ``leader_persistence_peer_strength_reward_min_theme_peer_count``: 2 to 5.
    - ``leader_persistence_peer_strength_reward_min_theme_peer_share``: 0.45 to 0.80.
    - ``leader_persistence_peer_strength_reward_min_relative_strength_score``: 88 to 96.
    - ``leader_persistence_peer_strength_reward_retain_fraction_boost``: 0.05 to 0.35.
    - ``leader_persistence_peer_strength_reward_max_weight_bonus``: 0.0025 to 0.0150.
    - ``leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker``:
      0.08 to 0.18.

    Circuit breaker: the peer-strength reward only applies when same-theme breadth,
    RS leadership, and a controlled drawdown all remain intact; otherwise the base
    persistence rule is kept without extra retained weight.
    """

    row = desired.copy()
    defaults = {
        "leader_persistence_applied": False,
        "leader_persistence_buffered_weight": float(row.get("target_weight", 0.0) or 0.0),
        "leader_persistence_incremental_weight": 0.0,
        "leader_persistence_circuit_breaker": "feature_off",
        "leader_persistence_peer_strength_reward_applied": False,
        "leader_persistence_peer_strength_reward_circuit_breaker": "feature_off",
        "leader_persistence_log": "",
    }
    for column, value in defaults.items():
        if column not in row.index:
            row[column] = value
    if not bool(getattr(config, "leader_persistence_overlay", False)):
        row["leader_persistence_circuit_breaker"] = "feature_off"
        return {"row": row}
    row["leader_persistence_circuit_breaker"] = "not_reduced_long"
    desired_weight = _safe_float(row.get("target_weight"), 0.0)
    if current_weight <= 0 or desired_weight <= 0 or desired_weight >= current_weight:
        row["leader_persistence_buffered_weight"] = desired_weight
        return {"row": row}
    if _safe_float(row.get("signal"), 0.0) <= 0:
        row["leader_persistence_circuit_breaker"] = "signal_not_long"
        return {"row": row}
    if bool(getattr(config, "leader_persistence_require_benchmark_risk_on", True)) and not bool(row.get("benchmark_risk_on", True)):
        row["leader_persistence_circuit_breaker"] = "benchmark_not_risk_on"
        return {"row": row}
    if bool(getattr(config, "leader_persistence_require_theme_active", True)) and not bool(row.get("theme_active", False)):
        row["leader_persistence_circuit_breaker"] = "theme_inactive"
        return {"row": row}
    event_risk = _safe_float(row.get("event_risk_score"), 0.0)
    if event_risk > float(getattr(config, "leader_persistence_max_event_risk_score_circuit_breaker", 20.0)):
        row["leader_persistence_circuit_breaker"] = "event_risk_high"
        return {"row": row}
    gap_risk = _safe_float(row.get("overnight_gap_risk_score"), 0.0)
    if gap_risk > float(getattr(config, "leader_persistence_max_overnight_gap_risk_score_circuit_breaker", 30.0)):
        row["leader_persistence_circuit_breaker"] = "gap_risk_high"
        return {"row": row}
    final_score = _safe_float(row.get("final_score"), 0.0)
    if final_score < float(getattr(config, "leader_persistence_min_final_score", 68.0)):
        row["leader_persistence_circuit_breaker"] = "final_score_low"
        return {"row": row}
    rs_score = _safe_float(row.get("relative_strength_score"), 0.0)
    if rs_score < float(getattr(config, "leader_persistence_min_relative_strength_score", 88.0)):
        row["leader_persistence_circuit_breaker"] = "relative_strength_low"
        return {"row": row}
    theme_score = _safe_float(row.get("theme_score"), 50.0)
    if theme_score < float(getattr(config, "leader_persistence_min_theme_score", 62.0)):
        row["leader_persistence_circuit_breaker"] = "theme_score_low"
        return {"row": row}
    mom_return = _safe_float(row.get("mom_return"), 0.0)
    if mom_return < float(getattr(config, "leader_persistence_min_mom_return", 0.15)):
        row["leader_persistence_circuit_breaker"] = "momentum_return_low"
        return {"row": row}
    peer_count = max(0, int(_safe_float(row.get("leader_persistence_theme_peer_count"), 0.0)))
    if peer_count < max(1, int(getattr(config, "leader_persistence_min_theme_peer_count", 2))):
        row["leader_persistence_circuit_breaker"] = "theme_peer_count_low"
        return {"row": row}
    peer_share = _safe_float(row.get("leader_persistence_theme_peer_share"), 0.0)
    if peer_share < float(getattr(config, "leader_persistence_min_theme_peer_share", 0.40)):
        row["leader_persistence_circuit_breaker"] = "theme_peer_share_low"
        return {"row": row}
    distance_to_high = _safe_float(row.get("distance_to_high_252"), 0.0)
    drawdown_from_high = abs(min(distance_to_high, 0.0))
    if bool(getattr(config, "leader_persistence_stability_overlay", False)):
        desired_weight_fraction = desired_weight / current_weight if current_weight > 0 else 0.0
        min_desired_weight_fraction = float(np.clip(getattr(config, "leader_persistence_min_desired_weight_fraction", 0.40), 0.0, 1.0))
        if desired_weight_fraction < min_desired_weight_fraction:
            row["leader_persistence_circuit_breaker"] = "desired_weight_fraction_low"
            return {"row": row}
        current_final_score = _safe_float(current.get("final_score"), final_score)
        current_rs_score = _safe_float(current.get("relative_strength_score"), rs_score)
        final_score_drop = max(0.0, current_final_score - final_score)
        max_final_score_drop = max(0.0, float(getattr(config, "leader_persistence_max_final_score_drop", 6.0)))
        if final_score_drop > max_final_score_drop:
            row["leader_persistence_circuit_breaker"] = "final_score_drop_too_large"
            return {"row": row}
        rs_score_drop = max(0.0, current_rs_score - rs_score)
        max_rs_score_drop = max(0.0, float(getattr(config, "leader_persistence_max_relative_strength_drop", 5.0)))
        if rs_score_drop > max_rs_score_drop:
            row["leader_persistence_circuit_breaker"] = "relative_strength_drop_too_large"
            return {"row": row}
    cut = max(0.0, current_weight - desired_weight)
    if cut <= 0:
        row["leader_persistence_circuit_breaker"] = "no_cut"
        return {"row": row}
    retain_fraction = float(np.clip(getattr(config, "leader_persistence_retain_fraction_of_cut", 0.35), 0.0, 1.0))
    bonus_cap = max(0.0, float(getattr(config, "leader_persistence_max_weight_bonus", 0.015)))
    peer_reward_applied = False
    peer_reward_circuit_breaker = "feature_off"
    if bool(getattr(config, "leader_persistence_peer_strength_reward_overlay", False)):
        peer_reward_circuit_breaker = "peer_strength_low"
        reward_min_peer_count = max(
            1,
            int(getattr(config, "leader_persistence_peer_strength_reward_min_theme_peer_count", 3)),
        )
        reward_min_peer_share = float(
            np.clip(getattr(config, "leader_persistence_peer_strength_reward_min_theme_peer_share", 0.55), 0.0, 1.0)
        )
        reward_min_rs = float(getattr(config, "leader_persistence_peer_strength_reward_min_relative_strength_score", 92.0))
        reward_fraction_boost = float(
            np.clip(getattr(config, "leader_persistence_peer_strength_reward_retain_fraction_boost", 0.20), 0.0, 1.0)
        )
        reward_bonus_cap = max(
            0.0,
            float(getattr(config, "leader_persistence_peer_strength_reward_max_weight_bonus", 0.0075)),
        )
        reward_max_drawdown = max(
            0.0,
            float(
                getattr(
                    config,
                    "leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker",
                    0.14,
                )
            ),
        )
        if peer_count < reward_min_peer_count:
            peer_reward_circuit_breaker = "peer_count_low"
        elif peer_share < reward_min_peer_share:
            peer_reward_circuit_breaker = "peer_share_low"
        elif rs_score < reward_min_rs:
            peer_reward_circuit_breaker = "relative_strength_low"
        elif drawdown_from_high > reward_max_drawdown:
            peer_reward_circuit_breaker = "drawdown_from_high_high"
        elif reward_fraction_boost <= 0 or reward_bonus_cap <= 0:
            peer_reward_circuit_breaker = "reward_zero"
        else:
            retain_fraction = min(1.0, retain_fraction + reward_fraction_boost)
            bonus_cap += reward_bonus_cap
            peer_reward_applied = True
            peer_reward_circuit_breaker = "applied"
    incremental = min(cut * retain_fraction, bonus_cap)
    if incremental <= 0:
        row["leader_persistence_circuit_breaker"] = "buffer_zero"
        row["leader_persistence_peer_strength_reward_applied"] = peer_reward_applied
        row["leader_persistence_peer_strength_reward_circuit_breaker"] = peer_reward_circuit_breaker
        return {"row": row}
    buffered_weight = min(current_weight, desired_weight + incremental, float(config.max_position_weight))
    incremental = max(0.0, buffered_weight - desired_weight)
    if incremental <= 0:
        row["leader_persistence_circuit_breaker"] = "buffer_zero"
        row["leader_persistence_peer_strength_reward_applied"] = peer_reward_applied
        row["leader_persistence_peer_strength_reward_circuit_breaker"] = peer_reward_circuit_breaker
        return {"row": row}
    row["target_weight"] = buffered_weight
    row["leader_persistence_applied"] = True
    row["leader_persistence_buffered_weight"] = buffered_weight
    row["leader_persistence_incremental_weight"] = incremental
    row["leader_persistence_circuit_breaker"] = "applied"
    row["leader_persistence_peer_strength_reward_applied"] = peer_reward_applied
    row["leader_persistence_peer_strength_reward_circuit_breaker"] = peer_reward_circuit_breaker
    row["leader_persistence_log"] = json.dumps(
        {
            "benchmark_risk_on": bool(row.get("benchmark_risk_on", True)),
            "buffered_weight": float(buffered_weight),
            "current_weight": float(current_weight),
            "drawdown_from_high": float(drawdown_from_high),
            "desired_weight": float(desired_weight),
            "event_risk_score": event_risk,
            "final_score": final_score,
            "incremental_weight": float(incremental),
            "max_weight_bonus": bonus_cap,
            "mom_return": mom_return,
            "overnight_gap_risk_score": gap_risk,
            "peer_strength_reward_applied": peer_reward_applied,
            "peer_strength_reward_circuit_breaker": peer_reward_circuit_breaker,
            "peer_strength_reward_max_drawdown_from_high_circuit_breaker": float(
                getattr(
                    config,
                    "leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker",
                    0.14,
                )
            ),
            "peer_strength_reward_max_weight_bonus": float(
                getattr(config, "leader_persistence_peer_strength_reward_max_weight_bonus", 0.0075)
            ),
            "peer_strength_reward_min_relative_strength_score": float(
                getattr(config, "leader_persistence_peer_strength_reward_min_relative_strength_score", 92.0)
            ),
            "peer_strength_reward_min_theme_peer_count": int(
                getattr(config, "leader_persistence_peer_strength_reward_min_theme_peer_count", 3)
            ),
            "peer_strength_reward_min_theme_peer_share": float(
                getattr(config, "leader_persistence_peer_strength_reward_min_theme_peer_share", 0.55)
            ),
            "peer_strength_reward_retain_fraction_boost": float(
                getattr(config, "leader_persistence_peer_strength_reward_retain_fraction_boost", 0.20)
            ),
            "stability_overlay": bool(getattr(config, "leader_persistence_stability_overlay", False)),
            "relative_strength_score": rs_score,
            "retain_fraction_of_cut": retain_fraction,
            "desired_weight_fraction": float(desired_weight / current_weight) if current_weight > 0 else 0.0,
            "final_score_drop": max(0.0, _safe_float(current.get("final_score"), final_score) - final_score),
            "max_final_score_drop": float(getattr(config, "leader_persistence_max_final_score_drop", 6.0)),
            "relative_strength_score_drop": max(0.0, _safe_float(current.get("relative_strength_score"), rs_score) - rs_score),
            "max_relative_strength_score_drop": float(getattr(config, "leader_persistence_max_relative_strength_drop", 5.0)),
            "theme_peer_count": peer_count,
            "theme_peer_share": peer_share,
            "theme_score": theme_score,
        },
        sort_keys=True,
    )
    row["reason_for_entry"] = (
        f"{str(row.get('reason_for_entry', '')).strip()} "
        f"Leader persistence retained {incremental:.2%} of weight on same-theme leadership strength."
    ).strip()
    return {"row": row}


def _event_context_changed(current: pd.Series, desired: pd.Series) -> bool:
    """Treat event-risk changes as rebalance triggers even when target weights are stable."""

    current_risk = _safe_float(current.get("event_risk_score"), 0.0)
    desired_risk = _safe_float(desired.get("event_risk_score"), 0.0)
    if abs(desired_risk - current_risk) >= 5.0:
        return True
    current_days = _safe_float(current.get("days_to_earnings"), float("nan"))
    desired_days = _safe_float(desired.get("days_to_earnings"), float("nan"))
    if np.isfinite(desired_days) and abs(desired_days) <= 7:
        if not np.isfinite(current_days) or int(current_days) != int(desired_days):
            return True
    return False


def _safe_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default
