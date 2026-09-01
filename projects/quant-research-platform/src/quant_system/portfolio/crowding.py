"""Theme, sector, and industry crowding controls."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from quant_system.config import RegimeConfig
from quant_system.universe.thematic import load_thematic_baskets

LOGGER = logging.getLogger(__name__)


def apply_crowding_risk_to_targets(targets: pd.DataFrame, config: RegimeConfig, theme_baskets_path: str | Path) -> pd.DataFrame:
    """Scale targets when one sector/theme dominates gross exposure.

    The overlay uses target-date metadata only. If sector or theme information is
    unavailable, it leaves weights untouched rather than pretending UNKNOWN is a
    real industry.
    """

    if targets.empty or not config.enabled or not config.crowding_overlay:
        return targets
    out = _attach_theme_memberships(targets.copy(), theme_baskets_path)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["crowding_multiplier"] = 1.0
    out["crowding_risk_score"] = 0.0
    out["theme_gross_exposure"] = 0.0
    out["sector_gross_exposure"] = 0.0
    out["industry_gross_exposure"] = 0.0
    out["crowding_relief_applied"] = False
    out["crowding_relief_weight_delta"] = 0.0
    out["crowding_relief_bucket"] = ""
    out["crowding_relief_circuit_breaker"] = "feature_off"
    out["crowding_relief_log"] = ""
    for date, day in out.groupby("date"):
        index = day.index
        multipliers = pd.Series(1.0, index=index)
        scores = pd.Series(0.0, index=index)
        for column, default_cap, cap_map, exposure_column in (
            ("primary_theme", float(config.max_theme_gross_exposure), config.theme_gross_exposure_caps, "theme_gross_exposure"),
            ("sector", float(config.max_sector_gross_exposure), config.sector_gross_exposure_caps, "sector_gross_exposure"),
            ("industry", float(config.max_industry_gross_exposure), config.industry_gross_exposure_caps, "industry_gross_exposure"),
        ):
            if column not in day.columns:
                continue
            labels = day[column].fillna("UNKNOWN").astype(str)
            usable = labels.ne("") & labels.ne("UNKNOWN") & labels.ne("unclassified")
            if not usable.any():
                continue
            gross = day.loc[usable].groupby(labels.loc[usable])["target_weight"].apply(lambda s: s.abs().sum())
            for label, group_gross in gross.items():
                cap = _label_cap(str(label), default_cap, cap_map)
                if cap <= 0:
                    continue
                members = index[labels.eq(label)]
                out.loc[members, exposure_column] = float(group_gross)
                if group_gross > cap:
                    multiplier = max(float(config.crowding_reduce_multiplier_floor), cap / float(group_gross))
                    column_multipliers = pd.Series(multiplier, index=members, dtype=float)
                    if column == "primary_theme":
                        relief = _apply_theme_leader_relief(day.loc[members], column_multipliers, str(label), float(cap), config)
                        column_multipliers = relief["crowding_multiplier"]
                        for relief_column in (
                            "crowding_relief_applied",
                            "crowding_relief_weight_delta",
                            "crowding_relief_bucket",
                            "crowding_relief_circuit_breaker",
                            "crowding_relief_log",
                        ):
                            out.loc[members, relief_column] = relief[relief_column]
                    multipliers.loc[members] = multipliers.loc[members].combine(column_multipliers, min)
                    scores.loc[members] = scores.loc[members].clip(lower=min(100.0, (float(group_gross) / cap - 1.0) * 100.0))
        out.loc[index, "crowding_multiplier"] = multipliers
        out.loc[index, "crowding_risk_score"] = scores
    out["target_weight"] = out["target_weight"].astype(float) * out["crowding_multiplier"].astype(float)
    if "reason_for_entry" in out.columns:
        mask = out["crowding_multiplier"].astype(float) < 1.0
        out.loc[mask, "reason_for_entry"] = (
            out.loc[mask, "reason_for_entry"].fillna("").astype(str)
            + " Crowding budget reduced exposure by theme/sector/industry concentration."
        ).str.strip()
        relief_mask = out["crowding_relief_applied"].fillna(False).astype(bool)
        out.loc[relief_mask, "reason_for_entry"] = (
            out.loc[relief_mask, "reason_for_entry"].fillna("").astype(str)
            + " Leader crowding relief preserved part of the strongest theme exposure under a capped restore budget."
        ).str.strip()
    return out


def _apply_theme_leader_relief(
    frame: pd.DataFrame,
    base_multipliers: pd.Series,
    bucket: str,
    cap: float,
    config: RegimeConfig,
) -> pd.DataFrame:
    """Restore part of a crowded theme haircut to the strongest leaders.

    Feature flag: ``leader_crowding_relief_overlay`` defaults to false.
    Optional peer exception flag:
    ``leader_crowding_relief_same_theme_peer_exception_overlay`` defaults to
    false and only operates when the benchmark is not risk-on. Reasonable
    ranges:
    - ``leader_crowding_relief_same_theme_peer_min_count``: 2 to 5.
    - ``leader_crowding_relief_same_theme_peer_min_active_share``: 0.35 to 0.75.
    - ``leader_crowding_relief_same_theme_peer_min_avg_mom_return``: 0.05 to 0.30.
    Optional peer restore flag:
    ``leader_crowding_relief_same_theme_peer_restore_overlay`` defaults to
    false and only restores crowding weight when the crowded theme still has
    broad same-theme leadership. Reasonable ranges:
    - ``leader_crowding_relief_same_theme_peer_restore_min_count``: 2 to 5.
    - ``leader_crowding_relief_same_theme_peer_restore_min_active_share``: 0.40 to 0.75.
    - ``leader_crowding_relief_same_theme_peer_restore_min_avg_mom_return``: 0.05 to 0.25.
    Optional controlled-pullback priority flag:
    ``leader_crowding_relief_controlled_pullback_overlay`` defaults to false.
    It only changes which already-eligible crowded leaders receive restore
    weight. Reasonable ranges:
    - ``leader_crowding_relief_controlled_pullback_min_final_score``: 55 to 75.
    - ``leader_crowding_relief_controlled_pullback_min_distance_from_high``: 0.03 to 0.10.
    - ``leader_crowding_relief_controlled_pullback_max_distance_from_high``: 0.10 to 0.25.
    - ``leader_crowding_relief_controlled_pullback_max_above_ma20_pct``: 0.02 to 0.10.
    - ``leader_crowding_relief_controlled_pullback_max_mom_return``: 0.00 to 0.20.

    Circuit breakers: total theme caps remain unchanged, row-level event/gap
    risk limits still apply, the peer exception only activates when the
    crowded bucket itself still has broad, strong same-theme leadership, and
    the controlled-pullback priority only applies inside an explicit drawdown
    and extension window.
    """

    result = pd.DataFrame(index=frame.index)
    result["crowding_multiplier"] = base_multipliers.astype(float)
    result["crowding_relief_applied"] = False
    result["crowding_relief_weight_delta"] = 0.0
    result["crowding_relief_bucket"] = ""
    result["crowding_relief_circuit_breaker"] = "feature_off"
    result["crowding_relief_log"] = ""
    if not bool(config.leader_crowding_relief_overlay):
        return result

    result["crowding_relief_bucket"] = str(bucket)
    long_mask = frame["target_weight"].astype(float) > 0
    benchmark_risk_on = frame.get("benchmark_risk_on", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    final_score = pd.to_numeric(frame["final_score"], errors="coerce").fillna(0.0) if "final_score" in frame.columns else pd.Series(0.0, index=frame.index)
    rs_score = pd.to_numeric(frame.get("relative_strength_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    theme_score = pd.to_numeric(frame.get("theme_score", pd.Series(50.0, index=frame.index)), errors="coerce").fillna(50.0)
    event_risk = pd.to_numeric(frame.get("event_risk_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(frame.get("overnight_gap_risk_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    mom_return = pd.to_numeric(frame.get("mom_return", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    drawdown_from_high = pd.to_numeric(
        frame.get("distance_to_high_252", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0).abs()
    above_ma20 = pd.to_numeric(
        frame.get("retest_distance_ma_pct", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    low_risk_exception = (
        base_multipliers.astype(float).ge(float(config.leader_crowding_relief_min_base_multiplier_circuit_breaker))
        & final_score.ge(float(config.leader_crowding_relief_min_final_score_exception))
        & event_risk.le(float(config.leader_crowding_relief_max_event_risk_score_circuit_breaker))
        & gap_risk.le(float(config.leader_crowding_relief_max_overnight_gap_risk_score_circuit_breaker))
    )
    same_theme_peer_exception_enabled = bool(config.leader_crowding_relief_same_theme_peer_exception_overlay)
    same_theme_peer_active = (
        long_mask
        & rs_score.ge(float(config.leader_crowding_relief_min_relative_strength_score))
        & theme_score.ge(float(config.leader_crowding_relief_min_theme_score))
    )
    same_theme_peer_count = int(long_mask.sum())
    same_theme_peer_active_count = int(same_theme_peer_active.sum())
    same_theme_peer_active_share = (
        same_theme_peer_active_count / same_theme_peer_count if same_theme_peer_count > 0 else 0.0
    )
    same_theme_peer_avg_mom_return = (
        float(mom_return.loc[same_theme_peer_active].mean()) if same_theme_peer_active_count > 0 else 0.0
    )
    same_theme_peer_exception_qualified = (
        same_theme_peer_exception_enabled
        and same_theme_peer_count >= int(config.leader_crowding_relief_same_theme_peer_min_count)
        and same_theme_peer_active_share >= float(config.leader_crowding_relief_same_theme_peer_min_active_share)
        and same_theme_peer_avg_mom_return >= float(config.leader_crowding_relief_same_theme_peer_min_avg_mom_return)
    )
    same_theme_peer_restore_enabled = bool(config.leader_crowding_relief_same_theme_peer_restore_overlay)
    same_theme_peer_restore_qualified = (
        same_theme_peer_restore_enabled
        and same_theme_peer_count >= int(config.leader_crowding_relief_same_theme_peer_restore_min_count)
        and same_theme_peer_active_share >= float(config.leader_crowding_relief_same_theme_peer_restore_min_active_share)
        and same_theme_peer_avg_mom_return >= float(config.leader_crowding_relief_same_theme_peer_restore_min_avg_mom_return)
    )
    same_theme_peer_exception = pd.Series(False, index=frame.index)
    if same_theme_peer_exception_qualified:
        same_theme_peer_exception = (~benchmark_risk_on) & low_risk_exception
        if bool(same_theme_peer_exception.any()):
            LOGGER.info(
                "leader_crowding_relief_same_theme_peer_exception_eligible",
                extra={
                    "bucket": str(bucket),
                    "cap": float(cap),
                    "peer_count": same_theme_peer_count,
                    "peer_active_count": same_theme_peer_active_count,
                    "peer_active_share": float(same_theme_peer_active_share),
                    "peer_avg_mom_return": float(same_theme_peer_avg_mom_return),
                },
            )
    if same_theme_peer_restore_enabled and not same_theme_peer_restore_qualified:
        result["crowding_relief_circuit_breaker"] = "same_theme_peer_restore_blocked"
        return result
    relief_context = pd.Series(True, index=frame.index)
    if bool(config.leader_crowding_relief_require_benchmark_risk_on):
        relief_context = benchmark_risk_on.copy()
        if bool(config.leader_crowding_relief_allow_non_risk_on_exception):
            relief_context |= (~benchmark_risk_on) & low_risk_exception
        if same_theme_peer_exception_enabled:
            relief_context |= same_theme_peer_exception
        if not bool(relief_context.any()):
            if same_theme_peer_exception_enabled and not bool(benchmark_risk_on.any()):
                result["crowding_relief_circuit_breaker"] = "same_theme_peer_exception_blocked"
            else:
                result["crowding_relief_circuit_breaker"] = (
                    "non_risk_on_exception_blocked"
                    if bool(config.leader_crowding_relief_allow_non_risk_on_exception)
                    else "benchmark_not_risk_on"
                )
            return result

    eligible = (
        long_mask
        & relief_context
        & rs_score.ge(float(config.leader_crowding_relief_min_relative_strength_score))
        & theme_score.ge(float(config.leader_crowding_relief_min_theme_score))
    )
    if same_theme_peer_restore_enabled:
        eligible &= same_theme_peer_active
    if not bool(eligible.any()):
        if bool(config.leader_crowding_relief_require_benchmark_risk_on) and not bool(benchmark_risk_on.any()):
            if same_theme_peer_exception_enabled and not bool(same_theme_peer_exception.any()):
                result["crowding_relief_circuit_breaker"] = "same_theme_peer_exception_blocked"
            else:
                result["crowding_relief_circuit_breaker"] = (
                    "non_risk_on_exception_blocked"
                    if bool(config.leader_crowding_relief_allow_non_risk_on_exception)
                    else "benchmark_not_risk_on"
                )
        elif same_theme_peer_restore_enabled:
            result["crowding_relief_circuit_breaker"] = "same_theme_peer_restore_no_eligible_leaders"
        else:
            result["crowding_relief_circuit_breaker"] = "no_eligible_leaders"
        return result

    controlled_pullback_overlay = bool(config.leader_crowding_relief_controlled_pullback_overlay)
    pullback_min_final_score = float(config.leader_crowding_relief_controlled_pullback_min_final_score)
    pullback_min_distance = max(float(config.leader_crowding_relief_controlled_pullback_min_distance_from_high), 0.0)
    pullback_max_distance = max(float(config.leader_crowding_relief_controlled_pullback_max_distance_from_high), pullback_min_distance)
    pullback_max_above_ma20 = max(float(config.leader_crowding_relief_controlled_pullback_max_above_ma20_pct), 0.0)
    pullback_max_mom_return = float(config.leader_crowding_relief_controlled_pullback_max_mom_return)
    controlled_pullback_eligible = (
        eligible
        & final_score.ge(pullback_min_final_score)
        & drawdown_from_high.ge(pullback_min_distance)
        & drawdown_from_high.le(pullback_max_distance)
        & above_ma20.le(pullback_max_above_ma20)
        & mom_return.le(pullback_max_mom_return)
        & event_risk.le(float(config.leader_crowding_relief_max_event_risk_score_circuit_breaker))
        & gap_risk.le(float(config.leader_crowding_relief_max_overnight_gap_risk_score_circuit_breaker))
    )

    max_names = max(0, int(config.leader_crowding_relief_max_names_per_bucket))
    if max_names <= 0:
        result["crowding_relief_circuit_breaker"] = "max_names_zero"
        return result

    ranked = frame.loc[eligible].assign(
        _controlled_pullback_priority=controlled_pullback_eligible.loc[eligible].astype(int) if controlled_pullback_overlay else 0,
        _rs_rank=rs_score.loc[eligible],
        _final_score_rank=final_score.loc[eligible],
    )
    sort_columns = ["_rs_rank", "_final_score_rank"]
    if controlled_pullback_overlay:
        sort_columns = ["_controlled_pullback_priority", *sort_columns]
    selected = ranked.sort_values(sort_columns, ascending=False).head(max_names).index
    donors = frame.index[long_mask & ~frame.index.isin(selected)]
    if len(selected) == 0:
        result["crowding_relief_circuit_breaker"] = "selection_empty"
        return result
    if len(donors) == 0:
        result["crowding_relief_circuit_breaker"] = "no_donors"
        return result

    original_abs = frame["target_weight"].astype(float).abs()
    base_abs = original_abs * base_multipliers.astype(float)
    haircut = (original_abs - base_abs).clip(lower=0.0)
    restore = haircut.loc[selected] * float(config.leader_crowding_relief_weight)
    restore = restore.clip(lower=0.0, upper=float(config.leader_crowding_relief_max_weight_per_symbol))
    total_restore = min(float(restore.sum()), float(config.leader_crowding_relief_max_bucket_weight_restore))
    if total_restore <= 0:
        result["crowding_relief_circuit_breaker"] = "restore_budget_zero"
        return result

    if float(restore.sum()) > total_restore:
        scale = total_restore / float(restore.sum())
        restore *= scale

    donor_capacity = base_abs.loc[donors].clip(lower=0.0)
    available = float(donor_capacity.sum())
    if available <= 0:
        result["crowding_relief_circuit_breaker"] = "donor_capacity_zero"
        return result
    if total_restore > available:
        scale = available / total_restore
        restore *= scale
        total_restore = float(restore.sum())
    if total_restore <= 0:
        result["crowding_relief_circuit_breaker"] = "scaled_restore_zero"
        return result

    donor_cut = donor_capacity / available * total_restore
    adjusted_abs = base_abs.copy()
    adjusted_abs.loc[selected] = adjusted_abs.loc[selected] + restore
    adjusted_abs.loc[donors] = adjusted_abs.loc[donors] - donor_cut
    adjusted_abs = adjusted_abs.clip(lower=0.0)
    base_multiplier_series = pd.to_numeric(base_multipliers, errors="coerce").fillna(1.0).astype(float)
    denominator = original_abs.where(original_abs.ne(0.0))
    crowding_multiplier = adjusted_abs.div(denominator)
    crowding_multiplier = crowding_multiplier.where(denominator.notna(), base_multiplier_series)
    result["crowding_multiplier"] = crowding_multiplier.astype(float)
    weight_delta = adjusted_abs - base_abs
    result["crowding_relief_weight_delta"] = weight_delta.where(long_mask, 0.0).astype(float)
    result["crowding_relief_applied"] = result["crowding_relief_weight_delta"].ne(0.0)
    result["crowding_relief_circuit_breaker"] = "applied"
    activation_mode = "benchmark_risk_on"
    if not bool(benchmark_risk_on.any()):
        activation_mode = (
            "same_theme_peer_exception"
            if bool(same_theme_peer_exception.any())
            else "non_risk_on_exception"
        )
    if bool(result["crowding_relief_applied"].any()):
        LOGGER.info(
            "leader_crowding_relief_applied",
            extra={
                "bucket": str(bucket),
                "cap": float(cap),
                "activation_mode": activation_mode,
                "peer_count": same_theme_peer_count,
                "peer_active_count": same_theme_peer_active_count,
                "peer_active_share": float(same_theme_peer_active_share),
                "peer_avg_mom_return": float(same_theme_peer_avg_mom_return),
                "peer_restore_enabled": bool(same_theme_peer_restore_enabled),
                "peer_restore_qualified": bool(same_theme_peer_restore_qualified),
                "controlled_pullback_overlay": bool(controlled_pullback_overlay),
                "controlled_pullback_eligible_count": int(controlled_pullback_eligible.sum()),
                "controlled_pullback_selected_count": int(controlled_pullback_eligible.loc[selected].sum()),
                "selected_count": int(len(selected)),
                "restore_budget": float(total_restore),
            },
        )
    for idx in result.index[result["crowding_relief_applied"]]:
        payload = {
            "activation_mode": activation_mode,
            "base_multiplier": float(base_multipliers.loc[idx]),
            "bucket": str(bucket),
            "cap": float(cap),
            "benchmark_risk_on": bool(benchmark_risk_on.loc[idx]),
            "same_theme_peer_active_count": same_theme_peer_active_count,
            "same_theme_peer_active_share": float(same_theme_peer_active_share),
            "same_theme_peer_avg_mom_return": float(same_theme_peer_avg_mom_return),
            "same_theme_peer_exception_qualified": bool(same_theme_peer_exception_qualified),
            "same_theme_peer_restore_enabled": bool(same_theme_peer_restore_enabled),
            "same_theme_peer_restore_qualified": bool(same_theme_peer_restore_qualified),
            "controlled_pullback_overlay": bool(controlled_pullback_overlay),
            "controlled_pullback_eligible": bool(controlled_pullback_eligible.loc[idx]),
            "distance_to_high_252_abs": float(drawdown_from_high.loc[idx]),
            "retest_distance_ma_pct": float(above_ma20.loc[idx]),
            "mom_return": float(mom_return.loc[idx]),
            "role": "leader" if idx in selected else "donor",
            "restore_delta": float(result.at[idx, "crowding_relief_weight_delta"]),
            "event_risk_score": float(event_risk.loc[idx]),
            "final_score": float(final_score.loc[idx]),
            "overnight_gap_risk_score": float(gap_risk.loc[idx]),
            "rs_score": float(rs_score.loc[idx]),
            "theme_score": float(theme_score.loc[idx]),
            "circuit_breaker": "applied",
        }
        result.at[idx, "crowding_relief_log"] = json.dumps(payload, sort_keys=True)
    return result


def _label_cap(label: str, default_cap: float, cap_map: dict[str, float] | None) -> float:
    """Return a per-label gross cap, falling back to the global cap."""

    if not cap_map:
        return float(default_cap)
    normalized = str(label).lower()
    for key, value in cap_map.items():
        if str(key).lower() == normalized:
            return float(value)
    return float(default_cap)


def _attach_theme_memberships(targets: pd.DataFrame, theme_baskets_path: str | Path) -> pd.DataFrame:
    baskets = load_thematic_baskets(theme_baskets_path)
    if not baskets:
        targets["theme_memberships"] = ""
        targets["primary_theme"] = "unclassified"
        return targets
    membership: dict[str, list[str]] = {}
    for theme, symbols in baskets.items():
        for symbol in symbols:
            membership.setdefault(symbol.upper(), []).append(theme)
    targets["theme_memberships"] = targets["symbol"].astype(str).str.upper().map(lambda symbol: ",".join(membership.get(symbol, [])))
    targets["primary_theme"] = targets["theme_memberships"].str.split(",").str[0].replace("", "unclassified")
    return targets
