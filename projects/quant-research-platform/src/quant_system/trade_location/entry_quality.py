"""Trade-location scoring.

The alpha layer answers "is this a strong asset?".  This module answers the
separate question "is the current close a good place to initiate or add?".
It intentionally uses only same-close information; the backtest engine still
executes target weights on the next session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.config import PortfolioConfig


def attach_trade_location_scores(frame: pd.DataFrame, config: PortfolioConfig, *, positive: bool) -> pd.DataFrame:
    """Attach entry quality, reward/risk, and sizing multipliers.

    The overlay is conservative by design: high-alpha names that are too far
    above short-term trend are marked as hold/avoid instead of being treated as
    fresh add candidates.
    """

    out = frame.copy()
    if out.empty:
        return out
    if not bool(getattr(config, "trade_location_overlay", False)) or not positive:
        out["entry_quality_score"] = 100.0
        out["reward_risk_estimate"] = np.nan
        out["trade_location_type"] = "LOCATION_LAYER_OFF"
        out["trade_location_multiplier"] = 1.0
        out["trade_location_note"] = ""
        return out

    alpha = _num(out, "final_score", 50.0)
    rs = _num(out, "relative_strength_score", 50.0)
    theme = _num(out, "theme_score", 50.0)
    gap_risk = _num(out, "overnight_gap_risk_score", 0.0)
    event_risk = _num(out, "event_risk_score", 0.0)
    rsi = _num(out, "rsi_14", 50.0)
    zscore = _num(out, "zscore_20", 0.0)
    volume = _num(out, "mom_volume_expansion", 1.0)
    ret_5d = _num(out, "ret_5d", np.nan)
    ret_20d = _num(out, "ret_20d", np.nan)
    adj_close = _num(out, "adj_close", np.nan).replace(0, np.nan)
    ma20 = _num(out, "ma_20", np.nan).replace(0, np.nan)
    atr = _num(out, "atr_14", np.nan)

    fallback_above_ma20 = _num(out, "retest_distance_ma_pct", np.nan)
    above_ma20 = (adj_close / ma20 - 1.0).replace([np.inf, -np.inf], np.nan).fillna(fallback_above_ma20).fillna(0.0)
    prior_high_gap = _num(out, "distance_to_prior_high_252", _num(out, "mom_distance_high", 0.0)).fillna(0.0)
    drawdown_from_high = (-prior_high_gap.clip(upper=0.0)).fillna(0.0)
    atr_pct = (atr / adj_close).replace([np.inf, -np.inf], np.nan).clip(lower=0.005, upper=0.18).fillna(0.055)

    min_pullback = float(getattr(config, "trade_location_pullback_min_drawdown", 0.02))
    max_pullback = float(getattr(config, "trade_location_pullback_max_drawdown", 0.16))
    ideal_above_ma20 = float(getattr(config, "trade_location_ideal_above_ma20_pct", 0.03))
    chase_max = float(getattr(config, "trade_location_chase_max_above_ma20_pct", 0.12))
    gap_reduce = float(getattr(config, "trade_location_gap_risk_reduce", 65.0))
    gap_block = float(getattr(config, "trade_location_gap_risk_block", 90.0))
    event_block = float(getattr(config, "trade_location_event_risk_block", 70.0))
    min_quality = float(getattr(config, "trade_location_min_entry_quality", 55.0))
    min_rr = float(getattr(config, "trade_location_min_reward_risk", 1.5))

    controlled_pullback = (
        drawdown_from_high.between(min_pullback, max_pullback, inclusive="both")
        & above_ma20.le(chase_max)
        & volume.le(1.25)
        & alpha.ge(58.0)
        & rs.ge(60.0)
    )
    reclaim = (
        drawdown_from_high.between(min_pullback, max_pullback, inclusive="both")
        & above_ma20.between(-0.01, chase_max, inclusive="both")
        & ret_5d.fillna(0.0).gt(0.0)
        & alpha.ge(58.0)
        & rs.ge(60.0)
    )
    extended = (
        above_ma20.gt(chase_max)
        | prior_high_gap.gt(0.10)
        | rsi.gt(76.0)
        | zscore.gt(2.25)
    )
    clean_continuation = (
        ~extended
        & above_ma20.le(max(chase_max, ideal_above_ma20))
        & alpha.ge(62.0)
        & rs.ge(65.0)
        & theme.ge(55.0)
    )

    stop_distance = (2.0 * atr_pct).clip(lower=0.04, upper=0.18)
    pullback_upside = drawdown_from_high + 0.05
    continuation_upside = (0.06 + _num(out, "mom_return", 0.0).clip(lower=0.0, upper=1.0) * 0.05).clip(upper=0.14)
    expected_upside = continuation_upside.where(~controlled_pullback & ~reclaim, pullback_upside.clip(lower=0.06, upper=0.22))
    expected_upside = expected_upside.where(~extended, expected_upside * 0.55)
    expected_upside = expected_upside.where(gap_risk.lt(gap_reduce), expected_upside * 0.75)
    reward_risk = (expected_upside / stop_distance).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    quality = (
        40.0
        + (alpha - 55.0).clip(lower=0.0, upper=35.0) * 0.45
        + (rs - 60.0).clip(lower=0.0, upper=40.0) * 0.25
        + (theme - 55.0).clip(lower=0.0, upper=40.0) * 0.15
        + controlled_pullback.astype(float) * 22.0
        + reclaim.astype(float) * 18.0
        + clean_continuation.astype(float) * 8.0
        - (above_ma20.clip(lower=0.0) / max(chase_max, 1e-9)).clip(upper=2.0) * 18.0
        - gap_risk.clip(lower=0.0, upper=100.0) * 0.18
        - event_risk.clip(lower=0.0, upper=100.0) * 0.15
        - rsi.sub(70.0).clip(lower=0.0) * 0.45
    ).clip(lower=0.0, upper=100.0)

    setup = pd.Series("WAIT_FOR_LOCATION", index=out.index, dtype=object)
    setup.loc[clean_continuation] = "BUY_CONTINUATION"
    setup.loc[reclaim] = "BUY_RECLAIM"
    setup.loc[controlled_pullback] = "BUY_PULLBACK"
    setup.loc[extended] = "HOLD_STRONG_EXTENDED"
    weak_location = quality.lt(min_quality) & reward_risk.lt(max(0.50, min_rr * 0.50)) & ~extended
    setup.loc[weak_location & alpha.ge(60.0)] = "AVOID_CHASE"
    setup.loc[gap_risk.ge(gap_block) | event_risk.ge(event_block)] = "RISK_BLOCK"

    multiplier = pd.Series(0.0, index=out.index, dtype=float)
    multiplier.loc[setup.eq("BUY_PULLBACK")] = float(getattr(config, "trade_location_pullback_multiplier", 1.15))
    multiplier.loc[setup.eq("BUY_RECLAIM")] = 1.0
    multiplier.loc[setup.eq("BUY_CONTINUATION")] = float(getattr(config, "trade_location_continuation_multiplier", 0.80))
    multiplier.loc[setup.eq("HOLD_STRONG_EXTENDED")] = float(getattr(config, "trade_location_hold_extended_multiplier", 0.35))
    multiplier.loc[setup.eq("AVOID_CHASE") | setup.eq("RISK_BLOCK") | setup.eq("WAIT_FOR_LOCATION")] = 0.0
    multiplier = multiplier.clip(lower=0.0, upper=1.25)

    out["entry_quality_score"] = quality
    out["reward_risk_estimate"] = reward_risk
    out["trade_location_type"] = setup
    out["trade_location_multiplier"] = multiplier
    out["trade_location_note"] = [
        (
            f"{kind}: entry_quality={q:.1f}, R/R={rr:.2f}, above_ma20={ma:.1%}, "
            f"drawdown_from_high={dd:.1%}, gap_risk={gap:.0f}, volume={vol:.2f}x"
        )
        for kind, q, rr, ma, dd, gap, vol in zip(setup, quality, reward_risk, above_ma20, drawdown_from_high, gap_risk, volume)
    ]
    return out


def apply_trade_location_to_targets(targets: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Apply long-entry location multipliers after risk overlays are available.

    Portfolio construction ranks alpha first.  This post-construction overlay
    runs after event/gap/crowding context has been attached, so the actual
    target size reflects whether the close is a reasonable add point rather
    than merely whether the asset is strong.
    """

    if targets.empty:
        return targets
    out = targets.copy()
    if "target_weight" not in out.columns:
        return out
    out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0)
    out["trade_location_weight_before"] = out["target_weight"].astype(float)
    if not bool(getattr(config, "trade_location_overlay", False)):
        if "trade_location_multiplier" not in out.columns:
            out["trade_location_multiplier"] = 1.0
        out["trade_location_weight_after"] = out["target_weight"].astype(float)
        return out

    if "date" not in out.columns or "symbol" not in out.columns:
        out["trade_location_weight_after"] = out["target_weight"].astype(float)
        return out

    location_cols = [
        "entry_quality_score",
        "reward_risk_estimate",
        "trade_location_type",
        "trade_location_multiplier",
        "trade_location_note",
    ]
    allow_new_types = {"BUY_PULLBACK", "BUY_RECLAIM", "BUY_CONTINUATION"}
    block_new_extended = bool(getattr(config, "trade_location_block_new_extended_entries", True))
    active_long_weights: dict[str, float] = {}
    rows: list[pd.DataFrame] = []
    for _, day in out.sort_values(["date", "symbol"]).groupby("date", sort=True):
        day = day.copy()
        long_mask = day["target_weight"].gt(0.0)
        if long_mask.any():
            longs = attach_trade_location_scores(day.loc[long_mask].copy(), config, positive=True)
            for column in location_cols:
                if column in longs.columns:
                    day.loc[long_mask, column] = longs[column].to_numpy()
            day["fresh_entry_allowed"] = day.get("fresh_entry_allowed", False)
            day["trade_action_intent"] = day.get("trade_action_intent", "")
            for idx in day.index[long_mask]:
                symbol = str(day.at[idx, "symbol"]).upper()
                before = float(day.at[idx, "target_weight"])
                setup = str(day.at[idx, "trade_location_type"])
                multiplier = float(np.clip(pd.to_numeric(pd.Series([day.at[idx, "trade_location_multiplier"]]), errors="coerce").fillna(1.0).iloc[0], 0.0, 1.25))
                was_active = active_long_weights.get(symbol, 0.0) > 1e-12
                fresh_allowed = setup in allow_new_types
                if block_new_extended and not was_active and not fresh_allowed:
                    after = 0.0
                    intent = "BLOCKED_RISK" if setup == "RISK_BLOCK" else "WAIT_FOR_PULLBACK_NO_NEW_ENTRY"
                    multiplier = 0.0
                else:
                    after = before * multiplier
                    current = active_long_weights.get(symbol, 0.0)
                    if not was_active:
                        intent = "OPEN_ALLOWED_BY_LOCATION"
                    elif after > current + 0.0025:
                        intent = "ADD_ALLOWED_BY_LOCATION"
                    elif after < current - 0.0025:
                        intent = "REDUCE_FOR_LOCATION_OR_RISK"
                    else:
                        intent = "HOLD_EXISTING"
                day.at[idx, "target_weight"] = after
                day.at[idx, "trade_location_multiplier"] = multiplier
                day.at[idx, "fresh_entry_allowed"] = bool(fresh_allowed and setup != "RISK_BLOCK")
                day.at[idx, "trade_action_intent"] = intent
            if "reason_for_entry" in day.columns:
                note = day.loc[long_mask, "trade_location_note"].fillna("").astype(str)
                mult = pd.to_numeric(day.loc[long_mask, "trade_location_multiplier"], errors="coerce").fillna(1.0)
                intent = day.loc[long_mask, "trade_action_intent"].fillna("").astype(str)
                msg = (
                    " Trade location overlay: "
                    + note
                    + "; action="
                    + intent
                    + "; multiplier="
                    + mult.map(lambda value: f"{value:.2f}").astype(str)
                    + "."
                )
                day.loc[long_mask, "reason_for_entry"] = (day.loc[long_mask, "reason_for_entry"].fillna("").astype(str) + msg.to_numpy()).str.strip()
        day["trade_location_weight_after"] = day["target_weight"].astype(float)
        active_long_weights = {
            str(row["symbol"]).upper(): float(row["target_weight"])
            for _, row in day.loc[day["target_weight"].gt(0.0)].iterrows()
        }
        rows.append(day)
    return pd.concat(rows, ignore_index=True) if rows else out


def _num(frame: pd.DataFrame, column: str, default: float | pd.Series) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    if isinstance(default, pd.Series):
        return default.reindex(frame.index)
    return pd.Series(default, index=frame.index, dtype=float)
