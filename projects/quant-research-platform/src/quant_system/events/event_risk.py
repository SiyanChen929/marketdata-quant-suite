"""Event risk scoring."""

from __future__ import annotations

import pandas as pd


EVENT_RISK_FACTORS = [
    "earnings_within_7d",
    "expected_move",
    "historical_earnings_gap_volatility",
    "recent_guidance_change",
    "dilution_event",
    "secondary_offering",
    "regulatory_event",
    "FDA_event",
    "lockup_expiration",
    "debt_maturity",
    "macro_event_exposure",
]


def score_event_risk(aligned_events: pd.DataFrame) -> pd.DataFrame:
    """Ensure event risk score exists and add earnings proximity penalty."""

    if aligned_events.empty:
        return aligned_events
    out = aligned_events.copy()
    base_raw = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    base = base_raw.copy()
    active_event = pd.Series(True, index=out.index)
    if "days_to_earnings" in out.columns:
        days = pd.to_numeric(out["days_to_earnings"], errors="coerce")
        pre_event = days.ge(0) & days.le(7)
        post_event = days.lt(0) & days.ge(-2)
        active_event = pre_event | post_event
        base = base_raw.where(active_event, 0.0)
        earnings_penalty = pd.Series(0.0, index=out.index)
        earnings_penalty.loc[pre_event] = (7 - days.loc[pre_event].clip(lower=0, upper=7)) / 7 * 40
        earnings_penalty.loc[post_event] = (3 - days.loc[post_event].abs().clip(lower=1, upper=2)) / 2 * 40
        base = base + earnings_penalty.fillna(0.0)
    if "expected_move" in out.columns:
        expected = pd.to_numeric(out["expected_move"], errors="coerce").fillna(0.0).abs()
        base = base + (expected.clip(0, 0.30) / 0.30 * 25.0).where(active_event, 0.0)
    if "historical_earnings_gap_volatility" in out.columns:
        gap_vol = pd.to_numeric(out["historical_earnings_gap_volatility"], errors="coerce").fillna(0.0).abs()
        base = base + (gap_vol.clip(0, 0.25) / 0.25 * 20.0).where(active_event, 0.0)
    for flag in ("dilution_event", "secondary_offering", "regulatory_event", "FDA_event", "lockup_expiration", "debt_maturity"):
        if flag in out.columns:
            values = out[flag]
            if values.dtype == bool:
                base = base + values.fillna(False).astype(float) * 25.0
            else:
                base = base + pd.to_numeric(values, errors="coerce").fillna(0.0).clip(0, 1) * 25.0
    if "recent_guidance_change" in out.columns:
        guidance = pd.to_numeric(out["recent_guidance_change"], errors="coerce").fillna(0.0)
        base = base + guidance.lt(0).astype(float) * guidance.abs().clip(0, 0.20) / 0.20 * 20.0
    out["event_risk_score"] = base.clip(0, 100)
    return out
