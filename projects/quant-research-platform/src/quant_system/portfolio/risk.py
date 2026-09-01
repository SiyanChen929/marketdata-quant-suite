"""Portfolio risk controls."""

from __future__ import annotations

import pandas as pd

from quant_system.config import EventsConfig, RiskConfig


def drawdown_exposure_multiplier(equity: float, peak: float, config: RiskConfig) -> float:
    """Return gross exposure multiplier based on current drawdown."""

    if peak <= 0:
        return 1.0
    drawdown = 1.0 - equity / peak
    if drawdown >= config.max_drawdown_cash_mode:
        return 0.0
    if drawdown >= config.max_drawdown_reduce_exposure:
        return config.drawdown_reduction_multiplier
    return 1.0


def apply_event_risk_to_targets(targets: pd.DataFrame, events_config: EventsConfig) -> pd.DataFrame:
    """Reduce or block targets based on event risk score and days to earnings."""

    if targets.empty or "event_risk_score" not in targets.columns:
        return targets
    out = targets.copy()
    risk = out["event_risk_score"].fillna(0.0)
    if events_config.block_high_event_risk:
        out.loc[risk >= events_config.max_event_risk_score_for_normal_strategies, "target_weight"] = 0.0
    if "days_to_earnings" in out.columns and events_config.earnings_mode == "risk_reduce":
        days = out["days_to_earnings"].abs()
        mask = days <= events_config.reduce_position_before_earnings_days
        out.loc[mask, "target_weight"] *= events_config.earnings_risk_multiplier
    return out
