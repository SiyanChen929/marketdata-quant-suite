"""Convert equity signals into conservative options overlay guidance."""

from __future__ import annotations

import pandas as pd

from quant_system.config import OptionsOverlayConfig


def build_options_overlay_recommendations(
    manual_blotter: pd.DataFrame,
    equity: float,
    config: OptionsOverlayConfig,
) -> pd.DataFrame:
    """Return option structure suggestions for actionable long equity signals.

    This module does not price contracts. It translates the equity model's
    conviction and risk flags into a target structure, DTE, delta, and premium
    budget cap so a live option-chain adapter can fill actual contracts later.
    Rows are not executable orders until option-chain liquidity, IV, and spread
    checks have passed.
    """

    if manual_blotter.empty or not config.enabled:
        return pd.DataFrame()
    rows: list[dict] = []
    available_budget = max(float(equity), 0.0) * float(config.max_option_premium_pct_equity)
    used_budget = 0.0
    data = manual_blotter.copy()
    for column in [
        "final_score",
        "relative_strength_score",
        "event_risk_score",
        "overnight_gap_risk_score",
        "minute_pretrade_risk_score",
        "delta_weight",
    ]:
        data[column] = _numeric_column(data, column)
    candidates = data[
        data.get("action_label", "").isin(["新开多", "加仓"])
        & data["delta_weight"].gt(0)
        & data["final_score"].ge(float(config.min_underlying_score_for_calls))
        & data["relative_strength_score"].ge(float(config.min_relative_strength_for_calls))
    ].sort_values(["final_score", "relative_strength_score"], ascending=False)
    for _, row in candidates.iterrows():
        risk_score = max(
            float(row.get("event_risk_score", 0.0) or 0.0),
            float(row.get("overnight_gap_risk_score", 0.0) or 0.0),
            float(row.get("minute_pretrade_risk_score", 0.0) or 0.0),
        )
        if risk_score >= 80:
            structure = "No new option; use stock or wait"
            max_premium = 0.0
            target_delta = 0.0
            note = "Risk score is high; options would amplify gap/intraday risk."
        elif risk_score >= 50:
            structure = "Call debit spread"
            max_premium = min(
                available_budget - used_budget,
                float(equity) * float(config.high_risk_max_premium_pct_equity),
            )
            target_delta = float(config.high_risk_delta)
            note = "Use defined-risk spread because event/gap/minute risk is elevated."
        else:
            structure = "Long call or call debit spread"
            max_premium = min(
                available_budget - used_budget,
                float(equity) * float(config.max_single_option_premium_pct_equity),
                abs(float(row.get("estimated_trade_notional", 0.0) or 0.0)),
            )
            target_delta = float(config.target_delta)
            note = "Equity signal is strong; options budget should still be capped by premium-at-risk."
        if max_premium <= 0 and structure != "No new option; use stock or wait":
            continue
        used_budget += max_premium
        rows.append(
            {
                "symbol": row.get("symbol"),
                "signal_date": row.get("signal_date"),
                "equity_action": row.get("action_label"),
                "option_structure": structure,
                "target_dte_min": int(config.target_dte_min),
                "target_dte_max": int(config.target_dte_max),
                "target_delta": target_delta,
                "max_premium_budget": max_premium,
                "executable": False,
                "chain_validation_status": "missing_option_chain_validation",
                "underlying_close": row.get("close"),
                "final_score": row.get("final_score"),
                "relative_strength_score": row.get("relative_strength_score"),
                "fundamental_score": row.get("fundamental_score"),
                "event_risk_score": row.get("event_risk_score"),
                "overnight_gap_risk_score": row.get("overnight_gap_risk_score"),
                "minute_pretrade_risk_score": row.get("minute_pretrade_risk_score"),
                "primary_theme": row.get("primary_theme"),
                "overlay_note": f"{note} Not executable until option-chain bid/ask, OI, IV, skew, and event liquidity are validated.",
            }
        )
        if used_budget >= available_budget:
            break
    return pd.DataFrame(rows)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
