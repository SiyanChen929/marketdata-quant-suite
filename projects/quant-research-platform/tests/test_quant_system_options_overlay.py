from __future__ import annotations

import pandas as pd

from quant_system.config import OptionsOverlayConfig
from quant_system.options.overlay import build_options_overlay_recommendations


def test_options_overlay_caps_premium_budget() -> None:
    blotter = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "signal_date": "2026-05-12",
                "action_label": "加仓",
                "delta_weight": 0.08,
                "estimated_trade_notional": 20_000,
                "close": 100,
                "final_score": 80,
                "relative_strength_score": 90,
                "fundamental_score": 60,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 0,
                "minute_pretrade_risk_score": 0,
            },
            {
                "symbol": "BBB",
                "signal_date": "2026-05-12",
                "action_label": "加仓",
                "delta_weight": 0.08,
                "estimated_trade_notional": 20_000,
                "close": 100,
                "final_score": 79,
                "relative_strength_score": 88,
                "fundamental_score": 60,
                "event_risk_score": 0,
                "overnight_gap_risk_score": 0,
                "minute_pretrade_risk_score": 0,
            },
        ]
    )

    out = build_options_overlay_recommendations(
        blotter,
        equity=100_000,
        config=OptionsOverlayConfig(enabled=True, max_option_premium_pct_equity=0.03, max_single_option_premium_pct_equity=0.02),
    )

    assert not out.empty
    assert out["max_premium_budget"].sum() <= 3_000
    assert out["option_structure"].iloc[0] == "Long call or call debit spread"


def test_options_overlay_blocks_high_risk_options() -> None:
    blotter = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "signal_date": "2026-05-12",
                "action_label": "新开多",
                "delta_weight": 0.08,
                "estimated_trade_notional": 20_000,
                "close": 100,
                "final_score": 80,
                "relative_strength_score": 90,
                "event_risk_score": 85,
            }
        ]
    )

    out = build_options_overlay_recommendations(blotter, equity=100_000, config=OptionsOverlayConfig(enabled=True))

    assert out["option_structure"].iloc[0] == "No new option; use stock or wait"
    assert out["max_premium_budget"].iloc[0] == 0.0
