from __future__ import annotations

import pandas as pd

from quant_system.config import AppConfig
from quant_system.portfolio.data_quality_gate import apply_latest_data_quality_gate_to_targets


def test_live_data_quality_gate_blocks_only_latest_targets() -> None:
    targets = pd.DataFrame(
        [
            {"date": "2026-05-11", "symbol": "BAD", "target_weight": 0.04},
            {"date": "2026-05-12", "symbol": "BAD", "target_weight": 0.05},
            {"date": "2026-05-12", "symbol": "GOOD", "target_weight": 0.05},
        ]
    )
    quality = pd.DataFrame(
        [
            {
                "symbol": "BAD",
                "latest_date": "2026-05-12",
                "blocked_from_trading": True,
                "block_reason": "stale_price",
                "status": "stale_price",
            }
        ]
    )

    out, warnings = apply_latest_data_quality_gate_to_targets(targets, quality, AppConfig())

    blocked = out[(out["date"].eq(pd.Timestamp("2026-05-12"))) & (out["symbol"].eq("BAD"))].iloc[0]
    prior = out[(out["date"].eq(pd.Timestamp("2026-05-11"))) & (out["symbol"].eq("BAD"))].iloc[0]
    good = out[(out["date"].eq(pd.Timestamp("2026-05-12"))) & (out["symbol"].eq("GOOD"))].iloc[0]
    assert blocked["target_weight"] == 0.0
    assert blocked["target_weight_before_data_quality_gate"] == 0.05
    assert bool(blocked["data_quality_blocked"]) is True
    assert prior["target_weight"] == 0.04
    assert good["target_weight"] == 0.05
    assert warnings


def test_live_data_quality_gate_skips_historical_windows() -> None:
    targets = pd.DataFrame([{"date": "2025-12-31", "symbol": "BAD", "target_weight": 0.05}])
    quality = pd.DataFrame(
        [
            {
                "symbol": "BAD",
                "latest_date": "2026-05-12",
                "blocked_from_trading": True,
                "block_reason": "stale_price",
                "status": "stale_price",
            }
        ]
    )

    out, warnings = apply_latest_data_quality_gate_to_targets(targets, quality, AppConfig())

    assert out.iloc[0]["target_weight"] == 0.05
    assert "historical window" in warnings[0]
