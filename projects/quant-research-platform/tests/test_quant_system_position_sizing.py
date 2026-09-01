from __future__ import annotations

import pandas as pd

from quant_system.config import PortfolioConfig
from quant_system.portfolio.construction import construct_target_weights
from quant_system.portfolio.sizing import shares_for_target_weight


def test_shares_for_target_weight() -> None:
    assert shares_for_target_weight(100_000, 0.1, 50) == 200


def test_construct_target_weights_respects_max_position() -> None:
    signals = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "A", "signal": 1, "final_score": 90, "technical_score": 90, "relative_strength_score": 90, "theme_score": 50, "fundamental_score": 60, "event_risk_score": 0, "reason_for_entry": "x"},
            {"date": "2024-01-01", "symbol": "B", "signal": 1, "final_score": 80, "technical_score": 80, "relative_strength_score": 80, "theme_score": 50, "fundamental_score": 55, "event_risk_score": 0, "reason_for_entry": "x"},
            {"date": "2024-01-01", "symbol": "C", "signal": -1, "final_score": 10, "technical_score": 10, "relative_strength_score": 10, "theme_score": 50, "fundamental_score": 35, "event_risk_score": 0, "reason_for_entry": "x"},
        ]
    )
    weights = construct_target_weights(signals, PortfolioConfig(target_gross_exposure=1.0, target_net_exposure=0.0, max_position_weight=0.25))
    assert weights["target_weight"].abs().max() <= 0.25
    assert round(weights["target_weight"].abs().sum(), 6) <= 0.75
