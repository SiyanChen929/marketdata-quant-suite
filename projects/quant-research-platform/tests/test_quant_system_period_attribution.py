from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from quant_system.backtest.institutional_attribution import compute_period_institutional_attribution


def test_period_institutional_attribution_splits_theme_contribution() -> None:
    result = SimpleNamespace(
        positions=pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "weight": 0.50},
                {"date": "2024-01-03", "symbol": "AAA", "weight": 0.50},
                {"date": "2024-01-04", "symbol": "BBB", "weight": 0.40},
                {"date": "2024-01-05", "symbol": "BBB", "weight": 0.40},
            ]
        ),
        prices=pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "adj_close": 100.0},
                {"date": "2024-01-03", "symbol": "AAA", "adj_close": 110.0},
                {"date": "2024-01-04", "symbol": "BBB", "adj_close": 50.0},
                {"date": "2024-01-05", "symbol": "BBB", "adj_close": 45.0},
            ]
        ),
        targets=pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "symbol": "AAA",
                    "primary_theme": "semiconductors",
                    "sector": "Technology",
                    "industry": "Chips",
                    "reason_for_entry": "Long momentum: strong tape",
                    "final_score": 80.0,
                },
                {
                    "date": "2024-01-03",
                    "symbol": "AAA",
                    "primary_theme": "semiconductors",
                    "sector": "Technology",
                    "industry": "Chips",
                    "reason_for_entry": "Long momentum: strong tape",
                    "final_score": 80.0,
                },
                {
                    "date": "2024-01-04",
                    "symbol": "BBB",
                    "primary_theme": "software",
                    "sector": "Technology",
                    "industry": "Software",
                    "reason_for_entry": "Trend following: weak",
                    "final_score": 60.0,
                },
                {
                    "date": "2024-01-05",
                    "symbol": "BBB",
                    "primary_theme": "software",
                    "sector": "Technology",
                    "industry": "Software",
                    "reason_for_entry": "Trend following: weak",
                    "final_score": 60.0,
                },
            ]
        ),
        signals=pd.DataFrame(),
    )

    tables = compute_period_institutional_attribution(
        result,
        {
            "train": ("2024-01-02", "2024-01-03"),
            "validation": ("2024-01-04", "2024-01-05"),
        },
    )
    theme = tables["primary_theme"]
    train_semi = theme[(theme["period"].eq("train")) & (theme["bucket"].eq("semiconductors"))].iloc[0]
    validation_software = theme[(theme["period"].eq("validation")) & (theme["bucket"].eq("software"))].iloc[0]

    assert round(train_semi["total_contribution"], 6) == 0.05
    assert round(validation_software["total_contribution"], 6) == -0.04
    assert set(theme["period"]) == {"train", "validation"}
