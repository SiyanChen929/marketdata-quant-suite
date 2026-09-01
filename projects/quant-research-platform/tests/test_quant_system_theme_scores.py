from __future__ import annotations

import pandas as pd

from quant_system.universe.thematic import dynamic_theme_scores


def test_dynamic_theme_scores_are_not_neutral_placeholders() -> None:
    rows = []
    for i in range(80):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        rows.extend(
            [
                {"date": date, "symbol": "SPY", "adj_close": 100 + i * 0.1, "close": 100 + i * 0.1, "volume": 1_000_000},
                {"date": date, "symbol": "LEADER", "adj_close": 50 + i * 1.0, "close": 50 + i * 1.0, "volume": 2_000_000 + i * 10_000},
                {"date": date, "symbol": "LAGGARD", "adj_close": 80 - i * 0.2, "close": 80 - i * 0.2, "volume": 1_000_000},
            ]
        )
    scores = dynamic_theme_scores(pd.DataFrame(rows), benchmark="SPY")
    latest = scores[scores["date"].eq(scores["date"].max())].set_index("symbol")
    assert latest.loc["LEADER", "theme_score"] > latest.loc["LAGGARD", "theme_score"]
    assert scores["theme_score"].nunique() > 1
