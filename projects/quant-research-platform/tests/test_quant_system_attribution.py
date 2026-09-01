from __future__ import annotations

import pandas as pd

from quant_system.backtest.attribution import (
    build_factor_forward_return_panel,
    compute_attribution_tables,
    daily_rank_ic,
)


def test_daily_rank_ic_uses_forward_returns_by_date() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 10 + ["2026-01-02"] * 10),
            "factor": list(range(10)) * 2,
            "fwd_1d": list(range(10)) * 2,
        }
    )

    ic = daily_rank_ic(panel, "factor", "fwd_1d")

    assert len(ic) == 2
    assert (ic.round(8) == 1.0).all()


def test_attribution_tables_include_factor_and_family_ic() -> None:
    dates = pd.bdate_range("2026-01-01", periods=35)
    symbols = [f"S{i:02d}" for i in range(12)]
    price_rows = []
    signal_rows = []
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(symbols):
            price = 100 + 0.05 * date_idx + symbol_idx * 0.01 + date_idx * symbol_idx * 0.02
            price_rows.append({"date": date, "symbol": symbol, "adj_close": price, "close": price})
            signal_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "final_score": symbol_idx,
                    "technical_score": symbol_idx,
                    "relative_strength_score": symbol_idx,
                    "fundamental_score": 50,
                    "event_risk_score": 0,
                    "score_decomposition": f"return={symbol_idx}.0%; risk_adj={symbol_idx}; breakout={symbol_idx}; rs={symbol_idx}; price_vs_slow={symbol_idx}.0%; adx={symbol_idx}",
                }
            )
    prices = pd.DataFrame(price_rows)
    signals = pd.DataFrame(signal_rows)

    panel = build_factor_forward_return_panel(signals, prices, horizons=(1,))
    factor_ic, family_ic = compute_attribution_tables(signals, prices, forward_start="2026-01-01", horizons=(1,))

    assert "fwd_1d" in panel.columns
    assert not factor_ic.empty
    assert not family_ic.empty
    assert (factor_ic["mean_rank_ic"] > 0).any()
    assert "ensemble_topline" in set(family_ic["family"])
