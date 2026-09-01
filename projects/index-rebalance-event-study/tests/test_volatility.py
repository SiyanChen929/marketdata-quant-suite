from __future__ import annotations

import pandas as pd
import pytest

from index_rebalance_event_study.volatility import (
    breakout_proxy,
    hypothetical_straddle_return,
    window_statistics,
)


def _session(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    clocks = [f"2035-01-12 15:{minute:02d}:00" for minute in range(49, 60)]
    opens = [100.0, *closes[:-1]]
    highs = highs or [max(open_, close) for open_, close in zip(opens, closes, strict=True)]
    lows = lows or [min(open_, close) for open_, close in zip(opens, closes, strict=True)]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(clocks),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def test_window_statistics_uses_full_path() -> None:
    session = _session([100.1, 100.2, 100.1, 100.3, 100.4, 100.5, 100.4, 100.6, 100.7, 100.8, 101.0])
    result = window_statistics(session)
    assert result is not None
    assert result["sqrt_realized_variance_11m"] > 0


def test_breakout_waits_until_next_bar_and_charges_cost() -> None:
    closes = [100.2, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9, 101.0, 101.1, 101.2, 101.3]
    opens = [100.0, *closes[:-1]]
    highs = [100.30] + [max(open_, close) + 0.05 for open_, close in zip(opens[1:], closes[1:], strict=True)]
    lows = [99.95] + [min(open_, close) - 0.05 for open_, close in zip(opens[1:], closes[1:], strict=True)]
    result = breakout_proxy(_session(closes, highs, lows), barrier_bps=25, cost_bps=10)
    assert result is not None and result["triggered"]
    assert result["fill_clock"] == "15:50"
    assert result["net_return"] == pytest.approx(101.3 / 100.25 - 1 - 0.001)


def test_ambiguous_bar_assigns_worse_direction() -> None:
    result = breakout_proxy(_session([100.0] * 11, [100.4] + [100.2] * 10, [99.6] + [99.8] * 10))
    assert result is not None and result["ambiguous_same_bar"]
    assert result["gross_return"] <= 0


def test_incoherent_ohlc_window_is_rejected() -> None:
    session = _session([100.0] * 11)
    session.loc[3, "high"] = 99.0
    assert window_statistics(session) is None


def test_hypothetical_straddle_is_explicit_model_sensitivity() -> None:
    flat = hypothetical_straddle_return(0.0, dte=7, entry_volatility=0.5)
    moved = hypothetical_straddle_return(0.03, dte=7, entry_volatility=0.5)
    with_friction = hypothetical_straddle_return(
        0.03,
        dte=7,
        entry_volatility=0.5,
        friction_fraction_of_premium=0.10,
    )
    assert moved > flat
    assert with_friction == pytest.approx(moved - 0.10)
