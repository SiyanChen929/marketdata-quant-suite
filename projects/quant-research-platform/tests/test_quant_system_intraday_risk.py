from __future__ import annotations

import pandas as pd

from quant_system.config import RegimeConfig
from quant_system.regime.intraday_risk import apply_intraday_risk_overlay, detect_intraday_proxy_risk


def _stress_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for symbol in ["SPY", "AAA", "BBB", "CCC", "DDD"]:
        for idx, date in enumerate(dates):
            prev = 100 + idx
            if idx == len(dates) - 1:
                open_price = prev * (0.96 if symbol == "SPY" else 1.00)
                close = open_price * (0.95 if symbol in {"SPY", "AAA", "BBB", "CCC"} else 1.01)
                low = min(open_price, close) * 0.98
                high = max(open_price, close) * 1.01
                volume = 3_000_000
            else:
                open_price = prev
                close = prev * 1.002
                low = prev * 0.995
                high = prev * 1.005
                volume = 1_000_000
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def test_intraday_proxy_detects_gap_breadth_and_volume_stress() -> None:
    config = RegimeConfig(
        intraday_risk_overlay=True,
        intraday_reduce_score=50,
        intraday_cash_score=90,
        intraday_benchmark_drop_reduce=-0.04,
        intraday_benchmark_gap_reduce=-0.03,
    )
    risk = detect_intraday_proxy_risk(_stress_prices(), "SPY", config)
    latest = risk.iloc[-1]
    assert latest["intraday_risk_status"] in {"reduce", "cash"}
    assert latest["intraday_risk_score"] >= 50
    assert latest["pct_down_from_open"] >= 0.60
    assert latest["pct_volume_expansion"] >= 0.60


def test_intraday_proxy_overlay_scales_target_weights() -> None:
    prices = _stress_prices()
    latest_date = prices["date"].max()
    targets = pd.DataFrame(
        {
            "date": [latest_date],
            "symbol": ["AAA"],
            "target_weight": [0.20],
            "reason_for_entry": ["test"],
        }
    )
    config = RegimeConfig(
        intraday_risk_overlay=True,
        intraday_reduce_score=50,
        intraday_cash_score=120,
        intraday_reduce_multiplier=0.5,
    )
    out = apply_intraday_risk_overlay(targets, prices, "SPY", config)
    assert out["target_weight"].iloc[0] == 0.10
    assert out["intraday_risk_multiplier"].iloc[0] == 0.5
    assert "Intraday proxy risk" in out["reason_for_entry"].iloc[0]


def test_intraday_proxy_disabled_is_noop() -> None:
    targets = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "symbol": ["AAA"], "target_weight": [0.10]})
    out = apply_intraday_risk_overlay(targets, _stress_prices(), "SPY", RegimeConfig(intraday_risk_overlay=False))
    pd.testing.assert_frame_equal(out, targets)


def test_intraday_leader_exception_restores_part_of_reduce_cut_for_strong_pullback_leader() -> None:
    prices = _stress_prices()
    latest_date = prices["date"].max()
    targets = pd.DataFrame(
        {
            "date": [latest_date],
            "symbol": ["AAA"],
            "target_weight": [0.20],
            "reason_for_entry": ["test"],
            "final_score": [76.0],
            "relative_strength_score": [94.0],
            "theme_score": [82.0],
            "mom_return": [0.24],
            "distance_to_prior_high_252": [-0.08],
            "retest_distance_ma_pct": [0.03],
            "trend_adx": [26.0],
            "event_risk_score": [8.0],
            "overnight_gap_risk_score": [15.0],
            "trend_price_vs_slow_pct": [0.22],
            "theme_active": [True],
        }
    )
    config = RegimeConfig(
        intraday_risk_overlay=True,
        intraday_reduce_score=50,
        intraday_cash_score=120,
        intraday_reduce_multiplier=0.5,
        intraday_leader_exception_overlay=True,
        intraday_leader_exception_restore_multiplier=0.85,
        intraday_leader_exception_min_final_score=70.0,
        intraday_leader_exception_min_relative_strength_score=88.0,
        intraday_leader_exception_min_theme_score=65.0,
        intraday_leader_exception_min_mom_return=0.10,
        intraday_leader_exception_min_drawdown_from_high=0.03,
        intraday_leader_exception_max_drawdown_from_high=0.14,
        intraday_leader_exception_max_above_ma20_pct=0.05,
        intraday_leader_exception_min_adx_circuit_breaker=18.0,
        intraday_leader_exception_max_event_risk_score_circuit_breaker=18.0,
        intraday_leader_exception_max_overnight_gap_risk_score_circuit_breaker=25.0,
    )
    out = apply_intraday_risk_overlay(targets, prices, "SPY", config)
    assert out["intraday_risk_status"].iloc[0] == "reduce"
    assert out["intraday_risk_multiplier"].iloc[0] == 0.85
    assert out["target_weight"].iloc[0] == 0.17
    assert bool(out["intraday_leader_exception_applied"].iloc[0]) is True
    assert out["intraday_leader_exception_circuit_breaker"].iloc[0] == "applied"
    assert "Strong-leader intraday exception" in out["reason_for_entry"].iloc[0]
    assert "\"circuit_breaker\": \"applied\"" in out["intraday_leader_exception_log"].iloc[0]


def test_intraday_leader_exception_never_overrides_cash_state() -> None:
    prices = _stress_prices()
    latest_date = prices["date"].max()
    targets = pd.DataFrame(
        {
            "date": [latest_date],
            "symbol": ["AAA"],
            "target_weight": [0.20],
            "reason_for_entry": ["test"],
            "final_score": [80.0],
            "relative_strength_score": [95.0],
            "theme_score": [84.0],
            "mom_return": [0.26],
            "distance_to_prior_high_252": [-0.06],
            "retest_distance_ma_pct": [0.02],
            "trend_adx": [28.0],
            "event_risk_score": [5.0],
            "overnight_gap_risk_score": [12.0],
            "trend_price_vs_slow_pct": [0.30],
            "theme_active": [True],
        }
    )
    config = RegimeConfig(
        intraday_risk_overlay=True,
        intraday_reduce_score=50,
        intraday_cash_score=60,
        intraday_reduce_multiplier=0.5,
        intraday_cash_multiplier=0.25,
        intraday_leader_exception_overlay=True,
        intraday_leader_exception_restore_multiplier=0.85,
    )
    out = apply_intraday_risk_overlay(targets, prices, "SPY", config)
    assert out["intraday_risk_status"].iloc[0] == "cash"
    assert out["intraday_risk_multiplier"].iloc[0] == 0.25
    assert out["target_weight"].iloc[0] == 0.05
    assert bool(out["intraday_leader_exception_applied"].iloc[0]) is False
    assert out["intraday_leader_exception_circuit_breaker"].iloc[0] == "cash_state"
