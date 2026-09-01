from __future__ import annotations

import pandas as pd

from quant_system.config import RegimeConfig
from quant_system.regime.detector import apply_regime_exposure_overlay, detect_overbought_risk


def test_regime_overlay_scales_targets_only_in_weak_regime() -> None:
    dates = pd.bdate_range("2024-01-01", periods=260)
    spy = pd.DataFrame(
        {
            "date": dates,
            "symbol": "SPY",
            "adj_close": [100 - i * 0.1 for i in range(len(dates))],
            "volume": 1_000_000,
        }
    )
    targets = pd.DataFrame(
        {
            "date": [dates[-1]],
            "symbol": ["AAA"],
            "target_weight": [0.10],
            "reason_for_entry": ["test"],
        }
    )
    config = RegimeConfig(
        exposure_overlay=True,
        risk_off_exposure_multiplier=0.25,
        partial_risk_on_exposure_multiplier=0.5,
    )
    out = apply_regime_exposure_overlay(targets, spy, "SPY", config)
    assert out["target_weight"].iloc[0] == 0.025
    assert out["regime_exposure_multiplier"].iloc[0] == 0.25


def test_regime_overlay_disabled_is_noop() -> None:
    targets = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "symbol": ["AAA"], "target_weight": [0.10]})
    out = apply_regime_exposure_overlay(targets, pd.DataFrame(), "SPY", RegimeConfig(exposure_overlay=False))
    pd.testing.assert_frame_equal(out, targets)


def test_analog_overlay_uses_pit_momentum_state() -> None:
    dates = pd.bdate_range("2023-01-01", periods=220)
    rows = []
    for symbol, drift in [("SPY", 0.001), ("AAA", 0.004), ("BBB", 0.003)]:
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_close": 100 * (1 + drift) ** idx,
                    "volume": 1_000_000 + idx * 1000,
                    "rs_63d": 0.2 if symbol != "SPY" and idx > 126 else 0.01,
                    "volume_expansion": 1.5 if idx > 126 else 0.9,
                }
            )
    prices = pd.DataFrame(rows)
    targets = pd.DataFrame({"date": [dates[-1]], "symbol": ["AAA"], "target_weight": [0.10]})
    config = RegimeConfig(
        exposure_overlay=True,
        risk_off_exposure_multiplier=1.0,
        analog_momentum_overlay=True,
        analog_inactive_exposure_multiplier=0.25,
        analog_active_exposure_multiplier=1.0,
        analog_lookback_days=80,
        analog_quantile=0.5,
    )
    out = apply_regime_exposure_overlay(targets, prices, "SPY", config)
    assert out["target_weight"].iloc[0] == 0.10
    assert out["analog_exposure_multiplier"].iloc[0] == 1.0


def test_overbought_cash_signal_can_zero_targets() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    rows = []
    for symbol in ["SPY", "AAA", "BBB", "CCC"]:
        for idx, date in enumerate(dates):
            close = 100 + idx * 2
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000,
                    "rsi_14": 82,
                    "zscore_20": 2.5,
                    "distance_to_high_252": -0.01,
                }
            )
    prices = pd.DataFrame(rows)
    targets = pd.DataFrame({"date": [dates[-1]], "symbol": ["AAA"], "target_weight": [0.20], "reason_for_entry": ["test"]})
    config = RegimeConfig(
        exposure_overlay=False,
        overbought_overlay=True,
        overbought_cash_score=50,
        overbought_cash_multiplier=0.0,
    )
    risk = detect_overbought_risk(prices, "SPY", config)
    assert bool(risk.iloc[-1]["cash_signal"])
    out = apply_regime_exposure_overlay(targets, prices, "SPY", config)
    assert out["target_weight"].iloc[0] == 0.0
    assert out["overbought_status"].iloc[0] == "cash"


def test_regime_leader_exception_restores_part_of_partial_cut_for_strong_leader() -> None:
    dates = pd.bdate_range("2024-01-01", periods=260)
    prices = pd.DataFrame(
        {
            "date": dates,
            "symbol": "SPY",
            "adj_close": [100.0] * 240 + [100.2 + 0.02 * i for i in range(20)],
            "volume": 1_000_000,
        }
    )
    targets = pd.DataFrame(
        {
            "date": [dates[-1]],
            "symbol": ["AAA"],
            "target_weight": [0.20],
            "reason_for_entry": ["test"],
            "final_score": [78.0],
            "relative_strength_score": [93.0],
            "theme_score": [76.0],
            "mom_return": [0.20],
            "distance_to_prior_high_252": [-0.07],
            "retest_distance_ma_pct": [0.03],
            "trend_adx": [26.0],
            "event_risk_score": [10.0],
            "overnight_gap_risk_score": [12.0],
            "trend_price_vs_slow_pct": [0.18],
            "theme_active": [True],
        }
    )
    config = RegimeConfig(
        exposure_overlay=True,
        risk_off_exposure_multiplier=0.0,
        partial_risk_on_exposure_multiplier=0.5,
        partial_exposure_regime_score=45.0,
        full_exposure_regime_score=101.0,
        regime_leader_exception_overlay=True,
        regime_leader_exception_restore_multiplier=0.75,
        regime_leader_exception_min_regime_score=50.0,
        regime_leader_exception_min_final_score=72.0,
        regime_leader_exception_min_relative_strength_score=90.0,
        regime_leader_exception_min_theme_score=62.0,
        regime_leader_exception_min_mom_return=0.12,
        regime_leader_exception_min_drawdown_from_high=0.03,
        regime_leader_exception_max_drawdown_from_high=0.14,
        regime_leader_exception_max_above_ma20_pct=0.05,
        regime_leader_exception_min_adx_circuit_breaker=18.0,
        regime_leader_exception_max_event_risk_score_circuit_breaker=18.0,
        regime_leader_exception_max_overnight_gap_risk_score_circuit_breaker=25.0,
        regime_leader_exception_min_benchmark_ret_20d_circuit_breaker=-0.04,
        regime_leader_exception_max_realized_vol_20d_circuit_breaker=0.35,
        regime_leader_exception_min_benchmark_drawdown_52w_circuit_breaker=-0.18,
    )
    out = apply_regime_exposure_overlay(targets, prices, "SPY", config)
    assert out["regime_exposure_multiplier"].iloc[0] == 0.75
    assert abs(out["target_weight"].iloc[0] - 0.15) < 1e-12
    assert bool(out["regime_leader_exception_applied"].iloc[0]) is True
    assert out["regime_leader_exception_circuit_breaker"].iloc[0] == "applied"
    assert "Strong-leader regime exception" in out["reason_for_entry"].iloc[0]
    assert "\"circuit_breaker\": \"applied\"" in out["regime_leader_exception_log"].iloc[0]


def test_regime_leader_exception_never_overrides_risk_off_state() -> None:
    dates = pd.bdate_range("2024-01-01", periods=260)
    prices = pd.DataFrame(
        {
            "date": dates,
            "symbol": "SPY",
            "adj_close": [140 - i * 0.2 for i in range(len(dates))],
            "volume": 1_000_000,
        }
    )
    targets = pd.DataFrame(
        {
            "date": [dates[-1]],
            "symbol": ["AAA"],
            "target_weight": [0.20],
            "reason_for_entry": ["test"],
            "final_score": [82.0],
            "relative_strength_score": [95.0],
            "theme_score": [80.0],
            "mom_return": [0.24],
            "distance_to_prior_high_252": [-0.06],
            "retest_distance_ma_pct": [0.02],
            "trend_adx": [28.0],
            "event_risk_score": [8.0],
            "overnight_gap_risk_score": [10.0],
            "trend_price_vs_slow_pct": [0.20],
            "theme_active": [True],
        }
    )
    config = RegimeConfig(
        exposure_overlay=True,
        risk_off_exposure_multiplier=0.0,
        partial_risk_on_exposure_multiplier=0.5,
        regime_leader_exception_overlay=True,
        regime_leader_exception_restore_multiplier=0.75,
    )
    out = apply_regime_exposure_overlay(targets, prices, "SPY", config)
    assert out["regime_exposure_multiplier"].iloc[0] == 0.0
    assert out["target_weight"].iloc[0] == 0.0
    assert bool(out["regime_leader_exception_applied"].iloc[0]) is False
    assert out["regime_leader_exception_circuit_breaker"].iloc[0] == "risk_off_state"
