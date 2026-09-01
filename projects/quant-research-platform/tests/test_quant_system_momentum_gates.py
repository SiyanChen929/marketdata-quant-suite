from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.strategies.base import StrategyContext
from quant_system.strategies.momentum_longs_short import (
    CrossSectionalMomentumLongShort,
    _apply_compound_leader_score_credit_overlay,
    _apply_exit_quality_rank_credit_overlay,
    _apply_medium_term_leader_pullback_overlay,
    _apply_pullback_reclaim_overlay,
    _momentum_final_score,
)


def _features(risk_on: bool = True) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=90)
    rows = []
    symbols = {
        "SPY": 100.0,
        "AAA": 20.0,
        "BBB": 60.0,
        "CCC": 15.0,
    }
    for i, date in enumerate(dates):
        for symbol, start in symbols.items():
            if symbol == "AAA":
                close = start * (1.012 ** i)
                theme = 85.0
                rs = 0.30
            elif symbol == "CCC":
                close = start * (1.015 ** i)
                theme = 35.0
                rs = 0.35
            elif symbol == "BBB":
                close = start * (0.990 ** i)
                theme = 20.0
                rs = -0.30
            else:
                close = start * ((1.004 if risk_on else 0.996) ** i)
                theme = 50.0
                rs = 0.0
            ma_50 = close * (0.96 if symbol in {"AAA", "CCC"} else 1.04)
            ma_200 = close * (0.90 if symbol in {"AAA", "CCC"} else 1.08)
            if symbol == "SPY":
                ma_50 = close * (0.97 if risk_on else 1.03)
                ma_200 = close * (0.94 if risk_on else 1.06)
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
                    "return_1d": 0.0,
                    "ma_20": close * 0.98,
                    "ma_50": ma_50,
                    "ma_200": ma_200,
                    "ret_63d": 0.10 if risk_on or symbol != "SPY" else -0.10,
                    "rs_63d": rs,
                    "distance_to_high_252": 0.0,
                    "distance_to_prior_high_252": 0.0,
                    "volume_expansion": 1.2,
                    "theme_score": theme,
                    "fundamental_score": 50.0,
                    "event_risk_score": 0.0,
                    "trend_adx": 32.0 if symbol == "AAA" else 18.0 if symbol == "CCC" else 20.0,
                }
            )
    out = pd.DataFrame(rows).sort_values(["symbol", "date"])
    out["return_1d"] = out.groupby("symbol")["adj_close"].pct_change().fillna(0.0)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _same_theme_substitution_features(high_vol_candidate: bool = False, activation_ready_candidate: bool = False) -> pd.DataFrame:
    features = _features()
    features["primary_theme"] = "other"
    features["industry"] = "other"
    features["theme_active"] = True
    features["overnight_gap_risk_score"] = 8.0
    features["vol_20d"] = 0.03
    features["theme_strength_delta_score"] = 10.0

    leader = features["symbol"].eq("AAA")
    features.loc[leader, "primary_theme"] = "semiconductors"
    features.loc[leader, "industry"] = "semiconductors"
    features.loc[leader, "theme_score"] = 64.0
    features.loc[leader, "rs_63d"] = 0.16
    features.loc[leader, "vol_20d"] = 0.04

    replacement = features[features["symbol"].eq("AAA")].copy()
    replacement["symbol"] = "DDD"
    replacement["theme_score"] = 68.0
    replacement["rs_63d"] = 0.26
    replacement["primary_theme"] = "semiconductors"
    replacement["industry"] = "semiconductors"
    replacement["theme_active"] = True
    replacement["trend_adx"] = 30.0
    replacement["volume_expansion"] = 0.78
    replacement["theme_strength_delta_score"] = 28.0
    replacement["overnight_gap_risk_score"] = 6.0
    replacement["vol_20d"] = 0.12 if high_vol_candidate else 0.03
    replacement = replacement.sort_values("date").reset_index(drop=True)
    tail = replacement.index[-8:]
    replacement.loc[tail, "adj_close"] = replacement.loc[tail, "adj_close"] * np.linspace(0.98, 0.88, len(tail))
    replacement["open"] = replacement["adj_close"]
    replacement["close"] = replacement["adj_close"]
    replacement["high"] = replacement["adj_close"] * 1.01
    replacement["low"] = replacement["adj_close"] * 0.99
    replacement["ma_20"] = replacement["adj_close"] / 1.01
    replacement["ma_50"] = replacement["adj_close"] * 0.96
    replacement["ma_200"] = replacement["adj_close"] * 0.90
    replacement["distance_to_prior_high_252"] = 0.0
    replacement.loc[tail, "distance_to_prior_high_252"] = -0.06
    if activation_ready_candidate:
        replacement.loc[tail, "volume_expansion"] = 1.25
    replacement["return_1d"] = replacement["adj_close"].pct_change().fillna(0.0)
    features = pd.concat([features, replacement], ignore_index=True)
    return features.sort_values(["date", "symbol"]).reset_index(drop=True)


def _compound_leader_credit_features() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=330)
    specs = {
        "LEAD": {"base": 80.0, "daily": 0.0050, "tail": 0.0010, "theme": 84.0, "rs": 0.34},
        "WEAK": {"base": 75.0, "daily": 0.0005, "tail": 0.0002, "theme": 72.0, "rs": 0.18},
        "CHASE": {"base": 78.0, "daily": 0.0038, "tail": 0.0100, "theme": 84.0, "rs": 0.33},
        "SPY": {"base": 100.0, "daily": 0.0010, "tail": 0.0008, "theme": 50.0, "rs": 0.0},
    }
    rows = []
    for symbol, spec in specs.items():
        returns = np.full(len(dates), spec["daily"], dtype=float)
        returns[-10:] = spec["tail"]
        close = spec["base"] * np.cumprod(1.0 + returns)
        ma20 = pd.Series(close).rolling(20, min_periods=1).mean().to_numpy()
        ma50 = pd.Series(close).rolling(50, min_periods=1).mean().to_numpy()
        high252 = pd.Series(close).rolling(252, min_periods=1).max().to_numpy()
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_close": close[idx],
                    "open": close[idx],
                    "high": close[idx] * 1.01,
                    "low": close[idx] * 0.99,
                    "close": close[idx],
                    "return_1d": returns[idx],
                    "ma_20": ma20[idx],
                    "ma_50": ma50[idx],
                    "distance_to_high_252": close[idx] / high252[idx] - 1.0,
                    "distance_to_prior_high_252": close[idx] / high252[idx] - 1.0,
                    "technical_score": 69.0 if symbol == "LEAD" else 64.0 if symbol == "WEAK" else 68.0 if symbol == "CHASE" else 52.0,
                    "final_score": 69.0 if symbol == "LEAD" else 63.0 if symbol == "WEAK" else 69.0 if symbol == "CHASE" else 50.0,
                    "relative_strength_score": 90.0 if symbol == "LEAD" else 74.0 if symbol == "WEAK" else 89.0 if symbol == "CHASE" else 50.0,
                    "theme_score": spec["theme"],
                    "mom_return": 0.22 if symbol == "LEAD" else 0.09 if symbol == "WEAK" else 0.24 if symbol == "CHASE" else 0.03,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 6.0,
                    "benchmark_ret_63d": 0.08,
                    "benchmark_risk_on": True,
                    "theme_active": symbol != "SPY",
                    "trend_adx": 28.0 if symbol in {"LEAD", "CHASE"} else 22.0,
                    "primary_theme": "semiconductors" if symbol != "SPY" else "benchmark_etfs",
                    "theme_reason": "semiconductors" if symbol != "SPY" else "benchmark_etfs",
                    "rs_63d": spec["rs"],
                }
            )
    features = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    features["return_1d"] = features.groupby("symbol")["adj_close"].pct_change().fillna(0.0)
    return features


def _pullback_reclaim_same_theme_gap_features(peer_gap_risk: float = 6.0) -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", periods=7)
    rows = []
    for idx, date in enumerate(dates):
        for symbol in ["LEAD", "PEER1", "PEER2", "OTHER"]:
            if symbol == "LEAD":
                close = 98.0 if idx < len(dates) - 1 else 101.0
                theme = "semiconductors"
                rs = 88.0
                theme_score = 68.0
                gap_risk = 8.0
                final_score = 70.0
                volume_expansion = 1.20
            elif symbol in {"PEER1", "PEER2"}:
                close = 102.0
                theme = "semiconductors"
                rs = 84.0
                theme_score = 66.0
                gap_risk = peer_gap_risk
                final_score = 62.0
                volume_expansion = 1.05
            else:
                close = 101.0
                theme = "other"
                rs = 74.0
                theme_score = 55.0
                gap_risk = 12.0
                final_score = 60.0
                volume_expansion = 1.00
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_close": close,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "ma_20": 100.0,
                    "ma_50": 95.0,
                    "ma_200": 90.0,
                    "mom_distance_high": -0.08 if symbol == "LEAD" else -0.03,
                    "mom_return": 0.16 if symbol == "LEAD" else 0.10,
                    "mom_volume_expansion": volume_expansion,
                    "relative_strength_score": rs,
                    "theme_score": theme_score,
                    "final_score": final_score,
                    "trend_adx": 28.0 if symbol == "LEAD" else 24.0,
                    "overnight_gap_risk_score": gap_risk,
                    "benchmark_risk_on": True,
                    "benchmark_ret_63d": 0.08,
                    "theme_active": True,
                    "primary_theme": theme,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    frame["return_1d"] = frame.groupby("symbol")["adj_close"].pct_change().fillna(0.0)
    return frame


def _post_earnings_drift_features() -> pd.DataFrame:
    features = _features()
    features["primary_theme"] = "unclassified"
    features["theme_active"] = False
    features["overnight_gap_risk_score"] = 10.0
    features["days_to_earnings"] = np.nan
    features["surprise_eps_pct"] = 0.0
    features["expected_move"] = 0.0

    latest_date = features["date"].max()
    aaa_tail = features["date"].ge(latest_date - pd.Timedelta(days=5)) & features["symbol"].eq("AAA")
    aaa_latest = features["date"].eq(latest_date) & features["symbol"].eq("AAA")
    features.loc[aaa_tail, "primary_theme"] = "semiconductors"
    features.loc[aaa_tail, "theme_active"] = True
    features.loc[aaa_tail, "days_to_earnings"] = -2
    features.loc[aaa_tail, "surprise_eps_pct"] = 18.0
    features.loc[aaa_tail, "expected_move"] = 0.04
    features.loc[aaa_tail, "overnight_gap_risk_score"] = 8.0
    features.loc[aaa_tail, "theme_score"] = 68.0
    features.loc[aaa_tail, "trend_adx"] = 28.0
    features.loc[aaa_latest, "ma_20"] = features.loc[aaa_latest, "adj_close"] / 1.03

    ccc_latest = features["date"].eq(latest_date) & features["symbol"].eq("CCC")
    features.loc[ccc_latest, "primary_theme"] = "optical_networking"
    features.loc[ccc_latest, "theme_active"] = True
    features.loc[ccc_latest, "theme_score"] = 66.0
    features.loc[ccc_latest, "ma_20"] = features.loc[ccc_latest, "adj_close"] / 1.04
    return features.sort_values(["date", "symbol"]).reset_index(drop=True)


def _boundary_rank_promotion_features(risk_on: bool = True, high_gap: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=330)
    specs = {
        "LEAD": {"base": 72.0, "daily": 0.0125, "tail": 0.0040, "theme": 76.0, "rs": 0.22, "adx": 28.0},
        "PROMO": {"base": 66.0, "daily": 0.0112, "tail": -0.0015, "theme": 88.0, "rs": 0.36, "adx": 30.0},
        "MID": {"base": 58.0, "daily": 0.0116, "tail": 0.0025, "theme": 60.0, "rs": 0.16, "adx": 24.0},
        "ALT": {"base": 54.0, "daily": 0.0102, "tail": 0.0010, "theme": 56.0, "rs": 0.12, "adx": 22.0},
        "LOW": {"base": 50.0, "daily": 0.0080, "tail": 0.0005, "theme": 52.0, "rs": 0.05, "adx": 20.0},
        "SPY": {"base": 100.0, "daily": 0.0030 if risk_on else -0.0020, "tail": 0.0020 if risk_on else -0.0020, "theme": 50.0, "rs": 0.0, "adx": 20.0},
    }
    rows = []
    for symbol, spec in specs.items():
        returns = np.full(len(dates), spec["daily"], dtype=float)
        returns[-8:] = spec["tail"]
        close = spec["base"] * np.cumprod(1.0 + returns)
        ma20 = pd.Series(close).rolling(20, min_periods=1).mean().to_numpy()
        ma50 = pd.Series(close).rolling(50, min_periods=1).mean().to_numpy()
        ma200 = pd.Series(close).rolling(200, min_periods=1).mean().to_numpy()
        high252 = pd.Series(close).rolling(252, min_periods=1).max().to_numpy()
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_close": close[idx],
                    "open": close[idx],
                    "high": close[idx] * 1.01,
                    "low": close[idx] * 0.99,
                    "close": close[idx],
                    "return_1d": returns[idx],
                    "ma_20": ma20[idx],
                    "ma_50": ma50[idx],
                    "ma_200": ma200[idx],
                    "distance_to_high_252": close[idx] / high252[idx] - 1.0,
                    "distance_to_prior_high_252": close[idx] / high252[idx] - 1.0,
                    "volume_expansion": 0.84 if symbol == "PROMO" else 0.92 if symbol == "MID" else 1.0,
                    "theme_score": spec["theme"],
                    "fundamental_score": 52.0,
                    "event_risk_score": 0.0,
                    "trend_adx": spec["adx"],
                    "rs_63d": spec["rs"],
                    "primary_theme": "semiconductors" if symbol != "SPY" else "benchmark_etfs",
                    "theme_active": symbol != "SPY",
                    "overnight_gap_risk_score": 40.0 if high_gap and symbol == "PROMO" else 6.0 if symbol == "PROMO" else 9.0,
                }
            )
    features = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    features["return_1d"] = features.groupby("symbol")["adj_close"].pct_change().fillna(0.0)
    return features


def test_momentum_long_gate_requires_theme_and_relative_strength() -> None:
    strategy = CrossSectionalMomentumLongShort(
        {
            "lookback_returns": 21,
            "skip_recent_days": 3,
            "long_quantile": 0.50,
            "short_quantile": 0.0,
            "max_positions": 10,
            "long_min_theme_score": 60,
            "long_min_relative_strength_score": 60,
            "long_require_price_above_ma50": True,
            "long_require_price_above_ma200": True,
        }
    )

    signals = strategy.generate_signals(_features(), StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    longs = latest[latest["signal"] > 0]

    assert not longs.empty
    assert set(longs["symbol"]) == {"AAA"}
    assert (longs["theme_score"] >= 60).all()
    assert (longs["relative_strength_score"] >= 60).all()


def test_momentum_custom_final_score_weights_prioritize_tech_and_relative_strength() -> None:
    frame = pd.DataFrame(
        {
            "technical_score": [90.0, 55.0],
            "relative_strength_score": [88.0, 52.0],
            "theme_score": [70.0, 70.0],
            "fundamental_score": [25.0, 95.0],
            "event_risk_score": [0.0, 0.0],
        }
    )

    default_scores = _momentum_final_score(frame.copy(), {})
    custom_frame = frame.copy()
    custom_scores = _momentum_final_score(
        custom_frame,
        {
            "momentum_custom_final_score_weights": True,
            "momentum_final_score_technical_weight": 0.42,
            "momentum_final_score_relative_strength_weight": 0.33,
            "momentum_final_score_theme_weight": 0.20,
            "momentum_final_score_fundamental_weight": 0.05,
            "momentum_final_score_event_risk_penalty": 0.05,
        },
    )
    blended_frame = frame.copy()
    blended_scores = _momentum_final_score(
        blended_frame,
        {
            "momentum_custom_final_score_weights": True,
            "momentum_final_score_technical_weight": 0.42,
            "momentum_final_score_relative_strength_weight": 0.33,
            "momentum_final_score_theme_weight": 0.20,
            "momentum_final_score_fundamental_weight": 0.05,
            "momentum_final_score_event_risk_penalty": 0.05,
            "momentum_custom_final_score_blend": 0.65,
        },
    )

    assert default_scores.iloc[0] < custom_scores.iloc[0]
    assert custom_scores.iloc[0] > custom_scores.iloc[1]
    assert default_scores.iloc[0] < blended_scores.iloc[0] < custom_scores.iloc[0]
    assert default_scores.iloc[1] < blended_scores.iloc[1] < custom_scores.iloc[1]
    assert custom_frame["momentum_score_weight_profile"].str.startswith("blend1.00_tech0.42_rs0.33").all()
    assert blended_frame["momentum_score_weight_profile"].str.startswith("blend0.65_tech0.42_rs0.33").all()


def test_momentum_post_earnings_drift_overlay_boosts_recent_positive_surprise_leader() -> None:
    features = _post_earnings_drift_features()
    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.50,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 55,
        "long_min_relative_strength_score": 55,
        "long_require_price_above_ma50": True,
    }
    overlay_params = {
        **base_params,
        "post_earnings_drift_overlay": True,
        "post_earnings_drift_min_surprise_eps_pct": 8.0,
        "post_earnings_drift_min_days_since_earnings": 1,
        "post_earnings_drift_max_days_since_earnings": 6,
        "post_earnings_drift_min_relative_strength_score": 68.0,
        "post_earnings_drift_min_theme_score": 58.0,
        "post_earnings_drift_min_mom_return": 0.05,
        "post_earnings_drift_max_above_ma20_pct": 0.06,
        "post_earnings_drift_max_overnight_gap_risk_score_circuit_breaker": 30.0,
        "post_earnings_drift_max_expected_move_pct_circuit_breaker": 0.18,
        "post_earnings_drift_min_benchmark_ret63d_circuit_breaker": 0.0,
        "post_earnings_drift_require_benchmark_risk_on": True,
        "post_earnings_drift_require_theme_active": True,
        "post_earnings_drift_require_price_above_ma50": True,
        "post_earnings_drift_max_score_boost": 1.75,
    }

    base_signals = CrossSectionalMomentumLongShort(base_params).generate_signals(features.copy(), StrategyContext(benchmark="SPY"))
    overlay_signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features.copy(), StrategyContext(benchmark="SPY"))
    latest_base = base_signals[base_signals["date"] == base_signals["date"].max()]
    latest_overlay = overlay_signals[overlay_signals["date"] == overlay_signals["date"].max()]
    aaa_base = latest_base[latest_base["symbol"].eq("AAA")].iloc[0]
    aaa_overlay = latest_overlay[latest_overlay["symbol"].eq("AAA")].iloc[0]

    assert aaa_overlay["final_score"] > aaa_base["final_score"]
    assert "post_earnings_drift=" in aaa_overlay["score_decomposition"]
    assert "post_earnings_block=0" in aaa_overlay["score_decomposition"]
    assert aaa_overlay["signal"] > 0


def test_momentum_post_earnings_drift_overlay_respects_extension_circuit_breaker() -> None:
    features = _post_earnings_drift_features()
    latest_date = features["date"].max()
    aaa_latest = features["date"].eq(latest_date) & features["symbol"].eq("AAA")
    features.loc[aaa_latest, "ma_20"] = features.loc[aaa_latest, "adj_close"] / 1.15
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.50,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 55,
        "long_min_relative_strength_score": 55,
        "long_require_price_above_ma50": True,
        "post_earnings_drift_overlay": True,
        "post_earnings_drift_min_surprise_eps_pct": 8.0,
        "post_earnings_drift_min_days_since_earnings": 1,
        "post_earnings_drift_max_days_since_earnings": 6,
        "post_earnings_drift_min_relative_strength_score": 68.0,
        "post_earnings_drift_min_theme_score": 58.0,
        "post_earnings_drift_min_mom_return": 0.05,
        "post_earnings_drift_max_above_ma20_pct": 0.06,
        "post_earnings_drift_max_overnight_gap_risk_score_circuit_breaker": 30.0,
        "post_earnings_drift_max_expected_move_pct_circuit_breaker": 0.18,
        "post_earnings_drift_min_benchmark_ret63d_circuit_breaker": 0.0,
        "post_earnings_drift_require_benchmark_risk_on": True,
        "post_earnings_drift_require_theme_active": True,
        "post_earnings_drift_require_price_above_ma50": True,
        "post_earnings_drift_max_score_boost": 1.75,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    aaa_row = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert "post_earnings_block=1" in aaa_row["score_decomposition"]


def test_boundary_rank_promotion_overlay_promotes_near_cutoff_candidate() -> None:
    features = _boundary_rank_promotion_features()
    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.10,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 55.0,
        "long_min_technical_score": 35.0,
        "long_min_relative_strength_score": 50.0,
        "long_min_theme_score": 50.0,
        "require_long_theme_active": True,
        "long_require_price_above_ma50": True,
    }
    overlay_params = {
        **base_params,
        "boundary_rank_promotion_overlay": True,
        "boundary_rank_promotion_rank_buffer_below": 0.12,
        "boundary_rank_promotion_min_relative_strength_score": 86.0,
        "boundary_rank_promotion_min_theme_score": 72.0,
        "boundary_rank_promotion_min_technical_score": 35.0,
        "boundary_rank_promotion_min_mom_return": 0.05,
        "boundary_rank_promotion_min_drawdown_from_high": 0.01,
        "boundary_rank_promotion_max_drawdown_from_high": 0.12,
        "boundary_rank_promotion_max_above_ma20_pct": 0.05,
        "boundary_rank_promotion_max_final_score_deficit": 10.0,
        "boundary_rank_promotion_max_promotions_per_date": 1,
        "boundary_rank_promotion_promoted_score_step": 0.10,
        "boundary_rank_promotion_require_benchmark_risk_on": False,
        "boundary_rank_promotion_require_theme_active": True,
        "boundary_rank_promotion_require_price_above_ma50": True,
        "boundary_rank_promotion_max_event_risk_score_circuit_breaker": 15.0,
        "boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker": -1.0,
        "boundary_rank_promotion_min_adx_circuit_breaker": 18.0,
    }

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features.copy(), StrategyContext(benchmark="SPY"))
    overlay = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features.copy(), StrategyContext(benchmark="SPY"))
    latest_base = base[base["date"] == base["date"].max()]
    latest_overlay = overlay[overlay["date"] == overlay["date"].max()]
    promo_base = latest_base[latest_base["symbol"].eq("PROMO")].iloc[0]
    promo_overlay = latest_overlay[latest_overlay["symbol"].eq("PROMO")].iloc[0]

    assert promo_base["signal"] == 0.0
    assert promo_overlay["signal"] == 1.0
    assert promo_overlay["final_score"] > promo_base["final_score"]
    assert "boundary_promo=1" in promo_overlay["score_decomposition"]
    assert "boundary_promo_block=0" in promo_overlay["score_decomposition"]


def test_boundary_rank_promotion_overlay_respects_gap_risk_circuit_breaker() -> None:
    features = _boundary_rank_promotion_features(high_gap=True)
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.10,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 55.0,
        "long_min_technical_score": 35.0,
        "long_min_relative_strength_score": 50.0,
        "long_min_theme_score": 50.0,
        "require_long_theme_active": True,
        "long_require_price_above_ma50": True,
        "boundary_rank_promotion_overlay": True,
        "boundary_rank_promotion_rank_buffer_below": 0.12,
        "boundary_rank_promotion_min_relative_strength_score": 86.0,
        "boundary_rank_promotion_min_theme_score": 72.0,
        "boundary_rank_promotion_min_technical_score": 35.0,
        "boundary_rank_promotion_min_mom_return": 0.05,
        "boundary_rank_promotion_min_drawdown_from_high": 0.01,
        "boundary_rank_promotion_max_drawdown_from_high": 0.12,
        "boundary_rank_promotion_max_above_ma20_pct": 0.05,
        "boundary_rank_promotion_max_final_score_deficit": 10.0,
        "boundary_rank_promotion_max_promotions_per_date": 1,
        "boundary_rank_promotion_promoted_score_step": 0.10,
        "boundary_rank_promotion_require_benchmark_risk_on": False,
        "boundary_rank_promotion_require_theme_active": True,
        "boundary_rank_promotion_require_price_above_ma50": True,
        "boundary_rank_promotion_max_event_risk_score_circuit_breaker": 15.0,
        "boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker": -1.0,
        "boundary_rank_promotion_min_adx_circuit_breaker": 18.0,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    promo = latest[latest["symbol"].eq("PROMO")].iloc[0]

    assert promo["signal"] == 0.0
    assert "boundary_promo=0" in promo["score_decomposition"]
    assert "boundary_promo_block=1" in promo["score_decomposition"]


def test_boundary_rank_promotion_compound_pullback_overlay_requires_long_horizon_leadership() -> None:
    features = _boundary_rank_promotion_features()
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.10,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 55.0,
        "long_min_technical_score": 35.0,
        "long_min_relative_strength_score": 50.0,
        "long_min_theme_score": 50.0,
        "require_long_theme_active": True,
        "long_require_price_above_ma50": True,
        "boundary_rank_promotion_overlay": True,
        "boundary_rank_promotion_rank_buffer_below": 0.12,
        "boundary_rank_promotion_min_relative_strength_score": 86.0,
        "boundary_rank_promotion_min_theme_score": 72.0,
        "boundary_rank_promotion_min_technical_score": 35.0,
        "boundary_rank_promotion_min_mom_return": 0.05,
        "boundary_rank_promotion_min_drawdown_from_high": 0.01,
        "boundary_rank_promotion_max_drawdown_from_high": 0.12,
        "boundary_rank_promotion_max_above_ma20_pct": 0.05,
        "boundary_rank_promotion_max_final_score_deficit": 10.0,
        "boundary_rank_promotion_max_promotions_per_date": 1,
        "boundary_rank_promotion_promoted_score_step": 0.10,
        "boundary_rank_promotion_require_benchmark_risk_on": False,
        "boundary_rank_promotion_require_theme_active": True,
        "boundary_rank_promotion_require_price_above_ma50": True,
        "boundary_rank_promotion_max_event_risk_score_circuit_breaker": 15.0,
        "boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker": -1.0,
        "boundary_rank_promotion_min_adx_circuit_breaker": 18.0,
        "boundary_rank_promotion_compound_pullback_overlay": True,
        "boundary_rank_promotion_compound_pullback_min_ret_100d_rank": 0.50,
        "boundary_rank_promotion_compound_pullback_min_252d_voladj_rank": 0.40,
        "boundary_rank_promotion_compound_pullback_max_ret_5d": 0.01,
        "boundary_rank_promotion_compound_pullback_max_ret_10d": 0.02,
        "boundary_rank_promotion_compound_pullback_max_volume_expansion": 0.90,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    promo = latest[latest["symbol"].eq("PROMO")].iloc[0]

    assert promo["signal"] == 1.0
    assert "boundary_promo=1" in promo["score_decomposition"]
    assert "boundary_promo_compound=1" in promo["score_decomposition"]


def test_exit_quality_rank_credit_only_promotes_near_cutoff_leaders() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01", "2026-05-01", "2026-05-01"]),
            "symbol": ["LEAD", "WEAK", "LOWRANK"],
            "score_rank": [0.78, 0.78, 0.60],
            "final_score": [80.0, 80.0, 84.0],
            "technical_score": [72.0, 72.0, 78.0],
            "relative_strength_score": [92.0, 70.0, 95.0],
            "theme_score": [70.0, 70.0, 72.0],
            "mom_return": [0.24, 0.24, 0.30],
            "mom_distance_high": [-0.06, -0.06, -0.05],
            "adj_close": [104.0, 104.0, 104.0],
            "ma_20": [100.0, 100.0, 100.0],
            "ma_50": [95.0, 95.0, 95.0],
            "event_risk_score": [0.0, 0.0, 0.0],
            "overnight_gap_risk_score": [5.0, 5.0, 5.0],
            "benchmark_ret_63d": [0.05, 0.05, 0.05],
            "benchmark_risk_on": [True, True, True],
            "trend_adx": [26.0, 26.0, 26.0],
            "theme_active": [True, True, True],
        }
    )

    out = _apply_exit_quality_rank_credit_overlay(
        frame,
        {
            "exit_quality_rank_credit_overlay": True,
            "exit_quality_rank_credit_rank_buffer_below": 0.07,
            "exit_quality_rank_credit_rank_buffer_above": 0.01,
            "exit_quality_rank_credit_min_final_score": 68.0,
            "exit_quality_rank_credit_min_relative_strength_score": 88.0,
            "exit_quality_rank_credit_min_theme_score": 62.0,
            "exit_quality_rank_credit_min_technical_score": 60.0,
            "exit_quality_rank_credit_min_mom_return": 0.12,
            "exit_quality_rank_credit_max_drawdown_from_high": 0.18,
            "exit_quality_rank_credit_max_above_ma20_pct": 0.08,
            "exit_quality_rank_credit_require_benchmark_risk_on": True,
            "exit_quality_rank_credit_require_theme_active": False,
            "exit_quality_rank_credit_require_price_above_ma50": True,
            "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": 20.0,
            "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
            "exit_quality_rank_credit_min_adx_circuit_breaker": 18.0,
            "exit_quality_rank_credit_max_score_rank_credit": 0.04,
        },
        long_quantile=0.20,
    ).set_index("symbol")

    assert bool(out.loc["LEAD", "exit_quality_rank_credit_eligible"]) is True
    assert out.loc["LEAD", "exit_quality_effective_score_rank"] >= 0.80
    assert out.loc["LEAD", "final_score"] == frame.loc[0, "final_score"]
    assert bool(out.loc["WEAK", "exit_quality_rank_credit_eligible"]) is False
    assert out.loc["WEAK", "exit_quality_effective_score_rank"] == out.loc["WEAK", "score_rank"]
    assert bool(out.loc["LOWRANK", "exit_quality_rank_credit_eligible"]) is False


def test_exit_quality_rank_credit_theme_support_overlay_requires_improving_theme_support() -> None:
    dates = pd.bdate_range("2026-04-01", periods=7)
    rows = []
    for i, date in enumerate(dates):
        ai_theme = 58.0 + i * 2.5
        ai_rs = 92.0 + i * 0.5
        stale_theme = 72.0 - i * 1.5
        stale_rs = 92.0 - i * 0.75
        rows.extend(
            [
                {
                    "date": date,
                    "symbol": "LEAD" if i == len(dates) - 1 else f"AI{i}",
                    "score_rank": 0.78 if i == len(dates) - 1 else 0.70,
                    "final_score": 80.0,
                    "technical_score": 72.0,
                    "relative_strength_score": ai_rs,
                    "theme_score": ai_theme,
                    "mom_return": 0.24,
                    "mom_distance_high": -0.06,
                    "adj_close": 104.0,
                    "ma_20": 100.0,
                    "ma_50": 95.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 5.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 26.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                },
                {
                    "date": date,
                    "symbol": f"AIP{i}",
                    "score_rank": 0.55,
                    "final_score": 74.0,
                    "technical_score": 66.0,
                    "relative_strength_score": ai_rs + 1.5,
                    "theme_score": ai_theme + 2.0,
                    "mom_return": 0.16,
                    "mom_distance_high": -0.08,
                    "adj_close": 102.0,
                    "ma_20": 99.0,
                    "ma_50": 94.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 6.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 24.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                },
                {
                    "date": date,
                    "symbol": "STALE" if i == len(dates) - 1 else f"ST{i}",
                    "score_rank": 0.78 if i == len(dates) - 1 else 0.69,
                    "final_score": 80.0,
                    "technical_score": 72.0,
                    "relative_strength_score": stale_rs,
                    "theme_score": stale_theme,
                    "mom_return": 0.24,
                    "mom_distance_high": -0.06,
                    "adj_close": 104.0,
                    "ma_20": 100.0,
                    "ma_50": 95.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 5.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 26.0,
                    "theme_active": True,
                    "primary_theme": "stale_theme",
                },
                {
                    "date": date,
                    "symbol": f"STP{i}",
                    "score_rank": 0.54,
                    "final_score": 74.0,
                    "technical_score": 66.0,
                    "relative_strength_score": stale_rs - 1.0,
                    "theme_score": stale_theme - 2.0,
                    "mom_return": 0.15,
                    "mom_distance_high": -0.09,
                    "adj_close": 101.0,
                    "ma_20": 99.0,
                    "ma_50": 94.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 6.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 24.0,
                    "theme_active": True,
                    "primary_theme": "stale_theme",
                },
            ]
        )
    out = _apply_exit_quality_rank_credit_overlay(
        pd.DataFrame(rows),
        {
            "exit_quality_rank_credit_overlay": True,
            "exit_quality_rank_credit_theme_support_overlay": True,
            "exit_quality_rank_credit_rank_buffer_below": 0.07,
            "exit_quality_rank_credit_rank_buffer_above": 0.01,
            "exit_quality_rank_credit_min_final_score": 68.0,
            "exit_quality_rank_credit_min_relative_strength_score": 88.0,
            "exit_quality_rank_credit_min_theme_score": 60.0,
            "exit_quality_rank_credit_min_technical_score": 60.0,
            "exit_quality_rank_credit_min_mom_return": 0.12,
            "exit_quality_rank_credit_max_drawdown_from_high": 0.18,
            "exit_quality_rank_credit_max_above_ma20_pct": 0.08,
            "exit_quality_rank_credit_require_benchmark_risk_on": True,
            "exit_quality_rank_credit_require_theme_active": True,
            "exit_quality_rank_credit_require_price_above_ma50": True,
            "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": 20.0,
            "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
            "exit_quality_rank_credit_min_adx_circuit_breaker": 18.0,
            "exit_quality_rank_credit_max_score_rank_credit": 0.05,
            "exit_quality_rank_credit_theme_support_lookback_days": 3,
            "exit_quality_rank_credit_theme_support_min_peer_count": 2,
            "exit_quality_rank_credit_theme_support_min_theme_score_prior": 60.0,
            "exit_quality_rank_credit_theme_support_min_peer_rs_prior": 92.0,
            "exit_quality_rank_credit_theme_support_min_combined_delta": 1.0,
        },
        long_quantile=0.20,
    )
    latest_out = out[out["date"] == out["date"].max()].set_index("symbol")

    assert bool(latest_out.loc["LEAD", "exit_quality_theme_support_eligible"]) is True
    assert latest_out.loc["LEAD", "exit_quality_theme_support_score"] > 0.0
    assert bool(latest_out.loc["LEAD", "exit_quality_rank_credit_eligible"]) is True
    assert latest_out.loc["LEAD", "exit_quality_effective_score_rank"] > latest_out.loc["LEAD", "score_rank"]
    assert bool(latest_out.loc["STALE", "exit_quality_theme_support_eligible"]) is False
    assert bool(latest_out.loc["STALE", "exit_quality_rank_credit_eligible"]) is False
    assert latest_out.loc["STALE", "exit_quality_effective_score_rank"] == latest_out.loc["STALE", "score_rank"]


def test_compound_leader_score_credit_promotes_only_controlled_long_horizon_leaders() -> None:
    frame = _compound_leader_credit_features()

    out = _apply_compound_leader_score_credit_overlay(
        frame,
        {
            "compound_leader_score_credit_overlay": True,
            "compound_leader_score_credit_min_126d_voladj_rank": 0.50,
            "compound_leader_score_credit_min_252d_voladj_rank": 0.00,
            "compound_leader_score_credit_min_final_score": 60.0,
            "compound_leader_score_credit_min_relative_strength_score": 80.0,
            "compound_leader_score_credit_min_theme_score": 60.0,
            "compound_leader_score_credit_min_mom_return": 0.05,
            "compound_leader_score_credit_min_theme_peer_count": 2,
            "compound_leader_score_credit_max_ret_10d": 0.04,
            "compound_leader_score_credit_max_drawdown_from_high": 0.12,
            "compound_leader_score_credit_max_above_ma20_pct": 0.08,
            "compound_leader_score_credit_require_benchmark_risk_on": True,
            "compound_leader_score_credit_require_theme_active": True,
            "compound_leader_score_credit_require_price_above_ma50": True,
            "compound_leader_score_credit_max_event_risk_score_circuit_breaker": 15.0,
            "compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker": 25.0,
            "compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
            "compound_leader_score_credit_min_adx_circuit_breaker": 20.0,
            "compound_leader_score_credit_max_score_boost": 1.75,
        },
    )
    latest = out[out["date"] == out["date"].max()].set_index("symbol")

    assert bool(latest.loc["LEAD", "compound_leader_score_credit_eligible"]) is True
    assert latest.loc["LEAD", "compound_leader_score_credit_boost"] > 0.0
    assert latest.loc["LEAD", "final_score"] > frame.loc[frame["symbol"].eq("LEAD"), "final_score"].iloc[-1]
    assert bool(latest.loc["WEAK", "compound_leader_score_credit_eligible"]) is False
    assert latest.loc["WEAK", "compound_leader_score_credit_boost"] == 0.0
    assert bool(latest.loc["CHASE", "compound_leader_score_credit_eligible"]) is False
    assert latest.loc["CHASE", "compound_leader_score_credit_boost"] == 0.0


def test_compound_leader_score_credit_volume_confirmation_requires_ret100_rank_and_volume() -> None:
    frame = _compound_leader_credit_features()
    frame["volume_expansion"] = np.where(frame["symbol"].eq("LEAD"), 1.18, 0.92)

    params = {
        "compound_leader_score_credit_overlay": True,
        "compound_leader_score_credit_volume_confirmation_overlay": True,
        "compound_leader_score_credit_min_126d_voladj_rank": 0.50,
        "compound_leader_score_credit_min_252d_voladj_rank": 0.00,
        "compound_leader_score_credit_min_final_score": 60.0,
        "compound_leader_score_credit_min_relative_strength_score": 80.0,
        "compound_leader_score_credit_min_theme_score": 60.0,
        "compound_leader_score_credit_min_mom_return": 0.05,
        "compound_leader_score_credit_min_theme_peer_count": 2,
        "compound_leader_score_credit_min_ret_100d_rank": 0.50,
        "compound_leader_score_credit_min_volume_expansion": 1.0,
        "compound_leader_score_credit_max_ret_10d": 0.04,
        "compound_leader_score_credit_max_drawdown_from_high": 0.12,
        "compound_leader_score_credit_max_above_ma20_pct": 0.08,
        "compound_leader_score_credit_require_benchmark_risk_on": True,
        "compound_leader_score_credit_require_theme_active": True,
        "compound_leader_score_credit_require_price_above_ma50": True,
        "compound_leader_score_credit_max_event_risk_score_circuit_breaker": 15.0,
        "compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
        "compound_leader_score_credit_min_adx_circuit_breaker": 20.0,
        "compound_leader_score_credit_max_score_boost": 1.75,
    }

    out = _apply_compound_leader_score_credit_overlay(frame, params)
    latest = out[out["date"] == out["date"].max()].set_index("symbol")

    assert bool(latest.loc["LEAD", "compound_leader_volume_confirmation_eligible"]) is True
    assert latest.loc["LEAD", "compound_leader_ret_100d_rank"] >= 0.50
    assert latest.loc["LEAD", "compound_leader_volume_expansion_signal"] >= 1.0
    assert bool(latest.loc["LEAD", "compound_leader_score_credit_eligible"]) is True
    assert bool(latest.loc["WEAK", "compound_leader_volume_confirmation_eligible"]) is False
    assert bool(latest.loc["WEAK", "compound_leader_score_credit_eligible"]) is False

    weak_volume = frame.copy()
    weak_volume.loc[weak_volume["symbol"].eq("LEAD"), "volume_expansion"] = 0.88
    weak_out = _apply_compound_leader_score_credit_overlay(weak_volume, params)
    weak_latest = weak_out[weak_out["date"] == weak_out["date"].max()].set_index("symbol")

    assert bool(weak_latest.loc["LEAD", "compound_leader_volume_confirmation_eligible"]) is False
    assert bool(weak_latest.loc["LEAD", "compound_leader_score_credit_eligible"]) is False


def test_compound_leader_score_credit_boundary_overlay_limits_boost_to_near_cutoff_names() -> None:
    frame = _compound_leader_credit_features()

    broad = _apply_compound_leader_score_credit_overlay(
        frame,
        {
            "compound_leader_score_credit_overlay": True,
            "compound_leader_score_credit_min_126d_voladj_rank": 0.50,
            "compound_leader_score_credit_min_252d_voladj_rank": 0.00,
            "compound_leader_score_credit_min_final_score": 60.0,
            "compound_leader_score_credit_min_relative_strength_score": 80.0,
            "compound_leader_score_credit_min_theme_score": 60.0,
            "compound_leader_score_credit_min_mom_return": 0.05,
            "compound_leader_score_credit_min_theme_peer_count": 2,
            "compound_leader_score_credit_max_ret_10d": 0.04,
            "compound_leader_score_credit_max_drawdown_from_high": 0.12,
            "compound_leader_score_credit_max_above_ma20_pct": 0.08,
            "compound_leader_score_credit_require_benchmark_risk_on": True,
            "compound_leader_score_credit_require_theme_active": True,
            "compound_leader_score_credit_require_price_above_ma50": True,
            "compound_leader_score_credit_max_event_risk_score_circuit_breaker": 15.0,
            "compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker": 25.0,
            "compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
            "compound_leader_score_credit_min_adx_circuit_breaker": 20.0,
            "compound_leader_score_credit_max_score_boost": 1.75,
            "long_quantile": 0.20,
        },
    )
    broad_latest = broad[broad["date"] == broad["date"].max()].set_index("symbol")
    assert bool(broad_latest.loc["LEAD", "compound_leader_score_credit_eligible"]) is True
    assert broad_latest.loc["LEAD", "compound_leader_score_credit_boost"] > 0.0

    bounded = _apply_compound_leader_score_credit_overlay(
        frame,
        {
            "compound_leader_score_credit_overlay": True,
            "compound_leader_score_credit_boundary_overlay": True,
            "compound_leader_score_credit_rank_buffer_below": 0.03,
            "compound_leader_score_credit_rank_buffer_above": 0.01,
            "compound_leader_score_credit_min_126d_voladj_rank": 0.50,
            "compound_leader_score_credit_min_252d_voladj_rank": 0.00,
            "compound_leader_score_credit_min_final_score": 60.0,
            "compound_leader_score_credit_min_relative_strength_score": 80.0,
            "compound_leader_score_credit_min_theme_score": 60.0,
            "compound_leader_score_credit_min_mom_return": 0.05,
            "compound_leader_score_credit_min_theme_peer_count": 2,
            "compound_leader_score_credit_max_ret_10d": 0.04,
            "compound_leader_score_credit_max_drawdown_from_high": 0.12,
            "compound_leader_score_credit_max_above_ma20_pct": 0.08,
            "compound_leader_score_credit_require_benchmark_risk_on": True,
            "compound_leader_score_credit_require_theme_active": True,
            "compound_leader_score_credit_require_price_above_ma50": True,
            "compound_leader_score_credit_max_event_risk_score_circuit_breaker": 15.0,
            "compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker": 25.0,
            "compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
            "compound_leader_score_credit_min_adx_circuit_breaker": 20.0,
            "compound_leader_score_credit_max_score_boost": 1.75,
            "long_quantile": 0.20,
        },
    )
    bounded_latest = bounded[bounded["date"] == bounded["date"].max()].set_index("symbol")

    assert bool(bounded_latest.loc["LEAD", "compound_leader_boundary_eligible"]) is False
    assert bool(bounded_latest.loc["LEAD", "compound_leader_score_credit_eligible"]) is False
    assert bounded_latest.loc["LEAD", "compound_leader_score_credit_boost"] == 0.0


def test_medium_term_leader_pullback_overlay_prefers_strong_ret100_leaders_without_short_term_chase() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01"] * 4),
            "symbol": ["LEAD", "CHASE", "WEAK", "BLOCK"],
            "final_score": [72.0, 72.0, 72.0, 72.0],
            "adj_close": [102.0, 109.0, 102.0, 102.0],
            "ma_20": [100.0, 100.0, 100.0, 100.0],
            "ma_50": [96.0, 96.0, 96.0, 96.0],
            "ret_100d": [0.50, 0.55, 0.12, 0.52],
            "ret_5d": [-0.01, 0.06, -0.01, -0.01],
            "ret_10d": [0.01, 0.08, 0.02, 0.01],
            "mom_distance_high": [-0.06, -0.01, -0.05, -0.06],
            "volume_expansion": [0.95, 1.35, 0.95, 0.95],
            "relative_strength_score": [86.0, 88.0, 62.0, 86.0],
            "theme_score": [72.0, 72.0, 52.0, 72.0],
            "mom_return": [0.18, 0.22, 0.08, 0.18],
            "event_risk_score": [0.0, 0.0, 0.0, 0.0],
            "overnight_gap_risk_score": [8.0, 8.0, 8.0, 40.0],
            "benchmark_ret_63d": [0.05, 0.05, 0.05, 0.05],
            "benchmark_risk_on": [True, True, True, True],
            "trend_adx": [24.0, 24.0, 24.0, 24.0],
            "theme_active": [True, True, True, True],
        }
    )

    out = _apply_medium_term_leader_pullback_overlay(
        frame,
        {
            "medium_term_leader_pullback_overlay": True,
            "medium_term_leader_pullback_min_ret_100d_rank": 0.50,
            "medium_term_leader_pullback_max_ret_5d": 0.015,
            "medium_term_leader_pullback_max_ret_10d": 0.03,
            "medium_term_leader_pullback_min_drawdown_from_high": 0.02,
            "medium_term_leader_pullback_max_drawdown_from_high": 0.16,
            "medium_term_leader_pullback_max_above_ma20_pct": 0.04,
            "medium_term_leader_pullback_max_volume_expansion": 1.15,
            "medium_term_leader_pullback_min_relative_strength_score": 68.0,
            "medium_term_leader_pullback_min_theme_score": 56.0,
            "medium_term_leader_pullback_min_mom_return": 0.05,
            "medium_term_leader_pullback_require_benchmark_risk_on": True,
            "medium_term_leader_pullback_require_theme_active": False,
            "medium_term_leader_pullback_require_price_above_ma50": True,
            "medium_term_leader_pullback_max_event_risk_score_circuit_breaker": 18.0,
            "medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker": 28.0,
            "medium_term_leader_pullback_min_benchmark_ret63d_circuit_breaker": 0.0,
            "medium_term_leader_pullback_min_adx_circuit_breaker": 18.0,
            "medium_term_leader_pullback_max_score_boost": 2.0,
        },
    ).set_index("symbol")

    assert bool(out.loc["LEAD", "medium_term_leader_pullback_eligible"]) is True
    assert out.loc["LEAD", "medium_term_leader_pullback_boost"] > 0.0
    assert out.loc["LEAD", "final_score"] > frame.loc[frame["symbol"].eq("LEAD"), "final_score"].iloc[0]
    assert bool(out.loc["CHASE", "medium_term_leader_pullback_eligible"]) is False
    assert out.loc["CHASE", "medium_term_leader_pullback_boost"] == 0.0
    assert bool(out.loc["WEAK", "medium_term_leader_pullback_eligible"]) is False
    assert out.loc["WEAK", "medium_term_leader_pullback_boost"] == 0.0
    assert bool(out.loc["BLOCK", "medium_term_leader_pullback_blocked"]) is True
    assert out.loc["BLOCK", "medium_term_leader_pullback_boost"] == 0.0


def test_exit_quality_rank_credit_reset_support_overlay_requires_reset_and_breadth_confirmation() -> None:
    dates = pd.bdate_range("2026-04-01", periods=7)
    rows = []
    for i, date in enumerate(dates):
        healthy_active = i < len(dates) - 1 or i % 2 == 0
        rows.extend(
            [
                {
                    "date": date,
                    "symbol": "LEAD" if i == len(dates) - 1 else f"LEAD{i}",
                    "score_rank": 0.78 if i == len(dates) - 1 else 0.72,
                    "final_score": 80.0,
                    "technical_score": 72.0,
                    "relative_strength_score": 92.0,
                    "theme_score": 68.0,
                    "mom_return": 0.24,
                    "mom_distance_high": -0.06,
                    "adj_close": 104.0,
                    "ma_20": 100.0,
                    "ma_50": 95.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 5.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 26.0,
                    "theme_active": healthy_active,
                    "pullback_short_term_reset_score": 36.0 if i == len(dates) - 1 else 18.0,
                    "pullback_short_term_reset_blocked": False,
                },
                {
                    "date": date,
                    "symbol": f"ALLY{i}",
                    "score_rank": 0.58,
                    "final_score": 74.0,
                    "technical_score": 66.0,
                    "relative_strength_score": 76.0,
                    "theme_score": 62.0,
                    "mom_return": 0.16,
                    "mom_distance_high": -0.08,
                    "adj_close": 102.0,
                    "ma_20": 99.0,
                    "ma_50": 94.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 6.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 24.0,
                    "theme_active": healthy_active,
                    "pullback_short_term_reset_score": 12.0,
                    "pullback_short_term_reset_blocked": False,
                },
            ]
        )
    latest_date = dates[-1]
    rows.extend(
        [
            {
                "date": latest_date,
                "symbol": "BREADTH0",
                "score_rank": 0.40,
                "final_score": 60.0,
                "technical_score": 55.0,
                "relative_strength_score": 50.0,
                "theme_score": 48.0,
                "mom_return": -0.02,
                "mom_distance_high": -0.18,
                "adj_close": 90.0,
                "ma_20": 95.0,
                "ma_50": 97.0,
                "event_risk_score": 0.0,
                "overnight_gap_risk_score": 8.0,
                "benchmark_ret_63d": 0.05,
                "benchmark_risk_on": True,
                "trend_adx": 20.0,
                "theme_active": False,
                "pullback_short_term_reset_score": 0.0,
                "pullback_short_term_reset_blocked": False,
            },
            {
                "date": latest_date,
                "symbol": "BREADTH1",
                "score_rank": 0.39,
                "final_score": 59.0,
                "technical_score": 54.0,
                "relative_strength_score": 49.0,
                "theme_score": 47.0,
                "mom_return": -0.03,
                "mom_distance_high": -0.20,
                "adj_close": 88.0,
                "ma_20": 94.0,
                "ma_50": 96.0,
                "event_risk_score": 0.0,
                "overnight_gap_risk_score": 8.0,
                "benchmark_ret_63d": 0.05,
                "benchmark_risk_on": True,
                "trend_adx": 20.0,
                "theme_active": False,
                "pullback_short_term_reset_score": 0.0,
                "pullback_short_term_reset_blocked": False,
            },
        ]
    )

    out = _apply_exit_quality_rank_credit_overlay(
        pd.DataFrame(rows),
        {
            "exit_quality_rank_credit_overlay": True,
            "exit_quality_rank_credit_reset_support_overlay": True,
            "exit_quality_rank_credit_rank_buffer_below": 0.07,
            "exit_quality_rank_credit_rank_buffer_above": 0.01,
            "exit_quality_rank_credit_min_final_score": 68.0,
            "exit_quality_rank_credit_min_relative_strength_score": 88.0,
            "exit_quality_rank_credit_min_theme_score": 60.0,
            "exit_quality_rank_credit_min_technical_score": 60.0,
            "exit_quality_rank_credit_min_mom_return": 0.12,
            "exit_quality_rank_credit_max_drawdown_from_high": 0.18,
            "exit_quality_rank_credit_max_above_ma20_pct": 0.08,
            "exit_quality_rank_credit_require_benchmark_risk_on": True,
            "exit_quality_rank_credit_require_theme_active": True,
            "exit_quality_rank_credit_require_price_above_ma50": True,
            "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": 20.0,
            "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
            "exit_quality_rank_credit_min_adx_circuit_breaker": 18.0,
            "exit_quality_rank_credit_max_score_rank_credit": 0.05,
            "exit_quality_rank_credit_reset_support_min_reset_score": 20.0,
            "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days": 5,
            "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold": 0.55,
            "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold": 0.08,
            "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share": 0.55,
            "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion": False,
            "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold": 0.02,
        },
        long_quantile=0.20,
    )
    latest_out = out[out["date"] == latest_date].set_index("symbol")

    assert bool(latest_out.loc["LEAD", "exit_quality_reset_support_eligible"]) is False
    assert bool(latest_out.loc["LEAD", "exit_quality_rank_credit_eligible"]) is False
    assert latest_out.loc["LEAD", "exit_quality_reset_breadth_active_share"] < 0.55

    supportive = pd.DataFrame(rows)
    supportive.loc[supportive["date"].eq(latest_date), "theme_active"] = True
    supportive_out = _apply_exit_quality_rank_credit_overlay(
        supportive,
        {
            "exit_quality_rank_credit_overlay": True,
            "exit_quality_rank_credit_reset_support_overlay": True,
            "exit_quality_rank_credit_rank_buffer_below": 0.07,
            "exit_quality_rank_credit_rank_buffer_above": 0.01,
            "exit_quality_rank_credit_min_final_score": 68.0,
            "exit_quality_rank_credit_min_relative_strength_score": 88.0,
            "exit_quality_rank_credit_min_theme_score": 60.0,
            "exit_quality_rank_credit_min_technical_score": 60.0,
            "exit_quality_rank_credit_min_mom_return": 0.12,
            "exit_quality_rank_credit_max_drawdown_from_high": 0.18,
            "exit_quality_rank_credit_max_above_ma20_pct": 0.08,
            "exit_quality_rank_credit_require_benchmark_risk_on": True,
            "exit_quality_rank_credit_require_theme_active": True,
            "exit_quality_rank_credit_require_price_above_ma50": True,
            "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": 20.0,
            "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
            "exit_quality_rank_credit_min_adx_circuit_breaker": 18.0,
            "exit_quality_rank_credit_max_score_rank_credit": 0.05,
            "exit_quality_rank_credit_reset_support_min_reset_score": 20.0,
            "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days": 5,
            "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold": 0.55,
            "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold": 0.08,
            "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share": 0.55,
            "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion": False,
            "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold": 0.02,
        },
        long_quantile=0.20,
    )
    supportive_latest = supportive_out[supportive_out["date"] == latest_date].set_index("symbol")

    assert bool(supportive_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is True
    assert bool(supportive_latest.loc["LEAD", "exit_quality_rank_credit_eligible"]) is True
    assert supportive_latest.loc["LEAD", "exit_quality_effective_score_rank"] > supportive_latest.loc["LEAD", "score_rank"]


def test_exit_quality_rank_credit_reset_support_can_use_same_theme_peer_breadth() -> None:
    dates = pd.bdate_range("2026-04-01", periods=7)
    rows = []
    for i, date in enumerate(dates):
        rows.extend(
            [
                {
                    "date": date,
                    "symbol": "LEAD" if i == len(dates) - 1 else f"LEAD{i}",
                    "score_rank": 0.78 if i == len(dates) - 1 else 0.73,
                    "final_score": 80.0,
                    "technical_score": 72.0,
                    "relative_strength_score": 92.0,
                    "theme_score": 68.0,
                    "mom_return": 0.24,
                    "mom_distance_high": -0.06,
                    "adj_close": 104.0,
                    "ma_20": 100.0,
                    "ma_50": 95.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 5.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 26.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                    "pullback_short_term_reset_score": 38.0 if i == len(dates) - 1 else 18.0,
                    "pullback_short_term_reset_blocked": False,
                },
                {
                    "date": date,
                    "symbol": f"ALLY{i}",
                    "score_rank": 0.58,
                    "final_score": 74.0,
                    "technical_score": 66.0,
                    "relative_strength_score": 76.0,
                    "theme_score": 62.0,
                    "mom_return": 0.16,
                    "mom_distance_high": -0.08,
                    "adj_close": 102.0,
                    "ma_20": 99.0,
                    "ma_50": 94.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 6.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 24.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                    "pullback_short_term_reset_score": 12.0,
                    "pullback_short_term_reset_blocked": False,
                },
            ]
        )
    latest_date = dates[-1]
    rows.extend(
        [
            {
                "date": latest_date,
                "symbol": "BREADTH0",
                "score_rank": 0.40,
                "final_score": 60.0,
                "technical_score": 55.0,
                "relative_strength_score": 50.0,
                "theme_score": 48.0,
                "mom_return": -0.02,
                "mom_distance_high": -0.18,
                "adj_close": 90.0,
                "ma_20": 95.0,
                "ma_50": 97.0,
                "event_risk_score": 0.0,
                "overnight_gap_risk_score": 8.0,
                "benchmark_ret_63d": 0.05,
                "benchmark_risk_on": True,
                "trend_adx": 20.0,
                "theme_active": False,
                "primary_theme": "other_theme",
                "pullback_short_term_reset_score": 0.0,
                "pullback_short_term_reset_blocked": False,
            },
            {
                "date": latest_date,
                "symbol": "BREADTH1",
                "score_rank": 0.39,
                "final_score": 59.0,
                "technical_score": 54.0,
                "relative_strength_score": 49.0,
                "theme_score": 47.0,
                "mom_return": -0.03,
                "mom_distance_high": -0.20,
                "adj_close": 88.0,
                "ma_20": 94.0,
                "ma_50": 96.0,
                "event_risk_score": 0.0,
                "overnight_gap_risk_score": 8.0,
                "benchmark_ret_63d": 0.05,
                "benchmark_risk_on": True,
                "trend_adx": 20.0,
                "theme_active": False,
                "primary_theme": "other_theme",
                "pullback_short_term_reset_score": 0.0,
                "pullback_short_term_reset_blocked": False,
            },
        ]
    )
    params = {
        "exit_quality_rank_credit_overlay": True,
        "exit_quality_rank_credit_reset_support_overlay": True,
        "exit_quality_rank_credit_rank_buffer_below": 0.045,
        "exit_quality_rank_credit_rank_buffer_above": 0.008,
        "exit_quality_rank_credit_min_final_score": 68.0,
        "exit_quality_rank_credit_min_relative_strength_score": 88.0,
        "exit_quality_rank_credit_min_theme_score": 60.0,
        "exit_quality_rank_credit_min_technical_score": 60.0,
        "exit_quality_rank_credit_min_mom_return": 0.12,
        "exit_quality_rank_credit_max_drawdown_from_high": 0.18,
        "exit_quality_rank_credit_max_above_ma20_pct": 0.08,
        "exit_quality_rank_credit_require_benchmark_risk_on": True,
        "exit_quality_rank_credit_require_theme_active": True,
        "exit_quality_rank_credit_require_price_above_ma50": True,
        "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": 20.0,
        "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
        "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
        "exit_quality_rank_credit_min_adx_circuit_breaker": 18.0,
        "exit_quality_rank_credit_max_score_rank_credit": 0.05,
        "exit_quality_rank_credit_reset_support_min_reset_score": 20.0,
        "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days": 5,
        "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold": 0.55,
        "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold": 0.08,
        "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share": 0.55,
        "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion": False,
        "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold": 0.02,
    }
    frame = pd.DataFrame(rows)

    broad_out = _apply_exit_quality_rank_credit_overlay(frame, params, long_quantile=0.20)
    broad_latest = broad_out[broad_out["date"] == latest_date].set_index("symbol")
    assert bool(broad_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is False
    assert broad_latest.loc["LEAD", "exit_quality_reset_breadth_active_share"] < 0.55

    local_params = dict(params)
    local_params["exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay"] = True
    local_params["exit_quality_rank_credit_reset_support_same_theme_peer_min_count"] = 2
    local_out = _apply_exit_quality_rank_credit_overlay(frame, local_params, long_quantile=0.20)
    local_latest = local_out[local_out["date"] == latest_date].set_index("symbol")

    assert bool(local_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is True
    assert bool(local_latest.loc["LEAD", "exit_quality_rank_credit_eligible"]) is True
    assert local_latest.loc["LEAD", "exit_quality_effective_score_rank"] > local_latest.loc["LEAD", "score_rank"]
    assert local_latest.loc["LEAD", "exit_quality_reset_breadth_active_share"] == 1.0
    assert local_latest.loc["LEAD", "exit_quality_reset_breadth_same_theme_peer_count"] == 2


def test_exit_quality_rank_credit_reset_support_can_require_same_theme_peer_reset_participation() -> None:
    dates = pd.bdate_range("2026-04-01", periods=7)
    rows = []
    for i, date in enumerate(dates):
        rows.extend(
            [
                {
                    "date": date,
                    "symbol": "LEAD" if i == len(dates) - 1 else f"LEAD{i}",
                    "score_rank": 0.78 if i == len(dates) - 1 else 0.73,
                    "final_score": 80.0,
                    "technical_score": 72.0,
                    "relative_strength_score": 92.0,
                    "theme_score": 68.0,
                    "mom_return": 0.24,
                    "mom_distance_high": -0.06,
                    "adj_close": 104.0,
                    "ma_20": 100.0,
                    "ma_50": 95.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 5.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 26.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                    "pullback_short_term_reset_score": 38.0 if i == len(dates) - 1 else 18.0,
                    "pullback_short_term_reset_blocked": False,
                },
                {
                    "date": date,
                    "symbol": f"ALLY{i}",
                    "score_rank": 0.58,
                    "final_score": 74.0,
                    "technical_score": 66.0,
                    "relative_strength_score": 76.0,
                    "theme_score": 62.0,
                    "mom_return": 0.16,
                    "mom_distance_high": -0.08,
                    "adj_close": 102.0,
                    "ma_20": 99.0,
                    "ma_50": 94.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 6.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 24.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                    "pullback_short_term_reset_score": 12.0,
                    "pullback_short_term_reset_blocked": False,
                },
            ]
        )
    latest_date = dates[-1]
    frame = pd.DataFrame(rows)
    params = {
        "exit_quality_rank_credit_overlay": True,
        "exit_quality_rank_credit_reset_support_overlay": True,
        "exit_quality_rank_credit_rank_buffer_below": 0.04,
        "exit_quality_rank_credit_rank_buffer_above": 0.008,
        "exit_quality_rank_credit_min_final_score": 68.0,
        "exit_quality_rank_credit_min_relative_strength_score": 88.0,
        "exit_quality_rank_credit_min_theme_score": 60.0,
        "exit_quality_rank_credit_min_technical_score": 60.0,
        "exit_quality_rank_credit_min_mom_return": 0.12,
        "exit_quality_rank_credit_max_drawdown_from_high": 0.18,
        "exit_quality_rank_credit_max_above_ma20_pct": 0.08,
        "exit_quality_rank_credit_require_benchmark_risk_on": True,
        "exit_quality_rank_credit_require_theme_active": True,
        "exit_quality_rank_credit_require_price_above_ma50": True,
        "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": 20.0,
        "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
        "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
        "exit_quality_rank_credit_min_adx_circuit_breaker": 18.0,
        "exit_quality_rank_credit_max_score_rank_credit": 0.05,
        "exit_quality_rank_credit_reset_support_min_reset_score": 20.0,
        "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days": 5,
        "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold": 0.55,
        "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold": 0.08,
        "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share": 0.55,
        "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion": False,
        "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold": 0.02,
        "exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay": True,
        "exit_quality_rank_credit_reset_support_same_theme_peer_min_count": 2,
    }

    peer_breadth_out = _apply_exit_quality_rank_credit_overlay(frame, params, long_quantile=0.20)
    peer_breadth_latest = peer_breadth_out[peer_breadth_out["date"] == latest_date].set_index("symbol")
    assert bool(peer_breadth_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is True

    reset_params = dict(params)
    reset_params["exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay"] = True
    reset_params["exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share"] = 0.60
    reset_params["exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count"] = 2
    reset_out = _apply_exit_quality_rank_credit_overlay(frame, reset_params, long_quantile=0.20)
    reset_latest = reset_out[reset_out["date"] == latest_date].set_index("symbol")

    assert bool(reset_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is False
    assert reset_latest.loc["LEAD", "exit_quality_reset_breadth_same_theme_peer_reset_share"] == 0.5
    assert reset_latest.loc["LEAD", "exit_quality_reset_breadth_same_theme_peer_reset_count"] == 1

    supportive = frame.copy()
    supportive.loc[supportive["symbol"].eq("ALLY6"), "pullback_short_term_reset_score"] = 28.0
    supportive_out = _apply_exit_quality_rank_credit_overlay(supportive, reset_params, long_quantile=0.20)
    supportive_latest = supportive_out[supportive_out["date"] == latest_date].set_index("symbol")

    assert bool(supportive_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is True
    assert bool(supportive_latest.loc["LEAD", "exit_quality_rank_credit_eligible"]) is True
    assert supportive_latest.loc["LEAD", "exit_quality_reset_breadth_same_theme_peer_reset_share"] == 1.0
    assert supportive_latest.loc["LEAD", "exit_quality_reset_breadth_same_theme_peer_reset_count"] == 2


def test_exit_quality_rank_credit_reset_support_same_theme_leader_guard_blocks_stale_local_nonleaders() -> None:
    dates = pd.bdate_range("2026-04-01", periods=7)
    rows = []
    for i, date in enumerate(dates):
        is_latest = i == len(dates) - 1
        rows.extend(
            [
                {
                    "date": date,
                    "symbol": "LEAD" if is_latest else f"LEAD{i}",
                    "score_rank": 0.78 if is_latest else 0.73,
                    "final_score": 80.0,
                    "technical_score": 72.0,
                    "relative_strength_score": 92.0,
                    "theme_score": 68.0,
                    "mom_return": 0.24,
                    "mom_distance_high": -0.06,
                    "adj_close": 104.0,
                    "ma_20": 100.0,
                    "ma_50": 95.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 5.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 26.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                    "pullback_short_term_reset_score": 38.0 if is_latest else 18.0,
                    "pullback_short_term_reset_blocked": False,
                },
                {
                    "date": date,
                    "symbol": f"ALLY{i}",
                    "score_rank": 0.58,
                    "final_score": 74.0,
                    "technical_score": 66.0,
                    "relative_strength_score": 76.0,
                    "theme_score": 62.0,
                    "mom_return": 0.16,
                    "mom_distance_high": -0.08,
                    "adj_close": 102.0,
                    "ma_20": 99.0,
                    "ma_50": 94.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 6.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 24.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                    "pullback_short_term_reset_score": 28.0 if is_latest else 12.0,
                    "pullback_short_term_reset_blocked": False,
                },
            ]
        )
    latest_date = dates[-1]
    frame = pd.DataFrame(rows)
    base_params = {
        "exit_quality_rank_credit_overlay": True,
        "exit_quality_rank_credit_reset_support_overlay": True,
        "exit_quality_rank_credit_rank_buffer_below": 0.04,
        "exit_quality_rank_credit_rank_buffer_above": 0.008,
        "exit_quality_rank_credit_min_final_score": 68.0,
        "exit_quality_rank_credit_min_relative_strength_score": 88.0,
        "exit_quality_rank_credit_min_theme_score": 60.0,
        "exit_quality_rank_credit_min_technical_score": 60.0,
        "exit_quality_rank_credit_min_mom_return": 0.12,
        "exit_quality_rank_credit_max_drawdown_from_high": 0.18,
        "exit_quality_rank_credit_max_above_ma20_pct": 0.08,
        "exit_quality_rank_credit_require_benchmark_risk_on": True,
        "exit_quality_rank_credit_require_theme_active": True,
        "exit_quality_rank_credit_require_price_above_ma50": True,
        "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": 20.0,
        "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
        "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
        "exit_quality_rank_credit_min_adx_circuit_breaker": 18.0,
        "exit_quality_rank_credit_max_score_rank_credit": 0.05,
        "exit_quality_rank_credit_reset_support_min_reset_score": 20.0,
        "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days": 5,
        "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold": 0.55,
        "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold": 0.08,
        "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share": 0.55,
        "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion": False,
        "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold": 0.02,
        "exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay": True,
        "exit_quality_rank_credit_reset_support_same_theme_peer_min_count": 2,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay": True,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share": 0.34,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count": 2,
    }
    base_out = _apply_exit_quality_rank_credit_overlay(frame, base_params, long_quantile=0.20)
    base_latest = base_out[base_out["date"] == latest_date].set_index("symbol")
    assert bool(base_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is True

    guarded_params = dict(base_params)
    guarded_params["exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay"] = True
    guarded_params["exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct"] = 0.80
    guarded_params["exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap"] = 6.0
    guarded_params["exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count"] = 2

    crowded = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "date": latest_date,
                        "symbol": "PEERWIN",
                        "score_rank": 0.92,
                        "final_score": 90.0,
                        "technical_score": 86.0,
                        "relative_strength_score": 97.0,
                        "theme_score": 74.0,
                        "mom_return": 0.32,
                        "mom_distance_high": -0.02,
                        "adj_close": 110.0,
                        "ma_20": 105.0,
                        "ma_50": 98.0,
                        "event_risk_score": 0.0,
                        "overnight_gap_risk_score": 4.0,
                        "benchmark_ret_63d": 0.05,
                        "benchmark_risk_on": True,
                        "trend_adx": 31.0,
                        "theme_active": True,
                        "primary_theme": "ai_compute",
                        "pullback_short_term_reset_score": 30.0,
                        "pullback_short_term_reset_blocked": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    guarded_out = _apply_exit_quality_rank_credit_overlay(crowded, guarded_params, long_quantile=0.20)
    guarded_latest = guarded_out[guarded_out["date"] == latest_date].set_index("symbol")

    assert bool(guarded_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is False
    assert bool(guarded_latest.loc["LEAD", "exit_quality_reset_same_theme_leader_guard_eligible"]) is False
    assert guarded_latest.loc["LEAD", "exit_quality_reset_same_theme_leader_rank_pct"] < 0.80
    assert guarded_latest.loc["LEAD", "exit_quality_reset_same_theme_leader_score_gap"] > 6.0

    supportive_out = _apply_exit_quality_rank_credit_overlay(frame, guarded_params, long_quantile=0.20)
    supportive_latest = supportive_out[supportive_out["date"] == latest_date].set_index("symbol")

    assert bool(supportive_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is True
    assert bool(supportive_latest.loc["LEAD", "exit_quality_reset_same_theme_leader_guard_eligible"]) is True
    assert supportive_latest.loc["LEAD", "exit_quality_reset_same_theme_leader_rank_pct"] >= 0.80
    assert supportive_latest.loc["LEAD", "exit_quality_reset_same_theme_leader_score_gap"] <= 6.0


def test_exit_quality_rank_credit_reset_support_can_require_high_quality_reset_peers() -> None:
    dates = pd.bdate_range("2026-04-01", periods=7)
    rows = []
    for i, date in enumerate(dates):
        rows.extend(
            [
                {
                    "date": date,
                    "symbol": "LEAD" if i == len(dates) - 1 else f"LEAD{i}",
                    "score_rank": 0.78 if i == len(dates) - 1 else 0.73,
                    "final_score": 80.0,
                    "technical_score": 72.0,
                    "relative_strength_score": 92.0,
                    "theme_score": 68.0,
                    "mom_return": 0.24,
                    "mom_distance_high": -0.06,
                    "adj_close": 104.0,
                    "ma_20": 100.0,
                    "ma_50": 95.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 5.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 26.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                    "pullback_short_term_reset_score": 38.0 if i == len(dates) - 1 else 18.0,
                    "pullback_short_term_reset_blocked": False,
                },
                {
                    "date": date,
                    "symbol": f"ALLY{i}",
                    "score_rank": 0.58,
                    "final_score": 74.0,
                    "technical_score": 66.0,
                    "relative_strength_score": 76.0,
                    "theme_score": 62.0,
                    "mom_return": 0.16,
                    "mom_distance_high": -0.08,
                    "adj_close": 102.0,
                    "ma_20": 99.0,
                    "ma_50": 94.0,
                    "event_risk_score": 0.0,
                    "overnight_gap_risk_score": 6.0,
                    "benchmark_ret_63d": 0.05,
                    "benchmark_risk_on": True,
                    "trend_adx": 24.0,
                    "theme_active": True,
                    "primary_theme": "ai_compute",
                    "pullback_short_term_reset_score": 28.0 if i == len(dates) - 1 else 12.0,
                    "pullback_short_term_reset_blocked": False,
                },
            ]
        )
    latest_date = dates[-1]
    params = {
        "exit_quality_rank_credit_overlay": True,
        "exit_quality_rank_credit_reset_support_overlay": True,
        "exit_quality_rank_credit_rank_buffer_below": 0.04,
        "exit_quality_rank_credit_rank_buffer_above": 0.008,
        "exit_quality_rank_credit_min_final_score": 68.0,
        "exit_quality_rank_credit_min_relative_strength_score": 88.0,
        "exit_quality_rank_credit_min_theme_score": 60.0,
        "exit_quality_rank_credit_min_technical_score": 60.0,
        "exit_quality_rank_credit_min_mom_return": 0.12,
        "exit_quality_rank_credit_max_drawdown_from_high": 0.18,
        "exit_quality_rank_credit_max_above_ma20_pct": 0.08,
        "exit_quality_rank_credit_require_benchmark_risk_on": True,
        "exit_quality_rank_credit_require_theme_active": True,
        "exit_quality_rank_credit_require_price_above_ma50": True,
        "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": 20.0,
        "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": 30.0,
        "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
        "exit_quality_rank_credit_min_adx_circuit_breaker": 18.0,
        "exit_quality_rank_credit_max_score_rank_credit": 0.05,
        "exit_quality_rank_credit_reset_support_min_reset_score": 20.0,
        "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days": 5,
        "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold": 0.55,
        "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold": 0.08,
        "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share": 0.55,
        "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion": False,
        "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold": 0.02,
        "exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay": True,
        "exit_quality_rank_credit_reset_support_same_theme_peer_min_count": 2,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay": True,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share": 0.60,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count": 2,
        "exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay": True,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs": 85.0,
    }
    frame = pd.DataFrame(rows)

    weak_out = _apply_exit_quality_rank_credit_overlay(frame, params, long_quantile=0.20)
    weak_latest = weak_out[weak_out["date"] == latest_date].set_index("symbol")

    assert bool(weak_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is False
    assert weak_latest.loc["LEAD", "exit_quality_reset_breadth_same_theme_peer_reset_share"] == 1.0
    assert weak_latest.loc["LEAD", "exit_quality_reset_breadth_same_theme_peer_reset_count"] == 2
    assert weak_latest.loc["LEAD", "exit_quality_reset_breadth_same_theme_peer_reset_rs_mean"] == 84.0

    strong = frame.copy()
    strong.loc[strong["symbol"].eq("ALLY6"), "relative_strength_score"] = 88.0
    strong_out = _apply_exit_quality_rank_credit_overlay(strong, params, long_quantile=0.20)
    strong_latest = strong_out[strong_out["date"] == latest_date].set_index("symbol")

    assert bool(strong_latest.loc["LEAD", "exit_quality_reset_support_eligible"]) is True
    assert bool(strong_latest.loc["LEAD", "exit_quality_rank_credit_eligible"]) is True
    assert strong_latest.loc["LEAD", "exit_quality_effective_score_rank"] > strong_latest.loc["LEAD", "score_rank"]
    assert strong_latest.loc["LEAD", "exit_quality_reset_breadth_same_theme_peer_reset_rs_mean"] > 85.0


def test_momentum_custom_final_score_controlled_entry_overlay_falls_back_for_stretched_names() -> None:
    frame = pd.DataFrame(
        {
            "technical_score": [90.0, 92.0],
            "relative_strength_score": [88.0, 89.0],
            "theme_score": [72.0, 73.0],
            "fundamental_score": [50.0, 50.0],
            "event_risk_score": [0.0, 0.0],
            "overnight_gap_risk_score": [5.0, 5.0],
            "benchmark_ret_63d": [0.10, 0.10],
            "benchmark_risk_on": [True, True],
            "theme_active": [True, True],
            "adj_close": [105.0, 120.0],
            "ma_20": [100.0, 100.0],
            "ma_50": [98.0, 98.0],
        }
    )

    default_scores = _momentum_final_score(frame.copy(), {})
    controlled_frame = frame.copy()
    controlled_scores = _momentum_final_score(
        controlled_frame,
        {
            "momentum_custom_final_score_weights": True,
            "momentum_final_score_technical_weight": 0.42,
            "momentum_final_score_relative_strength_weight": 0.33,
            "momentum_final_score_theme_weight": 0.20,
            "momentum_final_score_fundamental_weight": 0.05,
            "momentum_final_score_event_risk_penalty": 0.05,
            "momentum_custom_final_score_blend": 1.0,
            "momentum_custom_score_controlled_entry_overlay": True,
            "momentum_custom_score_min_relative_strength_score": 62.0,
            "momentum_custom_score_min_theme_score": 58.0,
            "momentum_custom_score_max_above_ma20_pct": 0.10,
            "momentum_custom_score_max_event_risk_score_circuit_breaker": 20.0,
            "momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "momentum_custom_score_min_benchmark_ret63d_circuit_breaker": 0.0,
            "momentum_custom_score_require_benchmark_risk_on": True,
            "momentum_custom_score_require_theme_active": True,
            "momentum_custom_score_require_price_above_ma50": True,
        },
    )

    assert controlled_scores.iloc[0] > default_scores.iloc[0]
    assert controlled_scores.iloc[1] == default_scores.iloc[1]
    assert bool(controlled_frame.loc[0, "momentum_custom_score_controlled_entry_eligible"]) is True
    assert bool(controlled_frame.loc[1, "momentum_custom_score_controlled_entry_eligible"]) is False
    assert controlled_frame.loc[0, "momentum_score_weight_profile"].endswith("_controlled")
    assert controlled_frame.loc[1, "momentum_score_weight_profile"] == "default_controlled_fallback"


def test_momentum_custom_final_score_controlled_entry_overlay_respects_risk_on_breaker() -> None:
    frame = pd.DataFrame(
        {
            "technical_score": [90.0],
            "relative_strength_score": [88.0],
            "theme_score": [72.0],
            "fundamental_score": [50.0],
            "event_risk_score": [0.0],
            "overnight_gap_risk_score": [5.0],
            "benchmark_ret_63d": [0.10],
            "benchmark_risk_on": [False],
            "theme_active": [True],
            "adj_close": [105.0],
            "ma_20": [100.0],
            "ma_50": [98.0],
        }
    )

    default_scores = _momentum_final_score(frame.copy(), {})
    controlled_frame = frame.copy()
    controlled_scores = _momentum_final_score(
        controlled_frame,
        {
            "momentum_custom_final_score_weights": True,
            "momentum_final_score_technical_weight": 0.42,
            "momentum_final_score_relative_strength_weight": 0.33,
            "momentum_final_score_theme_weight": 0.20,
            "momentum_final_score_fundamental_weight": 0.05,
            "momentum_final_score_event_risk_penalty": 0.05,
            "momentum_custom_final_score_blend": 1.0,
            "momentum_custom_score_controlled_entry_overlay": True,
            "momentum_custom_score_require_benchmark_risk_on": True,
        },
    )

    assert controlled_scores.iloc[0] == default_scores.iloc[0]
    assert bool(controlled_frame.loc[0, "momentum_custom_score_controlled_entry_eligible"]) is False
    assert controlled_frame.loc[0, "momentum_score_weight_profile"] == "default_controlled_fallback"


def test_momentum_short_gate_can_require_risk_off_and_real_weakness() -> None:
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.50,
        "max_positions": 10,
        "short_min_weakness_conditions": 3,
        "short_only_when_benchmark_risk_off": True,
        "short_max_final_score": 55,
    }

    risk_on_signals = CrossSectionalMomentumLongShort(params).generate_signals(_features(risk_on=True), StrategyContext(benchmark="SPY"))
    assert risk_on_signals[risk_on_signals["signal"] < 0].empty

    risk_off_signals = CrossSectionalMomentumLongShort(params).generate_signals(_features(risk_on=False), StrategyContext(benchmark="SPY"))
    latest = risk_off_signals[risk_off_signals["date"] == risk_off_signals["date"].max()]
    shorts = latest[latest["signal"] < 0]

    assert "BBB" in set(shorts["symbol"])
    assert (shorts["short_weakness_count"] >= 3).all()


def test_momentum_pullback_entry_overlay_boosts_controlled_leaders_only() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    chase = features["symbol"].eq("CCC")
    features.loc[leader, "distance_to_prior_high_252"] = -0.10
    features.loc[chase, "distance_to_prior_high_252"] = -0.01
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.50,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_pullback_entry_overlay": True,
        "pullback_min_relative_strength_score": 60,
        "pullback_min_theme_score": 60,
        "pullback_min_drawdown_from_high": 0.03,
        "pullback_max_drawdown_from_high": 0.20,
        "pullback_max_above_ma20_pct": 0.08,
        "pullback_max_score_boost": 6.0,
    }

    base_params = dict(params)
    base_params["long_pullback_entry_overlay"] = False
    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]
    ccc = latest[latest["symbol"].eq("CCC")].iloc[0]

    assert "pullback_entry=" in aaa["score_decomposition"]
    assert aaa["final_score"] > base_aaa["final_score"]
    assert "pullback_entry=0.00" in ccc["score_decomposition"]


def test_momentum_theme_leader_tilt_overlay_boosts_same_theme_leader_only() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    follower = features["symbol"].eq("CCC")
    features.loc[leader | follower, "primary_theme"] = "semiconductors"
    features.loc[leader | follower, "theme_active"] = True
    features.loc[leader, "theme_score"] = 86.0
    features.loc[follower, "theme_score"] = 82.0
    features.loc[leader, "rs_63d"] = 0.80
    features.loc[follower, "rs_63d"] = 0.05
    features.loc[leader | follower, "overnight_gap_risk_score"] = 5.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "theme_leader_tilt_overlay": True,
            "theme_leader_tilt_min_theme_score": 60.0,
            "theme_leader_tilt_min_relative_strength_score": 78.0,
            "theme_leader_tilt_min_theme_peer_count": 2,
            "theme_leader_tilt_min_within_theme_rank_pct": 0.75,
            "theme_leader_tilt_max_score_boost": 2.5,
            "theme_leader_tilt_require_benchmark_risk_on": True,
            "theme_leader_tilt_require_theme_active": True,
            "theme_leader_tilt_max_event_risk_score_circuit_breaker": 15.0,
            "theme_leader_tilt_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "theme_leader_tilt_min_benchmark_ret63d_circuit_breaker": 0.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]
    base_ccc = base_latest[base_latest["symbol"].eq("CCC")].iloc[0]
    ccc = latest[latest["symbol"].eq("CCC")].iloc[0]

    assert aaa["final_score"] > base_aaa["final_score"]
    assert ccc["final_score"] == base_ccc["final_score"]
    assert "theme_leader_tilt=" in aaa["score_decomposition"]
    assert "theme_leader_tilt=0.00" in ccc["score_decomposition"]


def test_momentum_theme_leader_tilt_overlay_respects_risk_on_circuit_breaker() -> None:
    features = _features(risk_on=False)
    leader = features["symbol"].eq("AAA")
    follower = features["symbol"].eq("CCC")
    features.loc[leader | follower, "primary_theme"] = "semiconductors"
    features.loc[leader | follower, "theme_active"] = True
    features.loc[leader, "theme_score"] = 86.0
    features.loc[follower, "theme_score"] = 82.0
    features.loc[leader, "rs_63d"] = 0.80
    features.loc[follower, "rs_63d"] = 0.05
    features.loc[leader | follower, "overnight_gap_risk_score"] = 5.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "theme_leader_tilt_overlay": True,
            "theme_leader_tilt_min_theme_score": 60.0,
            "theme_leader_tilt_min_relative_strength_score": 78.0,
            "theme_leader_tilt_min_theme_peer_count": 2,
            "theme_leader_tilt_min_within_theme_rank_pct": 0.75,
            "theme_leader_tilt_max_score_boost": 2.5,
            "theme_leader_tilt_require_benchmark_risk_on": True,
            "theme_leader_tilt_require_theme_active": True,
            "theme_leader_tilt_max_event_risk_score_circuit_breaker": 15.0,
            "theme_leader_tilt_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "theme_leader_tilt_min_benchmark_ret63d_circuit_breaker": 0.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["final_score"] == base_aaa["final_score"]
    assert "theme_leader_tilt=0.00" in aaa["score_decomposition"]


def test_momentum_boundary_rs_theme_credit_overlay_boosts_high_rs_theme_candidate() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    features.loc[leader, "primary_theme"] = "semiconductors"
    features.loc[leader, "theme_active"] = True
    features.loc[leader, "theme_score"] = 84.0
    features.loc[leader, "rs_63d"] = 0.88
    features.loc[leader, "distance_to_prior_high_252"] = -0.05
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.98
    features.loc[leader, "ma_50"] = features.loc[leader, "adj_close"] * 0.95
    features.loc[leader, "trend_adx"] = 28.0
    features.loc[leader, "overnight_gap_risk_score"] = 5.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.50,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 50,
        "long_min_relative_strength_score": 50,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "boundary_rs_theme_credit_overlay": True,
            "boundary_rs_theme_credit_rank_buffer_below": 1.0,
            "boundary_rs_theme_credit_rank_buffer_above": 1.0,
            "boundary_rs_theme_credit_min_relative_strength_score": 88.0,
            "boundary_rs_theme_credit_min_theme_score": 70.0,
            "boundary_rs_theme_credit_min_technical_score": 50.0,
            "boundary_rs_theme_credit_min_mom_return": 0.0,
            "boundary_rs_theme_credit_min_drawdown_from_high": 0.0,
            "boundary_rs_theme_credit_max_drawdown_from_high": 0.12,
            "boundary_rs_theme_credit_max_above_ma20_pct": 0.08,
            "boundary_rs_theme_credit_require_benchmark_risk_on": True,
            "boundary_rs_theme_credit_require_theme_active": True,
            "boundary_rs_theme_credit_require_price_above_ma50": True,
            "boundary_rs_theme_credit_max_event_risk_score_circuit_breaker": 15.0,
            "boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker": 25.0,
            "boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
            "boundary_rs_theme_credit_min_adx_circuit_breaker": 18.0,
            "boundary_rs_theme_credit_max_score_boost": 1.75,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["final_score"] > base_aaa["final_score"]
    assert "boundary_credit=" in aaa["score_decomposition"]
    assert "boundary_credit=0.00" not in aaa["score_decomposition"]
    assert "boundary_block=0" in aaa["score_decomposition"]


def test_momentum_boundary_rs_theme_credit_overlay_respects_gap_risk_breaker() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    features.loc[leader, "primary_theme"] = "semiconductors"
    features.loc[leader, "theme_active"] = True
    features.loc[leader, "theme_score"] = 84.0
    features.loc[leader, "rs_63d"] = 0.88
    features.loc[leader, "distance_to_prior_high_252"] = -0.05
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.98
    features.loc[leader, "ma_50"] = features.loc[leader, "adj_close"] * 0.95
    features.loc[leader, "trend_adx"] = 28.0
    features.loc[leader, "overnight_gap_risk_score"] = 40.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.50,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 50,
        "long_min_relative_strength_score": 50,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "boundary_rs_theme_credit_overlay": True,
            "boundary_rs_theme_credit_rank_buffer_below": 1.0,
            "boundary_rs_theme_credit_rank_buffer_above": 1.0,
            "boundary_rs_theme_credit_min_relative_strength_score": 88.0,
            "boundary_rs_theme_credit_min_theme_score": 70.0,
            "boundary_rs_theme_credit_min_technical_score": 50.0,
            "boundary_rs_theme_credit_min_mom_return": 0.0,
            "boundary_rs_theme_credit_min_drawdown_from_high": 0.0,
            "boundary_rs_theme_credit_max_drawdown_from_high": 0.12,
            "boundary_rs_theme_credit_max_above_ma20_pct": 0.08,
            "boundary_rs_theme_credit_require_benchmark_risk_on": True,
            "boundary_rs_theme_credit_require_theme_active": True,
            "boundary_rs_theme_credit_require_price_above_ma50": True,
            "boundary_rs_theme_credit_max_event_risk_score_circuit_breaker": 15.0,
            "boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker": 25.0,
            "boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker": 0.0,
            "boundary_rs_theme_credit_min_adx_circuit_breaker": 18.0,
            "boundary_rs_theme_credit_max_score_boost": 1.75,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["final_score"] == base_aaa["final_score"]
    assert "boundary_credit=0.00" in aaa["score_decomposition"]
    assert "boundary_block=1" in aaa["score_decomposition"]


def test_momentum_theme_breadth_acceleration_overlay_uses_prior_theme_breadth() -> None:
    features = _features()
    dates = sorted(features["date"].unique())
    active_start = dates[-5]
    semis = features["symbol"].isin(["AAA", "CCC"])
    early_semis = semis & features["date"].lt(active_start)
    late_semis = semis & features["date"].ge(active_start)
    features.loc[semis, "primary_theme"] = "semiconductors"
    features.loc[semis, "theme_active"] = True
    features.loc[semis, "theme_score"] = 82.0
    features.loc[semis, "overnight_gap_risk_score"] = 5.0
    features.loc[features["symbol"].eq("AAA"), "rs_63d"] = 0.45
    features.loc[features["symbol"].eq("CCC"), "rs_63d"] = 0.35
    features.loc[early_semis, "ma_20"] = features.loc[early_semis, "adj_close"] * 1.05
    features.loc[late_semis, "ma_20"] = features.loc[late_semis, "adj_close"] * 0.98

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "theme_breadth_acceleration_overlay": True,
            "theme_breadth_acceleration_lookback_days": 5,
            "theme_breadth_acceleration_min_active_share": 0.50,
            "theme_breadth_acceleration_min_change": 0.40,
            "theme_breadth_acceleration_min_theme_peer_count": 2,
            "theme_breadth_acceleration_min_theme_score": 55.0,
            "theme_breadth_acceleration_min_relative_strength_score": 50.0,
            "theme_breadth_acceleration_max_score_boost": 2.0,
            "theme_breadth_acceleration_require_benchmark_risk_on": True,
            "theme_breadth_acceleration_require_theme_active": True,
            "theme_breadth_acceleration_max_event_risk_score_circuit_breaker": 20.0,
            "theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker": 35.0,
            "theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker": 0.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["final_score"] > base_aaa["final_score"]
    assert aaa["theme_breadth_acceleration_score"] > 0
    assert "theme_breadth_accel=" in aaa["score_decomposition"]


def test_momentum_theme_strength_delta_overlay_uses_prior_theme_improvement() -> None:
    features = _features()
    features.loc[features["symbol"].isin(["AAA", "CCC"]), "primary_theme"] = "semiconductors"
    features.loc[features["symbol"].isin(["AAA", "CCC"]), "theme_reason"] = "semiconductors"
    dates = sorted(features["date"].unique())
    for i, date in enumerate(dates):
        mask = features["date"].eq(date) & features["symbol"].isin(["AAA", "CCC"])
        features.loc[mask, "theme_score"] = 52.0 + min(i, 30) * 0.6
        features.loc[mask, "theme_active"] = True
        features.loc[mask, "overnight_gap_risk_score"] = 0.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.50,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 50,
        "long_min_relative_strength_score": 50,
    }
    overlay_params = {
        **base_params,
        "theme_strength_delta_overlay": True,
        "theme_strength_delta_lookback_days": 5,
        "theme_strength_delta_min_peer_count": 2,
        "theme_strength_delta_min_theme_score": 50.0,
        "theme_strength_delta_min_relative_strength_score": 50.0,
        "theme_strength_delta_min_theme_prior": 48.0,
        "theme_strength_delta_min_peer_rs_prior": 50.0,
        "theme_strength_delta_min_combined_delta": 0.0,
        "theme_strength_delta_max_score_boost": 1.5,
        "theme_strength_delta_require_benchmark_risk_on": True,
        "theme_strength_delta_require_theme_active": False,
        "theme_strength_delta_max_event_risk_score_circuit_breaker": 20.0,
        "theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "theme_strength_delta_min_benchmark_ret63d_circuit_breaker": 0.0,
    }

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    overlay = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))

    latest = overlay[overlay["date"] == overlay["date"].max()]
    base_latest = base[base["date"] == base["date"].max()]
    aaa = latest[latest["symbol"] == "AAA"].iloc[0]
    base_aaa = base_latest[base_latest["symbol"] == "AAA"].iloc[0]

    assert aaa["final_score"] > base_aaa["final_score"]
    assert aaa["theme_strength_delta_score"] > 0
    assert aaa["theme_strength_delta_eligible"]
    assert "theme_strength_delta=" in aaa["score_decomposition"]


def test_momentum_theme_breadth_acceleration_peer_strength_requires_improving_theme_rs() -> None:
    features = _features()
    dates = sorted(features["date"].unique())
    active_start = dates[-5]
    semis = features["symbol"].isin(["AAA", "CCC"])
    early_semis = semis & features["date"].lt(active_start)
    late_semis = semis & features["date"].ge(active_start)
    features.loc[semis, "primary_theme"] = "semiconductors"
    features.loc[semis, "theme_active"] = True
    features.loc[semis, "theme_score"] = 82.0
    features.loc[semis, "overnight_gap_risk_score"] = 5.0
    features.loc[features["symbol"].eq("SPY"), "rs_63d"] = 0.30
    aaa = features["symbol"].eq("AAA")
    ccc = features["symbol"].eq("CCC")
    features.loc[aaa & early_semis, "rs_63d"] = 0.18
    features.loc[ccc & early_semis, "rs_63d"] = -0.10
    features.loc[aaa & late_semis, "rs_63d"] = 0.90
    features.loc[ccc & late_semis, "rs_63d"] = 0.75
    features.loc[early_semis, "ma_20"] = features.loc[early_semis, "adj_close"] * 1.05
    features.loc[late_semis, "ma_20"] = features.loc[late_semis, "adj_close"] * 0.98

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "theme_breadth_acceleration_overlay": True,
            "theme_breadth_acceleration_lookback_days": 5,
            "theme_breadth_acceleration_min_active_share": 0.50,
            "theme_breadth_acceleration_min_change": 0.40,
            "theme_breadth_acceleration_min_theme_peer_count": 2,
            "theme_breadth_acceleration_min_theme_score": 55.0,
            "theme_breadth_acceleration_min_relative_strength_score": 50.0,
            "theme_breadth_acceleration_max_score_boost": 2.0,
            "theme_breadth_acceleration_require_benchmark_risk_on": True,
            "theme_breadth_acceleration_require_theme_active": True,
            "theme_breadth_acceleration_max_event_risk_score_circuit_breaker": 20.0,
            "theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker": 35.0,
            "theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker": 0.0,
            "theme_breadth_acceleration_peer_strength_overlay": True,
            "theme_breadth_acceleration_min_peer_rs_mean": 60.0,
            "theme_breadth_acceleration_min_peer_rs_change": 8.0,
            "theme_breadth_acceleration_peer_strength_weight": 0.40,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["final_score"] > base_aaa["final_score"]
    assert "theme_peer_rs_accel=" in aaa["score_decomposition"]
    assert "theme_peer_rs_prior=" in aaa["score_decomposition"]


def test_momentum_theme_breadth_acceleration_peer_strength_respects_peer_rs_circuit_breaker() -> None:
    features = _features()
    dates = sorted(features["date"].unique())
    active_start = dates[-5]
    semis = features["symbol"].isin(["AAA", "CCC"])
    early_semis = semis & features["date"].lt(active_start)
    late_semis = semis & features["date"].ge(active_start)
    features.loc[semis, "primary_theme"] = "semiconductors"
    features.loc[semis, "theme_active"] = True
    features.loc[semis, "theme_score"] = 82.0
    features.loc[semis, "overnight_gap_risk_score"] = 5.0
    features.loc[features["symbol"].eq("AAA"), "rs_63d"] = 0.30
    features.loc[features["symbol"].eq("CCC"), "rs_63d"] = 0.25
    features.loc[early_semis, "ma_20"] = features.loc[early_semis, "adj_close"] * 1.05
    features.loc[late_semis, "ma_20"] = features.loc[late_semis, "adj_close"] * 0.98

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "theme_breadth_acceleration_overlay": True,
            "theme_breadth_acceleration_lookback_days": 5,
            "theme_breadth_acceleration_min_active_share": 0.50,
            "theme_breadth_acceleration_min_change": 0.40,
            "theme_breadth_acceleration_min_theme_peer_count": 2,
            "theme_breadth_acceleration_min_theme_score": 55.0,
            "theme_breadth_acceleration_min_relative_strength_score": 50.0,
            "theme_breadth_acceleration_max_score_boost": 2.0,
            "theme_breadth_acceleration_require_benchmark_risk_on": True,
            "theme_breadth_acceleration_require_theme_active": True,
            "theme_breadth_acceleration_max_event_risk_score_circuit_breaker": 20.0,
            "theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker": 35.0,
            "theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker": 0.0,
            "theme_breadth_acceleration_peer_strength_overlay": True,
            "theme_breadth_acceleration_min_peer_rs_mean": 60.0,
            "theme_breadth_acceleration_min_peer_rs_change": 8.0,
            "theme_breadth_acceleration_peer_strength_weight": 0.40,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["final_score"] == base_aaa["final_score"]
    assert "theme_peer_rs_accel=0.00" in aaa["score_decomposition"]


def test_momentum_gap_adjusted_continuation_rewards_benign_or_improving_gap_risk() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    tail_start = features.loc[leader, "date"].nlargest(12).min()
    tail = leader & features["date"].ge(tail_start)

    features["primary_theme"] = "semiconductors"
    features["theme_active"] = True
    features["overnight_gap_risk_score"] = 22.0
    features.loc[tail, "distance_to_prior_high_252"] = -0.08
    features.loc[tail, "distance_to_high_252"] = -0.08
    features.loc[tail, "overnight_gap_risk_score"] = np.linspace(22.0, 10.0, int(tail.sum()))
    features.loc[tail, "volume_expansion"] = 0.95
    features.loc[leader, "theme_score"] = 82.0
    features.loc[leader, "rs_63d"] = 0.32
    features.loc[leader, "trend_adx"] = 30.0
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] / 1.02
    features.loc[leader, "ma_50"] = features.loc[leader, "adj_close"] * 0.96
    features.loc[leader, "ma_200"] = features.loc[leader, "adj_close"] * 0.90

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "gap_adjusted_continuation_overlay": True,
        "gap_adjusted_continuation_min_drawdown_from_high": 0.03,
        "gap_adjusted_continuation_max_drawdown_from_high": 0.16,
        "gap_adjusted_continuation_max_above_ma20_pct": 0.05,
        "gap_adjusted_continuation_min_relative_strength_score": 40.0,
        "gap_adjusted_continuation_min_theme_score": 60.0,
        "gap_adjusted_continuation_min_adx_circuit_breaker": 18.0,
        "gap_adjusted_continuation_max_event_risk_score_circuit_breaker": 18.0,
        "gap_adjusted_continuation_max_gap_risk_score_circuit_breaker": 28.0,
        "gap_adjusted_continuation_benign_gap_risk_score": 14.0,
        "gap_adjusted_continuation_gap_risk_lookback_days": 10,
        "gap_adjusted_continuation_min_gap_risk_improvement": 3.0,
        "gap_adjusted_continuation_min_volume_expansion": 0.60,
        "gap_adjusted_continuation_max_volume_expansion": 1.35,
        "gap_adjusted_continuation_require_benchmark_risk_on": True,
        "gap_adjusted_continuation_require_theme_active": True,
        "gap_adjusted_continuation_max_score_boost": 2.5,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()].set_index("symbol")

    assert latest.loc["AAA", "gap_adjusted_continuation_score"] > 0.0
    assert latest.loc["AAA", "gap_adjusted_continuation_gap_risk"] <= 14.0
    assert bool(latest.loc["AAA", "gap_adjusted_continuation_eligible"]) is True
    assert bool(latest.loc["AAA", "gap_adjusted_continuation_blocked"]) is False
    assert "gap_continuation=" in latest.loc["AAA", "score_decomposition"]


def test_momentum_gap_adjusted_continuation_blocks_high_non_improving_gap_risk() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    tail_start = features.loc[leader, "date"].nlargest(12).min()
    tail = leader & features["date"].ge(tail_start)

    features["primary_theme"] = "semiconductors"
    features["theme_active"] = True
    features["overnight_gap_risk_score"] = 30.0
    features.loc[tail, "distance_to_prior_high_252"] = -0.08
    features.loc[tail, "distance_to_high_252"] = -0.08
    features.loc[tail, "volume_expansion"] = 0.95
    features.loc[leader, "theme_score"] = 82.0
    features.loc[leader, "rs_63d"] = 0.32
    features.loc[leader, "trend_adx"] = 30.0
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] / 1.02
    features.loc[leader, "ma_50"] = features.loc[leader, "adj_close"] * 0.96
    features.loc[leader, "ma_200"] = features.loc[leader, "adj_close"] * 0.90

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "gap_adjusted_continuation_overlay": True,
        "gap_adjusted_continuation_min_drawdown_from_high": 0.03,
        "gap_adjusted_continuation_max_drawdown_from_high": 0.16,
        "gap_adjusted_continuation_max_above_ma20_pct": 0.05,
        "gap_adjusted_continuation_min_relative_strength_score": 40.0,
        "gap_adjusted_continuation_min_theme_score": 60.0,
        "gap_adjusted_continuation_max_gap_risk_score_circuit_breaker": 24.0,
        "gap_adjusted_continuation_benign_gap_risk_score": 12.0,
        "gap_adjusted_continuation_min_gap_risk_improvement": 4.0,
        "gap_adjusted_continuation_min_volume_expansion": 0.60,
        "gap_adjusted_continuation_max_volume_expansion": 1.35,
        "gap_adjusted_continuation_require_benchmark_risk_on": True,
        "gap_adjusted_continuation_require_theme_active": True,
        "gap_adjusted_continuation_max_score_boost": 2.5,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()].set_index("symbol")

    assert latest.loc["AAA", "gap_adjusted_continuation_score"] == 0.0
    assert bool(latest.loc["AAA", "gap_adjusted_continuation_eligible"]) is False
    assert bool(latest.loc["AAA", "gap_adjusted_continuation_blocked"]) is True


def test_momentum_gap_adjusted_continuation_peer_quality_requires_same_theme_support() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    leader_rows = features.loc[leader].copy()
    peers = []
    for symbol, rs_value in (("DDD", 0.33), ("EEE", 0.36), ("FFF", 0.34)):
        peer = leader_rows.copy()
        peer["symbol"] = symbol
        peer["rs_63d"] = rs_value
        peer["theme_score"] = 80.0
        peer["trend_adx"] = 28.0
        peers.append(peer)
    features = pd.concat([features, *peers], ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    semis = features["symbol"].isin(["AAA", "DDD", "EEE", "FFF"])
    tail_start = features.loc[semis, "date"].nlargest(12).min()
    tail = semis & features["date"].ge(tail_start)

    features["primary_theme"] = "other"
    features["theme_active"] = True
    features["overnight_gap_risk_score"] = 22.0
    features.loc[semis, "primary_theme"] = "semiconductors"
    features.loc[tail, "distance_to_prior_high_252"] = -0.08
    features.loc[tail, "distance_to_high_252"] = -0.08
    features.loc[tail, "overnight_gap_risk_score"] = np.linspace(22.0, 10.0, int(tail.sum()))
    features.loc[semis, "volume_expansion"] = 0.95
    features.loc[semis, "ma_20"] = features.loc[semis, "adj_close"] / 1.02
    features.loc[semis, "ma_50"] = features.loc[semis, "adj_close"] * 0.96
    features.loc[semis, "ma_200"] = features.loc[semis, "adj_close"] * 0.90
    features.loc[semis, "trend_adx"] = 30.0
    features.loc[semis, "theme_score"] = 82.0

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "gap_adjusted_continuation_overlay": True,
        "gap_adjusted_continuation_min_drawdown_from_high": 0.03,
        "gap_adjusted_continuation_max_drawdown_from_high": 0.16,
        "gap_adjusted_continuation_max_above_ma20_pct": 0.05,
        "gap_adjusted_continuation_min_relative_strength_score": 40.0,
        "gap_adjusted_continuation_min_theme_score": 60.0,
        "gap_adjusted_continuation_min_adx_circuit_breaker": 18.0,
        "gap_adjusted_continuation_max_event_risk_score_circuit_breaker": 18.0,
        "gap_adjusted_continuation_max_gap_risk_score_circuit_breaker": 28.0,
        "gap_adjusted_continuation_benign_gap_risk_score": 14.0,
        "gap_adjusted_continuation_gap_risk_lookback_days": 10,
        "gap_adjusted_continuation_min_gap_risk_improvement": 3.0,
        "gap_adjusted_continuation_min_volume_expansion": 0.60,
        "gap_adjusted_continuation_max_volume_expansion": 1.35,
        "gap_adjusted_continuation_require_benchmark_risk_on": True,
        "gap_adjusted_continuation_require_theme_active": True,
        "gap_adjusted_continuation_same_theme_peer_quality_overlay": True,
        "gap_adjusted_continuation_same_theme_peer_min_count": 3,
        "gap_adjusted_continuation_same_theme_peer_min_share": 0.75,
        "gap_adjusted_continuation_same_theme_peer_min_avg_rs": 65.0,
        "gap_adjusted_continuation_max_score_boost": 2.5,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()].set_index("symbol")

    assert latest.loc["AAA", "gap_adjusted_continuation_score"] > 0.0
    assert "gap_continuation_peer_count=4" in latest.loc["AAA", "score_decomposition"]
    assert "gap_continuation_peer_gate=1" in latest.loc["AAA", "score_decomposition"]


def test_momentum_gap_adjusted_continuation_peer_quality_blocks_when_theme_peers_are_weak() -> None:
    features = _features()
    leader_rows = features.loc[features["symbol"].eq("AAA")].copy()
    weak_peer = leader_rows.copy()
    weak_peer["symbol"] = "DDD"
    weak_peer["rs_63d"] = 0.05
    weak_peer["theme_score"] = 65.0
    features = pd.concat([features, weak_peer], ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    leader = features["symbol"].eq("AAA")
    semis = features["symbol"].isin(["AAA", "DDD"])
    tail_start = features.loc[semis, "date"].nlargest(12).min()
    tail = semis & features["date"].ge(tail_start)

    features["primary_theme"] = "other"
    features["theme_active"] = True
    features["overnight_gap_risk_score"] = 24.0
    features.loc[semis, "primary_theme"] = "semiconductors"
    features.loc[tail, "distance_to_prior_high_252"] = -0.08
    features.loc[tail, "distance_to_high_252"] = -0.08
    features.loc[leader & features["date"].ge(tail_start), "overnight_gap_risk_score"] = np.linspace(24.0, 11.0, int((leader & features["date"].ge(tail_start)).sum()))
    features.loc[features["symbol"].eq("DDD") & features["date"].ge(tail_start), "overnight_gap_risk_score"] = 24.0
    features.loc[semis, "volume_expansion"] = 0.95
    features.loc[semis, "ma_20"] = features.loc[semis, "adj_close"] / 1.02
    features.loc[semis, "ma_50"] = features.loc[semis, "adj_close"] * 0.96
    features.loc[semis, "ma_200"] = features.loc[semis, "adj_close"] * 0.90
    features.loc[semis, "trend_adx"] = 30.0
    features.loc[leader, "theme_score"] = 82.0
    features.loc[leader, "rs_63d"] = 0.32

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "gap_adjusted_continuation_overlay": True,
        "gap_adjusted_continuation_min_drawdown_from_high": 0.03,
        "gap_adjusted_continuation_max_drawdown_from_high": 0.16,
        "gap_adjusted_continuation_max_above_ma20_pct": 0.05,
        "gap_adjusted_continuation_min_relative_strength_score": 70.0,
        "gap_adjusted_continuation_min_theme_score": 60.0,
        "gap_adjusted_continuation_min_adx_circuit_breaker": 18.0,
        "gap_adjusted_continuation_max_event_risk_score_circuit_breaker": 18.0,
        "gap_adjusted_continuation_max_gap_risk_score_circuit_breaker": 28.0,
        "gap_adjusted_continuation_benign_gap_risk_score": 14.0,
        "gap_adjusted_continuation_gap_risk_lookback_days": 10,
        "gap_adjusted_continuation_min_gap_risk_improvement": 3.0,
        "gap_adjusted_continuation_min_volume_expansion": 0.60,
        "gap_adjusted_continuation_max_volume_expansion": 1.35,
        "gap_adjusted_continuation_require_benchmark_risk_on": True,
        "gap_adjusted_continuation_require_theme_active": True,
        "gap_adjusted_continuation_same_theme_peer_quality_overlay": True,
        "gap_adjusted_continuation_same_theme_peer_min_count": 2,
        "gap_adjusted_continuation_same_theme_peer_min_share": 0.80,
        "gap_adjusted_continuation_same_theme_peer_min_avg_rs": 80.0,
        "gap_adjusted_continuation_max_score_boost": 2.5,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()].set_index("symbol")

    assert latest.loc["AAA", "gap_adjusted_continuation_score"] == 0.0
    assert bool(latest.loc["AAA", "gap_adjusted_continuation_eligible"]) is False
    assert bool(latest.loc["AAA", "gap_adjusted_continuation_blocked"]) is True
    assert "gap_continuation_peer_gate=0" in latest.loc["AAA", "score_decomposition"]


def test_momentum_pullback_volume_contraction_overlay_boosts_reset_leaders_only() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    chase = features["symbol"].eq("CCC")
    features.loc[leader, "theme_score"] = 85.0
    features.loc[leader, "rs_63d"] = 0.35
    features.loc[leader, "distance_to_prior_high_252"] = -0.09
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[leader, "volume_expansion"] = 0.72
    features.loc[leader, "trend_adx"] = 26.0
    features.loc[chase, "theme_score"] = 85.0
    features.loc[chase, "rs_63d"] = 0.34
    features.loc[chase, "distance_to_prior_high_252"] = -0.09
    features.loc[chase, "ma_20"] = features.loc[chase, "adj_close"] * 0.99
    features.loc[chase, "volume_expansion"] = 1.18

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "long_pullback_volume_contraction_overlay": True,
            "pullback_volume_contraction_max_volume_expansion": 0.90,
            "pullback_volume_contraction_min_drawdown_from_high": 0.03,
            "pullback_volume_contraction_max_drawdown_from_high": 0.16,
            "pullback_volume_contraction_max_above_ma20_pct": 0.03,
            "pullback_volume_contraction_min_relative_strength_score": 68.0,
            "pullback_volume_contraction_min_theme_score": 58.0,
            "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
            "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
            "pullback_volume_contraction_max_score_boost": 3.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]
    ccc = latest[latest["symbol"].eq("CCC")].iloc[0]

    assert aaa["final_score"] > base_aaa["final_score"]
    assert "pullback_volume_reset=" in aaa["score_decomposition"]
    assert "pullback_volume_reset=0.00" in ccc["score_decomposition"]


def test_momentum_pullback_volume_contraction_overlay_respects_circuit_breaker() -> None:
    features = _features(risk_on=False)
    leader = features["symbol"].eq("AAA")
    features.loc[leader, "theme_score"] = 85.0
    features.loc[leader, "rs_63d"] = 0.35
    features.loc[leader, "distance_to_prior_high_252"] = -0.09
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[leader, "volume_expansion"] = 0.72
    features.loc[leader, "trend_adx"] = 16.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "long_pullback_volume_contraction_overlay": True,
            "pullback_volume_contraction_max_volume_expansion": 0.90,
            "pullback_volume_contraction_min_drawdown_from_high": 0.03,
            "pullback_volume_contraction_max_drawdown_from_high": 0.16,
            "pullback_volume_contraction_max_above_ma20_pct": 0.03,
            "pullback_volume_contraction_min_relative_strength_score": 68.0,
            "pullback_volume_contraction_min_theme_score": 58.0,
            "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
            "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
            "pullback_volume_contraction_max_score_boost": 3.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["final_score"] == base_aaa["final_score"]
    assert "pullback_volume_reset=0.00" in aaa["score_decomposition"]


def test_momentum_pullback_volume_contraction_overlay_blocks_high_gap_risk() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    features.loc[leader, "theme_score"] = 85.0
    features.loc[leader, "rs_63d"] = 0.35
    features.loc[leader, "distance_to_prior_high_252"] = -0.09
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[leader, "volume_expansion"] = 0.72
    features.loc[leader, "trend_adx"] = 28.0
    features.loc[leader, "overnight_gap_risk_score"] = 55.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "long_pullback_volume_contraction_overlay": True,
            "pullback_volume_contraction_max_volume_expansion": 0.90,
            "pullback_volume_contraction_min_drawdown_from_high": 0.03,
            "pullback_volume_contraction_max_drawdown_from_high": 0.16,
            "pullback_volume_contraction_max_above_ma20_pct": 0.03,
            "pullback_volume_contraction_min_relative_strength_score": 68.0,
            "pullback_volume_contraction_min_theme_score": 58.0,
            "pullback_volume_contraction_require_benchmark_risk_on": True,
            "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 35.0,
            "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
            "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
            "pullback_volume_contraction_max_score_boost": 3.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["final_score"] == base_aaa["final_score"]
    assert "pullback_volume_reset=0.00" in aaa["score_decomposition"]


def test_momentum_pullback_reclaim_overlay_boosts_recent_ma20_reclaims_only() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    chase = features["symbol"].eq("CCC")
    leader_idx = features.index[leader]
    chase_idx = features.index[chase]

    features.loc[leader, "theme_score"] = 88.0
    features.loc[leader, "rs_63d"] = 0.36
    features.loc[leader, "distance_to_prior_high_252"] = -0.08
    features.loc[leader, "trend_adx"] = 28.0
    features.loc[leader, "volume_expansion"] = 0.85
    features.loc[leader_idx[-3:-1], "ma_20"] = features.loc[leader_idx[-3:-1], "adj_close"] * 1.02
    features.loc[leader_idx[-1], "ma_20"] = features.loc[leader_idx[-1], "adj_close"] * 0.995
    features.loc[leader_idx[-1], "volume_expansion"] = 1.18

    features.loc[chase, "theme_score"] = 87.0
    features.loc[chase, "rs_63d"] = 0.35
    features.loc[chase, "distance_to_prior_high_252"] = -0.08
    features.loc[chase, "trend_adx"] = 28.0
    features.loc[chase, "volume_expansion"] = 1.20
    features.loc[chase_idx[-1], "ma_20"] = features.loc[chase_idx[-1], "adj_close"] * 0.98

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "long_pullback_reclaim_overlay": True,
            "pullback_reclaim_min_relative_strength_score": 68.0,
            "pullback_reclaim_min_theme_score": 60.0,
            "pullback_reclaim_min_drawdown_from_high": 0.03,
            "pullback_reclaim_max_drawdown_from_high": 0.18,
            "pullback_reclaim_recent_below_ma20_lookback_days": 5,
            "pullback_reclaim_min_recent_below_ma20_pct": 0.01,
            "pullback_reclaim_max_above_ma20_pct": 0.03,
            "pullback_reclaim_min_volume_expansion": 1.0,
            "pullback_reclaim_min_adx_circuit_breaker": 22.0,
            "pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker": 35.0,
            "pullback_reclaim_max_score_boost": 3.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]
    ccc = latest[latest["symbol"].eq("CCC")].iloc[0]

    assert aaa["final_score"] > base_aaa["final_score"]
    assert "pullback_reclaim=" in aaa["score_decomposition"]
    assert "pullback_reclaim=0.00" in ccc["score_decomposition"]


def test_momentum_pullback_reclaim_overlay_respects_circuit_breaker() -> None:
    features = _features(risk_on=False)
    leader = features["symbol"].eq("AAA")
    leader_idx = features.index[leader]

    features.loc[leader, "theme_score"] = 88.0
    features.loc[leader, "rs_63d"] = 0.36
    features.loc[leader, "distance_to_prior_high_252"] = -0.08
    features.loc[leader, "trend_adx"] = 18.0
    features.loc[leader, "volume_expansion"] = 0.85
    features.loc[leader_idx[-3:-1], "ma_20"] = features.loc[leader_idx[-3:-1], "adj_close"] * 1.02
    features.loc[leader_idx[-1], "ma_20"] = features.loc[leader_idx[-1], "adj_close"] * 0.995
    features.loc[leader_idx[-1], "volume_expansion"] = 1.18

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "long_pullback_reclaim_overlay": True,
            "pullback_reclaim_min_relative_strength_score": 68.0,
            "pullback_reclaim_min_theme_score": 60.0,
            "pullback_reclaim_min_drawdown_from_high": 0.03,
            "pullback_reclaim_max_drawdown_from_high": 0.18,
            "pullback_reclaim_recent_below_ma20_lookback_days": 5,
            "pullback_reclaim_min_recent_below_ma20_pct": 0.01,
            "pullback_reclaim_max_above_ma20_pct": 0.03,
            "pullback_reclaim_min_volume_expansion": 1.0,
            "pullback_reclaim_min_adx_circuit_breaker": 22.0,
            "pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker": 35.0,
            "pullback_reclaim_max_score_boost": 3.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["final_score"] == base_aaa["final_score"]
    assert "pullback_reclaim=0.00" in aaa["score_decomposition"]


def test_momentum_short_term_volume_tilt_boosts_reset_volume_confirmation() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    laggard = features["symbol"].eq("BBB")
    features.loc[leader, "theme_score"] = 80.0
    features.loc[leader, "rs_63d"] = 0.30
    features.loc[leader, "trend_adx"] = 30.0
    features.loc[leader, "volume_expansion"] = 1.35
    features.loc[leader, "ret_5d"] = -0.035
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[laggard, "volume_expansion"] = 1.35
    features.loc[laggard, "ret_5d"] = -0.04

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "short_term_volume_tilt_overlay": True,
            "short_term_volume_tilt_max_ret_5d": 0.00,
            "short_term_volume_tilt_min_volume_expansion": 1.05,
            "short_term_volume_tilt_min_relative_strength_score": 58.0,
            "short_term_volume_tilt_min_theme_score": 55.0,
            "short_term_volume_tilt_max_above_ma20_pct": 0.08,
            "short_term_volume_tilt_require_benchmark_risk_on": True,
            "short_term_volume_tilt_require_price_above_ma50": True,
            "short_term_volume_tilt_min_adx_circuit_breaker": 18.0,
            "short_term_volume_tilt_max_score_boost": 1.75,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]
    bbb = latest[latest["symbol"].eq("BBB")].iloc[0]

    assert aaa["final_score"] > base_aaa["final_score"]
    assert aaa["short_term_volume_tilt_score"] > 0
    assert bool(aaa["short_term_volume_tilt_candidate"]) is True
    assert bool(aaa["short_term_volume_tilt_circuit_ok"]) is True
    assert bool(aaa["short_term_volume_tilt_peer_confirmed"]) is True
    assert aaa["short_term_volume_tilt_block_reason"] == ""
    assert "short_term_volume_tilt=" in aaa["score_decomposition"]
    assert bbb["short_term_volume_tilt_score"] == 0


def test_momentum_short_term_volume_tilt_can_require_same_theme_peer_confirmation() -> None:
    features = _features()
    latest_date = features["date"].max()
    features["primary_theme"] = "other"
    features.loc[features["symbol"].eq("SPY"), "primary_theme"] = "benchmark_etfs"
    features.loc[features["symbol"].isin(["AAA", "BBB", "CCC"]), "primary_theme"] = "ai_software_data"

    leader = features["symbol"].eq("AAA")
    peer_two = features["symbol"].eq("BBB")
    support_peer = features["symbol"].eq("CCC")
    candidate_symbols = features["symbol"].isin(["AAA", "CCC"])

    features.loc[candidate_symbols, "theme_score"] = 80.0
    features.loc[candidate_symbols, "volume_expansion"] = 1.35
    features.loc[candidate_symbols, "ret_5d"] = -0.03
    features.loc[candidate_symbols, "ma_20"] = features.loc[candidate_symbols, "adj_close"] * 0.99
    features.loc[candidate_symbols, "trend_adx"] = 30.0
    features.loc[leader, "rs_63d"] = 0.30
    features.loc[support_peer, "rs_63d"] = 0.28
    features.loc[peer_two, "theme_score"] = 52.0
    features.loc[peer_two, "rs_63d"] = -0.25
    features.loc[peer_two, "volume_expansion"] = 1.20
    features.loc[peer_two, "ret_5d"] = -0.01

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "short_term_volume_tilt_overlay": True,
        "short_term_volume_tilt_max_ret_5d": 0.00,
        "short_term_volume_tilt_min_volume_expansion": 1.05,
        "short_term_volume_tilt_min_relative_strength_score": 58.0,
        "short_term_volume_tilt_min_theme_score": 55.0,
        "short_term_volume_tilt_max_above_ma20_pct": 0.08,
        "short_term_volume_tilt_require_benchmark_risk_on": True,
        "short_term_volume_tilt_require_price_above_ma50": True,
        "short_term_volume_tilt_min_adx_circuit_breaker": 18.0,
        "short_term_volume_tilt_max_score_boost": 1.75,
        "short_term_volume_tilt_same_theme_peer_confirmation_overlay": True,
        "short_term_volume_tilt_same_theme_peer_min_count": 2,
        "short_term_volume_tilt_same_theme_peer_min_share": 0.50,
        "short_term_volume_tilt_same_theme_peer_min_avg_mom_return": 0.12,
    }

    isolated = features.copy()
    isolated.loc[isolated["date"].eq(latest_date) & support_peer, "theme_score"] = 50.0
    isolated_signals = CrossSectionalMomentumLongShort(params).generate_signals(isolated, StrategyContext(benchmark="SPY"))
    isolated_latest = isolated_signals[isolated_signals["date"] == latest_date].set_index("symbol")
    assert isolated_latest.loc["AAA", "short_term_volume_tilt_score"] == 0.0
    assert bool(isolated_latest.loc["AAA", "short_term_volume_tilt_candidate"]) is True
    assert bool(isolated_latest.loc["AAA", "short_term_volume_tilt_blocked"]) is True
    assert isolated_latest.loc["AAA", "short_term_volume_tilt_block_reason"] == "same_theme_peer"

    supportive = features.copy()
    supportive_signals = CrossSectionalMomentumLongShort(params).generate_signals(supportive, StrategyContext(benchmark="SPY"))
    supportive_latest = supportive_signals[supportive_signals["date"] == latest_date].set_index("symbol")
    assert supportive_latest.loc["AAA", "short_term_volume_tilt_score"] > 0.0
    assert supportive_latest.loc["CCC", "short_term_volume_tilt_score"] > 0.0


def test_momentum_short_term_volume_tilt_can_use_same_theme_peer_volume_substitution() -> None:
    features = _features()
    latest_date = features["date"].max()
    features["primary_theme"] = "other"
    features.loc[features["symbol"].eq("SPY"), "primary_theme"] = "benchmark_etfs"
    features.loc[features["symbol"].isin(["AAA", "BBB", "CCC"]), "primary_theme"] = "semiconductors"

    leader = features["symbol"].eq("AAA")
    support_peer = features["symbol"].eq("CCC")
    peer_two = features["symbol"].eq("BBB")
    supportive_symbols = features["symbol"].isin(["AAA", "BBB", "CCC"])

    features.loc[supportive_symbols, "theme_score"] = 82.0
    features.loc[supportive_symbols, "ret_5d"] = -0.03
    features.loc[supportive_symbols, "ma_20"] = features.loc[
        supportive_symbols,
        "adj_close",
    ] * 0.99
    features.loc[supportive_symbols, "trend_adx"] = 30.0
    features.loc[leader, "rs_63d"] = 0.30
    features.loc[support_peer, "rs_63d"] = 0.28
    features.loc[leader, "volume_expansion"] = 0.82
    features.loc[support_peer, "volume_expansion"] = 1.35
    features.loc[peer_two, "theme_score"] = 80.0
    features.loc[peer_two, "rs_63d"] = 0.24
    features.loc[peer_two, "volume_expansion"] = 1.25

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "short_term_volume_tilt_overlay": True,
        "short_term_volume_tilt_max_ret_5d": 0.00,
        "short_term_volume_tilt_min_volume_expansion": 1.05,
        "short_term_volume_tilt_min_relative_strength_score": 58.0,
        "short_term_volume_tilt_min_theme_score": 55.0,
        "short_term_volume_tilt_max_above_ma20_pct": 0.08,
        "short_term_volume_tilt_require_benchmark_risk_on": True,
        "short_term_volume_tilt_require_price_above_ma50": True,
        "short_term_volume_tilt_min_adx_circuit_breaker": 18.0,
        "short_term_volume_tilt_max_score_boost": 1.75,
        "short_term_volume_tilt_same_theme_peer_confirmation_overlay": True,
        "short_term_volume_tilt_same_theme_peer_min_count": 2,
        "short_term_volume_tilt_same_theme_peer_min_share": 0.33,
        "short_term_volume_tilt_same_theme_peer_min_avg_mom_return": 0.12,
        "short_term_volume_tilt_same_theme_peer_volume_substitution_overlay": True,
        "short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion": 0.75,
        "short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall": 0.30,
        "short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale": 0.60,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == latest_date].set_index("symbol")

    assert latest.loc["AAA", "short_term_volume_tilt_score"] > 0.0
    assert bool(latest.loc["AAA", "short_term_volume_tilt_candidate"]) is True
    assert latest.loc["AAA", "short_term_volume_tilt_block_reason"] == ""
    assert "short_term_volume_mode=same_theme_peer_volume_substitution" in latest.loc["AAA", "score_decomposition"]


def test_momentum_short_term_volume_tilt_peer_volume_substitution_respects_volume_floor() -> None:
    features = _features()
    latest_date = features["date"].max()
    features["primary_theme"] = "other"
    features.loc[features["symbol"].eq("SPY"), "primary_theme"] = "benchmark_etfs"
    features.loc[features["symbol"].isin(["AAA", "BBB", "CCC"]), "primary_theme"] = "semiconductors"

    leader = features["symbol"].eq("AAA")
    support_peer = features["symbol"].eq("CCC")
    peer_two = features["symbol"].eq("BBB")

    supportive_symbols = features["symbol"].isin(["AAA", "BBB", "CCC"])

    features.loc[supportive_symbols, "theme_score"] = 82.0
    features.loc[supportive_symbols, "ret_5d"] = -0.03
    features.loc[supportive_symbols, "ma_20"] = features.loc[
        supportive_symbols,
        "adj_close",
    ] * 0.99
    features.loc[supportive_symbols, "trend_adx"] = 30.0
    features.loc[leader, "rs_63d"] = 0.30
    features.loc[support_peer, "rs_63d"] = 0.28
    features.loc[leader, "volume_expansion"] = 0.45
    features.loc[support_peer, "volume_expansion"] = 1.35
    features.loc[peer_two, "theme_score"] = 80.0
    features.loc[peer_two, "rs_63d"] = 0.24
    features.loc[peer_two, "volume_expansion"] = 1.25

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "short_term_volume_tilt_overlay": True,
        "short_term_volume_tilt_max_ret_5d": 0.00,
        "short_term_volume_tilt_min_volume_expansion": 1.05,
        "short_term_volume_tilt_min_relative_strength_score": 58.0,
        "short_term_volume_tilt_min_theme_score": 55.0,
        "short_term_volume_tilt_max_above_ma20_pct": 0.08,
        "short_term_volume_tilt_require_benchmark_risk_on": True,
        "short_term_volume_tilt_require_price_above_ma50": True,
        "short_term_volume_tilt_min_adx_circuit_breaker": 18.0,
        "short_term_volume_tilt_max_score_boost": 1.75,
        "short_term_volume_tilt_same_theme_peer_confirmation_overlay": True,
        "short_term_volume_tilt_same_theme_peer_min_count": 2,
        "short_term_volume_tilt_same_theme_peer_min_share": 0.50,
        "short_term_volume_tilt_same_theme_peer_min_avg_mom_return": 0.12,
        "short_term_volume_tilt_same_theme_peer_volume_substitution_overlay": True,
        "short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion": 0.70,
        "short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall": 0.30,
        "short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale": 0.60,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == latest_date].set_index("symbol")

    assert latest.loc["AAA", "short_term_volume_tilt_score"] == 0.0
    assert bool(latest.loc["AAA", "short_term_volume_tilt_candidate"]) is True
    assert bool(latest.loc["AAA", "short_term_volume_tilt_blocked"]) is True
    assert latest.loc["AAA", "short_term_volume_tilt_block_reason"] == "volume_confirmation"


def test_momentum_short_term_volume_tilt_can_use_same_theme_peer_non_risk_on_exception() -> None:
    features = _features(risk_on=False)
    latest_date = features["date"].max()
    features["primary_theme"] = "other"
    features.loc[features["symbol"].eq("SPY"), "primary_theme"] = "benchmark_etfs"
    features.loc[features["symbol"].isin(["AAA", "CCC"]), "primary_theme"] = "ai_software_data"
    features.loc[features["symbol"].eq("SPY"), "ret_63d"] = 0.05

    candidate_symbols = features["symbol"].isin(["AAA", "CCC"])
    features.loc[candidate_symbols, "theme_score"] = 80.0
    features.loc[candidate_symbols, "volume_expansion"] = 1.35
    features.loc[candidate_symbols, "ret_5d"] = -0.03
    features.loc[candidate_symbols, "ma_20"] = features.loc[candidate_symbols, "adj_close"] * 0.99
    features.loc[candidate_symbols, "trend_adx"] = 30.0
    features.loc[features["symbol"].eq("AAA"), "rs_63d"] = 0.30
    features.loc[features["symbol"].eq("CCC"), "rs_63d"] = 0.28

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "short_term_volume_tilt_overlay": True,
        "short_term_volume_tilt_max_ret_5d": 0.00,
        "short_term_volume_tilt_min_volume_expansion": 1.05,
        "short_term_volume_tilt_min_relative_strength_score": 58.0,
        "short_term_volume_tilt_min_theme_score": 55.0,
        "short_term_volume_tilt_max_above_ma20_pct": 0.08,
        "short_term_volume_tilt_require_benchmark_risk_on": True,
        "short_term_volume_tilt_require_price_above_ma50": True,
        "short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker": 0.0,
        "short_term_volume_tilt_min_adx_circuit_breaker": 18.0,
        "short_term_volume_tilt_max_score_boost": 1.75,
        "short_term_volume_tilt_same_theme_peer_confirmation_overlay": True,
        "short_term_volume_tilt_same_theme_peer_min_count": 2,
        "short_term_volume_tilt_same_theme_peer_min_share": 0.50,
        "short_term_volume_tilt_same_theme_peer_min_avg_mom_return": 0.12,
    }

    blocked_signals = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    blocked_latest = blocked_signals[blocked_signals["date"] == latest_date].set_index("symbol")
    assert blocked_latest.loc["AAA", "short_term_volume_tilt_score"] == 0.0
    assert bool(blocked_latest.loc["AAA", "short_term_volume_tilt_blocked"]) is True
    assert blocked_latest.loc["AAA", "short_term_volume_tilt_block_reason"] == "benchmark_risk_off"

    exception_params = dict(base_params)
    exception_params.update(
        {
            "short_term_volume_tilt_non_risk_on_exception_overlay": True,
            "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count": 2,
            "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share": 0.50,
            "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return": 0.12,
            "short_term_volume_tilt_non_risk_on_exception_score_boost_scale": 0.50,
        }
    )
    exception_signals = CrossSectionalMomentumLongShort(exception_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    exception_latest = exception_signals[exception_signals["date"] == latest_date].set_index("symbol")
    assert exception_latest.loc["AAA", "short_term_volume_tilt_score"] > 0.0
    assert bool(exception_latest.loc["AAA", "short_term_volume_tilt_blocked"]) is False
    assert exception_latest.loc["AAA", "short_term_volume_tilt_block_reason"] == ""
    assert "short_term_volume_mode=same_theme_peer_exception" in exception_latest.loc["AAA", "score_decomposition"]


def test_momentum_pullback_reset_inclusion_can_bypass_plain_rank_gate() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    features.loc[leader, "theme_score"] = 85.0
    features.loc[leader, "rs_63d"] = 0.35
    features.loc[leader, "distance_to_prior_high_252"] = -0.09
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[leader, "volume_expansion"] = 0.72
    features.loc[leader, "trend_adx"] = 28.0

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 99.0,
        "long_pullback_volume_contraction_overlay": True,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 3.0,
        "pullback_reset_inclusion_min_score": 30.0,
        "pullback_reset_inclusion_max_names_per_date": 1,
        "pullback_reset_inclusion_min_final_score": 50.0,
        "pullback_reset_inclusion_min_score_rank": 0.40,
    }
    without_inclusion = dict(params)
    without_inclusion["long_pullback_reset_inclusion_overlay"] = False

    base = CrossSectionalMomentumLongShort(without_inclusion).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert base_aaa["signal"] == 0.0
    assert aaa["signal"] == 1.0
    assert bool(aaa["pullback_reset_inclusion"])
    assert "reset_include=1" in aaa["score_decomposition"]


def test_momentum_pullback_reset_inclusion_activation_support_can_bypass_volume_reset_score() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    features.loc[leader, "theme_score"] = 85.0
    features.loc[leader, "rs_63d"] = 0.35
    features.loc[leader, "distance_to_prior_high_252"] = -0.02
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[leader, "volume_expansion"] = 1.30
    features.loc[leader, "ret_5d"] = -0.035
    features.loc[leader, "trend_adx"] = 28.0

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 99.0,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_reset_inclusion_min_score": 30.0,
        "pullback_reset_inclusion_max_names_per_date": 1,
        "pullback_reset_inclusion_min_final_score": 50.0,
        "pullback_reset_inclusion_min_score_rank": 0.40,
        "pullback_reset_inclusion_activation_support_overlay": True,
        "pullback_reset_inclusion_activation_min_score": 35.0,
        "pullback_reset_inclusion_activation_score_rank_credit": 0.08,
        "short_term_volume_tilt_max_ret_5d": 0.0,
        "short_term_volume_tilt_min_volume_expansion": 1.05,
        "short_term_volume_tilt_min_relative_strength_score": 60.0,
        "short_term_volume_tilt_min_theme_score": 56.0,
        "short_term_volume_tilt_max_above_ma20_pct": 0.08,
        "short_term_volume_tilt_require_benchmark_risk_on": True,
        "short_term_volume_tilt_require_theme_active": True,
        "short_term_volume_tilt_require_price_above_ma50": True,
        "short_term_volume_tilt_min_adx_circuit_breaker": 18.0,
    }
    without_support = dict(params)
    without_support["pullback_reset_inclusion_activation_support_overlay"] = False

    base = CrossSectionalMomentumLongShort(without_support).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert base_aaa["signal"] == 0.0
    assert aaa["signal"] == 1.0
    assert bool(aaa["pullback_reset_inclusion"])
    assert "reset_activation_gate=1" in aaa["score_decomposition"]


def test_momentum_pullback_reset_inclusion_activation_support_respects_circuit_breaker() -> None:
    features = _features(risk_on=False)
    leader = features["symbol"].eq("AAA")
    features.loc[leader, "theme_score"] = 85.0
    features.loc[leader, "rs_63d"] = 0.35
    features.loc[leader, "distance_to_prior_high_252"] = -0.02
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[leader, "volume_expansion"] = 1.30
    features.loc[leader, "ret_5d"] = -0.035
    features.loc[leader, "trend_adx"] = 28.0

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 99.0,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_reset_inclusion_min_score": 30.0,
        "pullback_reset_inclusion_max_names_per_date": 1,
        "pullback_reset_inclusion_min_final_score": 50.0,
        "pullback_reset_inclusion_min_score_rank": 0.40,
        "pullback_reset_inclusion_activation_support_overlay": True,
        "pullback_reset_inclusion_activation_min_score": 35.0,
        "pullback_reset_inclusion_activation_score_rank_credit": 0.08,
        "short_term_volume_tilt_max_ret_5d": 0.0,
        "short_term_volume_tilt_min_volume_expansion": 1.05,
        "short_term_volume_tilt_min_relative_strength_score": 60.0,
        "short_term_volume_tilt_min_theme_score": 56.0,
        "short_term_volume_tilt_max_above_ma20_pct": 0.08,
        "short_term_volume_tilt_require_benchmark_risk_on": True,
        "short_term_volume_tilt_require_theme_active": True,
        "short_term_volume_tilt_require_price_above_ma50": True,
        "short_term_volume_tilt_min_adx_circuit_breaker": 18.0,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["signal"] == 0.0
    assert "reset_activation_gate=0" in aaa["score_decomposition"]


def test_momentum_pullback_reset_inclusion_can_require_theme_breadth_expansion() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    latest_date = features["date"].max()
    features.loc[leader, "theme_score"] = 85.0
    features.loc[leader, "rs_63d"] = 0.35
    features.loc[leader, "distance_to_prior_high_252"] = -0.09
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[leader, "volume_expansion"] = 0.72
    features.loc[leader, "trend_adx"] = 28.0
    features["theme_active"] = False
    expansion_names = features["symbol"].isin(["AAA", "CCC", "SPY"]) & features["date"].eq(latest_date)
    features.loc[expansion_names, "theme_active"] = True

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 99.0,
        "long_pullback_volume_contraction_overlay": True,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 3.0,
        "pullback_reset_inclusion_min_score": 30.0,
        "pullback_reset_inclusion_max_names_per_date": 1,
        "pullback_reset_inclusion_min_final_score": 50.0,
        "pullback_reset_inclusion_min_score_rank": 0.40,
        "pullback_reset_inclusion_require_theme_breadth_expansion": True,
        "pullback_reset_inclusion_theme_breadth_lookback_days": 10,
        "pullback_reset_inclusion_theme_breadth_min_active_share": 0.60,
        "pullback_reset_inclusion_theme_breadth_expansion_threshold": 0.20,
    }

    expanding = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    expanding_latest = expanding[expanding["date"] == latest_date]
    expanding_aaa = expanding_latest[expanding_latest["symbol"].eq("AAA")].iloc[0]

    flat_features = features.copy()
    flat_features.loc[flat_features["date"].eq(latest_date), "theme_active"] = flat_features.loc[
        flat_features["date"].eq(latest_date), "symbol"
    ].eq("AAA")
    flat = CrossSectionalMomentumLongShort(params).generate_signals(flat_features, StrategyContext(benchmark="SPY"))
    flat_latest = flat[flat["date"] == latest_date]
    flat_aaa = flat_latest[flat_latest["symbol"].eq("AAA")].iloc[0]

    assert bool(expanding_aaa["pullback_reset_inclusion"])
    assert "reset_breadth_gate=1" in expanding_aaa["score_decomposition"]
    assert flat_aaa["signal"] == 0.0
    assert not bool(flat_aaa["pullback_reset_inclusion"])
    assert "reset_breadth_gate=0" in flat_aaa["score_decomposition"]


def test_momentum_pullback_reset_inclusion_can_require_rs_acceleration() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    latest_date = features["date"].max()
    features.loc[leader, "theme_score"] = 85.0
    features.loc[leader, "rs_63d"] = -0.10
    features.loc[leader & features["date"].eq(latest_date), "rs_63d"] = 0.60
    features.loc[leader, "distance_to_prior_high_252"] = -0.09
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[leader, "volume_expansion"] = 0.72
    features.loc[leader, "trend_adx"] = 28.0

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 99.0,
        "long_pullback_volume_contraction_overlay": True,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 3.0,
        "pullback_reset_inclusion_min_score": 30.0,
        "pullback_reset_inclusion_max_names_per_date": 1,
        "pullback_reset_inclusion_min_final_score": 50.0,
        "pullback_reset_inclusion_min_score_rank": 0.40,
        "pullback_reset_inclusion_require_rs_acceleration": True,
        "pullback_reset_inclusion_rs_acceleration_lookback_days": 10,
        "pullback_reset_inclusion_min_current_rs_score": 78.0,
        "pullback_reset_inclusion_min_rs_acceleration": 5.0,
    }

    accelerating = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    accelerating_latest = accelerating[accelerating["date"] == latest_date]
    accelerating_aaa = accelerating_latest[accelerating_latest["symbol"].eq("AAA")].iloc[0]

    flat_features = features.copy()
    flat_features.loc[leader, "rs_63d"] = 0.60
    flat = CrossSectionalMomentumLongShort(params).generate_signals(flat_features, StrategyContext(benchmark="SPY"))
    flat_latest = flat[flat["date"] == latest_date]
    flat_aaa = flat_latest[flat_latest["symbol"].eq("AAA")].iloc[0]

    assert bool(accelerating_aaa["pullback_reset_inclusion"])
    assert bool(accelerating_aaa["pullback_reset_rs_accelerating"])
    assert "reset_rs_gate=1" in accelerating_aaa["score_decomposition"]
    assert flat_aaa["signal"] == 0.0
    assert not bool(flat_aaa["pullback_reset_inclusion"])
    assert "reset_rs_gate=0" in flat_aaa["score_decomposition"]


def test_momentum_pullback_reset_inclusion_theme_strength_support_can_relax_rank_gate() -> None:
    features = _features()
    features.loc[features["symbol"].isin(["AAA", "CCC"]), "primary_theme"] = "semiconductors"
    features.loc[features["symbol"].isin(["AAA", "CCC"]), "theme_reason"] = "semiconductors"
    leader = features["symbol"].eq("AAA")
    dates = sorted(features["date"].unique())
    for i, date in enumerate(dates):
        semis = features["date"].eq(date) & features["symbol"].isin(["AAA", "CCC"])
        features.loc[semis, "theme_score"] = 52.0 + min(i, 30) * 0.6
        features.loc[semis, "theme_active"] = True
        features.loc[semis, "overnight_gap_risk_score"] = 0.0
    features.loc[leader, "distance_to_prior_high_252"] = -0.09
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.99
    features.loc[leader, "volume_expansion"] = 0.72
    features.loc[leader, "trend_adx"] = 28.0

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 99.0,
        "theme_strength_delta_overlay": True,
        "theme_strength_delta_lookback_days": 5,
        "theme_strength_delta_min_peer_count": 2,
        "theme_strength_delta_min_theme_score": 50.0,
        "theme_strength_delta_min_relative_strength_score": 50.0,
        "theme_strength_delta_min_theme_prior": 48.0,
        "theme_strength_delta_min_peer_rs_prior": 50.0,
        "theme_strength_delta_min_combined_delta": 0.0,
        "theme_strength_delta_max_score_boost": 1.5,
        "theme_strength_delta_require_benchmark_risk_on": True,
        "theme_strength_delta_require_theme_active": False,
        "theme_strength_delta_max_event_risk_score_circuit_breaker": 20.0,
        "theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "theme_strength_delta_min_benchmark_ret63d_circuit_breaker": 0.0,
        "long_pullback_volume_contraction_overlay": True,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 3.0,
        "pullback_reset_inclusion_min_score": 30.0,
        "pullback_reset_inclusion_max_names_per_date": 1,
        "pullback_reset_inclusion_min_final_score": 50.0,
        "pullback_reset_inclusion_min_score_rank": 0.78,
        "pullback_reset_inclusion_theme_strength_support_overlay": True,
        "pullback_reset_inclusion_min_theme_strength_delta_score": 20.0,
        "pullback_reset_inclusion_theme_strength_score_rank_credit": 0.20,
    }
    without_support = dict(params)
    without_support["pullback_reset_inclusion_theme_strength_support_overlay"] = False

    base = CrossSectionalMomentumLongShort(without_support).generate_signals(features, StrategyContext(benchmark="SPY"))
    supported = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest_date = supported["date"].max()
    base_latest = base[base["date"] == latest_date]
    supported_latest = supported[supported["date"] == latest_date]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    supported_aaa = supported_latest[supported_latest["symbol"].eq("AAA")].iloc[0]

    assert base_aaa["signal"] == 0.0
    assert supported_aaa["signal"] == 1.0
    assert bool(supported_aaa["pullback_reset_inclusion"])
    assert "reset_theme_strength_gate=1" in supported_aaa["score_decomposition"]
    assert "reset_rank_credit=" in supported_aaa["score_decomposition"]


def test_momentum_pullback_reset_inclusion_reclaim_support_can_relax_rank_gate() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    dates = sorted(features["date"].unique())
    leader_idx = features.index[leader]

    features.loc[:, "theme_active"] = features["symbol"].isin(["AAA", "CCC"])
    features.loc[leader, "theme_score"] = 86.0
    features.loc[leader, "rs_63d"] = 0.36
    features.loc[leader, "distance_to_prior_high_252"] = -0.09
    features.loc[leader, "trend_adx"] = 28.0
    features.loc[leader, "volume_expansion"] = 0.82
    features.loc[leader_idx[-3:-1], "ma_20"] = features.loc[leader_idx[-3:-1], "adj_close"] * 1.02
    features.loc[leader_idx[-1], "ma_20"] = features.loc[leader_idx[-1], "adj_close"] * 0.995
    features.loc[leader_idx[-1], "volume_expansion"] = 1.18

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 99.0,
        "long_pullback_volume_contraction_overlay": True,
        "long_pullback_reclaim_overlay": True,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 3.0,
        "pullback_reclaim_min_relative_strength_score": 68.0,
        "pullback_reclaim_min_theme_score": 60.0,
        "pullback_reclaim_min_drawdown_from_high": 0.03,
        "pullback_reclaim_max_drawdown_from_high": 0.18,
        "pullback_reclaim_recent_below_ma20_lookback_days": 5,
        "pullback_reclaim_min_recent_below_ma20_pct": 0.01,
        "pullback_reclaim_max_above_ma20_pct": 0.03,
        "pullback_reclaim_min_volume_expansion": 1.0,
        "pullback_reclaim_min_adx_circuit_breaker": 22.0,
        "pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "pullback_reclaim_max_score_boost": 3.0,
        "pullback_reset_inclusion_min_score": 30.0,
        "pullback_reset_inclusion_max_names_per_date": 1,
        "pullback_reset_inclusion_min_final_score": 50.0,
        "pullback_reset_inclusion_min_score_rank": 0.78,
        "pullback_reset_inclusion_reclaim_support_overlay": True,
        "pullback_reset_inclusion_min_reclaim_score": 20.0,
        "pullback_reset_inclusion_reclaim_score_rank_credit": 0.20,
        "pullback_reset_inclusion_require_theme_breadth_not_deteriorating": True,
        "pullback_reset_inclusion_theme_breadth_lookback_days": 10,
        "pullback_reset_inclusion_theme_breadth_active_share_threshold": 0.30,
        "pullback_reset_inclusion_theme_breadth_shortfall_threshold": 0.10,
    }
    without_support = dict(params)
    without_support["pullback_reset_inclusion_reclaim_support_overlay"] = False

    base = CrossSectionalMomentumLongShort(without_support).generate_signals(features, StrategyContext(benchmark="SPY"))
    supported = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest_date = dates[-1]
    base_latest = base[base["date"] == latest_date]
    supported_latest = supported[supported["date"] == latest_date]
    base_aaa = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    supported_aaa = supported_latest[supported_latest["symbol"].eq("AAA")].iloc[0]

    assert base_aaa["signal"] == 0.0
    assert supported_aaa["signal"] == 1.0
    assert bool(supported_aaa["pullback_reset_inclusion"])
    assert "reset_reclaim_gate=1" in supported_aaa["score_decomposition"]
    assert "reset_reclaim_credit=" in supported_aaa["score_decomposition"]


def test_momentum_pullback_reset_inclusion_reclaim_support_respects_theme_breadth_safety() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    follower = features["symbol"].eq("CCC")
    dates = sorted(features["date"].unique())
    leader_idx = features.index[leader]
    latest_date = dates[-1]

    features.loc[:, "theme_active"] = features["symbol"].isin(["AAA", "CCC"])
    features.loc[follower & features["date"].eq(latest_date), "theme_active"] = False
    features.loc[features["symbol"].eq("BBB"), "theme_active"] = False
    features.loc[features["symbol"].eq("SPY"), "theme_active"] = False
    features.loc[leader, "theme_score"] = 86.0
    features.loc[leader, "rs_63d"] = 0.36
    features.loc[leader, "distance_to_prior_high_252"] = -0.09
    features.loc[leader, "trend_adx"] = 28.0
    features.loc[leader, "volume_expansion"] = 0.82
    features.loc[leader_idx[-3:-1], "ma_20"] = features.loc[leader_idx[-3:-1], "adj_close"] * 1.02
    features.loc[leader_idx[-1], "ma_20"] = features.loc[leader_idx[-1], "adj_close"] * 0.995
    features.loc[leader_idx[-1], "volume_expansion"] = 1.18

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_final_score": 99.0,
        "long_pullback_volume_contraction_overlay": True,
        "long_pullback_reclaim_overlay": True,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 3.0,
        "pullback_reclaim_min_relative_strength_score": 68.0,
        "pullback_reclaim_min_theme_score": 60.0,
        "pullback_reclaim_min_drawdown_from_high": 0.03,
        "pullback_reclaim_max_drawdown_from_high": 0.18,
        "pullback_reclaim_recent_below_ma20_lookback_days": 5,
        "pullback_reclaim_min_recent_below_ma20_pct": 0.01,
        "pullback_reclaim_max_above_ma20_pct": 0.03,
        "pullback_reclaim_min_volume_expansion": 1.0,
        "pullback_reclaim_min_adx_circuit_breaker": 22.0,
        "pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker": 35.0,
        "pullback_reclaim_max_score_boost": 3.0,
        "pullback_reset_inclusion_min_score": 30.0,
        "pullback_reset_inclusion_max_names_per_date": 1,
        "pullback_reset_inclusion_min_final_score": 50.0,
        "pullback_reset_inclusion_min_score_rank": 0.78,
        "pullback_reset_inclusion_reclaim_support_overlay": True,
        "pullback_reset_inclusion_min_reclaim_score": 20.0,
        "pullback_reset_inclusion_reclaim_score_rank_credit": 0.20,
        "pullback_reset_inclusion_require_theme_breadth_not_deteriorating": True,
        "pullback_reset_inclusion_theme_breadth_lookback_days": 10,
        "pullback_reset_inclusion_theme_breadth_active_share_threshold": 0.40,
        "pullback_reset_inclusion_theme_breadth_shortfall_threshold": 0.10,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == latest_date]
    aaa = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert aaa["signal"] == 0.0
    assert not bool(aaa["pullback_reset_inclusion"])
    assert "reset_reclaim_gate=0" in aaa["score_decomposition"]
    assert "reset_breadth_safe=0" in aaa["score_decomposition"]


def test_momentum_reentry_discipline_overlay_penalizes_late_weak_volume_chase() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    chase = features["symbol"].eq("CCC")
    features.loc[leader, "theme_score"] = 85.0
    features.loc[leader, "rs_63d"] = 0.35
    features.loc[leader, "distance_to_prior_high_252"] = -0.10
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.98
    features.loc[leader, "volume_expansion"] = 1.05
    features.loc[leader, "trend_adx"] = 34.0
    features.loc[chase, "theme_score"] = 85.0
    features.loc[chase, "rs_63d"] = 0.33
    features.loc[chase, "distance_to_prior_high_252"] = 0.01
    features.loc[chase, "ma_20"] = features.loc[chase, "adj_close"] * 0.92
    features.loc[chase, "volume_expansion"] = 0.65
    features.loc[chase, "trend_adx"] = 14.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 60,
        "long_min_relative_strength_score": 60,
        "long_reentry_discipline_overlay": False,
    }
    disciplined = dict(base_params)
    disciplined.update(
        {
            "long_reentry_discipline_overlay": True,
            "reentry_discipline_near_high_threshold": 0.03,
            "reentry_discipline_min_above_ma20_pct": 0.04,
            "reentry_discipline_max_volume_expansion": 0.95,
            "reentry_discipline_min_adx_circuit_breaker": 25.0,
            "reentry_discipline_max_score_penalty": 4.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(disciplined).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_chase = base_latest[base_latest["symbol"].eq("CCC")].iloc[0]
    chase_row = latest[latest["symbol"].eq("CCC")].iloc[0]
    leader_row = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert chase_row["final_score"] < base_chase["final_score"]
    assert "reentry_penalty=-0.00" not in chase_row["score_decomposition"]
    assert "reentry_penalty=-0.00" in leader_row["score_decomposition"]


def test_momentum_reentry_discipline_can_require_theme_deterioration() -> None:
    features = _features()
    chase = features["symbol"].eq("CCC")
    features["theme_active"] = True
    features.loc[chase, "theme_score"] = 85.0
    features.loc[chase, "rs_63d"] = 0.33
    features.loc[chase, "distance_to_prior_high_252"] = 0.01
    features.loc[chase, "ma_20"] = features.loc[chase, "adj_close"] * 0.92
    features.loc[chase, "volume_expansion"] = 0.65
    features.loc[chase, "trend_adx"] = 14.0
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 50,
        "long_min_relative_strength_score": 60,
        "long_reentry_discipline_overlay": True,
        "reentry_discipline_near_high_threshold": 0.03,
        "reentry_discipline_min_above_ma20_pct": 0.04,
        "reentry_discipline_max_volume_expansion": 0.95,
        "reentry_discipline_min_adx_circuit_breaker": 25.0,
        "reentry_discipline_max_score_penalty": 4.0,
        "reentry_discipline_require_theme_deterioration": True,
        "reentry_discipline_theme_score_deterioration_threshold": 64.0,
        "reentry_discipline_penalize_theme_inactive": True,
    }

    strong_theme = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    strong_row = strong_theme[strong_theme["date"] == strong_theme["date"].max()]
    strong_chase = strong_row[strong_row["symbol"].eq("CCC")].iloc[0]

    weak_features = features.copy()
    weak_features.loc[chase, "theme_score"] = 55.0
    weak_features.loc[chase, "theme_active"] = False
    weak_theme = CrossSectionalMomentumLongShort(params).generate_signals(weak_features, StrategyContext(benchmark="SPY"))
    weak_row = weak_theme[weak_theme["date"] == weak_theme["date"].max()]
    weak_chase = weak_row[weak_row["symbol"].eq("CCC")].iloc[0]

    assert "reentry_penalty=-0.00" in strong_chase["score_decomposition"]
    assert "reentry_penalty=-0.00" not in weak_chase["score_decomposition"]


def test_momentum_reentry_discipline_can_require_theme_breadth_deterioration() -> None:
    features = _features()
    chase = features["symbol"].eq("CCC")
    latest_date = features["date"].max()
    features["theme_active"] = True
    features.loc[chase, "theme_score"] = 85.0
    features.loc[chase, "rs_63d"] = 0.33
    features.loc[chase, "distance_to_prior_high_252"] = 0.01
    features.loc[chase, "ma_20"] = features.loc[chase, "adj_close"] * 0.92
    features.loc[chase, "volume_expansion"] = 0.65
    features.loc[chase, "trend_adx"] = 14.0
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 50,
        "long_min_relative_strength_score": 60,
        "long_reentry_discipline_overlay": True,
        "reentry_discipline_near_high_threshold": 0.03,
        "reentry_discipline_min_above_ma20_pct": 0.04,
        "reentry_discipline_max_volume_expansion": 0.95,
        "reentry_discipline_min_adx_circuit_breaker": 25.0,
        "reentry_discipline_max_score_penalty": 4.0,
        "reentry_discipline_require_theme_breadth_deterioration": True,
        "reentry_discipline_theme_breadth_lookback_days": 10,
        "reentry_discipline_theme_breadth_active_share_threshold": 0.40,
        "reentry_discipline_theme_breadth_shortfall_threshold": 0.20,
    }

    healthy_theme = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    healthy_row = healthy_theme[healthy_theme["date"] == latest_date]
    healthy_chase = healthy_row[healthy_row["symbol"].eq("CCC")].iloc[0]

    weak_breadth_features = features.copy()
    weak_breadth = weak_breadth_features["symbol"].isin(["AAA", "BBB", "SPY"]) & weak_breadth_features["date"].eq(latest_date)
    weak_breadth_features.loc[weak_breadth, "theme_active"] = False
    weak_theme = CrossSectionalMomentumLongShort(params).generate_signals(weak_breadth_features, StrategyContext(benchmark="SPY"))
    weak_row = weak_theme[weak_theme["date"] == latest_date]
    weak_chase = weak_row[weak_row["symbol"].eq("CCC")].iloc[0]

    assert "breadth_gate=0" in healthy_chase["score_decomposition"]
    assert "reentry_penalty=-0.00" in healthy_chase["score_decomposition"]
    assert "breadth_gate=1" in weak_chase["score_decomposition"]
    assert "reentry_penalty=-0.00" not in weak_chase["score_decomposition"]


def test_momentum_reentry_discipline_overlay_respects_adx_circuit_breaker() -> None:
    features = _features()
    chase = features["symbol"].eq("CCC")
    features.loc[chase, "theme_score"] = 85.0
    features.loc[chase, "rs_63d"] = 0.33
    features.loc[chase, "distance_to_prior_high_252"] = 0.01
    features.loc[chase, "ma_20"] = features.loc[chase, "adj_close"] * 0.92
    features.loc[chase, "volume_expansion"] = 0.65
    features.loc[chase, "trend_adx"] = 38.0
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 60,
        "long_min_relative_strength_score": 60,
        "long_reentry_discipline_overlay": True,
        "reentry_discipline_near_high_threshold": 0.03,
        "reentry_discipline_min_above_ma20_pct": 0.04,
        "reentry_discipline_max_volume_expansion": 0.95,
        "reentry_discipline_min_adx_circuit_breaker": 25.0,
        "reentry_discipline_max_score_penalty": 4.0,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    chase_row = latest[latest["symbol"].eq("CCC")].iloc[0]

    assert "reentry_penalty=-0.00" in chase_row["score_decomposition"]


def test_momentum_reentry_discipline_overlay_exempts_true_volume_reset_leaders() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    chase = features["symbol"].eq("CCC")
    features.loc[leader, "theme_score"] = 88.0
    features.loc[leader, "rs_63d"] = 0.36
    features.loc[leader, "distance_to_prior_high_252"] = 0.01
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] * 0.965
    features.loc[leader, "volume_expansion"] = 0.55
    features.loc[leader, "trend_adx"] = 23.0
    features.loc[chase, "theme_score"] = 85.0
    features.loc[chase, "rs_63d"] = 0.33
    features.loc[chase, "distance_to_prior_high_252"] = 0.01
    features.loc[chase, "ma_20"] = features.loc[chase, "adj_close"] * 0.95
    features.loc[chase, "volume_expansion"] = 0.65
    features.loc[chase, "trend_adx"] = 18.0

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 60,
        "long_min_relative_strength_score": 60,
        "long_pullback_volume_contraction_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.0,
        "pullback_volume_contraction_max_drawdown_from_high": 0.04,
        "pullback_volume_contraction_max_above_ma20_pct": 0.06,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 3.0,
        "long_reentry_discipline_overlay": True,
        "reentry_discipline_near_high_threshold": 0.03,
        "reentry_discipline_min_above_ma20_pct": 0.04,
        "reentry_discipline_max_volume_expansion": 0.95,
        "reentry_discipline_min_adx_circuit_breaker": 25.0,
        "reentry_discipline_max_score_penalty": 4.0,
        "reentry_discipline_min_pullback_volume_reset_score_exemption": 10.0,
        "reentry_discipline_min_relative_strength_score_exemption": 75.0,
        "reentry_discipline_min_theme_score_exemption": 70.0,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    leader_row = latest[latest["symbol"].eq("AAA")].iloc[0]
    chase_row = latest[latest["symbol"].eq("CCC")].iloc[0]

    assert leader_row["pullback_volume_contraction_score"] > 0.0
    assert "reentry_penalty=-0.00" in leader_row["score_decomposition"]
    assert "reentry_penalty=-0.00" not in chase_row["score_decomposition"]


def test_momentum_short_rebound_avoidance_overlay_blocks_exhausted_shorts() -> None:
    features = _features(risk_on=False)
    weak = features["symbol"].eq("BBB")
    features.loc[weak, "distance_to_prior_high_252"] = -0.32
    features.loc[weak, "ma_20"] = features.loc[weak, "adj_close"] * 1.12
    features.loc[weak, "volume_expansion"] = 1.45
    features.loc[weak, "trend_adx"] = 18.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.50,
        "max_positions": 10,
        "short_min_weakness_conditions": 3,
        "short_only_when_benchmark_risk_off": True,
        "short_max_final_score": 55,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "short_rebound_avoidance_overlay": True,
            "short_rebound_avoidance_min_below_ma20_pct": 0.08,
            "short_rebound_avoidance_min_distance_from_high": 0.20,
            "short_rebound_avoidance_min_volume_expansion": 1.10,
            "short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker": -0.12,
            "short_rebound_avoidance_min_adx_circuit_breaker": 32.0,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_short = base_latest[base_latest["symbol"].eq("BBB")].iloc[0]
    short_row = latest[latest["symbol"].eq("BBB")].iloc[0]

    assert base_short["signal"] < 0
    assert short_row["signal"] == 0
    assert "short_block=1" in short_row["score_decomposition"]


def test_momentum_pullback_short_term_reset_overlay_boosts_controlled_resets() -> None:
    features = _features()
    leader = features["symbol"].eq("AAA")
    features.loc[leader, "distance_to_prior_high_252"] = -0.06
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] / 1.01
    features.loc[leader, "ret_5d"] = -0.015
    features.loc[leader, "ret_10d"] = -0.01
    features.loc[leader, "trend_adx"] = 28.0
    features.loc[leader, "overnight_gap_risk_score"] = 8.0

    base_params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 60,
        "long_min_relative_strength_score": 60,
    }
    overlay_params = dict(base_params)
    overlay_params.update(
        {
            "pullback_short_term_reset_overlay": True,
            "pullback_short_term_reset_min_drawdown_from_high": 0.03,
            "pullback_short_term_reset_max_drawdown_from_high": 0.16,
            "pullback_short_term_reset_max_above_ma20_pct": 0.03,
            "pullback_short_term_reset_max_ret_5d": 0.01,
            "pullback_short_term_reset_max_ret_10d": 0.02,
            "pullback_short_term_reset_min_relative_strength_score": 58.0,
            "pullback_short_term_reset_min_theme_score": 56.0,
            "pullback_short_term_reset_require_benchmark_risk_on": True,
            "pullback_short_term_reset_require_theme_active": True,
            "pullback_short_term_reset_max_event_risk_score_circuit_breaker": 20.0,
            "pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker": 0.0,
            "pullback_short_term_reset_min_adx_circuit_breaker": 20.0,
            "pullback_short_term_reset_max_score_boost": 2.5,
        }
    )

    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(overlay_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_row = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    leader_row = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert leader_row["final_score"] > base_row["final_score"]
    assert leader_row["signal"] > 0
    assert "pullback_short_term_block=0" in leader_row["score_decomposition"]
    assert "pullback_short_term_reset=0.00" not in leader_row["score_decomposition"]


def test_momentum_pullback_short_term_reset_overlay_respects_risk_on_circuit_breaker() -> None:
    features = _features(risk_on=False)
    leader = features["symbol"].eq("AAA")
    features.loc[leader, "distance_to_prior_high_252"] = -0.06
    features.loc[leader, "ma_20"] = features.loc[leader, "adj_close"] / 1.01
    features.loc[leader, "ret_5d"] = -0.015
    features.loc[leader, "ret_10d"] = -0.01
    features.loc[leader, "trend_adx"] = 28.0

    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.75,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 60,
        "long_min_relative_strength_score": 60,
        "pullback_short_term_reset_overlay": True,
        "pullback_short_term_reset_min_drawdown_from_high": 0.03,
        "pullback_short_term_reset_max_drawdown_from_high": 0.16,
        "pullback_short_term_reset_max_above_ma20_pct": 0.03,
        "pullback_short_term_reset_max_ret_5d": 0.01,
        "pullback_short_term_reset_max_ret_10d": 0.02,
        "pullback_short_term_reset_min_relative_strength_score": 58.0,
        "pullback_short_term_reset_min_theme_score": 56.0,
        "pullback_short_term_reset_require_benchmark_risk_on": True,
        "pullback_short_term_reset_require_theme_active": True,
        "pullback_short_term_reset_max_event_risk_score_circuit_breaker": 20.0,
        "pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker": 30.0,
        "pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_short_term_reset_min_adx_circuit_breaker": 20.0,
        "pullback_short_term_reset_max_score_boost": 2.5,
    }

    base = CrossSectionalMomentumLongShort(
        {
            "lookback_returns": 21,
            "skip_recent_days": 3,
            "long_quantile": 0.75,
            "short_quantile": 0.0,
            "max_positions": 10,
            "long_min_theme_score": 60,
            "long_min_relative_strength_score": 60,
        }
    ).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    base_row = base_latest[base_latest["symbol"].eq("AAA")].iloc[0]
    leader_row = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert leader_row["final_score"] == base_row["final_score"]
    assert "pullback_short_term_block=1" in leader_row["score_decomposition"]


def test_momentum_same_theme_reset_substitution_overlay_swaps_in_guarded_candidate() -> None:
    features = _same_theme_substitution_features()
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.35,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 55,
        "long_min_relative_strength_score": 55,
        "long_require_price_above_ma50": True,
        "long_pullback_volume_contraction_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 1.0,
        "pullback_reset_same_theme_substitution_overlay": True,
        "pullback_reset_same_theme_substitution_min_relative_strength_edge": 4.0,
        "pullback_reset_same_theme_substitution_min_theme_score_edge": 2.0,
        "pullback_reset_same_theme_substitution_min_reset_score": 20.0,
        "pullback_reset_same_theme_substitution_min_reclaim_score": 0.0,
        "pullback_reset_same_theme_substitution_min_theme_strength_delta_score": 0.0,
        "pullback_reset_same_theme_substitution_max_final_score_deficit": 6.0,
        "pullback_reset_same_theme_substitution_max_score_rank_gap": 0.12,
        "pullback_reset_same_theme_substitution_max_event_risk_score_circuit_breaker": 20.0,
        "pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker": 0.05,
        "pullback_reset_same_theme_substitution_max_promotions_per_date": 1,
        "pullback_reset_same_theme_substitution_require_theme_active": True,
    }

    base_params = dict(params)
    base_params["pullback_reset_same_theme_substitution_overlay"] = False
    base = CrossSectionalMomentumLongShort(base_params).generate_signals(features, StrategyContext(benchmark="SPY"))
    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))

    base_latest = base[base["date"] == base["date"].max()]
    latest = signals[signals["date"] == signals["date"].max()]
    assert base_latest[base_latest["symbol"].eq("AAA")].iloc[0]["signal"] > 0
    assert base_latest[base_latest["symbol"].eq("DDD")].iloc[0]["signal"] == 0

    replacement_row = latest[latest["symbol"].eq("DDD")].iloc[0]
    demoted_row = latest[latest["symbol"].eq("AAA")].iloc[0]
    assert replacement_row["signal"] > 0
    assert demoted_row["signal"] == 0
    assert "reset_sub=1" in replacement_row["score_decomposition"]
    assert "reset_sub_demote=1" in demoted_row["score_decomposition"]


def test_momentum_same_theme_reset_substitution_overlay_respects_volatility_circuit_breaker() -> None:
    features = _same_theme_substitution_features(high_vol_candidate=True)
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.35,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 55,
        "long_min_relative_strength_score": 55,
        "long_require_price_above_ma50": True,
        "long_pullback_volume_contraction_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 1.0,
        "pullback_reset_same_theme_substitution_overlay": True,
        "pullback_reset_same_theme_substitution_min_relative_strength_edge": 4.0,
        "pullback_reset_same_theme_substitution_min_theme_score_edge": 2.0,
        "pullback_reset_same_theme_substitution_min_reset_score": 20.0,
        "pullback_reset_same_theme_substitution_min_reclaim_score": 0.0,
        "pullback_reset_same_theme_substitution_min_theme_strength_delta_score": 0.0,
        "pullback_reset_same_theme_substitution_max_final_score_deficit": 6.0,
        "pullback_reset_same_theme_substitution_max_score_rank_gap": 0.12,
        "pullback_reset_same_theme_substitution_max_event_risk_score_circuit_breaker": 20.0,
        "pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker": 0.05,
        "pullback_reset_same_theme_substitution_max_promotions_per_date": 1,
        "pullback_reset_same_theme_substitution_require_theme_active": True,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    replacement_row = latest[latest["symbol"].eq("DDD")].iloc[0]
    demoted_row = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert replacement_row["signal"] == 0
    assert demoted_row["signal"] > 0
    assert "reset_sub=1" not in replacement_row["score_decomposition"]


def test_momentum_same_theme_reset_substitution_overlay_respects_theme_breadth_safety() -> None:
    features = _same_theme_substitution_features()
    latest_date = features["date"].max()
    weak_breadth = features["date"].eq(latest_date) & features["symbol"].isin(["AAA", "BBB", "CCC"])
    features.loc[weak_breadth, "theme_active"] = False
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.35,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 55,
        "long_min_relative_strength_score": 55,
        "long_require_price_above_ma50": True,
        "long_pullback_volume_contraction_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 1.0,
        "pullback_reset_same_theme_substitution_overlay": True,
        "pullback_reset_same_theme_substitution_min_relative_strength_edge": 4.0,
        "pullback_reset_same_theme_substitution_min_theme_score_edge": 2.0,
        "pullback_reset_same_theme_substitution_min_reset_score": 20.0,
        "pullback_reset_same_theme_substitution_min_reclaim_score": 0.0,
        "pullback_reset_same_theme_substitution_min_theme_strength_delta_score": 0.0,
        "pullback_reset_same_theme_substitution_max_final_score_deficit": 6.0,
        "pullback_reset_same_theme_substitution_max_score_rank_gap": 0.12,
        "pullback_reset_same_theme_substitution_max_event_risk_score_circuit_breaker": 20.0,
        "pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker": 0.05,
        "pullback_reset_same_theme_substitution_max_promotions_per_date": 1,
        "pullback_reset_same_theme_substitution_require_theme_active": True,
        "pullback_reset_same_theme_substitution_require_theme_breadth_not_deteriorating": True,
        "pullback_reset_same_theme_substitution_theme_breadth_lookback_days": 10,
        "pullback_reset_same_theme_substitution_theme_breadth_active_share_threshold": 0.40,
        "pullback_reset_same_theme_substitution_theme_breadth_shortfall_threshold": 0.08,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == latest_date]
    replacement_row = latest[latest["symbol"].eq("DDD")].iloc[0]
    demoted_row = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert replacement_row["signal"] == 0
    assert demoted_row["signal"] > 0
    assert "reset_breadth_safe=0" in replacement_row["score_decomposition"]


def test_momentum_same_theme_reset_substitution_overlay_can_require_activation_support() -> None:
    features = _same_theme_substitution_features(activation_ready_candidate=True)
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.35,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 55,
        "long_min_relative_strength_score": 55,
        "long_require_price_above_ma50": True,
        "long_pullback_volume_contraction_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 1.0,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_reset_inclusion_activation_support_overlay": True,
        "pullback_reset_inclusion_activation_min_score": 35.0,
        "pullback_reset_inclusion_activation_score_rank_credit": 0.08,
        "pullback_reset_same_theme_substitution_overlay": True,
        "pullback_reset_same_theme_substitution_require_activation_support": True,
        "pullback_reset_same_theme_substitution_min_relative_strength_edge": 4.0,
        "pullback_reset_same_theme_substitution_min_theme_score_edge": 2.0,
        "pullback_reset_same_theme_substitution_min_reset_score": 20.0,
        "pullback_reset_same_theme_substitution_min_reclaim_score": 0.0,
        "pullback_reset_same_theme_substitution_min_theme_strength_delta_score": 0.0,
        "pullback_reset_same_theme_substitution_max_final_score_deficit": 6.0,
        "pullback_reset_same_theme_substitution_max_score_rank_gap": 0.12,
        "pullback_reset_same_theme_substitution_max_event_risk_score_circuit_breaker": 20.0,
        "pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker": 0.05,
        "pullback_reset_same_theme_substitution_max_promotions_per_date": 1,
        "pullback_reset_same_theme_substitution_require_theme_active": True,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    replacement_row = latest[latest["symbol"].eq("DDD")].iloc[0]
    demoted_row = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert replacement_row["signal"] > 0
    assert demoted_row["signal"] == 0


def test_momentum_same_theme_reset_substitution_overlay_blocks_without_activation_support() -> None:
    features = _same_theme_substitution_features()
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.35,
        "short_quantile": 0.0,
        "max_positions": 10,
        "long_min_theme_score": 55,
        "long_min_relative_strength_score": 55,
        "long_require_price_above_ma50": True,
        "long_pullback_volume_contraction_overlay": True,
        "pullback_volume_contraction_max_volume_expansion": 0.90,
        "pullback_volume_contraction_min_drawdown_from_high": 0.03,
        "pullback_volume_contraction_max_drawdown_from_high": 0.16,
        "pullback_volume_contraction_max_above_ma20_pct": 0.03,
        "pullback_volume_contraction_min_relative_strength_score": 68.0,
        "pullback_volume_contraction_min_theme_score": 58.0,
        "pullback_volume_contraction_require_benchmark_risk_on": True,
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": 0.0,
        "pullback_volume_contraction_min_adx_circuit_breaker": 22.0,
        "pullback_volume_contraction_max_score_boost": 1.0,
        "long_pullback_reset_inclusion_overlay": True,
        "pullback_reset_inclusion_activation_support_overlay": True,
        "pullback_reset_inclusion_activation_min_score": 35.0,
        "pullback_reset_inclusion_activation_score_rank_credit": 0.08,
        "pullback_reset_same_theme_substitution_overlay": True,
        "pullback_reset_same_theme_substitution_require_activation_support": True,
        "pullback_reset_same_theme_substitution_min_relative_strength_edge": 4.0,
        "pullback_reset_same_theme_substitution_min_theme_score_edge": 2.0,
        "pullback_reset_same_theme_substitution_min_reset_score": 20.0,
        "pullback_reset_same_theme_substitution_min_reclaim_score": 0.0,
        "pullback_reset_same_theme_substitution_min_theme_strength_delta_score": 0.0,
        "pullback_reset_same_theme_substitution_max_final_score_deficit": 6.0,
        "pullback_reset_same_theme_substitution_max_score_rank_gap": 0.12,
        "pullback_reset_same_theme_substitution_max_event_risk_score_circuit_breaker": 20.0,
        "pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker": 25.0,
        "pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker": 0.05,
        "pullback_reset_same_theme_substitution_max_promotions_per_date": 1,
        "pullback_reset_same_theme_substitution_require_theme_active": True,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    replacement_row = latest[latest["symbol"].eq("DDD")].iloc[0]
    demoted_row = latest[latest["symbol"].eq("AAA")].iloc[0]

    assert replacement_row["signal"] == 0
    assert demoted_row["signal"] > 0
    assert "reset_activation_gate=0" in replacement_row["score_decomposition"]
    assert "reset_sub=1" not in replacement_row["score_decomposition"]


def test_momentum_short_rebound_avoidance_overlay_respects_circuit_breaker() -> None:
    features = _features(risk_on=False)
    weak = features["symbol"].eq("BBB")
    features.loc[weak, "distance_to_prior_high_252"] = -0.32
    features.loc[weak, "ma_20"] = features.loc[weak, "adj_close"] * 1.12
    features.loc[weak, "volume_expansion"] = 1.45
    features.loc[weak, "trend_adx"] = 38.0
    params = {
        "lookback_returns": 21,
        "skip_recent_days": 3,
        "long_quantile": 0.0,
        "short_quantile": 0.50,
        "max_positions": 10,
        "short_min_weakness_conditions": 3,
        "short_only_when_benchmark_risk_off": True,
        "short_max_final_score": 55,
        "short_rebound_avoidance_overlay": True,
        "short_rebound_avoidance_min_below_ma20_pct": 0.08,
        "short_rebound_avoidance_min_distance_from_high": 0.20,
        "short_rebound_avoidance_min_volume_expansion": 1.10,
        "short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker": -0.12,
        "short_rebound_avoidance_min_adx_circuit_breaker": 32.0,
    }

    signals = CrossSectionalMomentumLongShort(params).generate_signals(features, StrategyContext(benchmark="SPY"))
    latest = signals[signals["date"] == signals["date"].max()]
    short_row = latest[latest["symbol"].eq("BBB")].iloc[0]

    assert short_row["signal"] < 0
    assert "short_block=0" in short_row["score_decomposition"]


def test_pullback_reclaim_same_theme_gap_confirmation_rewards_supported_reclaims() -> None:
    frame = _pullback_reclaim_same_theme_gap_features()

    out = _apply_pullback_reclaim_overlay(
        frame,
        {
            "long_pullback_reclaim_overlay": True,
            "pullback_reclaim_same_theme_gap_confirmation_overlay": True,
            "pullback_reclaim_min_relative_strength_score": 74.0,
            "pullback_reclaim_min_theme_score": 60.0,
            "pullback_reclaim_min_drawdown_from_high": 0.04,
            "pullback_reclaim_max_drawdown_from_high": 0.18,
            "pullback_reclaim_recent_below_ma20_lookback_days": 5,
            "pullback_reclaim_min_recent_below_ma20_pct": 0.01,
            "pullback_reclaim_max_above_ma20_pct": 0.025,
            "pullback_reclaim_min_volume_expansion": 1.0,
            "pullback_reclaim_require_benchmark_risk_on": True,
            "pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "pullback_reclaim_min_benchmark_ret63d_circuit_breaker": 0.0,
            "pullback_reclaim_min_adx_circuit_breaker": 24.0,
            "pullback_reclaim_max_score_boost": 3.0,
            "pullback_reclaim_same_theme_gap_confirmation_min_peer_count": 3,
            "pullback_reclaim_same_theme_gap_confirmation_min_active_share": 0.75,
            "pullback_reclaim_same_theme_gap_confirmation_max_avg_gap_risk_score": 12.0,
            "pullback_reclaim_same_theme_gap_confirmation_min_avg_relative_strength_score": 80.0,
        },
    )
    latest = out[out["date"] == out["date"].max()].set_index("symbol")

    assert bool(latest.loc["LEAD", "pullback_reclaim_same_theme_gap_confirmation_ok"]) is True
    assert latest.loc["LEAD", "pullback_reclaim_same_theme_peer_count"] == 3
    assert latest.loc["LEAD", "pullback_reclaim_same_theme_active_share"] >= 1.0
    assert latest.loc["LEAD", "pullback_reclaim_score"] > 0.0
    assert latest.loc["LEAD", "pullback_reclaim_boost"] > 0.0


def test_pullback_reclaim_same_theme_gap_confirmation_blocks_high_gap_theme_reclaims() -> None:
    frame = _pullback_reclaim_same_theme_gap_features(peer_gap_risk=28.0)

    out = _apply_pullback_reclaim_overlay(
        frame,
        {
            "long_pullback_reclaim_overlay": True,
            "pullback_reclaim_same_theme_gap_confirmation_overlay": True,
            "pullback_reclaim_min_relative_strength_score": 74.0,
            "pullback_reclaim_min_theme_score": 60.0,
            "pullback_reclaim_min_drawdown_from_high": 0.04,
            "pullback_reclaim_max_drawdown_from_high": 0.18,
            "pullback_reclaim_recent_below_ma20_lookback_days": 5,
            "pullback_reclaim_min_recent_below_ma20_pct": 0.01,
            "pullback_reclaim_max_above_ma20_pct": 0.025,
            "pullback_reclaim_min_volume_expansion": 1.0,
            "pullback_reclaim_require_benchmark_risk_on": True,
            "pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker": 30.0,
            "pullback_reclaim_min_benchmark_ret63d_circuit_breaker": 0.0,
            "pullback_reclaim_min_adx_circuit_breaker": 24.0,
            "pullback_reclaim_max_score_boost": 3.0,
            "pullback_reclaim_same_theme_gap_confirmation_min_peer_count": 3,
            "pullback_reclaim_same_theme_gap_confirmation_min_active_share": 0.75,
            "pullback_reclaim_same_theme_gap_confirmation_max_avg_gap_risk_score": 12.0,
            "pullback_reclaim_same_theme_gap_confirmation_min_avg_relative_strength_score": 80.0,
        },
    )
    latest = out[out["date"] == out["date"].max()].set_index("symbol")

    assert bool(latest.loc["LEAD", "pullback_reclaim_same_theme_gap_confirmation_ok"]) is False
    assert latest.loc["LEAD", "pullback_reclaim_same_theme_avg_gap_risk"] > 12.0
    assert bool(latest.loc["LEAD", "pullback_reclaim_blocked"]) is True
    assert latest.loc["LEAD", "pullback_reclaim_score"] == 0.0
    assert latest.loc["LEAD", "pullback_reclaim_boost"] == 0.0
