"""Market regime detection."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from quant_system.config import RegimeConfig
from quant_system.regime.analog_regimes import identify_analog_regimes_pit

LOGGER = logging.getLogger(__name__)


def detect_market_regime(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """Classify benchmark risk-on/risk-off with simple moving-average rules."""

    bench = prices[prices["symbol"] == benchmark.upper()].sort_values("date").copy()
    if bench.empty:
        return pd.DataFrame(columns=["date", "risk_on", "regime_score"])
    for window in (20, 50, 200):
        bench[f"ma_{window}"] = bench["adj_close"].rolling(window, min_periods=max(5, window // 3)).mean()
    bench["ret_20d"] = bench["adj_close"].pct_change(20)
    bench["ret_63d"] = bench["adj_close"].pct_change(63)
    bench["realized_vol_20d"] = bench["adj_close"].pct_change().rolling(20, min_periods=5).std() * (252**0.5)
    bench["high_252"] = bench["adj_close"].rolling(252, min_periods=60).max()
    bench["drawdown_52w"] = bench["adj_close"] / bench["high_252"] - 1.0
    score = (
        (bench["adj_close"] > bench["ma_20"]).astype(float) * 20
        + (bench["adj_close"] > bench["ma_50"]).astype(float) * 20
        + (bench["adj_close"] > bench["ma_200"]).astype(float) * 25
        + (bench["ret_20d"] > 0).astype(float) * 15
        + (bench["ret_63d"] > 0).astype(float) * 15
        + (bench["realized_vol_20d"] < 0.30).astype(float) * 5
    )
    bench["regime_score"] = score
    bench["risk_on"] = bench["regime_score"] >= 60
    return bench[["date", "risk_on", "regime_score", "ret_20d", "ret_63d", "realized_vol_20d", "drawdown_52w"]]


def apply_regime_exposure_overlay(
    targets: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: str,
    config: RegimeConfig,
) -> pd.DataFrame:
    """Scale target weights by market regime using only same-day close data."""

    if targets.empty or not config.enabled or not (config.exposure_overlay or config.overbought_overlay):
        return targets
    regime = detect_market_regime(prices, benchmark)
    if regime.empty and config.exposure_overlay:
        return targets
    if config.exposure_overlay and not regime.empty:
        overlay = regime[["date", "regime_score", "risk_on", "ret_20d", "ret_63d", "realized_vol_20d", "drawdown_52w"]].copy()
        overlay["date"] = pd.to_datetime(overlay["date"]).dt.normalize()
        full_score = float(config.full_exposure_regime_score)
        partial_score = float(config.partial_exposure_regime_score)
        risk_off_multiplier = float(config.risk_off_exposure_multiplier)
        partial_multiplier = float(config.partial_risk_on_exposure_multiplier)
        overlay["base_regime_exposure_multiplier"] = risk_off_multiplier
        overlay.loc[overlay["regime_score"] >= partial_score, "base_regime_exposure_multiplier"] = partial_multiplier
        overlay.loc[overlay["regime_score"] >= full_score, "base_regime_exposure_multiplier"] = 1.0
        overlay["regime_exposure_multiplier"] = overlay["base_regime_exposure_multiplier"]
    else:
        overlay = pd.DataFrame(
            {
                "date": pd.to_datetime(targets["date"]).dt.normalize().unique(),
                "base_regime_exposure_multiplier": 1.0,
                "regime_exposure_multiplier": 1.0,
            }
        )
    if config.overbought_overlay:
        overbought = detect_overbought_risk(prices, benchmark, config)
        if not overbought.empty:
            overlay = overlay.merge(overbought, on="date", how="left")
            overlay["overbought_multiplier"] = overlay["overbought_multiplier"].fillna(1.0)
            overlay["regime_exposure_multiplier"] *= overlay["overbought_multiplier"].astype(float)
    out = targets.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(overlay, on="date", how="left")
    out["base_regime_exposure_multiplier"] = pd.to_numeric(
        out.get("base_regime_exposure_multiplier", out.get("regime_exposure_multiplier", 1.0)),
        errors="coerce",
    ).fillna(1.0)
    if "overbought_multiplier" in out.columns:
        out["overbought_multiplier"] = pd.to_numeric(out["overbought_multiplier"], errors="coerce").fillna(1.0)
    else:
        out["overbought_multiplier"] = 1.0
    out["regime_exposure_multiplier"] = out["regime_exposure_multiplier"].fillna(1.0)
    out["regime_leader_exception_applied"] = False
    out["regime_leader_exception_circuit_breaker"] = "feature_off"
    out["regime_leader_exception_log"] = ""
    out = _apply_regime_leader_exception(out, config)
    if config.analog_momentum_overlay:
        analog = identify_analog_regimes_pit(prices, config.analog_lookback_days, config.analog_quantile)
        if not analog.empty:
            analog = analog[["date", "analog_regime", "analog_score"]].copy()
            analog["date"] = pd.to_datetime(analog["date"]).dt.normalize()
            analog["analog_exposure_multiplier"] = float(config.analog_inactive_exposure_multiplier)
            analog.loc[analog["analog_regime"].astype(bool), "analog_exposure_multiplier"] = float(config.analog_active_exposure_multiplier)
            out = out.merge(analog, on="date", how="left")
            out["analog_exposure_multiplier"] = out["analog_exposure_multiplier"].fillna(1.0)
            out["regime_exposure_multiplier"] *= out["analog_exposure_multiplier"].astype(float)
    out["target_weight"] = out["target_weight"].astype(float) * out["regime_exposure_multiplier"].astype(float)
    if "reason_for_entry" in out.columns:
        risk_off = out["regime_exposure_multiplier"] < 1.0
        out.loc[risk_off, "reason_for_entry"] = (
            out.loc[risk_off, "reason_for_entry"].fillna("").astype(str)
            + " Regime overlay reduced exposure."
        ).str.strip()
        relief_mask = out["regime_leader_exception_applied"].fillna(False).astype(bool)
        out.loc[relief_mask, "reason_for_entry"] = (
            out.loc[relief_mask, "reason_for_entry"].fillna("").astype(str)
            + " Strong-leader regime exception restored part of a partial regime cut under benchmark and symbol risk breakers."
        ).str.strip()
        if "overbought_status" in out.columns:
            overbought_mask = out["overbought_status"].fillna("normal").ne("normal")
            out.loc[overbought_mask, "reason_for_entry"] = (
                out.loc[overbought_mask, "reason_for_entry"].fillna("").astype(str)
                + " Overbought risk layer reduced/blocked chasing."
            ).str.strip()
    return out


def _apply_regime_leader_exception(targets: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Restore part of a partial regime cut for strong leaders only.

    Feature flag: ``regime_leader_exception_overlay`` defaults to false.
    Reasonable ranges:
    - ``regime_leader_exception_restore_multiplier``: 0.65 to 0.90.
    - ``regime_leader_exception_min_regime_score``: 48 to 58.
    - ``regime_leader_exception_min_final_score``: 68 to 80.
    - ``regime_leader_exception_min_relative_strength_score``: 86 to 96.
    - ``regime_leader_exception_min_theme_score``: 58 to 72.
    - ``regime_leader_exception_min_mom_return``: 0.08 to 0.20.
    - ``regime_leader_exception_min_drawdown_from_high``: 0.02 to 0.06.
    - ``regime_leader_exception_max_drawdown_from_high``: 0.08 to 0.18.
    - ``regime_leader_exception_max_above_ma20_pct``: 0.02 to 0.08.

    Circuit breakers: the exception never overrides the risk-off multiplier,
    only applies inside the partial regime band, clips exposure to <= 1.0x,
    and still requires bounded benchmark volatility, bounded benchmark
    drawdown, bounded event/gap risk, and minimum trend strength.
    """

    out = targets.copy()
    if out.empty or not bool(config.regime_leader_exception_overlay):
        return out

    def _series(column: str, default: float | bool) -> pd.Series:
        if column in out.columns:
            return pd.Series(out[column], index=out.index)
        return pd.Series(default, index=out.index)

    base_multiplier = pd.to_numeric(_series("base_regime_exposure_multiplier", 1.0), errors="coerce").fillna(1.0)
    overbought_multiplier = pd.to_numeric(_series("overbought_multiplier", 1.0), errors="coerce").fillna(1.0)
    target_weight = pd.to_numeric(_series("target_weight", 0.0), errors="coerce").fillna(0.0)
    regime_score = pd.to_numeric(_series("regime_score", 0.0), errors="coerce").fillna(0.0)
    final_score = pd.to_numeric(_series("final_score", 0.0), errors="coerce").fillna(0.0)
    rs_score = pd.to_numeric(_series("relative_strength_score", 0.0), errors="coerce").fillna(0.0)
    theme_score = pd.to_numeric(_series("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(_series("mom_return", 0.0), errors="coerce").fillna(0.0)
    drawdown_from_high = pd.to_numeric(
        _series("distance_to_prior_high_252", 0.0 if "distance_to_high_252" not in out.columns else 0.0),
        errors="coerce",
    ).fillna(0.0).abs()
    if "distance_to_prior_high_252" not in out.columns and "distance_to_high_252" in out.columns:
        drawdown_from_high = pd.to_numeric(_series("distance_to_high_252", 0.0), errors="coerce").fillna(0.0).abs()
    above_ma20 = pd.to_numeric(_series("retest_distance_ma_pct", 0.0), errors="coerce").fillna(0.0).abs()
    adx = pd.to_numeric(_series("trend_adx", 0.0), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(_series("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(_series("overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    benchmark_ret_20d = pd.to_numeric(_series("ret_20d", 0.0), errors="coerce").fillna(0.0)
    realized_vol_20d = pd.to_numeric(_series("realized_vol_20d", 0.0), errors="coerce").fillna(0.0)
    benchmark_drawdown_52w = pd.to_numeric(_series("drawdown_52w", 0.0), errors="coerce").fillna(0.0)
    theme_active = _series("theme_active", False).fillna(False).astype(bool)
    price_vs_slow = pd.to_numeric(_series("trend_price_vs_slow_pct", 0.0), errors="coerce").fillna(0.0)

    partial_state = (
        regime_score.ge(float(config.partial_exposure_regime_score))
        & regime_score.lt(float(config.full_exposure_regime_score))
        & regime_score.ge(float(config.regime_leader_exception_min_regime_score))
        & base_multiplier.gt(float(config.risk_off_exposure_multiplier) + 1e-12)
        & base_multiplier.lt(1.0 - 1e-12)
    )
    not_overbought_cut = overbought_multiplier.ge(1.0 - 1e-12)
    candidate = (
        target_weight.gt(0.0)
        & partial_state
        & not_overbought_cut
        & final_score.ge(float(config.regime_leader_exception_min_final_score))
        & rs_score.ge(float(config.regime_leader_exception_min_relative_strength_score))
        & theme_score.ge(float(config.regime_leader_exception_min_theme_score))
        & mom_return.ge(float(config.regime_leader_exception_min_mom_return))
        & drawdown_from_high.ge(float(config.regime_leader_exception_min_drawdown_from_high))
        & drawdown_from_high.le(float(config.regime_leader_exception_max_drawdown_from_high))
        & above_ma20.le(float(config.regime_leader_exception_max_above_ma20_pct))
        & adx.ge(float(config.regime_leader_exception_min_adx_circuit_breaker))
        & event_risk.le(float(config.regime_leader_exception_max_event_risk_score_circuit_breaker))
        & gap_risk.le(float(config.regime_leader_exception_max_overnight_gap_risk_score_circuit_breaker))
        & benchmark_ret_20d.ge(float(config.regime_leader_exception_min_benchmark_ret_20d_circuit_breaker))
        & realized_vol_20d.le(float(config.regime_leader_exception_max_realized_vol_20d_circuit_breaker))
        & benchmark_drawdown_52w.ge(float(config.regime_leader_exception_min_benchmark_drawdown_52w_circuit_breaker))
    )
    if bool(config.regime_leader_exception_require_theme_active):
        candidate &= theme_active
    if bool(config.regime_leader_exception_require_price_above_ma50):
        candidate &= price_vs_slow.gt(0.0)

    out["regime_leader_exception_circuit_breaker"] = np.where(
        ~partial_state,
        np.where(base_multiplier.le(float(config.risk_off_exposure_multiplier) + 1e-12), "risk_off_state", "not_partial_state"),
        np.where(~not_overbought_cut, "overbought_cut_active", "not_strong_leader"),
    )
    restore_multiplier = float(np.clip(config.regime_leader_exception_restore_multiplier, 0.0, 1.0))
    effective_multiplier = np.clip(np.maximum(base_multiplier, restore_multiplier), 0.0, 1.0)
    apply_mask = candidate & base_multiplier.lt(restore_multiplier)
    out.loc[apply_mask, "base_regime_exposure_multiplier"] = effective_multiplier.loc[apply_mask]
    out.loc[apply_mask, "regime_exposure_multiplier"] = effective_multiplier.loc[apply_mask] * overbought_multiplier.loc[apply_mask]
    out.loc[apply_mask, "regime_leader_exception_applied"] = True
    out.loc[apply_mask, "regime_leader_exception_circuit_breaker"] = "applied"
    out["regime_leader_exception_log"] = [
        json.dumps(
            {
                "event": "regime_leader_exception",
                "applied": bool(applied),
                "restore_multiplier": restore_multiplier,
                "base_multiplier": float(base),
                "effective_multiplier": float(effective),
                "regime_score": float(regime_value),
                "final_score": float(score),
                "relative_strength_score": float(rs),
                "theme_score": float(theme),
                "mom_return": float(mom),
                "drawdown_from_high": float(drawdown),
                "above_ma20_pct": float(ma20),
                "trend_adx": float(adx_value),
                "event_risk_score": float(event_value),
                "overnight_gap_risk_score": float(gap_value),
                "benchmark_ret_20d": float(ret20),
                "benchmark_realized_vol_20d": float(vol20),
                "benchmark_drawdown_52w": float(drawdown_52w),
                "price_vs_slow_pct": float(slow_value),
                "theme_active": bool(theme_on),
                "circuit_breaker": breaker,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        for applied, base, effective, regime_value, score, rs, theme, mom, drawdown, ma20, adx_value, event_value, gap_value, ret20, vol20, drawdown_52w, slow_value, theme_on, breaker in zip(
            out["regime_leader_exception_applied"].fillna(False).astype(bool),
            base_multiplier,
            pd.to_numeric(out["regime_exposure_multiplier"], errors="coerce").fillna(1.0),
            regime_score,
            final_score,
            rs_score,
            theme_score,
            mom_return,
            drawdown_from_high,
            above_ma20,
            adx,
            event_risk,
            gap_risk,
            benchmark_ret_20d,
            realized_vol_20d,
            benchmark_drawdown_52w,
            price_vs_slow,
            theme_active,
            out["regime_leader_exception_circuit_breaker"],
        )
    ]
    if bool(apply_mask.any()):
        LOGGER.info(
            "regime_leader_exception_applied",
            extra={
                "event": "regime_leader_exception_applied",
                "rows": int(apply_mask.sum()),
                "avg_base_multiplier": float(base_multiplier.loc[apply_mask].mean()),
                "avg_effective_multiplier": float(pd.to_numeric(out.loc[apply_mask, "regime_exposure_multiplier"], errors="coerce").mean()),
                "restore_multiplier": restore_multiplier,
                "min_regime_score": float(config.regime_leader_exception_min_regime_score),
                "min_final_score": float(config.regime_leader_exception_min_final_score),
                "min_relative_strength_score": float(config.regime_leader_exception_min_relative_strength_score),
            },
        )
    return out


def detect_overbought_risk(prices: pd.DataFrame, benchmark: str, config: RegimeConfig) -> pd.DataFrame:
    """Detect market-wide overbought/cash states from point-in-time daily data."""

    if prices.empty:
        return pd.DataFrame()
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    for column in ("rsi_14", "zscore_20", "distance_to_high_252"):
        if column not in data.columns:
            data[column] = pd.NA
    bench = data[data["symbol"] == benchmark.upper()].sort_values("date").copy()
    if bench.empty:
        return pd.DataFrame()
    if "rsi_14" not in bench.columns:
        delta = bench["adj_close"].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
        bench["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, pd.NA)))
    if "ma_20" not in bench.columns:
        bench["ma_20"] = bench["adj_close"].rolling(20, min_periods=5).mean()
    if "ma_50" not in bench.columns:
        bench["ma_50"] = bench["adj_close"].rolling(50, min_periods=10).mean()
    bench["ret_5d"] = bench["adj_close"].pct_change(5)
    bench["ret_20d"] = bench["adj_close"].pct_change(20)
    bench["dist_ma20"] = bench["adj_close"] / bench["ma_20"] - 1.0
    bench["dist_ma50"] = bench["adj_close"] / bench["ma_50"] - 1.0
    daily = data.groupby("date").agg(
        pct_rsi_over_70=("rsi_14", lambda s: pd.to_numeric(s, errors="coerce").gt(70).mean() if "rsi_14" in data.columns else 0.0),
        pct_z_over_2=("zscore_20", lambda s: pd.to_numeric(s, errors="coerce").gt(2.0).mean() if "zscore_20" in data.columns else 0.0),
        pct_near_high=("distance_to_high_252", lambda s: pd.to_numeric(s, errors="coerce").gt(-0.03).mean() if "distance_to_high_252" in data.columns else 0.0),
    ).reset_index()
    out = bench[["date", "rsi_14", "ret_5d", "ret_20d", "dist_ma20", "dist_ma50"]].merge(daily, on="date", how="left")
    score = (
        out["rsi_14"].fillna(50).ge(75).astype(float) * 25
        + out["rsi_14"].fillna(50).between(68, 75, inclusive="left").astype(float) * 12
        + out["ret_5d"].fillna(0).ge(0.05).astype(float) * 12
        + out["ret_20d"].fillna(0).ge(0.12).astype(float) * 15
        + out["dist_ma20"].fillna(0).ge(0.06).astype(float) * 12
        + out["dist_ma50"].fillna(0).ge(0.12).astype(float) * 10
        + out["pct_rsi_over_70"].fillna(0).ge(0.35).astype(float) * 12
        + out["pct_z_over_2"].fillna(0).ge(0.20).astype(float) * 8
        + out["pct_near_high"].fillna(0).ge(0.45).astype(float) * 6
    )
    out["overbought_score"] = score.clip(0, 100)
    out["overbought_status"] = "normal"
    out.loc[out["overbought_score"] >= float(config.overbought_warning_score), "overbought_status"] = "warning"
    out.loc[out["overbought_score"] >= float(config.overbought_reduce_score), "overbought_status"] = "reduce"
    out.loc[out["overbought_score"] >= float(config.overbought_cash_score), "overbought_status"] = "cash"
    out["cash_signal"] = out["overbought_status"].eq("cash")
    out["overbought_multiplier"] = 1.0
    out.loc[out["overbought_status"].eq("reduce"), "overbought_multiplier"] = float(config.overbought_reduce_multiplier)
    out.loc[out["overbought_status"].eq("cash"), "overbought_multiplier"] = float(config.overbought_cash_multiplier)
    return out[
        [
            "date",
            "overbought_score",
            "overbought_status",
            "cash_signal",
            "overbought_multiplier",
            "rsi_14",
            "ret_5d",
            "ret_20d",
            "dist_ma20",
            "dist_ma50",
            "pct_rsi_over_70",
            "pct_z_over_2",
            "pct_near_high",
        ]
    ]
