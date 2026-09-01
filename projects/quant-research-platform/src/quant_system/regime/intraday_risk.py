"""Intraday risk overlays.

The MVP uses daily OHLCV as a point-in-time proxy: all scores are known only
after the close and are applied to the next executable rebalance. A real
minute-bar provider can replace the proxy inputs without changing the target
overlay contract.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from quant_system.config import RegimeConfig

LOGGER = logging.getLogger(__name__)


def detect_intraday_proxy_risk(prices: pd.DataFrame, benchmark: str, config: RegimeConfig) -> pd.DataFrame:
    """Score same-day gap, intraday loss, breadth damage, and volume crowding."""

    if prices.empty:
        return pd.DataFrame()
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values(["symbol", "date"])
    data["prev_close"] = data.groupby("symbol")["adj_close"].shift(1)
    data["intraday_return"] = data["close"] / data["open"].replace(0, np.nan) - 1.0
    data["gap_return"] = data["open"] / data["prev_close"].replace(0, np.nan) - 1.0
    data["low_from_open"] = data["low"] / data["open"].replace(0, np.nan) - 1.0
    data["high_low_range"] = data["high"] / data["low"].replace(0, np.nan) - 1.0
    data["avg_volume_20d"] = data.groupby("symbol")["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    data["volume_expansion"] = data["volume"] / data["avg_volume_20d"].replace(0, np.nan)

    bench = data[data["symbol"] == benchmark.upper()].copy()
    if bench.empty:
        return pd.DataFrame()
    bench = bench[["date", "intraday_return", "gap_return", "low_from_open", "high_low_range", "volume_expansion"]].rename(
        columns={
            "intraday_return": "benchmark_intraday_return",
            "gap_return": "benchmark_gap_return",
            "low_from_open": "benchmark_low_from_open",
            "high_low_range": "benchmark_high_low_range",
            "volume_expansion": "benchmark_volume_expansion",
        }
    )
    daily = (
        data.groupby("date")
        .agg(
            pct_down_from_open=("intraday_return", lambda s: pd.to_numeric(s, errors="coerce").lt(0).mean()),
            pct_large_intraday_loss=(
                "intraday_return",
                lambda s: pd.to_numeric(s, errors="coerce").le(float(config.intraday_large_loss_threshold)).mean(),
            ),
            pct_gap_down=("gap_return", lambda s: pd.to_numeric(s, errors="coerce").le(float(config.intraday_benchmark_gap_reduce)).mean()),
            pct_large_range=("high_low_range", lambda s: pd.to_numeric(s, errors="coerce").ge(0.08).mean()),
            pct_volume_expansion=(
                "volume_expansion",
                lambda s: pd.to_numeric(s, errors="coerce").ge(float(config.intraday_volume_expansion_threshold)).mean(),
            ),
            median_intraday_return=("intraday_return", "median"),
            median_volume_expansion=("volume_expansion", "median"),
        )
        .reset_index()
    )
    out = bench.merge(daily, on="date", how="left")
    warn_drop = float(config.intraday_benchmark_drop_warn)
    reduce_drop = float(config.intraday_benchmark_drop_reduce)
    gap_reduce = float(config.intraday_benchmark_gap_reduce)
    breadth_down = float(config.intraday_breadth_down_threshold)
    vol_expansion = float(config.intraday_volume_expansion_threshold)
    score = (
        out["benchmark_intraday_return"].fillna(0).le(reduce_drop).astype(float) * 25
        + out["benchmark_intraday_return"].fillna(0).between(reduce_drop, warn_drop, inclusive="right").astype(float) * 12
        + out["benchmark_gap_return"].fillna(0).le(gap_reduce).astype(float) * 15
        + out["benchmark_low_from_open"].fillna(0).le(reduce_drop * 1.25).astype(float) * 12
        + out["pct_down_from_open"].fillna(0).ge(breadth_down).astype(float) * 14
        + out["pct_down_from_open"].fillna(0).ge(0.75).astype(float) * 10
        + out["pct_large_intraday_loss"].fillna(0).ge(0.20).astype(float) * 12
        + out["median_intraday_return"].fillna(0).le(-0.025).astype(float) * 12
        + out["pct_volume_expansion"].fillna(0).ge(0.35).astype(float) * 10
        + out["median_volume_expansion"].fillna(1).ge(vol_expansion).astype(float) * 8
        + (
            out["pct_down_from_open"].fillna(0).ge(0.75)
            & out["pct_large_intraday_loss"].fillna(0).ge(0.25)
        ).astype(float)
        * 18
        + (
            out["pct_down_from_open"].fillna(0).ge(0.70)
            & out["pct_large_intraday_loss"].fillna(0).ge(0.25)
            & out["pct_volume_expansion"].fillna(0).ge(0.30)
        ).astype(float)
        * 15
        + (
            out["pct_down_from_open"].fillna(0).ge(breadth_down)
            & out["pct_volume_expansion"].fillna(0).ge(0.30)
            & out["benchmark_intraday_return"].fillna(0).lt(0)
        ).astype(float)
        * 14
    )
    out["intraday_risk_score"] = score.clip(0, 100)
    out["intraday_risk_status"] = "normal"
    out.loc[out["intraday_risk_score"] >= float(config.intraday_warning_score), "intraday_risk_status"] = "warning"
    out.loc[out["intraday_risk_score"] >= float(config.intraday_reduce_score), "intraday_risk_status"] = "reduce"
    out.loc[out["intraday_risk_score"] >= float(config.intraday_cash_score), "intraday_risk_status"] = "cash"
    out["intraday_cash_signal"] = out["intraday_risk_status"].eq("cash")
    out["intraday_risk_multiplier"] = 1.0
    out.loc[out["intraday_risk_status"].eq("reduce"), "intraday_risk_multiplier"] = float(config.intraday_reduce_multiplier)
    out.loc[out["intraday_risk_status"].eq("cash"), "intraday_risk_multiplier"] = float(config.intraday_cash_multiplier)
    return out[
        [
            "date",
            "intraday_risk_score",
            "intraday_risk_status",
            "intraday_cash_signal",
            "intraday_risk_multiplier",
            "benchmark_intraday_return",
            "benchmark_gap_return",
            "benchmark_low_from_open",
            "benchmark_high_low_range",
            "benchmark_volume_expansion",
            "pct_down_from_open",
            "pct_large_intraday_loss",
            "pct_gap_down",
            "pct_large_range",
            "pct_volume_expansion",
            "median_intraday_return",
            "median_volume_expansion",
        ]
    ]


def apply_intraday_risk_overlay(
    targets: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: str,
    config: RegimeConfig,
) -> pd.DataFrame:
    """Scale target weights when the close-time intraday proxy flags crowding risk."""

    if targets.empty or not config.enabled or not config.intraday_risk_overlay:
        return targets
    risk = detect_intraday_proxy_risk(prices, benchmark, config)
    if risk.empty:
        return targets
    overlay = risk[["date", "intraday_risk_score", "intraday_risk_status", "intraday_cash_signal", "intraday_risk_multiplier"]].copy()
    overlay["date"] = pd.to_datetime(overlay["date"]).dt.normalize()
    out = targets.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(overlay, on="date", how="left")
    out["intraday_risk_multiplier"] = out["intraday_risk_multiplier"].fillna(1.0)
    out["intraday_leader_exception_applied"] = False
    out["intraday_leader_exception_circuit_breaker"] = "feature_off"
    out["intraday_leader_exception_log"] = ""
    out = _apply_intraday_leader_exception(out, config)
    out["target_weight"] = out["target_weight"].astype(float) * out["intraday_risk_multiplier"].astype(float)
    if "reason_for_entry" in out.columns:
        mask = out["intraday_risk_multiplier"].astype(float) < 1.0
        out.loc[mask, "reason_for_entry"] = (
            out.loc[mask, "reason_for_entry"].fillna("").astype(str)
            + " Intraday proxy risk reduced exposure after close-time gap/breadth/volume stress."
        ).str.strip()
        relief_mask = out["intraday_leader_exception_applied"].fillna(False).astype(bool)
        out.loc[relief_mask, "reason_for_entry"] = (
            out.loc[relief_mask, "reason_for_entry"].fillna("").astype(str)
            + " Strong-leader intraday exception preserved part of the next-session size under capped pullback and risk breakers."
        ).str.strip()
    return out


def _apply_intraday_leader_exception(targets: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Restore part of an intraday proxy reduce cut for strong, low-risk leaders.

    Feature flag: ``intraday_leader_exception_overlay`` defaults to false.
    Reasonable ranges:
    - ``intraday_leader_exception_restore_multiplier``: 0.75 to 0.95.
    - ``intraday_leader_exception_min_final_score``: 68 to 78.
    - ``intraday_leader_exception_min_relative_strength_score``: 84 to 96.
    - ``intraday_leader_exception_min_theme_score``: 58 to 75.
    - ``intraday_leader_exception_min_mom_return``: 0.05 to 0.25.
    - ``intraday_leader_exception_min_drawdown_from_high``: 0.02 to 0.06.
    - ``intraday_leader_exception_max_drawdown_from_high``: 0.08 to 0.18.
    - ``intraday_leader_exception_max_above_ma20_pct``: 0.02 to 0.08.
    - ``intraday_leader_exception_min_adx_circuit_breaker``: 16 to 28.

    Circuit breakers: the exception never fires when the intraday state is
    ``cash``, never raises exposure above 1.0x, and still requires bounded
    event risk, bounded overnight gap risk, a controlled pullback window, and
    a minimum trend-strength check.
    """

    out = targets.copy()
    if out.empty or not bool(config.intraday_leader_exception_overlay):
        return out

    base_multiplier = pd.to_numeric(out.get("intraday_risk_multiplier", 1.0), errors="coerce").fillna(1.0)
    status = out.get("intraday_risk_status", pd.Series("normal", index=out.index)).fillna("normal").astype(str)
    long_mask = pd.to_numeric(out.get("target_weight", 0.0), errors="coerce").fillna(0.0).gt(0.0)
    final_score = pd.to_numeric(out.get("final_score", 0.0), errors="coerce").fillna(0.0)
    rs_score = pd.to_numeric(out.get("relative_strength_score", 0.0), errors="coerce").fillna(0.0)
    theme_score = pd.to_numeric(out.get("theme_score", 50.0), errors="coerce").fillna(50.0)
    mom_return = pd.to_numeric(out.get("mom_return", 0.0), errors="coerce").fillna(0.0)
    drawdown_from_high = pd.to_numeric(out.get("distance_to_prior_high_252", out.get("distance_to_high_252", 0.0)), errors="coerce").fillna(0.0).abs()
    above_ma20 = pd.to_numeric(out.get("retest_distance_ma_pct", 0.0), errors="coerce").fillna(0.0).abs()
    adx = pd.to_numeric(out.get("trend_adx", 0.0), errors="coerce").fillna(0.0)
    event_risk = pd.to_numeric(out.get("event_risk_score", 0.0), errors="coerce").fillna(0.0)
    gap_risk = pd.to_numeric(out.get("overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    theme_active = out.get("theme_active", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    price_vs_slow = pd.to_numeric(out.get("trend_price_vs_slow_pct", 0.0), errors="coerce").fillna(0.0)

    reduce_state = status.eq("reduce")
    cash_state = status.eq("cash")
    strong_leader = (
        long_mask
        & reduce_state
        & final_score.ge(float(config.intraday_leader_exception_min_final_score))
        & rs_score.ge(float(config.intraday_leader_exception_min_relative_strength_score))
        & theme_score.ge(float(config.intraday_leader_exception_min_theme_score))
        & mom_return.ge(float(config.intraday_leader_exception_min_mom_return))
        & drawdown_from_high.ge(float(config.intraday_leader_exception_min_drawdown_from_high))
        & drawdown_from_high.le(float(config.intraday_leader_exception_max_drawdown_from_high))
        & above_ma20.le(float(config.intraday_leader_exception_max_above_ma20_pct))
        & adx.ge(float(config.intraday_leader_exception_min_adx_circuit_breaker))
        & event_risk.le(float(config.intraday_leader_exception_max_event_risk_score_circuit_breaker))
        & gap_risk.le(float(config.intraday_leader_exception_max_overnight_gap_risk_score_circuit_breaker))
    )
    if bool(config.intraday_leader_exception_require_theme_active):
        strong_leader &= theme_active
    if bool(config.intraday_leader_exception_require_price_above_ma50):
        strong_leader &= price_vs_slow.gt(0.0)

    out["intraday_leader_exception_circuit_breaker"] = np.where(
        cash_state,
        "cash_state",
        np.where(reduce_state, "not_strong_leader", "intraday_not_reduce"),
    )
    restore_multiplier = float(np.clip(config.intraday_leader_exception_restore_multiplier, 0.0, 1.0))
    effective_multiplier = np.clip(np.maximum(base_multiplier, restore_multiplier), 0.0, 1.0)
    apply_mask = strong_leader & base_multiplier.lt(restore_multiplier) & ~cash_state
    out.loc[apply_mask, "intraday_risk_multiplier"] = effective_multiplier.loc[apply_mask]
    out.loc[apply_mask, "intraday_leader_exception_applied"] = True
    out.loc[apply_mask, "intraday_leader_exception_circuit_breaker"] = "applied"
    out["intraday_leader_exception_log"] = [
        json.dumps(
            {
                "event": "intraday_leader_exception",
                "applied": bool(applied),
                "restore_multiplier": restore_multiplier,
                "base_multiplier": float(base),
                "effective_multiplier": float(eff),
                "status": state,
                "final_score": float(score),
                "relative_strength_score": float(rs),
                "theme_score": float(theme),
                "mom_return": float(mom),
                "drawdown_from_high": float(drawdown),
                "above_ma20_pct": float(ma20),
                "trend_adx": float(adx_value),
                "event_risk_score": float(event_value),
                "overnight_gap_risk_score": float(gap_value),
                "price_vs_slow_pct": float(slow_value),
                "theme_active": bool(theme_on),
                "circuit_breaker": breaker,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        for applied, base, eff, state, score, rs, theme, mom, drawdown, ma20, adx_value, event_value, gap_value, slow_value, theme_on, breaker in zip(
            out["intraday_leader_exception_applied"].fillna(False).astype(bool),
            base_multiplier,
            pd.to_numeric(out["intraday_risk_multiplier"], errors="coerce").fillna(1.0),
            status,
            final_score,
            rs_score,
            theme_score,
            mom_return,
            drawdown_from_high,
            above_ma20,
            adx,
            event_risk,
            gap_risk,
            price_vs_slow,
            theme_active,
            out["intraday_leader_exception_circuit_breaker"],
        )
    ]
    if bool(apply_mask.any()):
        LOGGER.info(
            "intraday_leader_exception_applied",
            extra={
                "event": "intraday_leader_exception_applied",
                "rows": int(apply_mask.sum()),
                "avg_base_multiplier": float(base_multiplier.loc[apply_mask].mean()),
                "avg_effective_multiplier": float(pd.to_numeric(out.loc[apply_mask, 'intraday_risk_multiplier'], errors='coerce').mean()),
                "restore_multiplier": restore_multiplier,
                "min_final_score": float(config.intraday_leader_exception_min_final_score),
                "min_relative_strength_score": float(config.intraday_leader_exception_min_relative_strength_score),
                "min_theme_score": float(config.intraday_leader_exception_min_theme_score),
            },
        )
    return out


def detect_minute_pretrade_risk(intraday_prices: pd.DataFrame, benchmark: str, config: RegimeConfig) -> pd.DataFrame:
    """Score true minute/5-minute stress known by the target-date close."""

    if intraday_prices.empty:
        return pd.DataFrame()
    summary_cols = {"first_open", "last_close", "intraday_volume", "minute_intraday_return", "minute_low_from_open"}
    if summary_cols.issubset(intraday_prices.columns):
        daily_symbol = intraday_prices.copy()
    else:
        daily_symbol = intraday_daily_symbol_summary(intraday_prices)
    return score_minute_pretrade_risk_from_daily_summary(daily_symbol, benchmark, config)


def intraday_daily_symbol_summary(intraday_prices: pd.DataFrame) -> pd.DataFrame:
    """Collapse minute bars into symbol/date summaries for pretrade risk scoring."""

    if intraday_prices.empty:
        return pd.DataFrame()
    data = intraday_prices.copy()
    data["date"] = pd.to_datetime(data["date"], format="mixed", errors="coerce").dt.normalize()
    data["datetime"] = pd.to_datetime(data["datetime"], format="mixed", errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.upper()
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    daily_symbol = (
        data.dropna(subset=["date", "datetime", "symbol", "open", "high", "low", "close"])
        .sort_values(["symbol", "date", "datetime"])
        .groupby(["date", "symbol"])
        .agg(
            first_open=("open", "first"),
            last_close=("close", "last"),
            low=("low", "min"),
            high=("high", "max"),
            intraday_volume=("volume", "sum"),
        )
        .reset_index()
    )
    daily_symbol["minute_intraday_return"] = daily_symbol["last_close"] / daily_symbol["first_open"].replace(0, np.nan) - 1.0
    daily_symbol["minute_low_from_open"] = daily_symbol["low"] / daily_symbol["first_open"].replace(0, np.nan) - 1.0
    daily_symbol["minute_range"] = daily_symbol["high"] / daily_symbol["low"].replace(0, np.nan) - 1.0
    daily_symbol = daily_symbol.sort_values(["symbol", "date"])
    daily_symbol["minute_volume_ratio_20d"] = daily_symbol.groupby("symbol")["intraday_volume"].transform(
        lambda s: s / s.rolling(20, min_periods=5).mean().replace(0, np.nan)
    )
    return daily_symbol


def score_minute_pretrade_risk_from_daily_summary(daily_symbol: pd.DataFrame, benchmark: str, config: RegimeConfig) -> pd.DataFrame:
    """Score pretrade risk from precomputed symbol/date intraday summaries."""

    if daily_symbol.empty:
        return pd.DataFrame()
    daily_symbol = daily_symbol.copy()
    daily_symbol["date"] = pd.to_datetime(daily_symbol["date"], format="mixed", errors="coerce").dt.normalize()
    daily_symbol["symbol"] = daily_symbol["symbol"].astype(str).str.upper()
    bench = daily_symbol[daily_symbol["symbol"] == benchmark.upper()].rename(
        columns={
            "minute_intraday_return": "minute_benchmark_return",
            "minute_low_from_open": "minute_benchmark_low_from_open",
            "minute_range": "minute_benchmark_range",
            "minute_volume_ratio_20d": "minute_benchmark_volume_ratio",
        }
    )
    if bench.empty:
        return pd.DataFrame()
    daily = (
        daily_symbol.groupby("date")
        .agg(
            minute_pct_down=("minute_intraday_return", lambda s: pd.to_numeric(s, errors="coerce").lt(0).mean()),
            minute_pct_large_loss=(
                "minute_intraday_return",
                lambda s: pd.to_numeric(s, errors="coerce").le(float(config.intraday_large_loss_threshold)).mean(),
            ),
            minute_pct_low_break=(
                "minute_low_from_open",
                lambda s: pd.to_numeric(s, errors="coerce").le(float(config.intraday_benchmark_drop_reduce)).mean(),
            ),
            minute_pct_volume_expansion=(
                "minute_volume_ratio_20d",
                lambda s: pd.to_numeric(s, errors="coerce").ge(float(config.intraday_volume_expansion_threshold)).mean(),
            ),
            minute_median_return=("minute_intraday_return", "median"),
        )
        .reset_index()
    )
    out = bench[
        [
            "date",
            "minute_benchmark_return",
            "minute_benchmark_low_from_open",
            "minute_benchmark_range",
            "minute_benchmark_volume_ratio",
        ]
    ].merge(daily, on="date", how="left")
    score = (
        out["minute_benchmark_return"].fillna(0).le(float(config.intraday_benchmark_drop_reduce)).astype(float) * 25
        + out["minute_benchmark_low_from_open"].fillna(0).le(float(config.intraday_benchmark_drop_reduce) * 1.25).astype(float) * 15
        + out["minute_pct_down"].fillna(0).ge(float(config.intraday_breadth_down_threshold)).astype(float) * 18
        + out["minute_pct_down"].fillna(0).ge(0.75).astype(float) * 12
        + out["minute_pct_large_loss"].fillna(0).ge(0.20).astype(float) * 18
        + out["minute_pct_low_break"].fillna(0).ge(0.25).astype(float) * 12
        + out["minute_pct_volume_expansion"].fillna(0).ge(0.30).astype(float) * 10
        + out["minute_median_return"].fillna(0).le(-0.025).astype(float) * 10
    )
    out["minute_pretrade_risk_score"] = score.clip(0, 100)
    out["minute_pretrade_risk_status"] = "normal"
    out.loc[out["minute_pretrade_risk_score"] >= float(config.minute_pretrade_reduce_score), "minute_pretrade_risk_status"] = "reduce"
    out.loc[out["minute_pretrade_risk_score"] >= float(config.minute_pretrade_cash_score), "minute_pretrade_risk_status"] = "cash"
    out["minute_pretrade_multiplier"] = 1.0
    out.loc[out["minute_pretrade_risk_status"].eq("reduce"), "minute_pretrade_multiplier"] = float(config.minute_pretrade_reduce_multiplier)
    out.loc[out["minute_pretrade_risk_status"].eq("cash"), "minute_pretrade_multiplier"] = float(config.minute_pretrade_cash_multiplier)
    return out


def apply_minute_pretrade_budget_to_targets(
    targets: pd.DataFrame,
    intraday_prices: pd.DataFrame,
    benchmark: str,
    config: RegimeConfig,
) -> pd.DataFrame:
    """Reduce target weights before next-session execution using true minute bars."""

    if targets.empty or not config.enabled or not config.minute_pretrade_budget_overlay:
        return targets
    risk = detect_minute_pretrade_risk(intraday_prices, benchmark, config)
    if risk.empty:
        return targets
    overlay = risk[["date", "minute_pretrade_risk_score", "minute_pretrade_risk_status", "minute_pretrade_multiplier"]].copy()
    overlay["date"] = pd.to_datetime(overlay["date"]).dt.normalize()
    out = targets.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(overlay, on="date", how="left")
    out["minute_pretrade_multiplier"] = out["minute_pretrade_multiplier"].fillna(1.0)
    out["target_weight"] = out["target_weight"].astype(float) * out["minute_pretrade_multiplier"].astype(float)
    if "reason_for_entry" in out.columns:
        mask = out["minute_pretrade_multiplier"].astype(float) < 1.0
        out.loc[mask, "reason_for_entry"] = (
            out.loc[mask, "reason_for_entry"].fillna("").astype(str)
            + " Minute-level pretrade budget reduced next-session exposure."
        ).str.strip()
    return out
