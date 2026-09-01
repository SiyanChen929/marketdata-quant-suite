"""Overnight gap risk model."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_system.config import RegimeConfig

LOGGER = logging.getLogger(__name__)


def detect_overnight_gap_risk(prices: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Estimate symbol-level gap risk from historical open-to-prior-close gaps.

    For a target generated after close on date t, the model can use all gaps up
    to and including date t. It never uses t+1 open.
    """

    if prices.empty:
        return pd.DataFrame()
    data = prices.sort_values(["symbol", "date"]).copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["prev_close"] = data.groupby("symbol")["adj_close"].shift(1)
    data["daily_return"] = data.groupby("symbol")["adj_close"].pct_change()
    data["overnight_gap_return"] = data["open"] / data["prev_close"].replace(0, np.nan) - 1.0
    grouped = data.groupby("symbol", group_keys=False)
    data["realized_vol_63d"] = grouped["daily_return"].transform(lambda s: s.rolling(63, min_periods=20).std() * np.sqrt(252))
    data["overnight_gap_vol_20d"] = grouped["overnight_gap_return"].transform(lambda s: s.rolling(20, min_periods=5).std())
    data["overnight_gap_abs_p95_252d"] = grouped["overnight_gap_return"].transform(
        lambda s: s.abs().rolling(252, min_periods=40).quantile(0.95)
    )
    data["large_gap_count_20d"] = grouped["overnight_gap_return"].transform(
        lambda s: s.abs().gt(float(config.overnight_gap_tail_reduce)).rolling(20, min_periods=5).sum()
    )
    vol = data["overnight_gap_vol_20d"].fillna(0.0)
    tail = data["overnight_gap_abs_p95_252d"].fillna(0.0)
    large_count = data["large_gap_count_20d"].fillna(0.0)
    score = (
        vol.ge(float(config.overnight_gap_vol_warn)).astype(float) * 20
        + vol.ge(float(config.overnight_gap_vol_reduce)).astype(float) * 30
        + tail.ge(float(config.overnight_gap_tail_reduce)).astype(float) * 30
        + large_count.ge(2).astype(float) * 15
        + large_count.ge(4).astype(float) * 20
    )
    data["overnight_gap_risk_score"] = score.clip(0, 100)
    data["overnight_gap_risk_status"] = "normal"
    data.loc[data["overnight_gap_risk_score"] >= float(config.overnight_gap_reduce_score), "overnight_gap_risk_status"] = "reduce"
    data.loc[data["overnight_gap_risk_score"] >= float(config.overnight_gap_cash_score), "overnight_gap_risk_status"] = "cash"
    data["overnight_gap_multiplier"] = 1.0
    data.loc[data["overnight_gap_risk_status"].eq("reduce"), "overnight_gap_multiplier"] = float(config.overnight_gap_reduce_multiplier)
    data.loc[data["overnight_gap_risk_status"].eq("cash"), "overnight_gap_multiplier"] = float(config.overnight_gap_cash_multiplier)
    data["beta_to_benchmark_63d"] = np.nan
    benchmark_symbol = str(config.gap_beta_benchmark_symbol or "").upper()
    if benchmark_symbol:
        benchmark = data.loc[data["symbol"].astype(str).str.upper().eq(benchmark_symbol), ["date", "daily_return"]].rename(
            columns={"daily_return": "benchmark_return"}
        )
        if not benchmark.empty:
            data = data.merge(benchmark, on="date", how="left")

            def _rolling_beta(frame: pd.DataFrame) -> pd.Series:
                bench_var = frame["benchmark_return"].rolling(63, min_periods=20).var()
                covar = frame["daily_return"].rolling(63, min_periods=20).cov(frame["benchmark_return"])
                return covar / bench_var.replace(0, np.nan)

            beta_input = data[["symbol", "daily_return", "benchmark_return"]]
            data["beta_to_benchmark_63d"] = beta_input.groupby("symbol", group_keys=False)[
                ["daily_return", "benchmark_return"]
            ].apply(_rolling_beta)
    return data[
        [
            "date",
            "symbol",
            "overnight_gap_return",
            "overnight_gap_vol_20d",
            "overnight_gap_abs_p95_252d",
            "large_gap_count_20d",
            "realized_vol_63d",
            "beta_to_benchmark_63d",
            "overnight_gap_risk_score",
            "overnight_gap_risk_status",
            "overnight_gap_multiplier",
        ]
    ]


def apply_overnight_gap_risk_to_targets(targets: pd.DataFrame, prices: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Apply symbol-level gap risk budget to target weights before execution."""

    if targets.empty or not config.enabled or not config.overnight_gap_risk_overlay:
        return targets
    risk = detect_overnight_gap_risk(prices, config)
    if risk.empty:
        return targets
    out = targets.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(risk, on=["date", "symbol"], how="left")
    out["overnight_gap_multiplier"] = out["overnight_gap_multiplier"].fillna(1.0)
    out["target_weight"] = out["target_weight"].astype(float) * out["overnight_gap_multiplier"].astype(float)
    if "reason_for_entry" in out.columns:
        mask = out["overnight_gap_multiplier"].astype(float) < 1.0
        out.loc[mask, "reason_for_entry"] = (
            out.loc[mask, "reason_for_entry"].fillna("").astype(str)
            + " Overnight gap budget reduced exposure using historical gap volatility/tail risk."
        ).str.strip()
    out = apply_gap_beta_guard_to_targets(out, config)
    return out


def apply_gap_beta_guard_to_targets(targets: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Reduce long exposure when gap risk coincides with high beta/volatility.

    Feature flag: ``regime.gap_beta_guard_overlay`` defaults to ``false``.
    Reasonable ranges:
    - ``gap_beta_reduce_multiplier``: 0.50 to 0.90.
    - ``gap_beta_min_multiplier``: 0.25 to 0.75.
    - ``gap_beta_max_position_cut``: 0.10 to 0.75.
    - ``gap_beta_guard_require_benchmark_risk_off``: use ``true`` for
      offensive trend systems that should keep upside beta during risk-on tape.

    Circuit breaker: the guard never cuts an individual target by more than
    ``gap_beta_max_position_cut`` and never increases an existing target.
    """

    if targets.empty:
        return targets
    out = targets.copy()
    if not bool(config.gap_beta_guard_overlay):
        if "gap_beta_guard_multiplier" not in out.columns:
            out["gap_beta_guard_multiplier"] = 1.0
        return out

    score = pd.to_numeric(out.get("overnight_gap_risk_score", 0.0), errors="coerce").fillna(0.0)
    gap_vol = pd.to_numeric(out.get("overnight_gap_vol_20d", 0.0), errors="coerce").fillna(0.0)
    realized_vol = pd.to_numeric(out.get("realized_vol_63d", 0.0), errors="coerce").fillna(0.0)
    beta = pd.to_numeric(out.get("beta_to_benchmark_63d", 0.0), errors="coerce").abs().fillna(0.0)
    target = pd.to_numeric(out.get("target_weight", 0.0), errors="coerce").fillna(0.0)
    if bool(config.gap_beta_guard_require_benchmark_risk_off):
        benchmark_risk_on = out.get("benchmark_risk_on")
        if benchmark_risk_on is None:
            benchmark_filter = pd.Series(False, index=out.index)
        else:
            benchmark_filter = ~benchmark_risk_on.fillna(True).astype(bool)
    else:
        benchmark_filter = pd.Series(True, index=out.index)

    high_gap_risk = score.ge(float(config.gap_beta_min_gap_score)) | gap_vol.ge(float(config.gap_beta_min_gap_vol_20d))
    high_beta_or_vol = realized_vol.ge(float(config.gap_beta_min_realized_vol_63d)) | beta.ge(float(config.gap_beta_min_beta_63d))
    mask = target.gt(0.0) & high_gap_risk & high_beta_or_vol & benchmark_filter

    max_cut = float(np.clip(config.gap_beta_max_position_cut, 0.0, 0.95))
    circuit_floor = 1.0 - max_cut
    multiplier = max(float(config.gap_beta_reduce_multiplier), float(config.gap_beta_min_multiplier), circuit_floor)
    multiplier = float(np.clip(multiplier, 0.0, 1.0))
    out["gap_beta_guard_multiplier"] = 1.0
    out.loc[mask, "gap_beta_guard_multiplier"] = multiplier
    out["target_weight"] = target * out["gap_beta_guard_multiplier"].astype(float)
    out["gap_beta_guard_reason"] = ""
    out.loc[mask, "gap_beta_guard_reason"] = (
        "gap_beta_guard: overnight gap risk plus elevated beta/realized volatility"
        + (" during benchmark risk-off; " if bool(config.gap_beta_guard_require_benchmark_risk_off) else "; ")
        + f"multiplier={multiplier:.2f}"
    )
    if "reason_for_entry" in out.columns and mask.any():
        out.loc[mask, "reason_for_entry"] = (
            out.loc[mask, "reason_for_entry"].fillna("").astype(str) + " " + out.loc[mask, "gap_beta_guard_reason"].astype(str)
        ).str.strip()
    if mask.any():
        LOGGER.info(
            "gap_beta_guard_applied",
            extra={
                "event": "gap_beta_guard_applied",
                "rows": int(mask.sum()),
                "avg_multiplier": float(out.loc[mask, "gap_beta_guard_multiplier"].mean()),
                "max_position_cut": max_cut,
                "min_gap_score": float(config.gap_beta_min_gap_score),
                "min_beta_63d": float(config.gap_beta_min_beta_63d),
                "min_realized_vol_63d": float(config.gap_beta_min_realized_vol_63d),
                "require_benchmark_risk_off": bool(config.gap_beta_guard_require_benchmark_risk_off),
            },
        )
    return out
