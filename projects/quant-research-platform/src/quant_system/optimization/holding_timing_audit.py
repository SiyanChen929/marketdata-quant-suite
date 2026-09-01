"""Audit whether the strategy exits or cuts leaders too early.

The audit is deliberately read-only. It compares actual target-weight
reductions with the symbol's subsequent return so the evolution loop can decide
whether a hold-buffer rule is worth testing before touching live logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.config import AppConfig
from quant_system.portfolio.construction import construct_target_weights, rebalance_dates


def build_holding_timing_audit(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    config: AppConfig,
    *,
    horizon: int = 21,
    min_weight_drop: float = 0.005,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    max_dates: int | None = None,
) -> pd.DataFrame:
    """Measure forward returns after long target-weight reductions.

    Positive ``hypothetical_hold_lift`` means the cut symbol rose after the
    reduction, so the strategy may have left return on the table by exiting too
    early. This is not a proposed PnL series; it is an evidence screen.
    """

    prepared_signals = _prepare_signals(signals, config, start_date=start_date, end_date=end_date, max_dates=max_dates)
    prepared_prices = _prepare_forward_returns(prices, horizon)
    if prepared_signals.empty or prepared_prices.empty:
        return _empty_audit(horizon=horizon)
    targets = construct_target_weights(prepared_signals, config.portfolio)
    if targets.empty:
        return _empty_audit(horizon=horizon)
    targets = _prepare_targets(targets)
    context = _context_by_symbol_date(prepared_signals)
    forward = prepared_prices.set_index(["date", "symbol"])[f"forward_return_{horizon}d"].to_dict()
    rows: list[dict[str, object]] = []
    active: dict[str, float] = {}
    active_rows: dict[str, pd.Series] = {}
    for date, day in targets.groupby("date", sort=True):
        day_rows = {str(row["symbol"]): row for _, row in day.iterrows()}
        day_weights = {symbol: float(row.get("target_weight", 0.0) or 0.0) for symbol, row in day_rows.items()}
        for symbol in sorted(set(active) | set(day_weights)):
            prev_weight = float(active.get(symbol, 0.0))
            new_weight = float(day_weights.get(symbol, 0.0))
            drop = prev_weight - new_weight
            if prev_weight > 0 and drop >= float(min_weight_drop):
                row_context = day_rows.get(symbol)
                if row_context is None:
                    row_context = context.get((pd.Timestamp(date), symbol), active_rows.get(symbol, pd.Series(dtype=object)))
                rows.append(
                    _audit_row(
                        date=pd.Timestamp(date),
                        symbol=symbol,
                        prev_weight=prev_weight,
                        new_weight=new_weight,
                        context=row_context,
                        forward_return=forward.get((pd.Timestamp(date), symbol), np.nan),
                        horizon=horizon,
                    )
                )
            if abs(new_weight) > 0:
                active[symbol] = new_weight
                active_rows[symbol] = day_rows.get(symbol, context.get((pd.Timestamp(date), symbol), active_rows.get(symbol, pd.Series(dtype=object))))
            else:
                active.pop(symbol, None)
                active_rows.pop(symbol, None)
    if not rows:
        return _empty_audit(horizon=horizon)
    return pd.DataFrame(rows).sort_values(["date", "hypothetical_hold_lift"], ascending=[True, False]).reset_index(drop=True)


def summarize_holding_timing_audit(audit: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Summarize early-exit evidence across train/validation/forward periods."""

    rows = []
    for period, start, end in _periods(config):
        frame = audit.copy()
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            if start is not None:
                frame = frame[frame["date"].ge(start)]
            if end is not None:
                frame = frame[frame["date"].le(end)]
        rows.append(_summarize_period(frame, period))
    return pd.DataFrame(rows)


def save_holding_timing_audit(audit: pd.DataFrame, summary: pd.DataFrame, out_dir: str | Path) -> Path:
    """Persist CSV and HTML artifacts for the audit."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "holding_timing_audit.csv", index=False)
    summary.to_csv(out / "holding_timing_summary.csv", index=False)
    (out / "holding_timing_audit.html").write_text(_render_html(audit, summary), encoding="utf-8")
    return out


def _prepare_signals(
    signals: pd.DataFrame,
    config: AppConfig,
    *,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
    max_dates: int | None,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    dates = rebalance_dates(out["date"], config.portfolio.rebalance)
    if dates:
        out = out[out["date"].isin(dates)].copy()
    if start_date is not None:
        out = out[out["date"].ge(pd.Timestamp(start_date))]
    if end_date is not None:
        out = out[out["date"].le(pd.Timestamp(end_date))]
    if max_dates is not None and max_dates > 0 and not out.empty:
        keep_dates = sorted(out["date"].dropna().unique())[: int(max_dates)]
        out = out[out["date"].isin(keep_dates)].copy()
    defaults: dict[str, object] = {
        "signal": 0.0,
        "final_score": 50.0,
        "technical_score": 50.0,
        "relative_strength_score": 50.0,
        "theme_score": 50.0,
        "fundamental_score": 50.0,
        "event_risk_score": 0.0,
        "overnight_gap_risk_score": 0.0,
        "minute_pretrade_risk_score": 0.0,
        "mom_return": 0.0,
        "distance_to_high_252": np.nan,
        "retest_distance_ma_pct": np.nan,
        "vol_20d": np.nan,
        "benchmark_risk_on": True,
        "theme_active": False,
        "primary_theme": "unclassified",
        "reason_for_entry": "",
        "reason_for_exit": "",
    }
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
    numeric_columns = [
        "signal",
        "final_score",
        "technical_score",
        "relative_strength_score",
        "theme_score",
        "fundamental_score",
        "event_risk_score",
        "overnight_gap_risk_score",
        "minute_pretrade_risk_score",
        "mom_return",
        "distance_to_high_252",
        "retest_distance_ma_pct",
        "vol_20d",
    ]
    for column in numeric_columns:
        fallback = defaults[column]
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(fallback if np.isfinite(fallback) else np.nan)
    for column in ["benchmark_risk_on", "theme_active"]:
        out[column] = out[column].where(out[column].notna(), False).astype(bool)
    for column in ["primary_theme", "reason_for_entry", "reason_for_exit"]:
        out[column] = out[column].fillna("").astype(str)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _prepare_targets(targets: pd.DataFrame) -> pd.DataFrame:
    out = targets.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0)
    return out.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last").reset_index(drop=True)


def _prepare_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    price_col = "adj_close" if "adj_close" in out.columns else "close"
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.sort_values(["symbol", "date"])
    future = out.groupby("symbol")[price_col].shift(-int(horizon))
    out[f"forward_return_{horizon}d"] = future / out[price_col] - 1.0
    return out[["date", "symbol", f"forward_return_{horizon}d"]]


def _context_by_symbol_date(signals: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], pd.Series]:
    return {
        (pd.Timestamp(row["date"]), str(row["symbol"])): row
        for _, row in signals.drop_duplicates(["date", "symbol"], keep="last").iterrows()
    }


def _audit_row(
    *,
    date: pd.Timestamp,
    symbol: str,
    prev_weight: float,
    new_weight: float,
    context: pd.Series,
    forward_return: float,
    horizon: int,
) -> dict[str, object]:
    weight_drop = max(0.0, float(prev_weight) - float(new_weight))
    forward = float(forward_return) if pd.notna(forward_return) else np.nan
    exit_type = "full_exit" if float(new_weight) <= 0 else "partial_cut"
    return {
        "date": pd.Timestamp(date),
        "symbol": symbol,
        "exit_type": exit_type,
        "prev_weight": float(prev_weight),
        "new_weight": float(new_weight),
        "weight_drop": weight_drop,
        "forward_return": forward,
        "hypothetical_hold_lift": weight_drop * forward if np.isfinite(forward) else np.nan,
        "horizon_days": int(horizon),
        "final_score": _safe_float(context.get("final_score"), np.nan),
        "technical_score": _safe_float(context.get("technical_score"), np.nan),
        "relative_strength_score": _safe_float(context.get("relative_strength_score"), np.nan),
        "theme_score": _safe_float(context.get("theme_score"), np.nan),
        "fundamental_score": _safe_float(context.get("fundamental_score"), np.nan),
        "event_risk_score": _safe_float(context.get("event_risk_score"), np.nan),
        "overnight_gap_risk_score": _safe_float(context.get("overnight_gap_risk_score"), np.nan),
        "minute_pretrade_risk_score": _safe_float(context.get("minute_pretrade_risk_score"), np.nan),
        "mom_return": _safe_float(context.get("mom_return"), np.nan),
        "distance_to_high_252": _safe_float(context.get("distance_to_high_252"), np.nan),
        "retest_distance_ma_pct": _safe_float(context.get("retest_distance_ma_pct"), np.nan),
        "vol_20d": _safe_float(context.get("vol_20d"), np.nan),
        "benchmark_risk_on": bool(context.get("benchmark_risk_on", False)),
        "theme_active": bool(context.get("theme_active", False)),
        "primary_theme": str(context.get("primary_theme", "unclassified") or "unclassified"),
        "reason_for_entry": str(context.get("reason_for_entry", "") or ""),
        "reason_for_exit": str(context.get("reason_for_exit", "") or ""),
        "early_exit_pass": bool(np.isfinite(forward) and forward > 0),
    }


def _periods(config: AppConfig) -> list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]:
    regime = config.regime
    return [
        ("full", None, None),
        ("train", pd.Timestamp(regime.train_start), pd.Timestamp(regime.train_end)),
        ("validation", pd.Timestamp(regime.validation_start), pd.Timestamp(regime.validation_end)),
        ("forward_current", pd.Timestamp(regime.forward_start), None),
    ]


def _summarize_period(frame: pd.DataFrame, period: str) -> dict[str, object]:
    if frame.empty:
        return {
            "period": period,
            "n_events": 0,
            "full_exits": 0,
            "partial_cuts": 0,
            "symbols": 0,
            "themes": 0,
            "mean_forward_return": 0.0,
            "positive_forward_rate": 0.0,
            "mean_hypothetical_hold_lift": 0.0,
            "sum_hypothetical_hold_lift": 0.0,
            "promotion_ready": False,
        }
    forward = pd.to_numeric(frame["forward_return"], errors="coerce").dropna()
    lift = pd.to_numeric(frame["hypothetical_hold_lift"], errors="coerce").dropna()
    positive_rate = float((forward > 0).mean()) if not forward.empty else 0.0
    mean_lift = float(lift.mean()) if not lift.empty else 0.0
    return {
        "period": period,
        "n_events": int(len(frame)),
        "full_exits": int(frame["exit_type"].eq("full_exit").sum()),
        "partial_cuts": int(frame["exit_type"].eq("partial_cut").sum()),
        "symbols": int(frame["symbol"].nunique()),
        "themes": int(frame["primary_theme"].nunique()),
        "mean_forward_return": float(forward.mean()) if not forward.empty else 0.0,
        "positive_forward_rate": positive_rate,
        "mean_hypothetical_hold_lift": mean_lift,
        "sum_hypothetical_hold_lift": float(lift.sum()) if not lift.empty else 0.0,
        "promotion_ready": bool(len(frame) >= 30 and mean_lift > 0 and positive_rate >= 0.52),
    }


def _empty_audit(horizon: int = 21) -> pd.DataFrame:
    columns = [
        "date",
        "symbol",
        "exit_type",
        "prev_weight",
        "new_weight",
        "weight_drop",
        "forward_return",
        "hypothetical_hold_lift",
        "horizon_days",
        "final_score",
        "technical_score",
        "relative_strength_score",
        "theme_score",
        "fundamental_score",
        "event_risk_score",
        "overnight_gap_risk_score",
        "minute_pretrade_risk_score",
        "mom_return",
        "distance_to_high_252",
        "retest_distance_ma_pct",
        "vol_20d",
        "benchmark_risk_on",
        "theme_active",
        "primary_theme",
        "reason_for_entry",
        "reason_for_exit",
        "early_exit_pass",
    ]
    return pd.DataFrame(columns=columns).assign(horizon_days=int(horizon)).iloc[0:0]


def _render_html(audit: pd.DataFrame, summary: pd.DataFrame) -> str:
    top = audit.sort_values("hypothetical_hold_lift", ascending=False).head(100) if not audit.empty else audit
    by_theme = (
        audit.groupby("primary_theme", dropna=False)
        .agg(
            n_events=("symbol", "size"),
            mean_forward_return=("forward_return", "mean"),
            positive_forward_rate=("early_exit_pass", "mean"),
            sum_hypothetical_hold_lift=("hypothetical_hold_lift", "sum"),
        )
        .sort_values("sum_hypothetical_hold_lift", ascending=False)
        .reset_index()
        if not audit.empty
        else pd.DataFrame()
    )
    return (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>Holding Timing Audit</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:7px;text-align:right}"
        "th:first-child,td:first-child,td:nth-child(2),td:nth-child(3),td:nth-child(24){text-align:left}"
        ".note{color:#5b6472;line-height:1.5}</style></head><body>"
        "<h1>持有/减仓时机审计</h1>"
        "<p class='note'>这是只读报告：它统计系统降仓或清仓多头后，标的在后续窗口是否继续上涨。"
        "如果验证期样本显示正向 lift，下一轮才考虑做成 feature-flagged 持仓延迟规则。</p>"
        "<h2>分阶段汇总</h2>"
        f"{summary.to_html(index=False, border=0) if not summary.empty else '<p>暂无汇总。</p>'}"
        "<h2>按主题汇总</h2>"
        f"{by_theme.to_html(index=False, border=0) if not by_theme.empty else '<p>暂无主题样本。</p>'}"
        "<h2>疑似过早卖出 Top</h2>"
        f"{top.to_html(index=False, border=0) if not top.empty else '<p>没有符合条件的降仓样本。</p>'}"
        "</body></html>"
    )


def _safe_float(value: object, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if np.isfinite(parsed) else fallback
