"""Audit same-theme rejected leaders before turning them into a trading rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.config import AppConfig
from quant_system.portfolio.construction import construct_target_weights, rebalance_dates


def build_leader_substitution_audit(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    config: AppConfig,
    *,
    horizon: int = 21,
    max_rejected_per_theme: int = 3,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    max_dates: int | None = None,
) -> pd.DataFrame:
    """Compare selected longs with stronger same-theme rejected candidates.

    This is intentionally read-only. It does not create a target portfolio; it
    asks whether the current ranking leaves future return on the table by
    excluding same-theme names with better relative strength and comparable
    event/gap risk.
    """

    signals = _prepare_signals(signals, config, start_date=start_date, end_date=end_date, max_dates=max_dates)
    prices = _prepare_forward_returns(prices, horizon)
    if signals.empty or prices.empty:
        return _empty_audit()

    targets = construct_target_weights(signals, config.portfolio)
    if targets.empty:
        return _empty_audit()
    selected = _selected_longs(targets)
    if selected.empty:
        return _empty_audit()

    scored = signals.merge(
        prices[["date", "symbol", f"forward_return_{horizon}d"]],
        on=["date", "symbol"],
        how="left",
    )
    selected = selected.merge(
        scored,
        on=["date", "symbol"],
        how="left",
        suffixes=("_target", ""),
    )
    rows: list[dict[str, object]] = []
    for (date, theme), selected_theme in selected.groupby(["date", "primary_theme"], dropna=False):
        day = scored[scored["date"].eq(date)]
        rejected = _same_theme_rejected_candidates(
            day,
            selected_symbols=set(selected_theme["symbol"].astype(str)),
            theme=str(theme),
            max_names=max_rejected_per_theme,
        )
        if rejected.empty:
            continue
        weakest_selected = selected_theme.sort_values(
            ["final_score", "relative_strength_score", "theme_score"],
            ascending=[True, True, True],
        ).head(max(1, min(len(selected_theme), max_rejected_per_theme)))
        for _, selected_row in weakest_selected.iterrows():
            candidate = _best_replacement_for(selected_row, rejected)
            if candidate is None:
                continue
            rows.append(_audit_row(date, theme, selected_row, candidate, horizon))
    if not rows:
        return _empty_audit(horizon=horizon)
    return pd.DataFrame(rows).sort_values(["date", "theme", "replacement_lift"], ascending=[True, True, False]).reset_index(drop=True)


def summarize_leader_substitution_audit(audit: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Summarize replacement lift by validation period."""

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


def save_leader_substitution_audit(audit: pd.DataFrame, summary: pd.DataFrame, out_dir: str | Path) -> Path:
    """Persist CSV and HTML artifacts for the audit."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "leader_substitution_audit.csv", index=False)
    summary.to_csv(out / "leader_substitution_summary.csv", index=False)
    (out / "leader_substitution_audit.html").write_text(_render_html(audit, summary), encoding="utf-8")
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
    defaults = {
        "signal": 0.0,
        "final_score": 50.0,
        "relative_strength_score": 50.0,
        "theme_score": 50.0,
        "event_risk_score": 0.0,
        "overnight_gap_risk_score": 0.0,
        "primary_theme": "unclassified",
    }
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
    for column in [
        "signal",
        "final_score",
        "relative_strength_score",
        "theme_score",
        "event_risk_score",
        "overnight_gap_risk_score",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(defaults[column])
    out["primary_theme"] = out["primary_theme"].fillna("unclassified").astype(str)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


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


def _selected_longs(targets: pd.DataFrame) -> pd.DataFrame:
    out = targets.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0)
    return out[out["target_weight"].gt(0)].copy()


def _same_theme_rejected_candidates(
    day: pd.DataFrame,
    *,
    selected_symbols: set[str],
    theme: str,
    max_names: int,
) -> pd.DataFrame:
    candidates = day[
        day["primary_theme"].astype(str).eq(str(theme))
        & ~day["symbol"].astype(str).isin(selected_symbols)
        & day["final_score"].gt(50.0)
        & day["relative_strength_score"].gt(50.0)
        & day["theme_score"].ge(50.0)
        & day[f"forward_return_{_horizon_from_frame(day)}d"].notna()
    ].copy()
    if candidates.empty:
        return candidates
    candidates["replacement_quality"] = (
        candidates["relative_strength_score"]
        + candidates["theme_score"] * 0.5
        + candidates["final_score"] * 0.25
        - candidates["event_risk_score"] * 0.35
        - candidates["overnight_gap_risk_score"] * 0.25
    )
    return candidates.sort_values("replacement_quality", ascending=False).head(max(1, int(max_names)))


def _best_replacement_for(selected: pd.Series, rejected: pd.DataFrame) -> pd.Series | None:
    if rejected.empty:
        return None
    eligible = rejected[
        rejected["relative_strength_score"].ge(float(selected.get("relative_strength_score", 50.0)))
        & rejected["theme_score"].ge(float(selected.get("theme_score", 50.0)))
        & rejected["event_risk_score"].le(float(selected.get("event_risk_score", 0.0)) + 5.0)
        & rejected["overnight_gap_risk_score"].le(float(selected.get("overnight_gap_risk_score", 0.0)) + 7.5)
    ].copy()
    if eligible.empty:
        return None
    return eligible.sort_values("replacement_quality", ascending=False).iloc[0]


def _audit_row(date: pd.Timestamp, theme: str, selected: pd.Series, rejected: pd.Series, horizon: int) -> dict[str, object]:
    forward_col = f"forward_return_{horizon}d"
    selected_forward = float(selected.get(forward_col, np.nan))
    rejected_forward = float(rejected.get(forward_col, np.nan))
    return {
        "date": pd.Timestamp(date),
        "theme": str(theme),
        "selected_symbol": str(selected["symbol"]),
        "rejected_symbol": str(rejected["symbol"]),
        "selected_weight": float(selected.get("target_weight", 0.0)),
        "selected_final_score": float(selected.get("final_score", np.nan)),
        "selected_relative_strength_score": float(selected.get("relative_strength_score", np.nan)),
        "selected_theme_score": float(selected.get("theme_score", np.nan)),
        "selected_event_risk_score": float(selected.get("event_risk_score", np.nan)),
        "selected_overnight_gap_risk_score": float(selected.get("overnight_gap_risk_score", np.nan)),
        "rejected_final_score": float(rejected.get("final_score", np.nan)),
        "rejected_relative_strength_score": float(rejected.get("relative_strength_score", np.nan)),
        "rejected_theme_score": float(rejected.get("theme_score", np.nan)),
        "rejected_event_risk_score": float(rejected.get("event_risk_score", np.nan)),
        "rejected_overnight_gap_risk_score": float(rejected.get("overnight_gap_risk_score", np.nan)),
        "selected_forward_return": selected_forward,
        "rejected_forward_return": rejected_forward,
        "replacement_lift": rejected_forward - selected_forward,
        "replacement_pass": bool(rejected_forward > selected_forward),
        "reason": (
            "Rejected same-theme candidate had stronger relative strength/theme context "
            "without materially higher event or overnight-gap risk."
        ),
    }


def _horizon_from_frame(frame: pd.DataFrame) -> int:
    for column in frame.columns:
        if column.startswith("forward_return_") and column.endswith("d"):
            return int(column.removeprefix("forward_return_").removesuffix("d"))
    raise ValueError("Expected a forward_return_{horizon}d column")


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
            "n_pairs": 0,
            "themes": 0,
            "mean_replacement_lift": 0.0,
            "median_replacement_lift": 0.0,
            "positive_lift_rate": 0.0,
            "mean_selected_forward_return": 0.0,
            "mean_rejected_forward_return": 0.0,
            "promotion_ready": False,
        }
    lift = pd.to_numeric(frame["replacement_lift"], errors="coerce").dropna()
    return {
        "period": period,
        "n_pairs": int(len(frame)),
        "themes": int(frame["theme"].nunique()),
        "mean_replacement_lift": float(lift.mean()) if not lift.empty else 0.0,
        "median_replacement_lift": float(lift.median()) if not lift.empty else 0.0,
        "positive_lift_rate": float((lift > 0).mean()) if not lift.empty else 0.0,
        "mean_selected_forward_return": float(pd.to_numeric(frame["selected_forward_return"], errors="coerce").mean()),
        "mean_rejected_forward_return": float(pd.to_numeric(frame["rejected_forward_return"], errors="coerce").mean()),
        "promotion_ready": bool((not lift.empty) and lift.mean() > 0 and (lift > 0).mean() >= 0.52 and len(frame) >= 30),
    }


def _empty_audit(horizon: int = 21) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "theme",
            "selected_symbol",
            "rejected_symbol",
            "selected_weight",
            "selected_final_score",
            "selected_relative_strength_score",
            "selected_theme_score",
            "selected_event_risk_score",
            "selected_overnight_gap_risk_score",
            "rejected_final_score",
            "rejected_relative_strength_score",
            "rejected_theme_score",
            "rejected_event_risk_score",
            "rejected_overnight_gap_risk_score",
            "selected_forward_return",
            "rejected_forward_return",
            "replacement_lift",
            "replacement_pass",
            "reason",
        ]
    )


def _render_html(audit: pd.DataFrame, summary: pd.DataFrame) -> str:
    top = audit.sort_values("replacement_lift", ascending=False).head(80) if not audit.empty else audit
    return (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>Leader Substitution Audit</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:7px;text-align:right}"
        "th:first-child,td:first-child,td:nth-child(2),td:nth-child(3),td:nth-child(4){text-align:left}"
        ".note{color:#5b6472;line-height:1.5}</style></head><body>"
        "<h1>同主题候选替换审计</h1>"
        "<p class='note'>这是只读研究报告：它比较已选多头与同主题、同日期但未进入目标持仓的强势候选，"
        "用于判断是否值得在下一轮实现可交易的 leader substitution 规则。</p>"
        "<h2>分阶段汇总</h2>"
        f"{summary.to_html(index=False, border=0) if not summary.empty else '<p>暂无汇总。</p>'}"
        "<h2>替换机会 Top</h2>"
        f"{top.to_html(index=False, border=0) if not top.empty else '<p>没有符合条件的替换样本。</p>'}"
        "</body></html>"
    )
