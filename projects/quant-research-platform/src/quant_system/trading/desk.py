"""Trading-desk style run artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_trading_desk_tables(
    manual_blotter: pd.DataFrame,
    risk_overlay: pd.DataFrame,
    theme_divergence: pd.DataFrame,
    options_overlay: pd.DataFrame,
    data_quality_detail: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build pre-market plan, risk state, drift, crowding, and override tables."""

    quality = data_quality_detail if data_quality_detail is not None else pd.DataFrame()
    return {
        "premarket_plan": _premarket_plan(manual_blotter, quality),
        "risk_state": _risk_state(manual_blotter, risk_overlay, options_overlay, quality),
        "position_drift": _position_drift(manual_blotter),
        "crowding_watch": _crowding_watch(theme_divergence),
        "options_risk": _options_risk(options_overlay),
        "manual_override_log_template": manual_override_template(),
    }


def write_trading_desk_artifacts(tables: dict[str, pd.DataFrame], out: str | Path) -> None:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(target / f"{name}.csv", index=False)


def _premarket_plan(blotter: pd.DataFrame, data_quality: pd.DataFrame | None = None) -> pd.DataFrame:
    if blotter.empty:
        return pd.DataFrame()
    data = blotter.copy()
    if data_quality is not None and not data_quality.empty and "symbol" in data_quality:
        quality_cols = [
            col
            for col in ["symbol", "status", "blocked_from_trading", "block_reason", "broker_close_diff_pct", "earnings_calendar_status"]
            if col in data_quality.columns
        ]
        quality = data_quality[quality_cols].copy()
        quality["symbol"] = quality["symbol"].astype(str).str.upper()
        data["symbol"] = data["symbol"].astype(str).str.upper()
        data = data.merge(quality, on="symbol", how="left", suffixes=("", "_data_quality"))
    if "blocked_from_trading" not in data:
        data["blocked_from_trading"] = False
    data["blocked_from_trading"] = data["blocked_from_trading"].fillna(False).astype(bool)
    data["abs_trade_notional"] = pd.to_numeric(data.get("estimated_trade_notional", 0), errors="coerce").abs().fillna(0)
    data["abs_delta_weight"] = pd.to_numeric(data.get("delta_weight", 0), errors="coerce").abs().fillna(0)
    data["priority_score"] = (
        data["abs_delta_weight"] * 100
        + pd.to_numeric(data.get("final_score", 0), errors="coerce").fillna(0) / 100
        - pd.to_numeric(data.get("event_risk_score", 0), errors="coerce").fillna(0) / 100
        - pd.to_numeric(data.get("overnight_gap_risk_score", 0), errors="coerce").fillna(0) / 150
    )
    data["execution_condition"] = data.apply(_execution_condition, axis=1)
    data["risk_instruction"] = data.apply(_risk_instruction, axis=1)
    data.loc[data["blocked_from_trading"], "execution_condition"] = "暂停执行：数据质量门阻断，先人工核对价格/成交量/事件日历。"
    keep = [
        "signal_date",
        "symbol",
        "action_label",
        "target_weight",
        "current_weight",
        "delta_weight",
        "estimated_trade_notional",
        "estimated_shares",
        "close",
        "priority_score",
        "blocked_from_trading",
        "block_reason",
        "status",
        "broker_close_diff_pct",
        "earnings_calendar_status",
        "execution_condition",
        "risk_instruction",
        "decision_note",
    ]
    return data[[col for col in keep if col in data.columns]].sort_values("priority_score", ascending=False).reset_index(drop=True)


def _risk_state(
    blotter: pd.DataFrame,
    risk_overlay: pd.DataFrame,
    options_overlay: pd.DataFrame,
    data_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    latest = blotter.copy() if not blotter.empty else pd.DataFrame()
    if not latest.empty:
        rows.extend(
            [
                _risk_row("cash_signal", "空仓信号", bool(latest.get("cash_signal", pd.Series([False])).astype(str).str.lower().eq("true").any())),
                _risk_row("overbought", "组合过热", pd.to_numeric(latest.get("overbought_score", pd.Series(dtype=float)), errors="coerce").max()),
                _risk_row("crowding", "行业/主题拥挤", pd.to_numeric(latest.get("crowding_risk_score", pd.Series(dtype=float)), errors="coerce").max()),
                _risk_row("gap_risk", "隔夜 gap 风险", pd.to_numeric(latest.get("overnight_gap_risk_score", pd.Series(dtype=float)), errors="coerce").max()),
                _risk_row("minute_pretrade", "分钟级开仓前风险", pd.to_numeric(latest.get("minute_pretrade_risk_score", pd.Series(dtype=float)), errors="coerce").max()),
            ]
        )
    if not risk_overlay.empty:
        item = risk_overlay.copy()
        item["date"] = pd.to_datetime(item.get("date"), errors="coerce")
        latest_risk = item.sort_values("date").tail(1)
        if not latest_risk.empty:
            rows.append(_risk_row("regime_multiplier", "当前风险敞口乘数", latest_risk.get("combined_multiplier", pd.Series([pd.NA])).iloc[0]))
    if not options_overlay.empty:
        executable = int(options_overlay.get("executable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        rows.append(_risk_row("options_executable", "可执行期权合约数", executable))
    if data_quality is not None and not data_quality.empty:
        blocked = int(data_quality.get("blocked_from_trading", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        bad = int(data_quality.get("status", pd.Series(dtype=str)).fillna("ok").ne("ok").sum())
        rows.append(_risk_row("data_quality_blocked", "数据质量阻断标的", blocked))
        rows.append(_risk_row("data_quality_warnings", "数据质量警告标的", bad))
    return pd.DataFrame(rows)


def _position_drift(blotter: pd.DataFrame) -> pd.DataFrame:
    if blotter.empty:
        return pd.DataFrame()
    data = blotter.copy()
    data["abs_drift"] = pd.to_numeric(data.get("delta_weight", 0), errors="coerce").abs().fillna(0)
    keep = ["symbol", "action_label", "target_weight", "current_weight", "delta_weight", "abs_drift", "estimated_trade_notional", "primary_theme", "sector", "industry"]
    return data[[col for col in keep if col in data.columns]].sort_values("abs_drift", ascending=False).reset_index(drop=True)


def _crowding_watch(theme_divergence: pd.DataFrame) -> pd.DataFrame:
    if theme_divergence.empty:
        return pd.DataFrame()
    data = theme_divergence.copy()
    sort_col = "target_gross" if "target_gross" in data.columns else data.columns[0]
    keep = [
        "date",
        "level",
        "bucket",
        "target_gross",
        "target_net",
        "target_gross_change",
        "avg_relative_strength_score",
        "avg_event_risk_score",
        "leaders",
        "divergence_status",
        "suggested_tilt",
    ]
    return data[[col for col in keep if col in data.columns]].sort_values(sort_col, ascending=False).reset_index(drop=True)


def _options_risk(options_overlay: pd.DataFrame) -> pd.DataFrame:
    if options_overlay.empty:
        return pd.DataFrame()
    keep = [
        "symbol",
        "equity_action",
        "option_structure",
        "executable",
        "chain_validation_status",
        "contract_option_symbol",
        "contract_expiration",
        "contract_strike",
        "contract_mid",
        "contract_spread_pct_mid",
        "contract_open_interest",
        "contract_volume",
        "recommended_contracts",
        "estimated_option_premium",
        "premium_budget_utilization",
        "max_premium_budget",
        "overlay_note",
    ]
    return options_overlay[[col for col in keep if col in options_overlay.columns]].copy()


def manual_override_template() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "symbol",
            "override_action",
            "original_action",
            "override_weight",
            "reason",
            "risk_approved_by",
            "expires_at",
            "notes",
        ]
    )


def _execution_condition(row: pd.Series) -> str:
    if bool(row.get("blocked_from_trading", False)):
        return "暂停执行：数据质量门阻断，先人工核对价格/成交量/事件日历。"
    action = str(row.get("action", ""))
    gap = float(pd.to_numeric(pd.Series([row.get("overnight_gap_risk_score", 0)]), errors="coerce").fillna(0).iloc[0])
    minute = float(pd.to_numeric(pd.Series([row.get("minute_pretrade_risk_score", 0)]), errors="coerce").fillna(0).iloc[0])
    if action in {"OPEN LONG", "ADD"} and max(gap, minute) >= 65:
        return "只在开盘后风险降温时执行；若继续放量下跌，暂停加仓。"
    if action in {"EXIT", "REDUCE"}:
        return "优先执行；若流动性差，分批成交。"
    return "按计划执行；若价格偏离开盘参考价过大，降低成交比例。"


def _risk_instruction(row: pd.Series) -> str:
    event = float(pd.to_numeric(pd.Series([row.get("event_risk_score", 0)]), errors="coerce").fillna(0).iloc[0])
    crowding = float(pd.to_numeric(pd.Series([row.get("crowding_risk_score", 0)]), errors="coerce").fillna(0).iloc[0])
    if event >= 70:
        return "事件风险高，禁止加期权，股票仓位也需折扣。"
    if crowding >= 60:
        return "主题拥挤，新增仓位使用更小切片。"
    return "常规风险预算。"


def _risk_row(key: str, label: str, value) -> dict:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    status = "normal"
    if isinstance(value, bool):
        status = "block" if value else "normal"
    elif pd.notna(numeric):
        if numeric >= 80:
            status = "high"
        elif numeric >= 60:
            status = "watch"
    return {"risk_key": key, "label": label, "value": value, "status": status}
