"""Fast selected-row activation validation for strategy mutations.

This module answers one question before an expensive walk-forward run:
did a candidate scenario actually change selected names or target weights?
It intentionally uses cached signal frames and portfolio construction only;
it does not run execution, stops, or PnL simulation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_system.config import AppConfig
from quant_system.portfolio.construction import construct_target_weights, rebalance_dates


def selected_row_activation_summary(
    baseline_signals: pd.DataFrame,
    candidate_signals: pd.DataFrame,
    config: AppConfig,
    *,
    label: str = "candidate",
    include_target_weights: bool = True,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    max_dates: int | None = None,
) -> pd.DataFrame:
    """Summarize whether a candidate changes selected names and target weights.

    The comparison is intentionally pre-PnL and pre-execution. A scenario with
    zero changed selected names and zero target-weight deltas should usually be
    rejected before walk-forward, because it cannot alter realized returns under
    the same downstream engine.
    """

    baseline = _prepare_signals(baseline_signals, config, start_date=start_date, end_date=end_date, max_dates=max_dates)
    candidate = _prepare_signals(candidate_signals, config, start_date=start_date, end_date=end_date, max_dates=max_dates)
    if include_target_weights:
        base_targets = construct_target_weights(baseline, config.portfolio)
        candidate_targets = construct_target_weights(candidate, config.portfolio)
    else:
        base_targets = pd.DataFrame(columns=["date", "symbol", "target_weight"])
        candidate_targets = pd.DataFrame(columns=["date", "symbol", "target_weight"])
    daily = _daily_activation(
        baseline,
        candidate,
        base_targets,
        candidate_targets,
        target_weight_check_skipped=not include_target_weights,
    )
    if daily.empty:
        return pd.DataFrame(
            [
                {
                    "label": label,
                    "period": "full",
                    "n_dates": 0,
                    "selected_changed_dates": 0,
                    "selected_changed_date_rate": 0.0,
                    "mean_changed_names": 0.0,
                    "mean_changed_name_rate": 0.0,
                    "score_changed_rows": 0,
                    "target_changed_dates": 0,
                    "target_changed_date_rate": 0.0,
                    "mean_abs_target_weight_diff": 0.0,
                    "max_abs_target_weight_diff": 0.0,
                    "activation_pass": False,
                    "target_weight_check_skipped": not include_target_weights,
                }
            ]
        )
    rows = []
    for period, start, end in _periods(config):
        frame = daily.copy()
        if start is not None:
            frame = frame[frame["date"] >= start]
        if end is not None:
            frame = frame[frame["date"] <= end]
        rows.append(_summarize_period(frame, label=label, period=period))
    return pd.DataFrame(rows)


def save_activation_validation_report(summary: pd.DataFrame, out_dir: str | Path) -> Path:
    """Persist a CSV and compact HTML report for activation validation."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "activation_validation.csv", index=False)
    html = _render_html(summary)
    (out / "activation_validation.html").write_text(html, encoding="utf-8")
    return out


def _prepare_signals(
    signals: pd.DataFrame,
    config: AppConfig,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    max_dates: int | None = None,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=["date", "symbol", "signal", "final_score"])
    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    if "signal" not in out:
        out["signal"] = 0.0
    if "final_score" not in out:
        out["final_score"] = 50.0
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0.0)
    out["final_score"] = pd.to_numeric(out["final_score"], errors="coerce").fillna(50.0)
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
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _daily_activation(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    base_targets: pd.DataFrame,
    candidate_targets: pd.DataFrame,
    *,
    target_weight_check_skipped: bool,
) -> pd.DataFrame:
    dates = sorted(set(baseline.get("date", pd.Series(dtype="datetime64[ns]"))) | set(candidate.get("date", pd.Series(dtype="datetime64[ns]"))))
    rows = []
    for date in dates:
        base_day = baseline[baseline["date"].eq(date)]
        candidate_day = candidate[candidate["date"].eq(date)]
        base_selected = set(base_day.loc[base_day["signal"].abs() > 0, "symbol"].astype(str))
        candidate_selected = set(candidate_day.loc[candidate_day["signal"].abs() > 0, "symbol"].astype(str))
        selected_union = base_selected | candidate_selected
        selected_diff = base_selected ^ candidate_selected
        score_changed = _score_changed_rows(base_day, candidate_day)
        target_stats = _target_diff_for_date(base_targets, candidate_targets, date)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "baseline_selected": len(base_selected),
                "candidate_selected": len(candidate_selected),
                "changed_names": len(selected_diff),
                "changed_name_rate": len(selected_diff) / max(len(selected_union), 1),
                "score_changed_rows": score_changed,
                "target_weight_check_skipped": target_weight_check_skipped,
                **target_stats,
            }
        )
    return pd.DataFrame(rows)


def _score_changed_rows(base_day: pd.DataFrame, candidate_day: pd.DataFrame) -> int:
    base = base_day[["symbol", "final_score", "signal"]].rename(
        columns={"final_score": "baseline_final_score", "signal": "baseline_signal"}
    )
    candidate = candidate_day[["symbol", "final_score", "signal"]].rename(
        columns={"final_score": "candidate_final_score", "signal": "candidate_signal"}
    )
    merged = base.merge(candidate, on="symbol", how="inner")
    if merged.empty:
        return 0
    score_changed = (merged["baseline_final_score"] - merged["candidate_final_score"]).abs().gt(1e-9)
    signal_changed = (merged["baseline_signal"] - merged["candidate_signal"]).abs().gt(1e-9)
    return int((score_changed | signal_changed).sum())


def _target_diff_for_date(base_targets: pd.DataFrame, candidate_targets: pd.DataFrame, date: pd.Timestamp) -> dict[str, float | int]:
    if base_targets.empty and candidate_targets.empty:
        return {
            "target_symbols_union": 0,
            "target_changed": 0,
            "sum_abs_target_weight_diff": 0.0,
            "mean_abs_target_weight_diff": 0.0,
            "max_abs_target_weight_diff": 0.0,
        }
    base = _target_day(base_targets, date, "baseline_target_weight")
    candidate = _target_day(candidate_targets, date, "candidate_target_weight")
    merged = base.merge(candidate, on="symbol", how="outer").fillna(0.0)
    if merged.empty:
        return {
            "target_symbols_union": 0,
            "target_changed": 0,
            "sum_abs_target_weight_diff": 0.0,
            "mean_abs_target_weight_diff": 0.0,
            "max_abs_target_weight_diff": 0.0,
        }
    diff = (merged["baseline_target_weight"] - merged["candidate_target_weight"]).abs()
    return {
        "target_symbols_union": int(len(merged)),
        "target_changed": int(diff.gt(1e-9).any()),
        "sum_abs_target_weight_diff": float(diff.sum()),
        "mean_abs_target_weight_diff": float(diff.mean()),
        "max_abs_target_weight_diff": float(diff.max()),
    }


def _target_day(targets: pd.DataFrame, date: pd.Timestamp, weight_col: str) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame(columns=["symbol", weight_col])
    frame = targets.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    out = frame.loc[frame["date"].eq(pd.Timestamp(date)), ["symbol", "target_weight"]].copy()
    if out.empty:
        return pd.DataFrame(columns=["symbol", weight_col])
    out["symbol"] = out["symbol"].astype(str).str.upper()
    return out.rename(columns={"target_weight": weight_col})


def _periods(config: AppConfig) -> list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]:
    regime = config.regime
    return [
        ("full", None, None),
        ("train", pd.Timestamp(regime.train_start), pd.Timestamp(regime.train_end)),
        ("validation", pd.Timestamp(regime.validation_start), pd.Timestamp(regime.validation_end)),
        ("forward_current", pd.Timestamp(regime.forward_start), None),
    ]


def _summarize_period(frame: pd.DataFrame, *, label: str, period: str) -> dict[str, float | int | str | bool]:
    if frame.empty:
        return {
            "label": label,
            "period": period,
            "n_dates": 0,
            "selected_changed_dates": 0,
            "selected_changed_date_rate": 0.0,
            "mean_changed_names": 0.0,
            "mean_changed_name_rate": 0.0,
            "score_changed_rows": 0,
            "target_changed_dates": 0,
            "target_changed_date_rate": 0.0,
            "mean_abs_target_weight_diff": 0.0,
            "max_abs_target_weight_diff": 0.0,
            "activation_pass": False,
            "target_weight_check_skipped": bool(frame.get("target_weight_check_skipped", pd.Series([False])).any()),
        }
    selected_changed = frame["changed_names"].gt(0)
    target_changed = frame["target_changed"].gt(0)
    activation_pass = bool(selected_changed.any() or target_changed.any() or frame["score_changed_rows"].sum() > 0)
    return {
        "label": label,
        "period": period,
        "n_dates": int(len(frame)),
        "selected_changed_dates": int(selected_changed.sum()),
        "selected_changed_date_rate": float(selected_changed.mean()),
        "mean_changed_names": float(frame["changed_names"].mean()),
        "mean_changed_name_rate": float(frame["changed_name_rate"].mean()),
        "score_changed_rows": int(frame["score_changed_rows"].sum()),
        "target_changed_dates": int(target_changed.sum()),
        "target_changed_date_rate": float(target_changed.mean()),
        "mean_abs_target_weight_diff": float(frame["mean_abs_target_weight_diff"].mean()),
        "max_abs_target_weight_diff": float(frame["max_abs_target_weight_diff"].max()),
        "activation_pass": activation_pass,
        "target_weight_check_skipped": bool(frame.get("target_weight_check_skipped", pd.Series([False])).any()),
    }


def _render_html(summary: pd.DataFrame) -> str:
    table = summary.to_html(index=False, float_format=lambda x: f"{x:.6f}", classes="activation-table")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Activation Validation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; }}
    h1 {{ margin-bottom: 8px; }}
    .note {{ color: #536179; margin-bottom: 24px; max-width: 920px; line-height: 1.55; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e1ec; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f4f7fb; }}
  </style>
</head>
<body>
  <h1>Selected-Row Activation Validation</h1>
  <div class="note">
    这个报告只检查候选参数是否真的改变入选股票、信号分数或目标权重。
    如果 activation_pass 为 false，完整 walk-forward 通常不会产生经济差异，应该先回到因子/排序层继续修改。
  </div>
  {table}
</body>
</html>
"""
