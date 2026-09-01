"""Experiment management and optimizer diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.experiments.registry import load_registry


def build_experiment_management_tables(output_dir: str | Path = "runs", optimization_dir: str | Path = "runs/optimization") -> dict[str, pd.DataFrame]:
    """Build run comparison, parameter version, stability, and overfit-flag tables."""

    output = Path(output_dir)
    opt = Path(optimization_dir)
    registry = load_registry(output)
    parameter_results = _read_csv(opt / "parameter_results.csv")
    second_stage = _read_csv(opt / "second_stage_full_validation.csv")
    stability = _read_csv(opt / "parameter_stability.csv")
    return {
        "run_registry_ranked": _rank_registry(registry),
        "parameter_versions": _parameter_versions(parameter_results, second_stage),
        "parameter_stability_scored": _stability_scored(stability),
        "overfit_red_flags": _overfit_red_flags(parameter_results, second_stage),
        "topn_full_validation": second_stage,
    }


def write_experiment_management_report(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> Path:
    """Persist experiment-management tables and a compact HTML report."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    html = ["<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>Experiment Management</title>"]
    html.append(
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}th,td{border-bottom:1px solid #e5e7eb;padding:7px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}.warn{background:#fff7ed;border:1px solid #fdba74;padding:10px}</style></head><body>"
    )
    html.append("<h1>参数研究与实验管理</h1><p class='warn'>用于比较参数版本、稳定性、过拟合红旗和 top-N full validation。forward/current 年默认不应参与调参。</p>")
    for name, frame in tables.items():
        html.append(f"<h2>{name}</h2>")
        html.append(frame.head(80).to_html(index=False, border=0) if not frame.empty else "<p>暂无数据。</p>")
    html.append("</body></html>")
    path = out / "experiment_management.html"
    path.write_text("".join(html), encoding="utf-8")
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _rank_registry(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return registry
    data = registry.copy()
    for column in ["sharpe", "cagr", "max_drawdown", "trailing_one_year_return", "average_turnover", "average_gross_exposure"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["risk_adjusted_rank_score"] = (
        data.get("sharpe", 0).fillna(0)
        + data.get("cagr", 0).fillna(0)
        + data.get("max_drawdown", 0).fillna(0).clip(upper=0)
        - data.get("average_turnover", 0).fillna(0) * 0.25
    )
    return data.sort_values("risk_adjusted_rank_score", ascending=False).reset_index(drop=True)


def _parameter_versions(parameter_results: pd.DataFrame, second_stage: pd.DataFrame) -> pd.DataFrame:
    data = second_stage if not second_stage.empty else parameter_results
    if data.empty:
        return pd.DataFrame()
    version_cols = [
        "scenario",
        "rebalance",
        "strategy_profile",
        "constraint_profile",
        "target_gross_exposure",
        "target_net_exposure",
        "daily_turnover_cap",
        "lookback_returns",
        "skip_recent_days",
        "long_quantile",
        "short_quantile",
        "momentum_max_positions",
        "validation_sharpe",
        "validation_objective",
        "full_sharpe",
        "full_cagr",
        "full_max_drawdown",
        "validation_pass",
        "overfit_risk",
    ]
    out = data[[col for col in version_cols if col in data.columns]].copy()
    out.insert(0, "parameter_version", [f"pv_{idx + 1:03d}" for idx in range(len(out))])
    return out


def _stability_scored(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return stability
    data = stability.copy()
    data["objective_mean"] = pd.to_numeric(data.get("objective_mean"), errors="coerce")
    data["objective_std"] = pd.to_numeric(data.get("objective_std"), errors="coerce").fillna(0.0)
    data["count"] = pd.to_numeric(data.get("count"), errors="coerce").fillna(0)
    data["stability_score"] = data["objective_mean"].fillna(0) - data["objective_std"].fillna(0)
    data["sample_warning"] = np.where(data["count"] < 3, "too_few_parameter_observations", "ok")
    return data.sort_values("stability_score", ascending=False).reset_index(drop=True)


def _overfit_red_flags(parameter_results: pd.DataFrame, second_stage: pd.DataFrame) -> pd.DataFrame:
    data = second_stage if not second_stage.empty else parameter_results
    if data.empty:
        return pd.DataFrame()
    rows = []
    for _, row in data.iterrows():
        flags = []
        train = pd.to_numeric(pd.Series([row.get("train_objective")]), errors="coerce").iloc[0]
        val = pd.to_numeric(pd.Series([row.get("validation_objective")]), errors="coerce").iloc[0]
        gap = pd.to_numeric(pd.Series([row.get("objective_gap")]), errors="coerce").iloc[0]
        pass_rate = pd.to_numeric(pd.Series([row.get("validation_pass_rate")]), errors="coerce").iloc[0]
        turnover = pd.to_numeric(pd.Series([row.get("validation_turnover")]), errors="coerce").iloc[0]
        if pd.notna(train) and pd.notna(val) and train - val > 0.15:
            flags.append("train_validation_gap")
        if pd.notna(gap) and abs(gap) > 0.20:
            flags.append("unstable_objective_gap")
        if pd.notna(pass_rate) and pass_rate < 0.50:
            flags.append("low_walk_forward_pass_rate")
        if pd.notna(turnover) and turnover > 1.0:
            flags.append("high_turnover")
        if bool(row.get("overfit_risk", False)):
            flags.append("optimizer_overfit_flag")
        rows.append(
            {
                "scenario": row.get("scenario", ""),
                "red_flags": "|".join(flags) if flags else "ok",
                "flag_count": len(flags),
                "train_objective": train,
                "validation_objective": val,
                "objective_gap": gap,
                "validation_pass_rate": pass_rate,
                "validation_turnover": turnover,
            }
        )
    return pd.DataFrame(rows).sort_values(["flag_count", "validation_objective"], ascending=[False, False]).reset_index(drop=True)
