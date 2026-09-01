"""Optimization report outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_parameter_heatmap(
    results: pd.DataFrame,
    out_path: str | Path,
    x: str = "lookback_returns",
    y: str = "skip_recent_days",
    value: str = "validation_objective",
) -> None:
    """Save a compact parameter sensitivity heatmap."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if results.empty or not {x, y, value}.issubset(results.columns):
        out.write_text("Heatmap unavailable: missing optimization columns.", encoding="utf-8")
        return
    import matplotlib.pyplot as plt

    pivot = results.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=[str(i) for i in pivot.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Validation Objective Sensitivity")
    fig.colorbar(image, ax=ax, label=value)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def save_validation_forward_scatter(results: pd.DataFrame, out_path: str | Path) -> None:
    """Save validation-vs-forward diagnostic scatter."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    required = {"validation_objective", "forward_sharpe", "validation_pass"}
    if results.empty or not required.issubset(results.columns):
        out.write_text("Scatter unavailable: missing optimization columns.", encoding="utf-8")
        return
    import matplotlib.pyplot as plt

    colors = results["validation_pass"].map({True: "#2f9e44", False: "#c92a2a"}).fillna("#868e96")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(results["validation_objective"], results["forward_sharpe"], c=colors, alpha=0.75)
    ax.axvline(0, color="#adb5bd", lw=1)
    ax.axhline(0, color="#adb5bd", lw=1)
    ax.set_xlabel("Validation objective")
    ax.set_ylabel("Forward/current Sharpe (not optimized)")
    ax.set_title("Validation Robustness vs Forward Check")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def save_optimization_report(
    results: pd.DataFrame,
    stability: pd.DataFrame,
    warnings: list[str],
    out_dir: str | Path,
    pbo_summary: dict | None = None,
) -> None:
    """Save CSV/JSON/HTML optimization artifacts."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "parameter_results.csv", index=False)
    stability.to_csv(out / "parameter_stability.csv", index=False)
    pbo_summary = pbo_summary or {"status": "not_computed"}
    (out / "pbo_summary.json").write_text(json.dumps(pbo_summary, indent=2, default=str), encoding="utf-8")
    if isinstance(pbo_summary.get("selection_details"), list):
        pd.DataFrame(pbo_summary["selection_details"]).to_csv(out / "pbo_selection_details.csv", index=False)
    (out / "warnings.txt").write_text("\n".join(warnings), encoding="utf-8")
    save_parameter_heatmap(results, out / "parameter_stability_heatmap.png")
    save_validation_forward_scatter(results, out / "validation_forward_scatter.png")
    eligible = results[results.get("validation_pass", False).astype(bool)] if "validation_pass" in results else results
    best_pool = eligible if not eligible.empty else results
    best = best_pool.sort_values(["validation_objective", "train_objective"], ascending=False).head(1)
    summary = {
        "best": best.to_dict(orient="records")[0] if not best.empty else {},
        "warnings": warnings,
        "current_year_used_for_optimization": False,
        "pbo": {key: value for key, value in pbo_summary.items() if key != "selection_details"},
    }
    (out / "optimization_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    warning_html = "".join(f"<li>{warning}</li>" for warning in warnings)
    best_html = best.to_html(index=False) if not best.empty else "<p>No valid parameter result.</p>"
    top_html = results.sort_values("validation_objective", ascending=False).head(20).to_html(index=False) if not results.empty else "<p>No results.</p>"
    forward_html = results.sort_values("forward_sharpe", ascending=False).head(20).to_html(index=False) if "forward_sharpe" in results else "<p>No forward table.</p>"
    stability_html = stability.to_html(index=False) if not stability.empty else "<p>No stability table.</p>"
    pbo_html = pd.DataFrame([{key: value for key, value in pbo_summary.items() if key != "selection_details"}]).to_html(index=False)
    dsr_cols = [col for col in ["scenario", "validation_dsr", "train_dsr", "validation_sharpe", "validation_objective", "validation_pass"] if col in results]
    dsr_html = (
        results.sort_values("validation_dsr", ascending=False)[dsr_cols].head(20).to_html(index=False)
        if dsr_cols and "validation_dsr" in results
        else "<p>DSR unavailable.</p>"
    )
    (out / "robustness_report.html").write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>quant_system Optimization Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px; }}
    th {{ background: #f0f4f8; }}
    .warn {{ background: #fff7ed; border: 1px solid #fdba74; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Optimization And Robustness Report</h1>
  <div class="warn"><b>Warnings</b><ul>{warning_html}</ul></div>
  <p>Optimization uses train data for fitting and validation data for parameter acceptance. Forward/current-year data is not used unless explicitly enabled in config.</p>
  <p>The CLI also writes <code>best_config.yml</code> next to this report when a best validation row is available. Run it with <code>python -m quant_system.cli backtest --config runs/optimization/best_config.yml</code>.</p>
  <h2>DSR / PBO Overfit Diagnostics</h2>
  <p>DSR penalizes Sharpe for multiple trials and non-normal returns. PBO estimates how often parameter selection would pick a candidate that ranks poorly out of sample.</p>
  <h3>PBO Summary</h3>{pbo_html}
  <h3>Top DSR Rows</h3>{dsr_html}
  <h2>Best Validation Parameter Set</h2>{best_html}
  <h2>Top Validation Results</h2>{top_html}
  <h2>Forward/Current Regime Check</h2>
  <p>This section is diagnostic only. It is not used to choose parameters unless explicitly enabled in config.</p>
  {forward_html}
  <h2>Parameter Stability</h2>{stability_html}
  <h2>Heatmap</h2><img src="parameter_stability_heatmap.png" style="max-width: 900px; width: 100%;">
  <h2>Validation vs Forward</h2><img src="validation_forward_scatter.png" style="max-width: 900px; width: 100%;">
</body>
</html>""",
        encoding="utf-8",
    )


def save_topn_full_validation_report(
    second_stage: pd.DataFrame,
    out_dir: str | Path,
    *,
    top_n: int,
) -> None:
    """Save a dedicated top-N full validation report after stage-two reruns."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if second_stage.empty:
        (out / "topn_full_validation_report.html").write_text(
            "<!doctype html><html><body><h1>Top-N Full Validation</h1><p>No second-stage validation rows.</p></body></html>",
            encoding="utf-8",
        )
        return
    data = second_stage.copy()
    for column in [
        "validation_objective",
        "validation_sharpe",
        "validation_cagr",
        "validation_max_drawdown",
        "train_objective",
        "objective_gap",
        "validation_pass_rate",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["full_validation_rank"] = data.sort_values(
        ["validation_pass", "validation_objective", "validation_sharpe"],
        ascending=[False, False, False],
    ).reset_index().index + 1
    data = data.sort_values("full_validation_rank")
    data.to_csv(out / "topn_full_validation_ranked.csv", index=False)
    pass_html = data[data.get("validation_pass", False).astype(bool)].to_html(index=False) if "validation_pass" in data else "<p>No pass flag.</p>"
    all_html = data.to_html(index=False)
    red_flags = _topn_red_flags(data)
    red_flag_html = red_flags.to_html(index=False) if not red_flags.empty else "<p>No red flags.</p>"
    (out / "topn_full_validation_report.html").write_text(
        f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Top-N Full Validation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 30px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 24px; font-size: 12px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px; }}
    th {{ background: #eef2f7; }}
    .warn {{ background: #fff7ed; border: 1px solid #fdba74; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Top-{top_n} Full Validation 自动报告</h1>
  <div class="warn">第二阶段会用完整 intraday/fundamental/event/risk 设置重跑第一阶段筛出的 top-N 参数。forward/current year 仍只用于诊断，不应作为最终调参依据。</div>
  <h2>过拟合/执行红旗</h2>{red_flag_html}
  <h2>通过 validation gate 的参数</h2>{pass_html}
  <h2>全部 Top-N Full Validation</h2>{all_html}
</body>
</html>""",
        encoding="utf-8",
    )


def _topn_red_flags(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        flags = []
        if bool(row.get("overfit_risk", False)):
            flags.append("first_stage_overfit_flag")
        if pd.notna(row.get("objective_gap")) and float(row.get("objective_gap")) > 0.10:
            flags.append("train_validation_gap")
        if pd.notna(row.get("validation_max_drawdown")) and float(row.get("validation_max_drawdown")) < -0.25:
            flags.append("validation_drawdown_too_large")
        if pd.notna(row.get("validation_cagr")) and float(row.get("validation_cagr")) <= 0:
            flags.append("validation_cagr_non_positive")
        if pd.notna(row.get("validation_dsr")) and float(row.get("validation_dsr")) < 0.50:
            flags.append("low_deflated_sharpe_ratio")
        if pd.notna(row.get("trade_count")) and float(row.get("trade_count")) < 50:
            flags.append("too_few_trades")
        rows.append({"scenario": row.get("scenario", ""), "flag_count": len(flags), "red_flags": "|".join(flags) if flags else "ok"})
    return pd.DataFrame(rows).sort_values(["flag_count", "scenario"], ascending=[False, True]).reset_index(drop=True)
