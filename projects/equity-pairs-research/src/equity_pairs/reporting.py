"""Publication-quality charts and a self-contained HTML research report."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, select_autoescape


COLORS = {
    "ink": "#172033",
    "muted": "#68758A",
    "blue": "#176BCE",
    "cyan": "#0E9FAD",
    "green": "#14865A",
    "orange": "#E17B16",
    "red": "#C74343",
    "purple": "#7656C8",
    "grid": "#DDE3EC",
    "panel": "#F7F9FC",
}


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=COLORS["ink"], labelsize=9)
    ax.title.set_color(COLORS["ink"])
    ax.xaxis.label.set_color(COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])


def _save(fig: plt.Figure, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_cointegration_heatmap(top_pairs: pd.DataFrame, output: str | Path) -> Path:
    sectors = sorted(top_pairs["sector"].dropna().unique())
    max_rank = int(top_pairs["descriptive_rank_in_sector"].max())
    matrix = np.full((len(sectors), max_rank), np.nan)
    labels = np.full((len(sectors), max_rank), "", dtype=object)
    for row_index, sector in enumerate(sectors):
        group = top_pairs[top_pairs["sector"].eq(sector)]
        for row in group.itertuples(index=False):
            column = int(row.descriptive_rank_in_sector) - 1
            pvalue = max(float(row.pair_pvalue), 1e-16)
            matrix[row_index, column] = -math.log10(pvalue)
            labels[row_index, column] = str(row.pair).replace("__", "/")
    fig, ax = plt.subplots(figsize=(15, max(5.5, len(sectors) * 0.65)))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(sectors)), sectors)
    ax.set_xticks(np.arange(max_rank), [f"#{rank}" for rank in range(1, max_rank + 1)])
    for row in range(len(sectors)):
        for column in range(max_rank):
            if labels[row, column]:
                color = "white" if matrix[row, column] > np.nanmedian(matrix) else COLORS["ink"]
                ax.text(column, row, labels[row, column], ha="center", va="center", fontsize=6.8, color=color)
    ax.set_title("Eight-year descriptive cointegration leaders by current GICS sector", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Within-sector rank (lower orientation-adjusted Engle-Granger p-value is better)")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("−log10(pair p-value)")
    ax.grid(False)
    fig.text(
        0.01,
        0.01,
        "Descriptive full-window ranking only; not used to claim held-out performance.",
        color=COLORS["muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, output)


def plot_sector_performance(metrics: pd.DataFrame, output: str | Path) -> Path:
    successful = metrics[metrics.get("backtest_status", "failed").eq("ok")].copy()
    summary = successful.groupby("sector").agg(
        median_net_cagr=("net_cagr", "median"),
        mean_net_cagr=("net_cagr", "mean"),
        positive_pairs=("net_cagr", lambda values: int((values > 0).sum())),
        tested_pairs=("pair", "count"),
    )
    summary = summary.sort_values("median_net_cagr")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    positions = np.arange(len(summary))
    colors = np.where(summary["median_net_cagr"] >= 0, COLORS["green"], COLORS["red"])
    ax.barh(positions, summary["median_net_cagr"] * 100.0, color=colors, alpha=0.88)
    ax.set_yticks(positions, summary.index)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.9)
    ax.set_xlabel("Median held-out net CAGR (%)")
    ax.set_title("Formation-selected pair performance by sector", loc="left", fontsize=15, weight="bold")
    for y, value, positive, total in zip(
        positions,
        summary["median_net_cagr"] * 100.0,
        summary["positive_pairs"],
        summary["tested_pairs"],
        strict=True,
    ):
        offset = 0.12 if value >= 0 else -0.12
        ax.text(
            value + offset,
            y,
            f"{value:+.2f}%  ({positive}/{total} positive)",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color=COLORS["ink"],
        )
    _style_axes(ax)
    fig.tight_layout()
    return _save(fig, output)


def plot_portfolio_curves(curves: pd.DataFrame, output: str | Path) -> Path:
    equity_columns = [column for column in curves if column.endswith("_equity")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    palette = [COLORS["blue"], COLORS["green"], COLORS["purple"]]
    for column, color in zip(equity_columns, palette, strict=False):
        label = column.removesuffix("_equity").replace("_", " ").title()
        axes[0].plot(curves.index, curves[column], label=label, color=color, linewidth=2)
        drawdown = curves[column] / curves[column].cummax() - 1.0
        axes[1].plot(curves.index, drawdown * 100.0, label=label, color=color, linewidth=1.5)
    axes[0].axhline(1.0, color=COLORS["muted"], linewidth=0.8)
    axes[0].set_ylabel("Growth of $1, net")
    axes[0].set_title("Held-out portfolio equity curves after modeled costs", loc="left", fontsize=15, weight="bold")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    for ax in axes:
        _style_axes(ax)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    return _save(fig, output)


def plot_macro_history(macro: pd.DataFrame, output: str | Path) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    if "DFF" in macro:
        axes[0].plot(macro.index, macro["DFF"], color=COLORS["blue"], label="Fed funds")
    if "DGS2" in macro:
        axes[0].plot(macro.index, macro["DGS2"], color=COLORS["orange"], label="2-year")
    if "DGS10" in macro:
        axes[0].plot(macro.index, macro["DGS10"], color=COLORS["green"], label="10-year")
    axes[0].set_ylabel("Percent")
    axes[0].set_title("FRED macro regime context (current-vintage, attribution only)", loc="left", fontsize=15, weight="bold")
    axes[0].legend(frameon=False, ncol=3)
    if "yield_curve_2s10s" in macro:
        axes[1].plot(macro.index, macro["yield_curve_2s10s"], color=COLORS["purple"])
        axes[1].fill_between(
            macro.index,
            macro["yield_curve_2s10s"],
            0,
            where=macro["yield_curve_2s10s"] < 0,
            color=COLORS["red"],
            alpha=0.24,
        )
    axes[1].axhline(0, color=COLORS["ink"], linewidth=0.8)
    axes[1].set_ylabel("2s10s (pp)")
    if "VIXCLS" in macro:
        axes[2].plot(macro.index, macro["VIXCLS"], color=COLORS["red"])
        axes[2].axhline(25, color=COLORS["orange"], linestyle="--", linewidth=1)
    axes[2].set_ylabel("VIX")
    if "cpi_yoy" in macro:
        cpi = macro["cpi_yoy"].dropna()
        axes[3].plot(cpi.index, cpi, color=COLORS["orange"], label="CPI YoY")
        axes[3].axhline(3, color=COLORS["red"], linestyle="--", linewidth=1)
    if "UNRATE" in macro:
        unemployment = macro["UNRATE"].dropna()
        secondary = axes[3].twinx()
        secondary.plot(unemployment.index, unemployment, color=COLORS["cyan"], alpha=0.85, label="Unemployment")
        secondary.set_ylabel("Unemployment (%)", color=COLORS["cyan"])
        secondary.spines["top"].set_visible(False)
    axes[3].set_ylabel("CPI YoY (%)")
    axes[3].set_xlabel("Date")
    for ax in axes:
        _style_axes(ax)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    return _save(fig, output)


def plot_pair_dashboard(
    pair_signals: pd.DataFrame,
    metrics: Mapping[str, Any],
    output: str | Path,
) -> Path:
    frame = pair_signals.set_index("date").sort_index() if "date" in pair_signals else pair_signals.copy()
    pair = str(metrics["pair"])
    dependent = str(metrics["dependent"])
    independent = str(metrics["independent"])
    first_dep = frame["price_dependent"].dropna().iloc[0]
    first_ind = frame["price_independent"].dropna().iloc[0]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1.4, 1.2]})
    axes[0].plot(frame.index, frame["price_dependent"] / first_dep, label=dependent, color=COLORS["blue"])
    axes[0].plot(frame.index, frame["price_independent"] / first_ind, label=independent, color=COLORS["orange"])
    axes[0].set_ylabel("Normalized price")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title(
        f"{pair.replace('__', ' / ')} | net CAGR {float(metrics['net_cagr']):+.2%} | "
        f"Sharpe {float(metrics['sharpe']):.2f} | {int(metrics['trade_count'])} trades",
        loc="left",
        fontsize=14,
        weight="bold",
    )
    axes[1].plot(frame.index, frame["spread"], color=COLORS["purple"], linewidth=1.2, label="Log spread")
    axes[1].plot(frame.index, frame["spread_rolling_mean"], color=COLORS["ink"], linewidth=1, label="60d mean")
    axes[1].fill_between(
        frame.index,
        frame["spread_rolling_mean"] - 2 * frame["spread_rolling_std"],
        frame["spread_rolling_mean"] + 2 * frame["spread_rolling_std"],
        color=COLORS["purple"],
        alpha=0.10,
        label="±2 rolling σ",
    )
    axes[1].set_ylabel("Spread")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    axes[2].plot(frame.index, frame["zscore"], color=COLORS["ink"], linewidth=1)
    for level, color, style in [(2, COLORS["orange"], "--"), (-2, COLORS["orange"], "--"), (0.5, COLORS["green"], ":"), (-0.5, COLORS["green"], ":"), (4, COLORS["red"], "--"), (-4, COLORS["red"], "--")]:
        axes[2].axhline(level, color=color, linestyle=style, linewidth=0.8)
    entries = frame[frame["decision_event"].str.startswith("enter", na=False)]
    exits = frame[frame["decision_event"].str.startswith("exit", na=False)]
    axes[2].scatter(entries.index, entries["zscore"], marker="^", color=COLORS["blue"], s=34, label="Entry decision", zorder=4)
    axes[2].scatter(exits.index, exits["zscore"], marker="x", color=COLORS["red"], s=34, label="Exit decision", zorder=4)
    axes[2].fill_between(frame.index, -4.2, 4.2, where=frame["target_position"].eq(1), color=COLORS["green"], alpha=0.07)
    axes[2].fill_between(frame.index, -4.2, 4.2, where=frame["target_position"].eq(-1), color=COLORS["red"], alpha=0.06)
    axes[2].set_ylim(-5.2, 5.2)
    axes[2].set_ylabel("Z-score")
    axes[2].legend(frameon=False, ncol=2, fontsize=8)
    axes[3].plot(frame.index, frame["gross_equity"], color=COLORS["muted"], linewidth=1.1, label="Gross")
    axes[3].plot(frame.index, frame["net_equity"], color=COLORS["blue"], linewidth=1.8, label="Net")
    axes[3].axhline(1, color=COLORS["ink"], linewidth=0.7)
    axes[3].set_ylabel("Growth of $1")
    axes[3].set_xlabel("Date")
    axes[3].legend(frameon=False, ncol=2)
    for ax in axes:
        _style_axes(ax)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    return _save(fig, output)


def _format_frame(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return '<p class="muted">No rows available.</p>'
    shown = frame.head(max_rows) if max_rows else frame
    formatted = shown.copy()
    for column in formatted.columns:
        if any(token in column.lower() for token in ("cagr", "return", "drawdown", "win_rate", "fraction", "volatility")):
            formatted[column] = formatted[column].map(
                lambda value: f"{float(value):.2%}" if pd.notna(value) else ""
            )
        elif any(token in column.lower() for token in ("pvalue", "qvalue")):
            formatted[column] = formatted[column].map(
                lambda value: f"{float(value):.3g}" if pd.notna(value) else ""
            )
        elif pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: f"{float(value):.3f}" if pd.notna(value) else ""
            )
    return formatted.to_html(index=False, border=0, classes="data-table", escape=True)


REPORT_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Equity Pairs Trading Research — {{ as_of }}</title>
<style>
:root{--ink:#172033;--muted:#68758a;--blue:#176bce;--green:#14865a;--red:#c74343;--panel:#f7f9fc;--line:#dde3ec}
*{box-sizing:border-box}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:#fff;line-height:1.55}
main{max-width:1440px;margin:auto;padding:40px 46px 80px}h1{font-size:38px;line-height:1.08;margin:0 0 12px;letter-spacing:-.03em}h2{font-size:25px;margin:46px 0 14px;border-bottom:1px solid var(--line);padding-bottom:8px}h3{font-size:18px;margin:28px 0 10px}.eyebrow{color:var(--blue);text-transform:uppercase;letter-spacing:.12em;font-weight:750;font-size:12px}.subtitle{font-size:17px;color:var(--muted);max-width:980px}.warning{background:#fff5e8;border-left:5px solid #e17b16;padding:15px 18px;margin:25px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:13px;margin:24px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px}.card .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.card .value{font-size:24px;font-weight:750;margin-top:4px}.chart{width:100%;border:1px solid var(--line);border-radius:8px;margin:12px 0 25px}.data-table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0 24px}.data-table th{position:sticky;top:0;background:#edf2f8;text-align:left}.data-table th,.data-table td{padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}.table-wrap{max-height:580px;overflow:auto;border:1px solid var(--line);border-radius:7px;margin-bottom:22px}.table-wrap .data-table{margin:0}.muted{color:var(--muted)}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}pre{background:#101827;color:#e7ecf5;padding:16px;border-radius:7px;overflow:auto;font-size:12px}a{color:var(--blue)}ul{padding-left:22px}.method-table td:first-child{font-weight:700}.small{font-size:12px}.pill{display:inline-block;border-radius:999px;padding:3px 9px;background:#e9f2ff;color:var(--blue);font-size:11px;font-weight:700;margin-right:5px}
@media(max-width:700px){main{padding:26px 18px}h1{font-size:30px}.data-table{font-size:11px}}
</style>
</head>
<body><main>
<div class="eyebrow">Quantitative research package · generated {{ generated_at }}</div>
<h1>Sector cointegration and equity pairs trading</h1>
<p class="subtitle">Eight-year current-S&amp;P-500 study through {{ as_of }}, with exhaustive within-sector screening, a five-year formation window, a three-year held-out trading window, explicit trading frictions, daily signals, and FRED regime attribution.</p>

<div class="warning"><strong>Read this first.</strong> This is a research prototype, not a profit guarantee or investment recommendation. Today's constituents and today's GICS labels are projected backward, so results have survivorship and classification bias. The “best ten” are ranked after observing held-out results and are therefore an ex-post shortlist, not a live selection rule.</div>

<div class="grid">
{% for card in cards %}<div class="card"><div class="label">{{ card.label }}</div><div class="value">{{ card.value }}</div></div>{% endfor %}
</div>

<h2>Outcome</h2>
<p>The defensible aggregate result is the all-formation-selected or sector-balanced portfolio. The ex-post top-performer curve answers the requested winner analysis but is selection-biased. CAGR is net of the configured turnover cost and short-borrow proxy.</p>
<img class="chart" src="{{ chart_paths.portfolios }}" alt="Portfolio equity curves">
<h3>Portfolio metrics</h3><div class="table-wrap">{{ portfolio_table | safe }}</div>
<h3>Transaction-cost sensitivity</h3><p class="muted">The same realized trades are repriced at each one-way turnover-cost assumption; no signals are re-optimized.</p><div class="table-wrap">{{ cost_table | safe }}</div>
<h3>Top ten held-out pair performers by net CAGR</h3><div class="table-wrap">{{ best_table | safe }}</div>

<h2>What was most cointegrated over all eight years?</h2>
<p>The heatmap and table are descriptive full-window rankings. Pair p-values adjust the choice between the two Engle-Granger orientations; sector q-values additionally apply Benjamini-Hochberg false-discovery control.</p>
<img class="chart" src="{{ chart_paths.cointegration }}" alt="Cointegration leaders heatmap">
<div class="table-wrap">{{ full_top_table | safe }}</div>

<h2>Held-out backtest diagnostics</h2>
<div class="warning"><strong>Statistical gate:</strong> {{ strict_selection_count }} of {{ formation_selection_count }} sector selections passed every strict formation criterion; {{ fallback_selection_count }} were deterministic fills needed to provide the requested count per sector. Fallback does not mean profitable evidence passed the false-discovery gate.</div>
<img class="chart" src="{{ chart_paths.sectors }}" alt="Sector performance">
<div class="table-wrap">{{ formation_table | safe }}</div>

<h2>Trading signals and equity curves for the best ten</h2>
<p class="muted">Triangles and crosses mark close-time decisions. Holdings change one session later. Green/red shading denotes long/short spread target state.</p>
{% for dashboard in dashboards %}<img class="chart" src="{{ dashboard }}" alt="Pair dashboard">{% endfor %}

<h2>Strategy deep dive</h2>
<table class="data-table method-table"><thead><tr><th>Family</th><th>Core idea</th><th>Why practitioners use it</th><th>Principal risk</th></tr></thead><tbody>
<tr><td>Distance</td><td>Match normalized price paths by minimum squared distance.</td><td>Fast, intuitive, canonical.</td><td>No formal stationary equilibrium.</td></tr>
<tr><td>Cointegration</td><td>Trade a stationary linear combination of non-stationary prices.</td><td>Testable equilibrium and explicit hedge ratio.</td><td>Structural breaks, test asymmetry, multiplicity.</td></tr>
<tr><td>OU / time series</td><td>Fit mean-reverting spread dynamics and optimal bands.</td><td>Links half-life and volatility to holding rules.</td><td>Parameter error and distribution assumptions.</td></tr>
<tr><td>Kalman / state space</td><td>Allow alpha and beta to evolve through time.</td><td>Adapts to slow relationship drift.</td><td>Filter tuning can overfit and inflate turnover.</td></tr>
<tr><td>Factor / PCA</td><td>Trade mean reversion in factor-neutral residuals.</td><td>Scalable, diversified statistical arbitrage.</td><td>Factor instability, crowding, constraint complexity.</td></tr>
<tr><td>Copula</td><td>Model nonlinear and tail dependence.</td><td>Handles asymmetric joint behavior.</td><td>Model-selection and estimation instability.</td></tr>
<tr><td>ML / clustering</td><td>Preselect economic neighbors from rich features.</td><td>Reduces combinatorics and adds fundamentals.</td><td>Leakage, regime drift, validation burden.</td></tr>
<tr><td>Stochastic control</td><td>Optimize holdings and no-trade regions under costs.</td><td>Direct risk/cost objective.</td><td>Strong process assumptions and implementation complexity.</td></tr>
</tbody></table>

<h3>Implemented specification</h3>
<ul>
<li>Every different-issuer pair within each current GICS sector is tested; same-CIK share classes are excluded when CIK is available.</li>
<li>The first {{ formation_years }} years select ten pairs per sector. Strict selections require sufficient coverage, I(1)-like legs, sector FDR significance, positive beta, 2–252 day half-life, and Hurst below 0.5. Fallbacks are labeled.</li>
<li>The final {{ test_years }} years use frozen log-price OLS alpha/beta, a trailing 60-session z-score, ±2 entry, ±0.5 exit, ±4 stop, and 60-day maximum holding period.</li>
<li>Signals at close <em>t</em> affect return from close <em>t</em> to close <em>t+1</em>. Gross-normalized weights are rebalanced daily, including charged drift turnover; turnover costs {{ transaction_cost_bps }} bps per dollar traded and short notional costs {{ borrow_bps }} bps annualized.</li>
</ul>
<p>For formulas, references, and the productization checklist, see <a href="../../docs/methodology.md">docs/methodology.md</a>.</p>

<h2>FRED macro attribution</h2>
<p>FRED values below are today's revised vintage and are used only for ex-post regime attribution—not to gate or size trades. USREC is especially unsuitable as a contemporaneous signal without real-time vintage handling.</p>
<img class="chart" src="{{ chart_paths.macro }}" alt="FRED macro history">
<div class="table-wrap">{{ regime_table | safe }}</div>

<h2>Biases, risks, and production gap</h2>
<ul>
<li><strong>Universe:</strong> current constituents/current GICS create survivorship, historical membership, and classification bias. Point-in-time S&amp;P/CRSP data is required for a production claim.</li>
<li><strong>Selection:</strong> the full-window list is descriptive. The formation/test split is cleaner, but a single holdout is still only one historical path. Rolling, purged nested validation is the next research gate.</li>
<li><strong>Market data:</strong> confirmed MarketData daily bars omit executable bid/ask, market impact, locate constraints, symbol-specific borrow, and delisted-name coverage. Production use requires the appropriate data rights.</li>
<li><strong>Model:</strong> cointegration can break abruptly; static beta leaves factor and market exposures. Live controls need rolling break tests, exposure limits, liquidity/capacity constraints, and kill switches.</li>
<li><strong>Costs:</strong> configured costs are transparent scenario assumptions, not measured execution costs. Results should be rerun across materially harsher scenarios.</li>
</ul>

<h2>Quality checks and reproducibility</h2>
<div class="table-wrap">{{ quality_table | safe }}</div>
<h3>Configuration</h3><pre>{{ config_json }}</pre>
<h3>Primary sources</h3>
<ul>{% for source in sources %}<li><a href="{{ source.url }}">{{ source.label }}</a></li>{% endfor %}</ul>
<p class="small muted">Raw/processed CSV files, every selected-pair daily signal, the trade ledger, manifest, and chart assets sit beside this report. Credentials are not serialized.</p>
</main></body></html>
"""


def render_html_report(
    output: str | Path,
    *,
    context: Mapping[str, Any],
    best_pairs: pd.DataFrame,
    full_top_pairs: pd.DataFrame,
    formation_pairs: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    portfolio_cost_sensitivity: pd.DataFrame,
    regime_attribution: pd.DataFrame,
    quality_checks: Mapping[str, Any],
    chart_paths: Mapping[str, Any],
) -> Path:
    environment = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = environment.from_string(REPORT_TEMPLATE)
    best_columns = [
        "rank",
        "pair",
        "sector",
        "selection_status",
        "formation_pair_pvalue",
        "formation_fdr_qvalue",
        "net_cagr",
        "gross_cagr",
        "sharpe",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "transaction_cost_sum",
    ]
    top_columns = [
        "sector",
        "descriptive_rank_in_sector",
        "pair",
        "dependent",
        "independent",
        "pair_pvalue",
        "fdr_qvalue",
        "half_life_days",
        "hurst_exponent",
    ]
    formation_columns = [
        "sector",
        "selection_rank_in_sector",
        "pair",
        "selection_status",
        "pair_pvalue",
        "fdr_qvalue",
        "half_life_days",
        "beta",
    ]
    quality_frame = pd.DataFrame(
        [{"check": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in quality_checks.items()]
    )
    html_text = template.render(
        **context,
        best_table=_format_frame(best_pairs[[c for c in best_columns if c in best_pairs]], max_rows=10),
        full_top_table=_format_frame(full_top_pairs[[c for c in top_columns if c in full_top_pairs]]),
        formation_table=_format_frame(formation_pairs[[c for c in formation_columns if c in formation_pairs]]),
        portfolio_table=_format_frame(portfolio_metrics),
        cost_table=_format_frame(portfolio_cost_sensitivity),
        regime_table=_format_frame(regime_attribution, max_rows=100),
        quality_table=_format_frame(quality_frame),
        chart_paths=chart_paths,
        dashboards=chart_paths.get("dashboards", []),
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path
