"""Investor-facing report panel and daily monitor for quant_system.

This Streamlit app reads the saved run artifacts under runs/ and presents them
as a product-grade performance, risk, attribution, and trading-monitor panel.
It does not place orders.
"""

from __future__ import annotations

import json
import html
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from quant_system.backtest.dashboard import discover_runs, enrich_equity_curve, load_run, monthly_return_matrix


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/marketdata_ai_cycle_max_return.yml"
DEFAULT_RUN_DIR = ROOT / "runs" / "latest"


st.set_page_config(
    page_title="AI Cycle Quant Co-Pilot",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #122033;
            --muted: #687789;
            --line: #d9e1ea;
            --panel: #ffffff;
            --soft: #f5f7fb;
            --accent: #0f766e;
            --warn: #b7791f;
            --bad: #b91c1c;
            --good: #047857;
        }
        .stApp { background: #f5f7fb; color: var(--ink); }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        .hero {
            padding: 22px 24px;
            background: linear-gradient(120deg, #ffffff 0%, #eef7f4 52%, #f8fafc 100%);
            border: 1px solid var(--line);
            border-radius: 8px;
            margin-bottom: 14px;
        }
        .hero-title { font-size: 32px; line-height: 1.12; font-weight: 760; margin: 0; }
        .hero-sub { margin-top: 8px; max-width: 980px; color: var(--muted); font-size: 15px; }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 16px;
        }
        .kpi-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px 13px;
            min-height: 92px;
        }
        .kpi-label { color: var(--muted); font-size: 12px; text-transform: uppercase; }
        .kpi-value { font-size: 24px; font-weight: 760; margin-top: 8px; white-space: nowrap; }
        .kpi-note { color: var(--muted); font-size: 12px; margin-top: 4px; }
        .status-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 16px;
        }
        .status-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px 14px;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 650;
        }
        .badge-good { background: #dcfce7; color: #166534; }
        .badge-warn { background: #fef3c7; color: #92400e; }
        .badge-bad { background: #fee2e2; color: #991b1b; }
        .badge-neutral { background: #e7eef8; color: #334155; }
        .section-note { color: var(--muted); font-size: 13px; margin-top: -6px; margin-bottom: 8px; }
        .callout {
            border-left: 4px solid var(--accent);
            background: #ffffff;
            border-radius: 8px;
            padding: 13px 15px;
            margin: 8px 0 14px;
            border-top: 1px solid var(--line);
            border-right: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
        }
        .risk-callout { border-left-color: var(--warn); }
        .strategy-summary {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            margin: 8px 0 10px;
        }
        .strategy-mandate {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: flex-start;
            padding-bottom: 13px;
            border-bottom: 1px solid var(--line);
        }
        .strategy-kicker { color: var(--accent); font-size: 12px; font-weight: 750; text-transform: uppercase; }
        .strategy-headline { color: var(--ink); font-size: 20px; font-weight: 750; margin-top: 4px; }
        .strategy-objective { color: var(--muted); font-size: 14px; max-width: 820px; margin-top: 5px; }
        .strategy-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-top: 13px;
        }
        .strategy-step { background: var(--soft); border-radius: 6px; padding: 12px 13px; min-height: 130px; }
        .strategy-step-number { color: var(--accent); font-size: 11px; font-weight: 750; }
        .strategy-step-title { color: var(--ink); font-size: 15px; font-weight: 720; margin-top: 5px; }
        .strategy-step-copy { color: var(--muted); font-size: 13px; line-height: 1.45; margin-top: 5px; }
        .strategy-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 11px; }
        .strategy-chip { background: #e7f4f1; color: #115e59; border-radius: 999px; padding: 4px 8px; font-size: 11px; font-weight: 650; }
        .strategy-disclosure { color: var(--muted); font-size: 12px; margin-top: 11px; }
        @media (max-width: 1100px) {
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .status-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .strategy-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 640px) {
            .kpi-grid, .status-row, .strategy-grid { grid-template-columns: 1fr; }
            .strategy-mandate { display: block; }
            .hero-title { font-size: 24px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for column in ("date", "signal_date", "position_date", "next_execution", "month"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
    return df


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def fmt_pct(value: Any, digits: int = 1, signed: bool = True) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:,.{digits}f}"


def fmt_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"${number:,.0f}"


def badge(label: str, status: str = "neutral") -> str:
    status = (status or "neutral").lower()
    klass = {
        "normal": "good",
        "ok": "good",
        "good": "good",
        "healthy": "good",
        "warning": "warn",
        "warn": "warn",
        "high": "bad",
        "critical": "bad",
        "bad": "bad",
        "stale": "bad",
    }.get(status, "neutral")
    return f'<span class="badge badge-{klass}">{label}</span>'


def kpi_card(label: str, value: str, note: str = "") -> str:
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div>'
        "</div>"
    )


def first_existing_run(selected: str) -> Path:
    if selected == "latest":
        return DEFAULT_RUN_DIR
    path = ROOT / "runs" / selected
    return path if path.exists() else DEFAULT_RUN_DIR


def data_freshness(equity: pd.DataFrame, signals: pd.DataFrame) -> tuple[str, str, str]:
    dates: list[pd.Timestamp] = []
    if "date" in equity.columns and not equity.empty:
        dates.append(pd.to_datetime(equity["date"]).max())
    if "signal_date" in signals.columns and not signals.empty:
        dates.append(pd.to_datetime(signals["signal_date"]).max())
    if not dates:
        return "No dated artifacts", "bad", "No equity or signal timestamp was found."
    latest = max(dates)
    age_days = (pd.Timestamp.today(tz=None).normalize() - latest.tz_localize(None).normalize()).days
    if age_days <= 1:
        return latest.strftime("%Y-%m-%d"), "good", "Fresh for daily monitoring."
    if age_days <= 5:
        return latest.strftime("%Y-%m-%d"), "warn", f"Data is {age_days} calendar days old."
    return latest.strftime("%Y-%m-%d"), "stale", f"Data is {age_days} calendar days old. Refresh before using for live decisions."


def equity_drawdown_chart(curve: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=curve["date"],
            y=curve["normalized_equity"],
            mode="lines",
            name="Net asset value, indexed to 100",
            line=dict(color="#0f766e", width=2.4),
            hovertemplate="%{x|%Y-%m-%d}<br>NAV %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=curve["date"],
            y=curve["drawdown"] * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#b91c1c", width=1.2),
            hovertemplate="%{x|%Y-%m-%d}<br>DD %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Indexed NAV", secondary_y=False, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="Drawdown %", secondary_y=True, range=[min(-5, curve["drawdown"].min() * 120), 5])
    return fig


def exposure_chart(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, color in (("gross_exposure", "#2563eb"), ("net_exposure", "#0f766e"), ("turnover", "#b7791f")):
        if column in curve.columns:
            fig.add_trace(
                go.Scatter(
                    x=curve["date"],
                    y=pd.to_numeric(curve[column], errors="coerce"),
                    name=column.replace("_", " ").title(),
                    mode="lines",
                    line=dict(width=1.8, color=color),
                )
            )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#e5e7eb")
    return fig


def monthly_heatmap(equity: pd.DataFrame) -> go.Figure | None:
    matrix = monthly_return_matrix(equity)
    if matrix.empty:
        return None
    z = matrix.values * 100.0
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            y=matrix.index.astype(str),
            colorscale=[[0, "#b91c1c"], [0.5, "#f8fafc"], [1, "#047857"]],
            zmid=0,
            text=np.round(z, 1),
            texttemplate="%{text}%",
            hovertemplate="%{y} %{x}<br>%{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="white")
    return fig


def attribution_bar(df: pd.DataFrame, title: str, top_n: int = 10) -> go.Figure | None:
    if df.empty or not {"bucket", "total_contribution"}.issubset(df.columns):
        return None
    data = df.copy()
    data["total_contribution"] = pd.to_numeric(data["total_contribution"], errors="coerce")
    data = data.dropna(subset=["total_contribution"]).sort_values("total_contribution", ascending=False).head(top_n)
    if data.empty:
        return None
    fig = px.bar(
        data,
        x="total_contribution",
        y="bucket",
        orientation="h",
        color="total_contribution",
        color_continuous_scale=["#b91c1c", "#f8fafc", "#047857"],
        title=title,
        hover_data=[c for c in ("hit_rate", "avg_gross_weight", "symbols") if c in data.columns],
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10), yaxis=dict(autorange="reversed"), paper_bgcolor="white", plot_bgcolor="white")
    fig.update_coloraxes(showscale=False)
    return fig


def run_cli(args: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "quant_system.cli", *args],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            timeout=900,
        )
    except Exception as exc:  # pragma: no cover - UI guardrail
        return False, str(exc)
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return completed.returncode == 0, output.strip()


def display_dataframe(df: pd.DataFrame, columns: list[str], height: int = 360) -> None:
    if df.empty:
        st.info("No artifact available for this table.")
        return
    existing = [c for c in columns if c in df.columns]
    data = df[existing].copy() if existing else df.copy()
    st.dataframe(data, use_container_width=True, height=height, hide_index=True)


def build_status_cards(risk_state: pd.DataFrame, freshness_label: str, freshness_status: str, freshness_note: str) -> str:
    cards = [
        '<div class="status-card">'
        '<div class="kpi-label">Data Freshness</div>'
        f'<div style="margin-top:8px">{badge(freshness_label, freshness_status)}</div>'
        f'<div class="kpi-note">{freshness_note}</div>'
        "</div>"
    ]
    if not risk_state.empty and {"label", "value", "status"}.issubset(risk_state.columns):
        for row in risk_state.head(3).itertuples(index=False):
            cards.append(
                '<div class="status-card">'
                f'<div class="kpi-label">{getattr(row, "label")}</div>'
                f'<div style="margin-top:8px">{badge(str(getattr(row, "value")), str(getattr(row, "status")))}</div>'
                '<div class="kpi-note">Current portfolio monitor</div>'
                "</div>"
            )
    return '<div class="status-row">' + "".join(cards[:4]) + "</div>"


def _config_pct(value: Any, default: float = 0.0) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return f"{default:.0%}"


def build_strategy_intro(config: dict[str, Any]) -> str:
    universe = config.get("universe", {})
    execution = config.get("execution", {})
    portfolio = config.get("portfolio", {})
    risk = config.get("risk", {})
    regime = config.get("regime", {})
    strategies = config.get("strategies", {})
    live = config.get("live", {})

    symbols = universe.get("custom_symbols") or universe.get("symbols") or []
    universe_name = str(universe.get("base", "configured universe")).replace("_", " ").title()
    enabled = []
    for key, label in (("momentum", "Momentum"), ("breakout", "Breakout + Retest"), ("trend", "Trend"), ("mean_reversion", "Mean Reversion")):
        settings = strategies.get(key, {})
        if settings.get("enabled", False) and float(settings.get("weight", 0.0) or 0.0) > 0:
            enabled.append(f"{label} {_config_pct(settings.get('weight'))}")
    strategy_chips = "".join(f'<span class="strategy-chip">{html.escape(item)}</span>' for item in enabled)

    momentum = strategies.get("momentum", {})
    gross = float(portfolio.get("target_gross_exposure", 0.0) or 0.0)
    net = float(portfolio.get("target_net_exposure", 0.0) or 0.0)
    short_quantile = float(momentum.get("short_quantile", 0.0) or 0.0)
    mandate = "Aggressive long-biased trend mandate" if short_quantile <= 0 and net > 0 else "Long/short trend mandate"
    if abs(net) <= 0.1 and short_quantile > 0:
        mandate = "Market-neutral long/short mandate"

    universe_copy = (
        f"Start with {len(symbols):,} configured names from {universe_name}, then require price, liquidity, history, "
        "data-quality, and tradability checks on each date. Theme baskets guide ranking; they do not replace the base universe."
    )
    signal_copy = (
        f"Rank leaders using {int(momentum.get('lookback_returns', 21))}-day momentum with a "
        f"{int(momentum.get('skip_recent_days', 3))}-day skip, relative strength, MA trend, breakout/retest quality, "
        "volume, theme breadth, fundamentals, and event risk."
    )
    portfolio_copy = (
        f"Use {str(portfolio.get('construction', 'score_weighted')).replace('_', ' ')} sizing, up to "
        f"{_config_pct(portfolio.get('max_position_weight', 0.10))} per stock and {int(portfolio.get('max_total_positions', 0) or 0)} positions. "
        f"Current targets are {gross:.1f}x gross / {net:.1f}x net."
    )
    execution_copy = (
        f"Signals form after the close and simulate execution at the next open. Rebalancing is "
        f"{str(portfolio.get('rebalance', 'daily')).replace('_', ' ')}, with {float(execution.get('slippage_bps', 0) or 0):.0f} bps "
        "slippage plus commissions, spread, market impact, ADV capacity, and turnover limits."
    )
    protection_copy = (
        f"Regime, overbought, crowding, overnight-gap, and minute-risk overlays can reduce exposure. ATR/trailing stops apply; "
        f"portfolio drawdown controls reduce risk at {_config_pct(risk.get('max_drawdown_reduce_exposure', 0.18))} and target cash mode at "
        f"{_config_pct(risk.get('max_drawdown_cash_mode', 0.30))}."
    )
    live_status = "paper/manual approval" if live.get("require_manual_approval", True) else str(live.get("mode", "paper"))

    return (
        '<div class="strategy-summary">'
        '<div class="strategy-mandate"><div>'
        '<div class="strategy-kicker">Strategy at a glance</div>'
        f'<div class="strategy-headline">{html.escape(mandate)}</div>'
        '<div class="strategy-objective">Seek absolute return by owning persistent technology leadership when price trend, relative strength, '
        'theme participation, and risk conditions agree. The model prefers controlled pullbacks and confirmed continuation over indiscriminate chasing.</div>'
        f'</div><div class="strategy-chips">{strategy_chips}</div></div>'
        '<div class="strategy-grid">'
        f'<div class="strategy-step"><div class="strategy-step-number">01</div><div class="strategy-step-title">Define the opportunity set</div><div class="strategy-step-copy">{html.escape(universe_copy)}</div></div>'
        f'<div class="strategy-step"><div class="strategy-step-number">02</div><div class="strategy-step-title">Find confirmed leadership</div><div class="strategy-step-copy">{html.escape(signal_copy)}</div></div>'
        f'<div class="strategy-step"><div class="strategy-step-number">03</div><div class="strategy-step-title">Size the portfolio</div><div class="strategy-step-copy">{html.escape(portfolio_copy)}</div></div>'
        f'<div class="strategy-step"><div class="strategy-step-number">04</div><div class="strategy-step-title">Execute and protect</div><div class="strategy-step-copy">{html.escape(execution_copy)} {html.escape(protection_copy)}</div></div>'
        '</div>'
        f'<div class="strategy-disclosure">Current implementation: {html.escape(live_status)}; no autonomous order submission. '
        'This configuration is long-biased and does not demonstrate market-neutral or short-side alpha.</div>'
        '</div>'
    )


def main() -> None:
    inject_css()

    runs = discover_runs(ROOT / "runs", limit=200)
    run_options = ["latest"]
    if not runs.empty:
        run_options.extend([r for r in runs["run_id"].astype(str).tolist() if r != "latest"])

    with st.sidebar:
        st.header("Control Center")
        selected_run = st.selectbox("Report run", run_options, index=0)
        config_path = st.text_input("Config", DEFAULT_CONFIG)
        view_mode = st.radio("Mode", ["Investor", "Monitor", "Research"], horizontal=True, index=0)
        st.divider()
        if st.button("Refresh latest trading signal", use_container_width=True):
            ok, output = run_cli(["today", "--config", config_path])
            if ok:
                st.success("Signal refresh finished.")
            else:
                st.error("Signal refresh failed.")
            with st.expander("Command output", expanded=not ok):
                st.code(output or "(no output)")
        if st.button("Run backtest report", use_container_width=True):
            ok, output = run_cli(["backtest", "--config", config_path])
            if ok:
                st.success("Backtest finished. Reload the run selector.")
            else:
                st.error("Backtest failed.")
            with st.expander("Command output", expanded=not ok):
                st.code(output or "(no output)")
        auto_refresh = st.toggle("Auto-refresh monitor", value=False)
        refresh_seconds = st.number_input("Refresh seconds", min_value=30, max_value=3600, value=180, step=30)
        st.caption("Panel is read-only except explicit CLI refresh buttons. It never submits broker orders.")

    run_dir = first_existing_run(selected_run)
    selected_config_path = run_dir / "config.yml"
    if not selected_config_path.exists():
        selected_config_path = ROOT / config_path
    strategy_config = read_yaml(selected_config_path)
    loaded = load_run(run_dir)
    metrics = dict(loaded.get("metrics", {}))
    equity = enrich_equity_curve(loaded.get("equity_curve", pd.DataFrame()))
    raw_equity = loaded.get("equity_curve", pd.DataFrame())
    signals = read_csv(run_dir / "latest_manual_trading_signals.csv")
    premarket = read_csv(run_dir / "premarket_plan.csv")
    risk_state = read_csv(run_dir / "risk_state.csv")
    theme_div = read_csv(run_dir / "theme_divergence.csv")
    crowding = read_csv(run_dir / "crowding_watch.csv")
    quality = read_csv(run_dir / "daily_data_quality_summary.csv")
    stress = read_csv(run_dir / "portfolio_risk_stress_scenarios.csv")
    risk_summary = read_csv(run_dir / "portfolio_risk_model_summary.csv")
    risk_buckets = read_csv(run_dir / "portfolio_risk_model_buckets.csv")
    options_risk = read_csv(run_dir / "options_risk.csv")
    option_recs = read_csv(run_dir / "options_overlay_recommendations.csv")
    yearly = read_csv(run_dir / "yearly_breakdown.csv")
    monthly = read_csv(run_dir / "monthly_breakdown.csv")
    attr_theme = read_csv(run_dir / "institutional_attribution_primary_theme.csv")
    attr_strategy = read_csv(run_dir / "institutional_attribution_strategy_family.csv")
    attr_symbol = read_csv(run_dir / "institutional_attribution_symbol.csv")
    attr_factor = read_csv(run_dir / "institutional_attribution_factor_quintile.csv")
    audit = read_json(run_dir / "research_audit.json")

    freshness_label, freshness_status, freshness_note = data_freshness(raw_equity, signals)

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-title">AI Cycle Quant Co-Pilot</div>
          <div class="hero-sub">
            A deterministic momentum and trend-following research system with explainable signals, explicit risk controls,
            attribution, and daily trading monitor. AI is used as reviewer and reporting assistant, not as an autonomous order sender.
            Current run: <b>{run_dir.name}</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpis = [
        kpi_card("Total Return", fmt_pct(metrics.get("total_return")), "Backtest, not a promise"),
        kpi_card("Trailing 1Y", fmt_pct(metrics.get("trailing_one_year_return")), "Recent regime"),
        kpi_card("CAGR", fmt_pct(metrics.get("cagr")), "Annualized"),
        kpi_card("Sharpe", fmt_num(metrics.get("sharpe")), "Cost-adjusted backtest"),
        kpi_card("Max Drawdown", fmt_pct(metrics.get("max_drawdown")), "Peak to trough"),
        kpi_card("Avg Gross", fmt_pct(metrics.get("average_gross_exposure")), "Exposure used"),
    ]
    st.markdown('<div class="kpi-grid">' + "".join(kpis) + "</div>", unsafe_allow_html=True)
    st.markdown(build_status_cards(risk_state, freshness_label, freshness_status, freshness_note), unsafe_allow_html=True)

    if freshness_status in {"stale", "bad"}:
        st.warning("Monitoring data is stale. Use the refresh buttons or run the MarketData update before treating signals as current.")

    if view_mode == "Investor":
        st.markdown(build_strategy_intro(strategy_config), unsafe_allow_html=True)
        st.subheader("Performance Story")
        st.markdown(
            """
            <div class="callout">
            The panel emphasizes three investor questions: whether the strategy has captured the AI-cycle trend,
            whether the risk taken is visible and controlled, and whether each trade can be explained after costs.
            Backtest results are research evidence only; they are not a guarantee of future returns.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not equity.empty:
            st.plotly_chart(equity_drawdown_chart(equity), use_container_width=True)
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            fig = attribution_bar(attr_theme, "Contribution by Theme")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = attribution_bar(attr_strategy, "Contribution by Strategy Layer", top_n=8)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk and Capacity")
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            if not equity.empty:
                st.plotly_chart(exposure_chart(equity), use_container_width=True)
        with col2:
            display_dataframe(
                stress,
                ["scenario", "estimated_pnl_pct_equity", "description", "shock_assumption"],
                height=300,
            )

        st.subheader("Year and Month Breakdown")
        col1, col2 = st.columns([1.0, 1.0])
        with col1:
            display_dataframe(yearly, ["year", "return", "max_drawdown", "volatility", "sharpe", "avg_gross_exposure", "avg_turnover"], height=300)
        with col2:
            fig = monthly_heatmap(raw_equity)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_dataframe(monthly, ["month", "return", "max_drawdown", "avg_gross_exposure", "avg_turnover"], height=300)

    elif view_mode == "Monitor":
        st.subheader("Today Trading Desk")
        st.markdown('<div class="section-note">Designed for pre-market and intraday review. It is advisory and read-only.</div>', unsafe_allow_html=True)
        action_source = premarket if not premarket.empty else signals
        if not action_source.empty and "estimated_trade_notional" in action_source.columns:
            buys = action_source[pd.to_numeric(action_source["estimated_trade_notional"], errors="coerce") > 0]
            sells = action_source[pd.to_numeric(action_source["estimated_trade_notional"], errors="coerce") < 0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Buy/Add Notional", fmt_money(buys["estimated_trade_notional"].sum()))
            c2.metric("Reduce/Sell Notional", fmt_money(sells["estimated_trade_notional"].sum()))
            c3.metric("Actions", f"{len(action_source):,}")
            c4.metric("Blocked", f"{int(action_source.get('blocked_from_trading', pd.Series(dtype=bool)).fillna(False).sum()):,}" if "blocked_from_trading" in action_source.columns else "0")
        display_dataframe(
            action_source,
            [
                "signal_date",
                "symbol",
                "action_label",
                "target_weight",
                "current_weight",
                "delta_weight",
                "estimated_trade_notional",
                "close",
                "final_score",
                "technical_score",
                "relative_strength_score",
                "fundamental_score",
                "event_risk_score",
                "primary_theme",
                "status",
                "execution_condition",
                "risk_instruction",
                "decision_note",
            ],
            height=500,
        )

        st.subheader("Risk Monitor")
        col1, col2 = st.columns([1.0, 1.0])
        with col1:
            display_dataframe(risk_state, ["risk_key", "label", "value", "status"], height=220)
            display_dataframe(quality, ["status_item", "symbol_count", "symbol_pct"], height=220)
        with col2:
            display_dataframe(risk_summary, ["date", "portfolio_annualized_vol_63d", "portfolio_beta_to_benchmark_63d", "gross_exposure", "net_exposure", "top5_risk_contribution_pct", "top_risk_symbols", "risk_flags", "suggested_risk_action"], height=460)

        st.subheader("Crowding, Theme Rotation, and Options Overlay")
        tab_a, tab_b, tab_c = st.tabs(["Theme rotation", "Risk buckets", "Options"])
        with tab_a:
            display_dataframe(
                theme_div if not theme_div.empty else crowding,
                ["date", "level", "bucket", "target_gross", "target_gross_change", "avg_final_score", "avg_relative_strength_score", "leaders", "divergence_status", "suggested_tilt"],
                height=420,
            )
        with tab_b:
            display_dataframe(risk_buckets, ["dimension", "bucket", "gross_weight", "net_weight", "risk_contribution_pct", "avg_beta_to_benchmark_63d", "symbols", "symbol_count"], height=420)
        with tab_c:
            display_dataframe(
                option_recs if not option_recs.empty else options_risk,
                ["symbol", "equity_action", "option_structure", "target_dte_min", "target_dte_max", "target_delta", "max_premium_budget", "executable", "chain_validation_status", "overlay_note"],
                height=420,
            )

    else:
        st.subheader("Research Diagnostics")
        st.markdown(
            """
            <div class="risk-callout callout">
            This view is for model review: attribution stability, data quality, and overfit warnings matter more than making the
            backtest look smooth. If a result looks too good, audit the data freshness, universe construction, and execution assumptions first.
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([1.0, 1.0])
        with col1:
            fig = attribution_bar(attr_symbol, "Top Symbol Contributions", top_n=15)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if not attr_factor.empty:
                st.dataframe(attr_factor, use_container_width=True, height=330, hide_index=True)
            else:
                st.info("No factor quintile attribution artifact available.")
        st.subheader("Audit Snapshot")
        if audit:
            st.json(audit, expanded=False)
        else:
            st.info("No research audit JSON found for this run.")

    st.caption(
        f"Artifacts: {run_dir}. This panel is for research, client communication, and monitoring. It does not submit live orders."
    )

    if auto_refresh:
        time.sleep(int(refresh_seconds))
        st.rerun()


if __name__ == "__main__":
    main()
