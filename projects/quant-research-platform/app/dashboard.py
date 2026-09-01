"""Internal Streamlit workbench for backtests and market-data maintenance."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from quant_system.backtest.dashboard import (
    comparison_curves,
    discover_runs,
    enrich_equity_curve,
    load_run,
    monthly_return_matrix,
    trade_activity,
)
from quant_system.data.maintenance import (
    build_cleanup_plan,
    bytes_label,
    consolidate_daily_ohlcv,
    execute_cleanup,
    storage_inventory,
)


st.set_page_config(page_title="Quant System 内部工作台", page_icon="📈", layout="wide")
st.title("Quant System 内部回测工作台")
st.caption("研究回测、跨版本比较、市场数据整合与安全缓存治理")


@st.cache_data(ttl=30, show_spinner=False)
def cached_discover_runs(root: str) -> pd.DataFrame:
    return discover_runs(root, limit=200)


@st.cache_data(ttl=30, show_spinner=False)
def cached_load_run(path: str) -> dict[str, object]:
    return load_run(path)


@st.cache_data(ttl=30, show_spinner=False)
def cached_inventory(data_root: str, runs_root: str) -> pd.DataFrame:
    return storage_inventory(data_root, runs_root)


def run_cli(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "quant_system.cli", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def metric_value(metrics: dict[str, object], key: str, *, percent: bool = False) -> str:
    try:
        value = float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return f"{value:.2%}" if percent else f"{value:.2f}"


with st.sidebar:
    st.header("工作区")
    config_path = st.text_input("策略配置", "configs/marketdata.yml")
    runs_root = st.text_input("回测目录", "runs")
    runs_table = cached_discover_runs(runs_root)
    run_options = runs_table["run_id"].tolist() if not runs_table.empty else []
    default_run = "latest" if "latest" in run_options else (run_options[0] if run_options else "")
    selected_run_id = st.selectbox(
        "当前回测",
        run_options,
        index=run_options.index(default_run) if default_run in run_options else 0,
    ) if run_options else ""

    st.header("执行")
    action_cols = st.columns(2)
    run_backtest = action_cols[0].button("运行回测", type="primary", use_container_width=True)
    update_data = action_cols[1].button("更新数据", use_container_width=True)
    build_lake = st.button("构建 Parquet 数据湖", use_container_width=True)
    run_audit = st.button("运行数据质量审计", use_container_width=True)

    if run_backtest or update_data or build_lake or run_audit:
        if run_backtest:
            arguments = ["backtest", "--config", config_path]
        elif update_data:
            arguments = ["update-market-data", "--config", config_path]
        elif build_lake:
            arguments = ["build-lake", "--config", config_path]
        else:
            arguments = ["audit-data", "--config", config_path, "--out", "runs/data_audit"]
        with st.spinner("任务执行中…"):
            completed = run_cli(arguments)
        if completed.returncode == 0:
            st.success("任务完成")
            st.cache_data.clear()
        else:
            st.error(f"任务失败（退出码 {completed.returncode}）")
        st.code(completed.stdout or completed.stderr or "无输出", language="text")


selected_path = ""
if selected_run_id and not runs_table.empty:
    selected_path = str(runs_table.loc[runs_table["run_id"].eq(selected_run_id), "path"].iloc[0])
run = cached_load_run(selected_path) if selected_path else {
    "metrics": {},
    "equity_curve": pd.DataFrame(),
    "trades": pd.DataFrame(),
    "positions": pd.DataFrame(),
    "signals": pd.DataFrame(),
    "period_metrics": pd.DataFrame(),
}
metrics = dict(run.get("metrics", {}))
equity = enrich_equity_curve(run.get("equity_curve", pd.DataFrame()))

overview_tab, compare_tab, trading_tab, data_tab = st.tabs(["回测总览", "版本对比", "交易与持仓", "市场数据中心"])

with overview_tab:
    if not selected_path:
        st.info("尚未找到包含 metrics.json 与 equity_curve.csv 的回测。")
    else:
        st.subheader(selected_run_id)
        metric_cols = st.columns(7)
        metric_specs = [
            ("总收益", "total_return", True),
            ("CAGR", "cagr", True),
            ("Sharpe", "sharpe", False),
            ("Sortino", "sortino", False),
            ("最大回撤", "max_drawdown", True),
            ("胜率", "win_rate", True),
            ("平均换手", "average_turnover", True),
        ]
        for column, (label, key, percent) in zip(metric_cols, metric_specs):
            column.metric(label, metric_value(metrics, key, percent=percent))

        if not equity.empty:
            date_min = equity["date"].min().date()
            date_max = equity["date"].max().date()
            selected_dates = st.slider("观察区间", min_value=date_min, max_value=date_max, value=(date_min, date_max))
            view = equity[equity["date"].dt.date.between(selected_dates[0], selected_dates[1])].copy()
            figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.52, 0.24, 0.24])
            figure.add_trace(go.Scatter(x=view["date"], y=view["normalized_equity"], name="净值（起点=100）", line={"width": 2}), row=1, col=1)
            figure.add_trace(go.Scatter(x=view["date"], y=view["drawdown"], name="回撤", fill="tozeroy", line={"width": 1}), row=2, col=1)
            if "gross_exposure" in view:
                figure.add_trace(go.Scatter(x=view["date"], y=view["gross_exposure"], name="总敞口"), row=3, col=1)
            if "net_exposure" in view:
                figure.add_trace(go.Scatter(x=view["date"], y=view["net_exposure"], name="净敞口"), row=3, col=1)
            figure.update_yaxes(title_text="净值", row=1, col=1)
            figure.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
            figure.update_yaxes(title_text="敞口", tickformat=".0%", row=3, col=1)
            figure.update_layout(height=720, hovermode="x unified", margin={"l": 30, "r": 20, "t": 20, "b": 20}, legend={"orientation": "h"})
            st.plotly_chart(figure, use_container_width=True)

            risk_cols = st.columns(2)
            rolling = px.line(view, x="date", y=["rolling_volatility", "rolling_sharpe"], title="63 日滚动风险")
            rolling.update_layout(hovermode="x unified", legend_title_text="")
            risk_cols[0].plotly_chart(rolling, use_container_width=True)
            matrix = monthly_return_matrix(view)
            if not matrix.empty:
                heatmap = go.Figure(go.Heatmap(
                    z=matrix.values,
                    x=[f"{month}月" for month in matrix.columns],
                    y=matrix.index.astype(str),
                    colorscale="RdYlGn",
                    zmid=0,
                    colorbar={"tickformat": ".0%"},
                    hovertemplate="%{y} %{x}<br>%{z:.2%}<extra></extra>",
                ))
                heatmap.update_layout(title="月度收益", height=380, margin={"l": 30, "r": 20, "t": 45, "b": 20})
                risk_cols[1].plotly_chart(heatmap, use_container_width=True)

        period_metrics = run.get("period_metrics", pd.DataFrame())
        if isinstance(period_metrics, pd.DataFrame) and not period_metrics.empty:
            st.subheader("训练 / 验证 / 前瞻区间")
            st.dataframe(period_metrics, use_container_width=True, hide_index=True)

with compare_tab:
    if runs_table.empty:
        st.info("暂无可比较回测。")
    else:
        default_compare = run_options[: min(3, len(run_options))]
        compare_ids = st.multiselect("选择 2–8 个回测", run_options, default=default_compare, max_selections=8)
        compare_paths = runs_table.set_index("run_id").loc[compare_ids, "path"].tolist() if compare_ids else []
        curves, metric_table = comparison_curves(compare_paths)
        if not curves.empty:
            compare_figure = px.line(curves, x="date", y="normalized_equity", color="run_id", title="标准化净值对比")
            compare_figure.update_layout(height=520, hovermode="x unified", yaxis_title="净值（起点=100）", legend_title_text="回测")
            st.plotly_chart(compare_figure, use_container_width=True)
            dd_figure = px.line(curves, x="date", y="drawdown", color="run_id", title="回撤对比")
            dd_figure.update_yaxes(tickformat=".0%")
            dd_figure.update_layout(height=360, hovermode="x unified", legend_title_text="回测")
            st.plotly_chart(dd_figure, use_container_width=True)
        if not metric_table.empty:
            columns = [column for column in ["run_id", "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "win_rate", "average_turnover"] if column in metric_table]
            st.dataframe(metric_table[columns], use_container_width=True, hide_index=True)

with trading_tab:
    trades = run.get("trades", pd.DataFrame())
    positions = run.get("positions", pd.DataFrame())
    signals = run.get("signals", pd.DataFrame())
    trade_daily = trade_activity(trades) if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    if not trade_daily.empty:
        trade_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        trade_fig.add_trace(go.Bar(x=trade_daily["trade_date"], y=trade_daily["notional"], name="成交名义金额"), row=1, col=1)
        trade_fig.add_trace(go.Scatter(x=trade_daily["trade_date"], y=trade_daily["fees"] + trade_daily["liquidity_cost"], name="手续费+流动性成本"), row=2, col=1)
        trade_fig.update_layout(height=520, hovermode="x unified", legend={"orientation": "h"})
        st.plotly_chart(trade_fig, use_container_width=True)
    table_tabs = st.tabs(["成交", "最新持仓", "信号"])
    with table_tabs[0]:
        st.dataframe(trades, use_container_width=True, hide_index=True)
    with table_tabs[1]:
        if isinstance(positions, pd.DataFrame) and not positions.empty and "date" in positions:
            latest_date = positions["date"].max()
            latest_positions = positions[positions["date"].eq(latest_date)].copy()
            if "market_value" in latest_positions:
                latest_positions["abs_market_value"] = pd.to_numeric(latest_positions["market_value"], errors="coerce").abs()
                latest_positions = latest_positions.sort_values("abs_market_value", ascending=False)
            st.dataframe(latest_positions, use_container_width=True, hide_index=True)
        else:
            st.info("该回测没有持仓明细。")
    with table_tabs[2]:
        st.dataframe(signals, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("存储与数据分层")
    inventory = cached_inventory("data", runs_root)
    total_size = int(inventory["size_bytes"].sum()) if not inventory.empty else 0
    cache_size = int(inventory.loc[inventory["rebuildable_cache"], "size_bytes"].sum()) if not inventory.empty else 0
    inventory_cols = st.columns(3)
    inventory_cols[0].metric("已盘点存储", bytes_label(total_size))
    inventory_cols[1].metric("可重建缓存", bytes_label(cache_size))
    inventory_cols[2].metric("文件数", f"{int(inventory['file_count'].sum()):,}" if not inventory.empty else "0")
    if not inventory.empty:
        inv_fig = px.bar(inventory, x="category", y="size_bytes", color="rebuildable_cache", title="各层存储占用", hover_data=["size", "file_count", "path"])
        inv_fig.update_layout(yaxis_title="字节", xaxis_title="", legend_title_text="可重建")
        st.plotly_chart(inv_fig, use_container_width=True)
        st.dataframe(inventory.drop(columns="size_bytes"), use_container_width=True, hide_index=True)

    integration_col, cleanup_col = st.columns(2)
    with integration_col:
        st.subheader("日线市场数据整合")
        st.caption("按输入顺序合并，后面的来源覆盖同一标的同一交易日；输出经标准 OHLCV 校验并去重。")
        source_text = st.text_area("来源文件（每行一个 CSV/Parquet）", value="data/cache/marketdata_unused.csv")
        consolidated_out = st.text_input("整合输出", "data/processed/daily_ohlcv_consolidated.parquet")
        if st.button("整合并去重", use_container_width=True):
            sources = [line.strip() for line in source_text.splitlines() if line.strip()]
            try:
                with st.spinner("正在整合…"):
                    combined, audit = consolidate_daily_ohlcv(sources, consolidated_out)
                st.success(f"已写入 {len(combined):,} 行：{consolidated_out}")
                st.dataframe(audit, use_container_width=True, hide_index=True)
                st.cache_data.clear()
            except Exception as exc:
                st.error(str(exc))

    with cleanup_col:
        st.subheader("可重建缓存清理")
        keep_signals = st.number_input("保留最新信号缓存", min_value=1, max_value=500, value=20)
        keep_bundles = st.number_input("保留最新研究包缓存", min_value=1, max_value=100, value=5)
        min_age_days = st.number_input("最小缓存年龄（天）", min_value=1, max_value=365, value=14)
        plan = build_cleanup_plan("data/cache", keep_signals=int(keep_signals), keep_bundles=int(keep_bundles), min_age_days=int(min_age_days))
        reclaimable = int(plan["size_bytes"].sum()) if not plan.empty else 0
        st.metric("预计释放", bytes_label(reclaimable), f"{len(plan):,} 个文件")
        st.dataframe(plan[["category", "size", "modified_at", "path"]].head(100), use_container_width=True, hide_index=True)
        confirmation = st.text_input("输入 DELETE CACHE 确认", type="password")
        if st.button("删除计划内缓存", type="primary", disabled=confirmation != "DELETE CACHE" or plan.empty, use_container_width=True):
            result = execute_cleanup(plan, "data/cache", confirmed=True)
            deleted = result["status"].eq("deleted")
            reclaimed = int(result.loc[deleted, "size_bytes"].sum())
            st.success(f"已删除 {int(deleted.sum()):,} 个可重建缓存，释放 {bytes_label(reclaimed)}")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.cache_data.clear()
