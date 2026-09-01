"""Portfolio risk-model snapshots for trading review."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.config import AppConfig


def compute_portfolio_risk_model(
    result,
    config: AppConfig,
    *,
    lookback_days: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return symbol-level and summary risk-model diagnostics."""

    positions = _latest_positions(result)
    prices = _price_returns(result)
    targets = _latest_metadata(result)
    if positions.empty or prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    latest_date = positions["date"].max()
    window = prices[prices["date"].le(latest_date)].copy()
    dates = sorted(window["date"].dropna().unique())[-int(lookback_days) :]
    window = window[window["date"].isin(dates)]
    returns = window.pivot(index="date", columns="symbol", values="return").sort_index()
    symbols = [symbol for symbol in positions["symbol"] if symbol in returns.columns]
    if not symbols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    position_weights = positions.set_index("symbol").loc[symbols, "weight"].astype(float)
    cov = returns[symbols].fillna(0.0).cov()
    weights = position_weights.reindex(cov.index).fillna(0.0)
    variance = float(weights.T @ cov @ weights)
    daily_vol = float(np.sqrt(max(variance, 0.0)))
    annual_vol = daily_vol * np.sqrt(252)
    marginal = cov @ weights
    risk_contribution = weights * marginal / variance if variance > 1e-12 else pd.Series(0.0, index=weights.index)
    benchmark = str(config.universe.benchmark).upper()
    benchmark_returns = returns[benchmark] if benchmark in returns.columns else pd.Series(dtype=float)
    rows = []
    for symbol in symbols:
        ret = returns[symbol].dropna()
        beta = np.nan
        corr = np.nan
        if not benchmark_returns.empty:
            aligned = pd.concat([returns[symbol], benchmark_returns], axis=1).dropna()
            if len(aligned) >= 20 and aligned.iloc[:, 1].var() > 1e-12:
                beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())
                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        rows.append(
            {
                "date": latest_date,
                "symbol": symbol,
                "weight": float(weights.get(symbol, 0.0)),
                "annualized_vol_63d": float(ret.std() * np.sqrt(252)) if len(ret) >= 5 else np.nan,
                "beta_to_benchmark_63d": beta,
                "corr_to_benchmark_63d": corr,
                "risk_contribution_pct": float(risk_contribution.get(symbol, 0.0)),
                "marginal_daily_var": float(marginal.get(symbol, 0.0)),
            }
        )
    detail = pd.DataFrame(rows)
    if not targets.empty:
        detail = detail.merge(targets, on="symbol", how="left")
    summary = _summary(detail, returns, weights, benchmark, annual_vol)
    detail = detail.sort_values("risk_contribution_pct", ascending=False).reset_index(drop=True)
    buckets = _bucket_risk(detail)
    stress = _stress_scenarios(detail, summary, config)
    summary = _attach_risk_flags(summary, detail, buckets, stress, config)
    return detail, summary, buckets, stress


def _latest_positions(result) -> pd.DataFrame:
    positions = result.positions if isinstance(getattr(result, "positions", None), pd.DataFrame) else pd.DataFrame()
    equity = result.equity_curve if isinstance(getattr(result, "equity_curve", None), pd.DataFrame) else pd.DataFrame()
    if positions.empty or equity.empty or not {"date", "symbol", "market_value"}.issubset(positions.columns):
        return pd.DataFrame()
    pos = positions.copy()
    pos["date"] = pd.to_datetime(pos["date"], errors="coerce").dt.normalize()
    pos["symbol"] = pos["symbol"].astype(str).str.upper()
    latest = pos[pos["date"].eq(pos["date"].max())].copy()
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"], errors="coerce").dt.normalize()
    latest_equity = pd.to_numeric(eq[eq["date"].eq(latest["date"].max())]["equity"], errors="coerce")
    denominator = float(latest_equity.iloc[-1]) if not latest_equity.empty else np.nan
    latest["weight"] = pd.to_numeric(latest["market_value"], errors="coerce") / denominator
    return latest[["date", "symbol", "weight"]].dropna(subset=["date", "symbol", "weight"])


def _price_returns(result) -> pd.DataFrame:
    prices = result.prices if isinstance(getattr(result, "prices", None), pd.DataFrame) else pd.DataFrame()
    if prices.empty or not {"date", "symbol"}.issubset(prices.columns):
        return pd.DataFrame()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    data = prices[["date", "symbol", price_col]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data = data.dropna(subset=["date", "symbol", price_col]).sort_values(["symbol", "date"])
    data["return"] = data.groupby("symbol")[price_col].pct_change()
    return data.dropna(subset=["return"])


def _latest_metadata(result) -> pd.DataFrame:
    targets = result.targets if isinstance(getattr(result, "targets", None), pd.DataFrame) else pd.DataFrame()
    if targets.empty or not {"date", "symbol"}.issubset(targets.columns):
        return pd.DataFrame()
    data = targets.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    latest = data[data["date"].eq(data["date"].max())].copy()
    keep = [
        "symbol",
        "primary_theme",
        "sector",
        "industry",
        "final_score",
        "relative_strength_score",
        "fundamental_score",
        "event_risk_score",
        "crowding_risk_score",
        "overnight_gap_risk_score",
        "minute_pretrade_risk_score",
    ]
    return latest[[col for col in keep if col in latest.columns]].drop_duplicates("symbol")


def _summary(detail: pd.DataFrame, returns: pd.DataFrame, weights: pd.Series, benchmark: str, annual_vol: float) -> pd.DataFrame:
    portfolio_returns = returns.reindex(columns=weights.index).fillna(0.0).dot(weights)
    benchmark_returns = returns[benchmark] if benchmark in returns.columns else pd.Series(dtype=float)
    beta = np.nan
    corr = np.nan
    if not benchmark_returns.empty:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) >= 20 and aligned.iloc[:, 1].var() > 1e-12:
            beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    top_risk = detail.sort_values("risk_contribution_pct", ascending=False).head(5)
    return pd.DataFrame(
        [
            {
                "date": detail["date"].max() if "date" in detail else pd.NaT,
                "portfolio_annualized_vol_63d": annual_vol,
                "portfolio_beta_to_benchmark_63d": beta,
                "portfolio_corr_to_benchmark_63d": corr,
                "gross_exposure": float(weights.abs().sum()),
                "net_exposure": float(weights.sum()),
                "top5_risk_contribution_pct": float(top_risk["risk_contribution_pct"].sum()) if not top_risk.empty else np.nan,
                "top_risk_symbols": ", ".join(top_risk["symbol"].astype(str).tolist()) if not top_risk.empty else "",
            }
        ]
    )


def _bucket_risk(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk contribution by theme and industry dimensions."""

    if detail.empty:
        return pd.DataFrame()
    frames = []
    for dimension in ["primary_theme", "sector", "industry"]:
        if dimension not in detail:
            continue
        data = detail.copy()
        data[dimension] = data[dimension].fillna("unknown").astype(str)
        grouped = data.groupby(dimension, as_index=False).agg(
            gross_weight=("weight", lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum())),
            net_weight=("weight", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
            risk_contribution_pct=("risk_contribution_pct", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
            avg_beta_to_benchmark_63d=("beta_to_benchmark_63d", "mean"),
            symbols=("symbol", lambda s: ", ".join(s.astype(str).head(10))),
            symbol_count=("symbol", "nunique"),
        )
        grouped = grouped.rename(columns={dimension: "bucket"})
        grouped.insert(0, "dimension", dimension)
        frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["dimension", "risk_contribution_pct"], ascending=[True, False]).reset_index(drop=True)


def _stress_scenarios(detail: pd.DataFrame, summary: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Estimate simple one-day PnL shocks from current weights and metadata."""

    if detail.empty:
        return pd.DataFrame()
    data = detail.copy()
    data["weight"] = pd.to_numeric(data.get("weight"), errors="coerce").fillna(0.0)
    data["beta_to_benchmark_63d"] = pd.to_numeric(data.get("beta_to_benchmark_63d"), errors="coerce").fillna(1.0)
    data["primary_theme"] = _text_column(data, "primary_theme").str.lower()
    data["sector"] = _text_column(data, "sector").str.lower()
    high_beta_theme = data["primary_theme"].str.contains(
        "ai|semiconductor|optical|data|quantum|power|nuclear|crypto|space", regex=True, na=False
    )
    top_risk_symbol = str(data.sort_values("risk_contribution_pct", ascending=False)["symbol"].iloc[0])
    scenarios = [
        {
            "scenario": "benchmark_down_3pct",
            "description": f"{str(config.universe.benchmark).upper()} 单日 -3%，用63日beta估算",
            "estimated_pnl_pct_equity": float((data["weight"] * data["beta_to_benchmark_63d"] * -0.03).sum()),
            "shock_assumption": "symbol_shock = beta * -3%",
        },
        {
            "scenario": "ai_high_beta_theme_down_8pct",
            "description": "AI/半导体/光模块/电力/量子/高beta主题单日 -8%，其余 -2%",
            "estimated_pnl_pct_equity": float((data["weight"] * np.where(high_beta_theme, -0.08, -0.02)).sum()),
            "shock_assumption": "theme shock by primary_theme keyword",
        },
        {
            "scenario": "rates_up_growth_derating",
            "description": "成长股估值压缩：科技/AI主题 -6%，其余 -2%",
            "estimated_pnl_pct_equity": float(
                (data["weight"] * np.where(high_beta_theme | data["sector"].str.contains("technology|tech", na=False), -0.06, -0.02)).sum()
            ),
            "shock_assumption": "growth derating proxy",
        },
        {
            "scenario": "top_risk_symbol_down_10pct",
            "description": f"最大风险贡献单票 {top_risk_symbol} 单日 -10%",
            "estimated_pnl_pct_equity": float(
                (data["weight"] * np.where(data["symbol"].astype(str).eq(top_risk_symbol), -0.10, 0.0)).sum()
            ),
            "shock_assumption": "single-name idiosyncratic shock",
        },
    ]
    if not summary.empty and "gross_exposure" in summary:
        gross = float(pd.to_numeric(summary["gross_exposure"], errors="coerce").iloc[0])
        scenarios.append(
            {
                "scenario": "all_positions_gap_down_5pct",
                "description": "全部持仓同向隔夜跳空 -5%",
                "estimated_pnl_pct_equity": -0.05 * gross,
                "shock_assumption": "gross exposure * -5%",
            }
        )
    return pd.DataFrame(scenarios).sort_values("estimated_pnl_pct_equity").reset_index(drop=True)


def _text_column(data: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    if column not in data:
        return pd.Series(default, index=data.index, dtype="object")
    return data[column].fillna(default).astype(str)


def _attach_risk_flags(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    buckets: pd.DataFrame,
    stress: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """Add human-readable guardrail flags to the one-row risk summary."""

    if summary.empty:
        return summary
    out = summary.copy()
    flags: list[str] = []
    actions: list[str] = []
    beta = _first_float(out, "portfolio_beta_to_benchmark_63d")
    gross = _first_float(out, "gross_exposure")
    top5 = _first_float(out, "top5_risk_contribution_pct")
    worst_stress = _first_float(stress, "estimated_pnl_pct_equity")
    if pd.notna(beta) and beta > 1.5:
        flags.append(f"benchmark_beta_high:{beta:.2f}")
        actions.append("大盘下跌日避免追高，新增仓位优先等待盘中企稳。")
    if pd.notna(gross) and gross > float(config.portfolio.target_gross_exposure) * 1.05:
        flags.append(f"gross_above_target:{gross:.2f}")
        actions.append("总敞口高于目标，新增开仓前先做换仓或减低优先级仓位。")
    if pd.notna(top5) and top5 > 0.50:
        flags.append(f"top5_risk_concentrated:{top5:.1%}")
        actions.append("前五大风险贡献过高，加仓应优先避开已贡献最大风险的名字。")
    if pd.notna(worst_stress) and worst_stress < -0.05:
        flags.append(f"worst_stress_loss:{worst_stress:.1%}")
        actions.append("压力情景亏损超过5%，盘前需设置分批成交和失败撤退条件。")
    theme_bucket = _largest_bucket(buckets, "primary_theme")
    if theme_bucket is not None and float(theme_bucket.get("risk_contribution_pct", 0.0) or 0.0) > 0.45:
        bucket = str(theme_bucket.get("bucket", "unknown"))
        risk = float(theme_bucket.get("risk_contribution_pct", 0.0) or 0.0)
        flags.append(f"theme_concentration:{bucket}:{risk:.1%}")
        actions.append(f"{bucket} 主题风险贡献偏高，新增同主题仓位需要更强触发或用期权限损。")
    if not flags:
        flags.append("within_guardrails")
        actions.append("组合风险处在当前简化风控阈值内。")
    out["risk_flags"] = " | ".join(flags)
    out["suggested_risk_action"] = " ".join(dict.fromkeys(actions))
    return out


def _first_float(data: pd.DataFrame, column: str) -> float:
    if data.empty or column not in data:
        return np.nan
    series = pd.to_numeric(data[column], errors="coerce").dropna()
    return float(series.iloc[0]) if not series.empty else np.nan


def _largest_bucket(buckets: pd.DataFrame, dimension: str) -> dict | None:
    if buckets.empty or not {"dimension", "risk_contribution_pct"}.issubset(buckets.columns):
        return None
    data = buckets[buckets["dimension"].eq(dimension)].copy()
    if data.empty:
        return None
    data["risk_contribution_pct"] = pd.to_numeric(data["risk_contribution_pct"], errors="coerce")
    data = data.sort_values("risk_contribution_pct", ascending=False)
    return data.iloc[0].to_dict() if not data.empty else None
