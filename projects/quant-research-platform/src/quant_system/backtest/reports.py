"""Report writers for backtest runs."""

from __future__ import annotations

from html import escape
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from quant_marketdata import MarketDataStore

from quant_system.backtest.attribution import compute_attribution_tables, factor_correlation_table
from quant_system.backtest.institutional_attribution import compute_institutional_attribution, compute_period_institutional_attribution
from quant_system.backtest.metrics import equity_metrics, monthly_returns
from quant_system.config import AppConfig, dump_config
from quant_system.data.quality_monitor import run_daily_data_quality_monitor
from quant_system.options.chain import CSVOptionChainProvider, MarketDataOptionChainProvider, validate_overlay_against_chain
from quant_system.options.overlay import build_options_overlay_recommendations
from quant_system.portfolio.risk_model import compute_portfolio_risk_model
from quant_system.regime.reports import period_metrics
from quant_system.trading.desk import build_trading_desk_tables, write_trading_desk_artifacts
from quant_system.universe.thematic import load_thematic_baskets


def save_run(
    result,
    config: AppConfig,
    run_dir: str | Path,
    warnings: list[str] | None = None,
    *,
    full_report: bool = True,
    write_heavy_artifacts: bool = True,
) -> dict[str, float]:
    """Save all run artifacts and an HTML report."""

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(config, out / "config.yml")
    metrics = equity_metrics(result.equity_curve)
    data_provenance = build_data_provenance(config, result)
    (out / "data_provenance.json").write_text(
        json.dumps(data_provenance, indent=2, default=str),
        encoding="utf-8",
    )
    for field in ("data_provider", "data_max_date", "data_manifest_sha256"):
        if data_provenance.get(field) not in {None, ""}:
            metrics[field] = data_provenance[field]
    result.equity_curve.to_csv(out / "equity_curve.csv", index=False)
    if write_heavy_artifacts:
        result.positions.to_csv(out / "positions.csv", index=False)
        result.trades.to_csv(out / "trades.csv", index=False)
        result.signals.to_csv(out / "signals.csv", index=False)
    targets = result.targets if isinstance(getattr(result, "targets", None), pd.DataFrame) else pd.DataFrame()
    if write_heavy_artifacts:
        targets.to_csv(out / "target_weights.csv", index=False)
    risk_overlay = result.risk_overlay if isinstance(getattr(result, "risk_overlay", None), pd.DataFrame) else pd.DataFrame()
    if write_heavy_artifacts:
        risk_overlay.to_csv(out / "risk_overlay.csv", index=False)
    overlay_activation = compute_overlay_activation_summary(
        result.signals if isinstance(getattr(result, "signals", None), pd.DataFrame) else pd.DataFrame()
    )
    overlay_activation.to_csv(out / "overlay_activation_summary.csv", index=False)
    manual_blotter = latest_manual_trading_blotter(result)
    manual_blotter.to_csv(out / "latest_manual_trading_signals.csv", index=False)
    latest_equity = (
        float(result.equity_curve["equity"].iloc[-1])
        if isinstance(getattr(result, "equity_curve", None), pd.DataFrame) and not result.equity_curve.empty and "equity" in result.equity_curve
        else float(config.portfolio.initial_capital)
    )
    options_overlay = build_options_overlay_recommendations(manual_blotter, latest_equity, config.options_overlay)
    options_overlay = validate_options_overlay_with_chain(options_overlay, config, out)
    options_overlay.to_csv(out / "options_overlay_recommendations.csv", index=False)
    theme_divergence = theme_divergence_report(result, manual_blotter, config.universe.theme_baskets_path)
    theme_divergence.to_csv(out / "theme_divergence.csv", index=False)
    data_quality_detail, data_quality_summary = run_daily_data_quality_monitor(config)
    data_quality_detail.to_csv(out / "daily_data_quality_detail.csv", index=False)
    data_quality_summary.to_csv(out / "daily_data_quality_summary.csv", index=False)
    trading_desk_tables = build_trading_desk_tables(manual_blotter, risk_overlay, theme_divergence, options_overlay, data_quality_detail)
    write_trading_desk_artifacts(trading_desk_tables, out)
    portfolio_risk_detail, portfolio_risk_summary, portfolio_risk_buckets, portfolio_risk_stress = compute_portfolio_risk_model(result, config)
    portfolio_risk_detail.to_csv(out / "portfolio_risk_model_detail.csv", index=False)
    portfolio_risk_summary.to_csv(out / "portfolio_risk_model_summary.csv", index=False)
    portfolio_risk_buckets.to_csv(out / "portfolio_risk_model_buckets.csv", index=False)
    portfolio_risk_stress.to_csv(out / "portfolio_risk_stress_scenarios.csv", index=False)
    periods = {
        "train": (config.regime.train_start, config.regime.train_end),
        "validation": (config.regime.validation_start, config.regime.validation_end),
        "forward_current": (config.regime.forward_start, None),
        "ai_cycle": (config.regime.validation_start, None),
    }
    institutional_attribution = compute_institutional_attribution(result)
    for name, frame in institutional_attribution.items():
        frame.to_csv(out / f"institutional_attribution_{name}.csv", index=False)
    period_institutional_attribution = compute_period_institutional_attribution(result, periods)
    for name, frame in period_institutional_attribution.items():
        frame.to_csv(out / f"institutional_attribution_{name}_by_period.csv", index=False)
    monthly_returns(result.equity_curve).to_csv(out / "monthly_returns.csv", index=False)
    yearly_breakdown, monthly_breakdown = performance_breakdowns(result.equity_curve)
    yearly_breakdown.to_csv(out / "yearly_breakdown.csv", index=False)
    monthly_breakdown.to_csv(out / "monthly_breakdown.csv", index=False)
    period_table = period_metrics(result.equity_curve, periods)
    period_table.to_csv(out / "period_metrics.csv", index=False)
    research_audit = build_research_audit_summary(config, warnings or [], metrics, period_table, result)
    metrics = augment_metrics_with_research_audit(metrics, period_table, research_audit)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "research_audit.json").write_text(json.dumps(research_audit, indent=2), encoding="utf-8")
    exposure_decomposition = exposure_mode_decomposition(result, metrics, research_audit)
    exposure_decomposition.to_csv(out / "exposure_mode_decomposition.csv", index=False)
    if not full_report:
        (out / "report.html").write_text(
            render_fast_html_report(
                metrics,
                warnings or [],
                period_table,
                manual_blotter,
                options_overlay,
                data_quality_summary,
                data_quality_detail,
                trading_desk_tables,
                institutional_attribution,
                exposure_decomposition,
                portfolio_risk_detail,
                portfolio_risk_summary,
                portfolio_risk_buckets,
                portfolio_risk_stress,
            ),
            encoding="utf-8",
        )
        return metrics
    prices = result.prices if isinstance(getattr(result, "prices", None), pd.DataFrame) else pd.DataFrame()
    factor_ic, family_ic = compute_attribution_tables(
        result.signals,
        prices,
        train_start=config.regime.train_start,
        train_end=config.regime.train_end,
        validation_start=config.regime.validation_start,
        validation_end=config.regime.validation_end,
        forward_start=config.regime.forward_start,
    )
    factor_ic.to_csv(out / "factor_ic_attribution.csv", index=False)
    family_ic.to_csv(out / "strategy_family_ic_attribution.csv", index=False)
    factor_corr = factor_correlation_table(result.signals)
    factor_corr.to_csv(out / "factor_correlation.csv", index=False)
    drawdown_periods, drawdown_contributors = drawdown_component_analysis(result)
    drawdown_periods.to_csv(out / "drawdown_periods.csv", index=False)
    drawdown_contributors.to_csv(out / "drawdown_contributors.csv", index=False)
    charts = save_report_charts(result, out)
    manual_chart = save_manual_trading_chart(manual_blotter, out)
    if manual_chart:
        charts["manual_trading"] = manual_chart
    (out / "report.html").write_text(
        render_html_report(
            metrics,
            result,
            warnings or [],
            period_table,
            charts,
            manual_blotter,
            factor_ic,
            family_ic,
            factor_corr,
            yearly_breakdown,
            monthly_breakdown,
            drawdown_periods,
            drawdown_contributors,
            theme_divergence,
            options_overlay,
            institutional_attribution,
            exposure_decomposition,
            trading_desk_tables,
            data_quality_summary,
            data_quality_detail,
            portfolio_risk_detail,
            portfolio_risk_summary,
            portfolio_risk_buckets,
            portfolio_risk_stress,
        ),
        encoding="utf-8",
    )
    return metrics


def build_data_provenance(config: AppConfig, result) -> dict[str, object]:
    """Capture the exact formal input identity without copying vendor data."""

    prices = result.prices if isinstance(getattr(result, "prices", None), pd.DataFrame) else pd.DataFrame()
    dates = pd.to_datetime(prices.get("date"), errors="coerce") if "date" in prices else pd.Series(dtype="datetime64[ns]")
    sources = sorted(prices.get("source", pd.Series(dtype="string")).dropna().astype(str).unique().tolist())
    finalities = sorted(prices.get("finality", pd.Series(dtype="string")).dropna().astype(str).unique().tolist())
    payload: dict[str, object] = {
        "schema_version": "1.1",
        "data_provider": config.data_provider,
        "timeframe": config.timeframe,
        "rows": int(len(prices)),
        "symbols": int(prices["symbol"].nunique()) if "symbol" in prices else 0,
        "data_min_date": dates.min().isoformat() if dates.notna().any() else None,
        "data_max_date": dates.max().isoformat() if dates.notna().any() else None,
        "sources": sources,
        "finalities": finalities,
        "data_manifest_sha256": None,
        "manifest_snapshot": None,
        "reference_data_manifest_sha256": None,
        "reference_data_snapshots": [],
    }
    reference_snapshots = prices.attrs.get("reference_data_snapshots")
    if isinstance(reference_snapshots, list) and all(isinstance(item, dict) for item in reference_snapshots):
        normalized = sorted(reference_snapshots, key=lambda item: str(item.get("cache_key", "")))
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        payload["reference_data_snapshots"] = normalized
        payload["reference_data_manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    if config.data_provider == "marketdata":
        captured_snapshot = prices.attrs.get("marketdata_manifest_snapshot")
        captured_sha256 = prices.attrs.get("marketdata_manifest_sha256")
        if isinstance(captured_snapshot, dict) and isinstance(captured_sha256, str):
            payload["data_manifest_sha256"] = captured_sha256
            payload["manifest_snapshot"] = captured_snapshot
            payload["manifest_capture"] = "atomic_with_price_read"
        else:
            manifest_path = MarketDataStore().root / "manifests" / "marketdata" / "confirmed.json"
            if manifest_path.is_file():
                raw = manifest_path.read_bytes()
                payload["data_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
                payload["manifest_snapshot"] = json.loads(raw.decode("utf-8"))
                payload["manifest_capture"] = "save_time_fallback"
    else:
        source_path = Path(config.data_path)
        if source_path.is_file():
            payload["input_file"] = source_path.name
            payload["input_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return payload


def build_research_audit_summary(
    config: AppConfig,
    warnings: list[str],
    metrics: dict[str, float],
    period_table: pd.DataFrame,
    result,
) -> dict[str, object]:
    """Summarize research-bias and exposure-mode diagnostics for the report."""

    warning_text = " ".join(str(item).lower() for item in warnings)
    full_exploratory = bool(
        config.universe.custom_only
        or "selection bias" in warning_text
        or "survivorship bias" in warning_text
        or bool(config.universe.base_symbols_path)
        or not bool(config.universe.strict_backtest)
    )
    headline_period = "forward_current" if _period_row(period_table, "forward_current") else "validation"
    avg_gross = float(metrics.get("average_gross_exposure", 0.0) or 0.0)
    avg_net = float(metrics.get("average_net_exposure", 0.0) or 0.0)
    short_contribution = float(metrics.get("short_contribution", 0.0) or 0.0)
    if avg_gross > 0 and abs(avg_net) / avg_gross >= 0.85 and abs(short_contribution) < max(1.0, abs(float(metrics.get("long_contribution", 0.0) or 0.0)) * 0.01):
        exposure_mode = "long_only_beta_tilt"
    elif avg_gross > 0 and abs(avg_net) / avg_gross <= 0.20:
        exposure_mode = "market_neutral_like"
    else:
        exposure_mode = "directional_long_short"
    return {
        "strict_backtest": bool(config.universe.strict_backtest),
        "custom_only": bool(config.universe.custom_only),
        "base": config.universe.base,
        "base_symbols_path": config.universe.base_symbols_path,
        "full_period_metrics_are_exploratory": full_exploratory,
        "headline_metric_scope": headline_period,
        "headline_metric_rule": "show forward_current when available, otherwise validation; full-period metrics are retained below as diagnostics",
        "observed_exposure_mode": exposure_mode,
        "target_gross_exposure": float(config.portfolio.target_gross_exposure),
        "target_net_exposure": float(config.portfolio.target_net_exposure),
        "average_gross_exposure": avg_gross,
        "average_net_exposure": avg_net,
        "long_contribution": float(metrics.get("long_contribution", 0.0) or 0.0),
        "short_contribution": short_contribution,
    }


def augment_metrics_with_research_audit(metrics: dict[str, float], period_table: pd.DataFrame, audit: dict[str, object]) -> dict[str, float]:
    """Add explicit headline/OOS and full-period metric namespaces."""

    out: dict[str, object] = dict(metrics)
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            out[f"full_period_{key}"] = float(value)
    headline_period = str(audit.get("headline_metric_scope", "validation"))
    row = _period_row(period_table, headline_period) or _period_row(period_table, "validation")
    if row:
        for key, value in row.items():
            if key == "period":
                continue
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                out[f"headline_{key}"] = float(value)
                out[f"{headline_period}_{key}"] = float(value)
    validation = _period_row(period_table, "validation")
    forward = _period_row(period_table, "forward_current")
    for prefix, source in (("validation", validation), ("forward_current", forward)):
        if source:
            for key, value in source.items():
                if key == "period":
                    continue
                if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                    out[f"{prefix}_{key}"] = float(value)
    out["headline_metric_scope"] = headline_period
    out["full_period_metrics_are_exploratory"] = bool(audit.get("full_period_metrics_are_exploratory", False))
    out["observed_exposure_mode"] = str(audit.get("observed_exposure_mode", "unknown"))
    return out  # type: ignore[return-value]


def exposure_mode_decomposition(result, metrics: dict[str, float], audit: dict[str, object]) -> pd.DataFrame:
    """Write a compact long-only / long-short / market-neutral diagnostic table."""

    rows = [
        {
            "mode": audit.get("observed_exposure_mode", "unknown"),
            "avg_gross_exposure": metrics.get("average_gross_exposure", np.nan),
            "avg_net_exposure": metrics.get("average_net_exposure", np.nan),
            "long_contribution": metrics.get("long_contribution", np.nan),
            "short_contribution": metrics.get("short_contribution", np.nan),
            "interpretation": "Observed realized exposure of this backtest; it is not proof of market-neutral alpha unless short contribution and near-zero net exposure are present.",
        },
        {
            "mode": "market_neutral_reference",
            "avg_gross_exposure": np.nan,
            "avg_net_exposure": 0.0,
            "long_contribution": np.nan,
            "short_contribution": np.nan,
            "interpretation": "Reference bucket only. Run a separate config with target_net_exposure near zero to validate market-neutral alpha.",
        },
    ]
    return pd.DataFrame(rows)


def _period_row(period_table: pd.DataFrame, period: str) -> dict[str, object] | None:
    if period_table is None or period_table.empty or "period" not in period_table:
        return None
    rows = period_table[period_table["period"].astype(str).eq(period)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def validate_options_overlay_with_chain(options_overlay: pd.DataFrame, config: AppConfig, out: Path) -> pd.DataFrame:
    """Attach executable contract details to options overlay rows when a chain is available."""

    if options_overlay.empty or not config.options_overlay.enabled:
        return options_overlay
    provider_name = str(config.options_overlay.chain_provider).lower()
    if provider_name in {"none", "disabled", "off"}:
        return options_overlay
    symbols = sorted(set(options_overlay["symbol"].dropna().astype(str).str.upper())) if "symbol" in options_overlay else []
    dates = sorted(pd.to_datetime(options_overlay.get("signal_date"), errors="coerce").dropna().dt.normalize().unique())
    if not symbols or not dates:
        return options_overlay

    errors: list[str] = []
    all_errors: list[str] = []
    chain_frames: list[pd.DataFrame] = []
    providers = []
    if "marketdata" in provider_name:
        providers.append(
            MarketDataOptionChainProvider(
                cache_dir=config.options_overlay.option_chain_cache_dir,
                dte=config.options_overlay.option_chain_dte,
                side=config.options_overlay.option_chain_side,
                strike_limit=config.options_overlay.option_chain_strike_limit,
                min_open_interest=config.options_overlay.min_option_open_interest,
                min_volume=config.options_overlay.min_option_volume,
                max_bid_ask_spread_pct=config.options_overlay.max_option_spread_pct_mid,
            )
        )
    if "csv" in provider_name or provider_name == "csv":
        providers.append(CSVOptionChainProvider(config.options_overlay.option_chain_path))

    for provider in providers:
        chain_frames = []
        errors = []
        for quote_date in dates:
            quote = pd.Timestamp(quote_date).date().isoformat()
            for symbol in symbols:
                try:
                    frame = provider.get_chain(symbol, quote)
                except Exception as exc:  # pragma: no cover - real provider failure path
                    errors.append(f"{symbol} {quote}: {exc}")
                    continue
                if not frame.empty:
                    chain_frames.append(frame)
        if chain_frames:
            break
        all_errors.extend(errors)

    if not chain_frames:
        checked = options_overlay.copy()
        checked["executable"] = False
        status = "missing_option_chain"
        provider_errors = all_errors or errors
        if provider_errors:
            status = f"option_chain_provider_error: {provider_errors[0]}"
        checked["chain_validation_status"] = status
        if provider_errors:
            (out / "option_chain_errors.txt").write_text("\n".join(provider_errors[:200]), encoding="utf-8")
        return checked

    chain = pd.concat(chain_frames, ignore_index=True).drop_duplicates()
    chain.to_csv(out / "option_chain_snapshot.csv", index=False)
    return validate_overlay_against_chain(
        options_overlay,
        chain,
        min_open_interest=config.options_overlay.min_option_open_interest,
        min_volume=config.options_overlay.min_option_volume,
        max_spread_pct_mid=config.options_overlay.max_option_spread_pct_mid,
    )


OVERLAY_DIAGNOSTIC_SPECS = {
    "gap_adjusted_continuation": {
        "score": "gap_adjusted_continuation_score",
        "boost": "gap_adjusted_continuation_boost",
        "eligible": "gap_adjusted_continuation_eligible",
        "blocked": "gap_adjusted_continuation_blocked",
    },
    "short_term_volume_tilt": {
        "score": "short_term_volume_tilt_score",
        "boost": "short_term_volume_tilt_boost",
        "eligible": "short_term_volume_tilt_candidate",
        "blocked": "short_term_volume_tilt_blocked",
    },
    "pullback_short_term_reset": {
        "score": "pullback_short_term_reset_score",
        "boost": "pullback_short_term_reset_boost",
        "blocked": "pullback_short_term_reset_blocked",
    },
    "theme_breadth_acceleration": {
        "score": "theme_breadth_acceleration_score",
        "boost": "theme_breadth_acceleration_boost",
        "eligible": "theme_breadth_acceleration_eligible",
    },
    "theme_strength_delta": {
        "score": "theme_strength_delta_score",
        "boost": "theme_strength_delta_boost",
        "eligible": "theme_strength_delta_eligible",
    },
    "theme_leader_tilt": {
        "score": "theme_leader_tilt_score",
        "boost": "theme_leader_tilt_boost",
        "eligible": "theme_leader_tilt_eligible",
    },
    "boundary_rs_theme_credit": {
        "score": "boundary_rs_theme_credit_score",
        "boost": "boundary_rs_theme_credit_boost",
        "eligible": "boundary_rs_theme_credit_eligible",
        "blocked": "boundary_rs_theme_credit_blocked",
    },
    "boundary_rank_promotion": {
        "score": "boundary_rank_promotion_score",
        "eligible": "boundary_rank_promotion_promoted",
        "blocked": "boundary_rank_promotion_blocked",
    },
    "exit_quality_rank_credit": {
        "score": "exit_quality_rank_credit_score",
        "boost": "exit_quality_rank_credit",
        "eligible": "exit_quality_rank_credit_eligible",
        "blocked": "exit_quality_rank_credit_blocked",
    },
    "pullback_reclaim": {
        "score": "pullback_reclaim_score",
        "boost": "pullback_reclaim_boost",
        "blocked": "pullback_reclaim_blocked",
    },
}


def compute_overlay_activation_summary(signals: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether experimental overlays actually changed live signal rows."""

    columns = [
        "overlay",
        "rows",
        "active_rows",
        "active_row_rate",
        "eligible_rows",
        "blocked_rows",
        "active_signal_rows",
        "unique_active_symbols",
        "active_dates",
        "avg_score_active",
        "avg_boost_active",
        "avg_final_score_active",
        "avg_signal_active",
    ]
    if signals.empty:
        return pd.DataFrame(columns=columns)
    data = signals.copy()
    if "date" in data:
        data["date"] = pd.to_datetime(data["date"], format="mixed", errors="coerce").dt.normalize()
    rows: list[dict] = []
    for overlay, spec in OVERLAY_DIAGNOSTIC_SPECS.items():
        available = [column for column in spec.values() if column in data]
        if not available:
            continue
        score = _numeric_series(data, spec.get("score"))
        boost = _numeric_series(data, spec.get("boost"))
        eligible = _truthy_series(data, spec.get("eligible"))
        blocked = _truthy_series(data, spec.get("blocked"))
        active = score.abs().gt(1e-12) | boost.abs().gt(1e-12) | eligible
        active_frame = data.loc[active].copy()
        signal = _numeric_series(data, "signal")
        final_score = _numeric_series(data, "final_score")
        rows.append(
            {
                "overlay": overlay,
                "rows": int(len(data)),
                "active_rows": int(active.sum()),
                "active_row_rate": float(active.mean()) if len(active) else 0.0,
                "eligible_rows": int(eligible.sum()),
                "blocked_rows": int(blocked.sum()),
                "active_signal_rows": int((active & signal.abs().gt(1e-12)).sum()),
                "unique_active_symbols": int(active_frame["symbol"].nunique()) if "symbol" in active_frame else 0,
                "active_dates": int(active_frame["date"].nunique()) if "date" in active_frame else 0,
                "avg_score_active": float(score.loc[active].mean()) if active.any() else 0.0,
                "avg_boost_active": float(boost.loc[active].mean()) if active.any() else 0.0,
                "avg_final_score_active": float(final_score.loc[active].mean()) if active.any() else 0.0,
                "avg_signal_active": float(signal.loc[active].mean()) if active.any() else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["active_signal_rows", "active_rows"], ascending=False).reset_index(drop=True)


def _numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in frame:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def _truthy_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0.0).ne(0.0)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "on"})


def theme_divergence_report(result, manual_blotter: pd.DataFrame | None = None, theme_baskets_path: str | Path | None = None) -> pd.DataFrame:
    """Summarize latest theme/sector divergence for discretionary review."""

    targets = result.targets if isinstance(getattr(result, "targets", None), pd.DataFrame) else pd.DataFrame()
    if targets.empty or "date" not in targets or "symbol" not in targets:
        return pd.DataFrame()
    data = targets.copy()
    data["date"] = pd.to_datetime(data["date"], format="mixed", errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data = data.dropna(subset=["date", "symbol"])
    if data.empty:
        return pd.DataFrame()
    latest_date = data["date"].max()
    latest = data[data["date"].eq(latest_date)].copy()
    if latest.empty:
        return pd.DataFrame()
    theme_map = _symbol_theme_map(theme_baskets_path)
    latest["primary_theme"] = latest.get("primary_theme", pd.Series("unclassified", index=latest.index)).fillna("unclassified").replace("", "unclassified")
    if theme_map:
        mapped = latest["symbol"].map(theme_map)
        latest["primary_theme"] = mapped.fillna(latest["primary_theme"]).replace("", "unclassified")
    latest["target_weight"] = pd.to_numeric(latest.get("target_weight", 0.0), errors="coerce").fillna(0.0)
    for column in [
        "final_score",
        "technical_score",
        "relative_strength_score",
        "fundamental_score",
        "event_risk_score",
        "mom_return",
        "mom_volume_expansion",
        "distance_to_prior_high_252",
        "overnight_gap_risk_score",
        "minute_pretrade_risk_score",
    ]:
        if column in latest:
            latest[column] = pd.to_numeric(latest[column], errors="coerce")
    snapshot = _symbol_return_snapshot(result.prices if isinstance(getattr(result, "prices", None), pd.DataFrame) else pd.DataFrame(), latest_date)
    if not snapshot.empty:
        latest = latest.merge(snapshot, on="symbol", how="left")

    previous = data[data["date"].lt(latest_date)].copy()
    previous_gross = pd.DataFrame(columns=["primary_theme", "previous_target_gross"])
    if not previous.empty:
        previous_date = previous["date"].max()
        previous = previous[previous["date"].eq(previous_date)].copy()
        previous["primary_theme"] = previous.get("primary_theme", pd.Series("unclassified", index=previous.index)).fillna("unclassified").replace("", "unclassified")
        if theme_map:
            mapped = previous["symbol"].map(theme_map)
            previous["primary_theme"] = mapped.fillna(previous["primary_theme"]).replace("", "unclassified")
        previous["target_weight"] = pd.to_numeric(previous.get("target_weight", 0.0), errors="coerce").fillna(0.0)
        previous_gross = previous.groupby("primary_theme", as_index=False).agg(previous_target_gross=("target_weight", lambda s: s.abs().sum()))

    rows: list[dict] = []
    for theme, frame in latest.groupby("primary_theme", dropna=False):
        weights = frame["target_weight"].astype(float)
        leaders = (
            frame[frame["target_weight"].abs().gt(0)]
            .sort_values(["target_weight", "final_score"], ascending=[False, False])
            .head(5)
        )
        leader_text = ", ".join(
            f"{row.symbol}({_percent(row.target_weight)})" for row in leaders.itertuples(index=False) if hasattr(row, "symbol")
        )
        rows.append(
            {
                "date": latest_date,
                "level": "theme",
                "bucket": str(theme or "unclassified"),
                "symbols": int(frame["symbol"].nunique()),
                "target_gross": float(weights.abs().sum()),
                "target_net": float(weights.sum()),
                "avg_final_score": float(frame["final_score"].mean()) if "final_score" in frame else np.nan,
                "avg_technical_score": float(frame["technical_score"].mean()) if "technical_score" in frame else np.nan,
                "avg_relative_strength_score": float(frame["relative_strength_score"].mean()) if "relative_strength_score" in frame else np.nan,
                "avg_fundamental_score": float(frame["fundamental_score"].mean()) if "fundamental_score" in frame else np.nan,
                "avg_event_risk_score": float(frame["event_risk_score"].mean()) if "event_risk_score" in frame else np.nan,
                "avg_ret_5d": float(frame["ret_5d"].mean()) if "ret_5d" in frame else np.nan,
                "avg_ret_20d": float(frame["ret_20d"].mean()) if "ret_20d" in frame else np.nan,
                "avg_ret_63d": float(frame["ret_63d"].mean()) if "ret_63d" in frame else np.nan,
                "positive_20d_rate": float(pd.to_numeric(frame.get("ret_20d", pd.Series(dtype=float)), errors="coerce").gt(0).mean()) if "ret_20d" in frame else np.nan,
                "avg_momentum_return": float(frame["mom_return"].mean()) if "mom_return" in frame else np.nan,
                "avg_volume_expansion": float(frame["mom_volume_expansion"].median()) if "mom_volume_expansion" in frame else np.nan,
                "near_high_rate": float(pd.to_numeric(frame.get("distance_to_prior_high_252", pd.Series(dtype=float)), errors="coerce").ge(-0.05).mean()) if "distance_to_prior_high_252" in frame else np.nan,
                "avg_overnight_gap_risk_score": float(frame["overnight_gap_risk_score"].mean()) if "overnight_gap_risk_score" in frame else np.nan,
                "avg_minute_pretrade_risk_score": float(frame["minute_pretrade_risk_score"].mean()) if "minute_pretrade_risk_score" in frame else np.nan,
                "leaders": leader_text,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(previous_gross.rename(columns={"primary_theme": "bucket"}), on="bucket", how="left")
    out["previous_target_gross"] = pd.to_numeric(out.get("previous_target_gross"), errors="coerce").fillna(0.0)
    out["target_gross_change"] = out["target_gross"] - out["previous_target_gross"]
    action_summary = _theme_action_summary(manual_blotter, theme_map)
    if not action_summary.empty:
        out = out.merge(action_summary, on="bucket", how="left")
    for column in ["buy_notional", "sell_notional", "net_trade_notional", "open_add_count", "reduce_exit_count"]:
        if column not in out:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["divergence_status"] = out.apply(_theme_divergence_status, axis=1)
    out["suggested_tilt"] = out.apply(_theme_suggested_tilt, axis=1)
    return out.sort_values(["target_gross", "avg_relative_strength_score"], ascending=[False, False]).reset_index(drop=True)


def _symbol_return_snapshot(prices: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
    if prices.empty or not {"date", "symbol"}.issubset(prices.columns):
        return pd.DataFrame()
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], format="mixed", errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    price_column = "adj_close" if "adj_close" in data.columns else "close"
    if price_column not in data:
        return pd.DataFrame()
    data[price_column] = pd.to_numeric(data[price_column], errors="coerce")
    data = data[data["date"].le(pd.Timestamp(latest_date).normalize())].sort_values(["symbol", "date"])
    frames = []
    for symbol, group in data.groupby("symbol", sort=False):
        latest = group.tail(1).copy()
        if latest.empty:
            continue
        close = float(latest[price_column].iloc[0])
        row = {"symbol": symbol}
        for days, column in [(1, "ret_1d"), (5, "ret_5d"), (20, "ret_20d"), (63, "ret_63d")]:
            if len(group) > days and pd.notna(close):
                previous = float(group[price_column].iloc[-days - 1])
                row[column] = close / previous - 1.0 if previous else np.nan
            else:
                row[column] = np.nan
        frames.append(row)
    return pd.DataFrame(frames)


def _symbol_theme_map(theme_baskets_path: str | Path | None) -> dict[str, str]:
    if not theme_baskets_path:
        return {}
    baskets = load_thematic_baskets(theme_baskets_path)
    mapping: dict[str, str] = {}
    for theme, symbols in baskets.items():
        for symbol in symbols:
            mapping.setdefault(str(symbol).upper(), str(theme))
    return mapping


def _theme_action_summary(manual_blotter: pd.DataFrame | None, theme_map: dict[str, str] | None = None) -> pd.DataFrame:
    if manual_blotter is None or manual_blotter.empty:
        return pd.DataFrame()
    data = manual_blotter.copy()
    data["bucket"] = data.get("primary_theme", pd.Series("unclassified", index=data.index)).fillna("unclassified").replace("", "unclassified").astype(str)
    if theme_map and "symbol" in data:
        mapped = data["symbol"].astype(str).str.upper().map(theme_map)
        data["bucket"] = mapped.fillna(data["bucket"]).replace("", "unclassified")
    data["estimated_trade_notional"] = pd.to_numeric(data.get("estimated_trade_notional", 0.0), errors="coerce").fillna(0.0)
    data["action"] = data.get("action", "").astype(str)
    grouped = data.groupby("bucket")
    return grouped.agg(
        buy_notional=("estimated_trade_notional", lambda s: s[s.gt(0)].sum()),
        sell_notional=("estimated_trade_notional", lambda s: s[s.lt(0)].abs().sum()),
        net_trade_notional=("estimated_trade_notional", "sum"),
        open_add_count=("action", lambda s: s.isin(["OPEN LONG", "ADD"]).sum()),
        reduce_exit_count=("action", lambda s: s.isin(["REDUCE", "EXIT"]).sum()),
    ).reset_index()


def _theme_divergence_status(row: pd.Series) -> str:
    gross = _safe_float(row.get("target_gross"))
    gross_change = _safe_float(row.get("target_gross_change"))
    score = _safe_float(row.get("avg_final_score"), default=np.nan)
    rs = _safe_float(row.get("avg_relative_strength_score"), default=np.nan)
    ret20 = _safe_float(row.get("avg_ret_20d"), default=np.nan)
    near_high = _safe_float(row.get("near_high_rate"), default=np.nan)
    event = _safe_float(row.get("avg_event_risk_score"), default=0.0)
    if event >= 60:
        return "事件风险偏高"
    if gross >= 0.18 and rs >= 85 and score >= 70 and ret20 > 0:
        return "主线强势"
    if gross_change >= 0.03 and rs >= 80 and ret20 > 0:
        return "加速走强"
    if gross_change <= -0.03 or score < 66 or ret20 < -0.03:
        return "降温/轮出"
    if near_high >= 0.60 and ret20 > 0:
        return "趋势延续"
    return "观察"


def _theme_suggested_tilt(row: pd.Series) -> str:
    status = str(row.get("divergence_status", ""))
    gross = _safe_float(row.get("target_gross"))
    buy = _safe_float(row.get("buy_notional"))
    sell = _safe_float(row.get("sell_notional"))
    if status in {"主线强势", "加速走强"} and buy >= sell:
        return "优先增配"
    if status in {"主线强势", "趋势延续"} and gross > 0:
        return "持有核心"
    if status in {"降温/轮出", "事件风险偏高"}:
        return "减仓或等待"
    return "观察等待"


def latest_manual_trading_blotter(result) -> pd.DataFrame:
    """Build latest target-vs-current actions for manual trading."""

    targets = result.targets if isinstance(getattr(result, "targets", None), pd.DataFrame) else pd.DataFrame()
    equity_curve = result.equity_curve.copy()
    positions = result.positions.copy()
    if targets.empty or equity_curve.empty:
        return pd.DataFrame()
    targets = targets.copy()
    targets["date"] = pd.to_datetime(targets["date"]).dt.normalize()
    latest_target_date = targets["date"].max()
    latest_targets = targets[targets["date"] == latest_target_date].copy()
    if latest_targets.empty:
        return pd.DataFrame()
    equity_curve["date"] = pd.to_datetime(equity_curve["date"]).dt.normalize()
    latest_equity_date = equity_curve["date"].max()
    latest_equity = float(equity_curve.loc[equity_curve["date"] == latest_equity_date, "equity"].iloc[-1])
    current = pd.DataFrame(columns=["symbol", "current_quantity", "close", "current_market_value", "current_weight"])
    latest_prices = _latest_prices_for_manual_blotter(result, latest_equity_date)
    if not positions.empty:
        positions = positions.copy()
        positions["date"] = pd.to_datetime(positions["date"]).dt.normalize()
        latest_positions = positions[positions["date"] == latest_equity_date].copy()
        if not latest_positions.empty:
            current = latest_positions.rename(
                columns={
                    "quantity": "current_quantity",
                    "market_value": "current_market_value",
                }
            )[["symbol", "current_quantity", "close", "current_market_value"]]
            current["current_weight"] = current["current_market_value"].astype(float) / max(latest_equity, 1e-9)
    blotter = latest_targets.merge(current, on="symbol", how="outer")
    if not latest_prices.empty:
        blotter = blotter.merge(latest_prices, on="symbol", how="left", suffixes=("", "_latest"))
        if "close_latest" in blotter:
            blotter["close"] = blotter.get("close").fillna(blotter["close_latest"])
            blotter = blotter.drop(columns=["close_latest"])
    blotter["target_weight"] = pd.to_numeric(blotter.get("target_weight", 0.0), errors="coerce").fillna(0.0)
    blotter["current_weight"] = pd.to_numeric(blotter.get("current_weight", 0.0), errors="coerce").fillna(0.0)
    blotter["current_quantity"] = pd.to_numeric(blotter.get("current_quantity", 0.0), errors="coerce").fillna(0.0)
    blotter["close"] = pd.to_numeric(blotter.get("close", pd.NA), errors="coerce")
    blotter["delta_weight"] = blotter["target_weight"] - blotter["current_weight"]
    blotter["estimated_trade_notional"] = blotter["delta_weight"] * latest_equity
    blotter["estimated_shares"] = blotter["estimated_trade_notional"] / blotter["close"].where(blotter["close"].abs() > 1e-9)
    blotter["action"] = blotter.apply(_manual_action, axis=1)
    blotter["action_label"] = blotter["action"].map(_action_label)
    blotter["decision_note"] = blotter.apply(_manual_decision_note, axis=1)
    blotter["signal_date"] = latest_target_date
    blotter["position_date"] = latest_equity_date
    blotter["next_execution"] = "下一交易日开盘"
    columns = [
        "signal_date",
        "position_date",
        "next_execution",
        "symbol",
        "action",
        "action_label",
        "target_weight",
        "current_weight",
        "delta_weight",
        "estimated_trade_notional",
        "estimated_shares",
        "close",
        "final_score",
        "technical_score",
        "relative_strength_score",
        "fundamental_score",
        "moat_score",
        "fundamental_factor_coverage",
        "fundamental_missing_group_count",
        "fundamental_data_quality",
        "growth_score",
        "quality_score",
        "balance_sheet_score",
        "valuation_score",
        "revision_score",
        "sector",
        "industry",
        "fundamental_data_age_days",
        "event_risk_score",
        "days_to_earnings",
        "next_earnings_date",
        "expected_move",
        "mom_return",
        "mom_distance_high",
        "mom_volume_expansion",
        "retest_component",
        "retest_distance_ma_pct",
        "overbought_score",
        "overbought_status",
        "cash_signal",
        "regime_exposure_multiplier",
        "crowding_risk_score",
        "crowding_multiplier",
        "primary_theme",
        "theme_gross_exposure",
        "sector_gross_exposure",
        "industry_gross_exposure",
        "overnight_gap_risk_score",
        "overnight_gap_risk_status",
        "overnight_gap_multiplier",
        "minute_pretrade_risk_score",
        "minute_pretrade_risk_status",
        "minute_pretrade_multiplier",
        "mom_risk_adjusted",
        "distance_to_prior_high_252",
        "trend_price_vs_slow_pct",
        "trend_adx",
        "score_decomposition",
        "decision_note",
        "reason_for_entry",
        "reason_for_exit",
    ]
    available = []
    for column in columns:
        if column in blotter.columns and column not in available:
            available.append(column)
    out = blotter[available].copy()
    if "delta_weight" in out:
        out = out[out["delta_weight"].abs() >= 0.001]
        out = out.sort_values(["action", "delta_weight"], key=lambda s: s.abs() if s.name == "delta_weight" else s, ascending=False)
    return out.reset_index(drop=True)


def _latest_prices_for_manual_blotter(result, latest_date: pd.Timestamp) -> pd.DataFrame:
    prices = result.prices if isinstance(getattr(result, "prices", None), pd.DataFrame) else pd.DataFrame()
    if prices.empty or "date" not in prices.columns:
        return pd.DataFrame(columns=["symbol", "close"])
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    latest = data[data["date"] == pd.Timestamp(latest_date)].copy()
    if latest.empty:
        latest = data.sort_values("date").groupby("symbol", as_index=False).tail(1)
    price_col = "adj_close" if "adj_close" in latest.columns else "close"
    return latest[["symbol", price_col]].rename(columns={price_col: "close"})


def performance_breakdowns(equity_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return yearly and monthly performance diagnostics."""

    if equity_curve.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = equity_curve.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    data["daily_return"] = data["equity"].astype(float).pct_change().fillna(0.0)
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.to_period("M").astype(str)
    yearly_rows: list[dict] = []
    for year, frame in data.groupby("year"):
        curve = frame["equity"].astype(float)
        ret = curve.iloc[-1] / curve.iloc[0] - 1.0 if len(curve) else 0.0
        dd = curve / curve.cummax() - 1.0
        vol = frame["daily_return"].std() * np.sqrt(252)
        yearly_rows.append(
            {
                "year": int(year),
                "return": ret,
                "max_drawdown": dd.min(),
                "volatility": vol,
                "sharpe": (frame["daily_return"].mean() * 252 / vol) if vol and vol > 0 else np.nan,
                "best_day": frame["daily_return"].max(),
                "worst_day": frame["daily_return"].min(),
                "avg_gross_exposure": frame.get("gross_exposure", pd.Series(dtype=float)).mean(),
                "avg_turnover": frame.get("turnover", pd.Series(dtype=float)).mean(),
                "intraday_events": frame.get("intraday_risk_event_count", pd.Series(dtype=float)).sum(),
            }
        )
    monthly_rows: list[dict] = []
    for month, frame in data.groupby("month"):
        curve = frame["equity"].astype(float)
        ret = curve.iloc[-1] / curve.iloc[0] - 1.0 if len(curve) else 0.0
        dd = curve / curve.cummax() - 1.0
        monthly_rows.append(
            {
                "month": month,
                "return": ret,
                "max_drawdown": dd.min(),
                "best_day": frame["daily_return"].max(),
                "worst_day": frame["daily_return"].min(),
                "avg_gross_exposure": frame.get("gross_exposure", pd.Series(dtype=float)).mean(),
                "avg_turnover": frame.get("turnover", pd.Series(dtype=float)).mean(),
                "intraday_events": frame.get("intraday_risk_event_count", pd.Series(dtype=float)).sum(),
            }
        )
    return pd.DataFrame(yearly_rows), pd.DataFrame(monthly_rows)


def drawdown_component_analysis(result, top_days: int = 8, top_symbols: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze major drawdown periods and symbol-level contributors."""

    equity = result.equity_curve.copy()
    if equity.empty:
        return pd.DataFrame(), pd.DataFrame()
    equity["date"] = pd.to_datetime(equity["date"]).dt.normalize()
    equity = equity.sort_values("date").reset_index(drop=True)
    equity["daily_return"] = equity["equity"].astype(float).pct_change().fillna(0.0)
    equity["peak"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["peak"] - 1.0
    periods = _drawdown_periods(equity)
    contributors = _worst_day_contributors(result, equity.nsmallest(top_days, "daily_return")["date"].tolist(), top_symbols)
    return periods, contributors


def _drawdown_periods(equity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    in_dd = False
    start_idx = 0
    trough_idx = 0
    for idx, row in equity.iterrows():
        dd = float(row["drawdown"])
        if not in_dd and dd < 0:
            in_dd = True
            start_idx = max(0, idx - 1)
            trough_idx = idx
        if in_dd and dd < float(equity.loc[trough_idx, "drawdown"]):
            trough_idx = idx
        if in_dd and dd >= -1e-12:
            rows.append(_drawdown_period_row(equity, start_idx, trough_idx, idx))
            in_dd = False
    if in_dd:
        rows.append(_drawdown_period_row(equity, start_idx, trough_idx, len(equity) - 1))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("max_drawdown").reset_index(drop=True)


def _drawdown_period_row(equity: pd.DataFrame, start_idx: int, trough_idx: int, end_idx: int) -> dict:
    start = equity.loc[start_idx]
    trough = equity.loc[trough_idx]
    end = equity.loc[end_idx]
    recovered = float(end["drawdown"]) >= -1e-12
    return {
        "start_date": start["date"],
        "trough_date": trough["date"],
        "end_date": end["date"] if recovered else pd.NaT,
        "max_drawdown": float(trough["drawdown"]),
        "days_to_trough": int((pd.Timestamp(trough["date"]) - pd.Timestamp(start["date"])).days),
        "recovery_days": int((pd.Timestamp(end["date"]) - pd.Timestamp(trough["date"])).days) if recovered else pd.NA,
        "return_start_to_trough": float(trough["equity"] / start["equity"] - 1.0),
    }


def _worst_day_contributors(result, dates: list[pd.Timestamp], top_symbols: int) -> pd.DataFrame:
    prices = result.prices if isinstance(getattr(result, "prices", None), pd.DataFrame) else pd.DataFrame()
    positions = result.positions if isinstance(getattr(result, "positions", None), pd.DataFrame) else pd.DataFrame()
    equity = result.equity_curve.copy()
    if prices.empty or positions.empty or equity.empty:
        return pd.DataFrame()
    prices = prices.copy()
    positions = positions.copy()
    equity["date"] = pd.to_datetime(equity["date"]).dt.normalize()
    equity = equity.sort_values("date").reset_index(drop=True)
    equity["daily_return"] = equity["equity"].astype(float).pct_change().fillna(0.0)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    positions["date"] = pd.to_datetime(positions["date"]).dt.normalize()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    price_lookup = prices.set_index(["date", "symbol"])[price_col].sort_index()
    rows: list[dict] = []
    calendar = list(pd.to_datetime(sorted(equity["date"].unique())))
    for date in dates:
        date = pd.Timestamp(date).normalize()
        if date not in set(calendar):
            continue
        idx = calendar.index(date)
        if idx == 0:
            continue
        prev_date = calendar[idx - 1]
        prev_positions = positions[positions["date"] == prev_date].copy()
        if prev_positions.empty:
            continue
        prev_equity = float(equity.loc[equity["date"] == prev_date, "equity"].iloc[-1])
        daily_row = equity.loc[equity["date"] == date].iloc[-1]
        day_rows: list[dict] = []
        for _, pos in prev_positions.iterrows():
            symbol = str(pos["symbol"])
            qty = float(pos.get("quantity", 0.0) or 0.0)
            try:
                prev_price = float(price_lookup.loc[(prev_date, symbol)])
                curr_price = float(price_lookup.loc[(date, symbol)])
            except (KeyError, TypeError, ValueError):
                continue
            pnl = qty * (curr_price - prev_price)
            day_rows.append(
                {
                    "drawdown_date": date,
                    "symbol": symbol,
                    "previous_weight": (qty * prev_price) / max(prev_equity, 1e-9),
                    "symbol_return": curr_price / prev_price - 1.0 if prev_price else np.nan,
                    "pnl": pnl,
                    "portfolio_contribution": pnl / max(prev_equity, 1e-9),
                    "daily_return": float(daily_row.get("daily_return", np.nan)),
                    "gross_exposure": float(daily_row.get("gross_exposure", np.nan)),
                    "turnover": float(daily_row.get("turnover", np.nan)),
                    "intraday_risk_event_count": float(daily_row.get("intraday_risk_event_count", 0.0) or 0.0),
                    "intraday_exit_notional": float(daily_row.get("intraday_exit_notional", 0.0) or 0.0),
                }
            )
        if day_rows:
            worst = pd.DataFrame(day_rows).sort_values("portfolio_contribution").head(top_symbols)
            rows.extend(worst.to_dict("records"))
    return pd.DataFrame(rows)


def render_fast_html_report(
    metrics: dict[str, float],
    warnings: list[str],
    period_table: pd.DataFrame,
    manual_blotter: pd.DataFrame,
    options_overlay: pd.DataFrame,
    data_quality_summary: pd.DataFrame | None = None,
    data_quality_detail: pd.DataFrame | None = None,
    trading_desk_tables: dict[str, pd.DataFrame] | None = None,
    institutional_attribution: dict[str, pd.DataFrame] | None = None,
    exposure_decomposition: pd.DataFrame | None = None,
    portfolio_risk_detail: pd.DataFrame | None = None,
    portfolio_risk_summary: pd.DataFrame | None = None,
    portfolio_risk_buckets: pd.DataFrame | None = None,
    portfolio_risk_stress: pd.DataFrame | None = None,
) -> str:
    """Render a lightweight report for iterative research loops."""

    headline_scope = escape(str(metrics.get("headline_metric_scope", "full_period")))
    metric_items = "".join(
        f"<li><strong>{escape(str(key))}</strong>: {float(metrics.get('headline_' + key, metrics.get(key, 0.0))):.4f}</li>"
        for key in ["total_return", "trailing_one_year_return", "cagr", "sharpe", "max_drawdown", "average_gross_exposure", "average_turnover"]
        if isinstance(metrics.get("headline_" + key, metrics.get(key, 0.0)), (int, float, np.floating))
    )
    warning_items = "".join(f"<li>{escape(str(item))}</li>" for item in warnings[:25])
    period_html = period_table.to_html(index=False, classes="table", border=0, float_format=lambda x: f"{x:.4f}") if not period_table.empty else "<p>No period metrics.</p>"
    manual_cols = [
        col
        for col in [
            "symbol",
            "action_label",
            "target_weight",
            "current_weight",
            "final_score",
            "relative_strength_score",
            "fundamental_score",
            "fundamental_factor_coverage",
            "event_risk_score",
        ]
        if col in manual_blotter.columns
    ]
    manual_html = (
        manual_blotter[manual_cols].head(50).to_html(index=False, classes="table", border=0, float_format=lambda x: f"{x:.4f}")
        if manual_cols and not manual_blotter.empty
        else "<p>No manual trading actions.</p>"
    )
    option_cols = [
        col
        for col in [
            "symbol",
            "equity_action",
            "option_structure",
            "executable",
            "target_delta",
            "max_premium_budget",
            "chain_validation_status",
            "contract_option_symbol",
            "contract_chain_quote_source",
            "contract_expiration",
            "contract_strike",
            "contract_mid",
            "contract_delta",
            "contract_open_interest",
            "contract_volume",
            "contract_spread_pct_mid",
            "recommended_contracts",
            "estimated_option_premium",
            "premium_budget_utilization",
        ]
        if col in options_overlay.columns
    ]
    option_html = (
        options_overlay[option_cols].head(50).to_html(index=False, classes="table", border=0, float_format=lambda x: f"{x:.4f}")
        if option_cols and not options_overlay.empty
        else "<p>No options overlay recommendations.</p>"
    )
    data_quality_html = _data_quality_html(data_quality_summary if data_quality_summary is not None else pd.DataFrame(), data_quality_detail if data_quality_detail is not None else pd.DataFrame())
    trading_desk_html = _trading_desk_html(trading_desk_tables or {})
    portfolio_risk_html = _portfolio_risk_model_html(
        portfolio_risk_detail if portfolio_risk_detail is not None else pd.DataFrame(),
        portfolio_risk_summary if portfolio_risk_summary is not None else pd.DataFrame(),
        portfolio_risk_buckets if portfolio_risk_buckets is not None else pd.DataFrame(),
        portfolio_risk_stress if portfolio_risk_stress is not None else pd.DataFrame(),
    )
    institutional_attribution_html = _institutional_attribution_html(institutional_attribution or {})
    exposure_html = _exposure_mode_html(exposure_decomposition if exposure_decomposition is not None else pd.DataFrame())
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Fast Quant Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 28px; color: #172033; }}
    h1, h2 {{ margin: 0.8rem 0; }}
    .notice {{ background: #fff7e6; border: 1px solid #ffd58a; padding: 12px 14px; border-radius: 8px; }}
    .table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    .table th, .table td {{ border-bottom: 1px solid #e4e8f0; padding: 7px 8px; text-align: right; }}
    .table th:first-child, .table td:first-child {{ text-align: left; }}
    li {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <h1>快速回测报告</h1>
  <p class="notice">这是 fast 模式：跳过大表 CSV、归因图和完整 HTML 图表，用于快速迭代。最终参数仍应跑完整报告。</p>
	  <h2>核心指标</h2>
	  <p class="notice">Headline scope: {headline_scope}. Full-period metrics are diagnostics, not the primary validation number.</p>
	  <ul>{metric_items}</ul>
  <h2>分阶段表现</h2>
  {period_html}
  <h2>最新交易动作</h2>
  {manual_html}
  {data_quality_html}
  {trading_desk_html}
	  {portfolio_risk_html}
	  {exposure_html}
  <h2>期权 Overlay</h2>
  {option_html}
  {institutional_attribution_html}
  <h2>警告</h2>
  <ul>{warning_items}</ul>
</body>
</html>"""


def _manual_action(row: pd.Series) -> str:
    current = float(row.get("current_weight", 0.0) or 0.0)
    target = float(row.get("target_weight", 0.0) or 0.0)
    delta = target - current
    eps = 0.001
    if abs(target) < eps and abs(current) >= eps:
        return "EXIT"
    if abs(current) < eps and abs(target) >= eps:
        return "OPEN LONG" if target > 0 else "OPEN SHORT"
    if abs(delta) < eps:
        return "HOLD"
    if delta > 0 and target >= 0:
        return "ADD"
    if delta < 0 and target >= 0:
        return "REDUCE"
    if delta < 0 and target < 0:
        return "ADD SHORT"
    return "REDUCE SHORT"


def _action_label(action: str) -> str:
    return {
        "OPEN LONG": "新开多",
        "ADD": "加仓",
        "REDUCE": "减仓",
        "EXIT": "清仓",
        "OPEN SHORT": "新开空",
        "ADD SHORT": "加空",
        "REDUCE SHORT": "减空",
        "HOLD": "持有",
    }.get(str(action), str(action))


def _manual_decision_note(row: pd.Series) -> str:
    """Create a user-facing Chinese explanation for the latest manual action."""

    action = str(row.get("action", ""))
    action_cn = _action_label(action)
    target = _safe_float(row.get("target_weight"))
    current = _safe_float(row.get("current_weight"))
    delta = _safe_float(row.get("delta_weight"))
    final = _safe_float(row.get("final_score"))
    tech = _safe_float(row.get("technical_score"))
    rs = _safe_float(row.get("relative_strength_score"))
    fundamental = _safe_float(row.get("fundamental_score"))
    event = _safe_float(row.get("event_risk_score"))
    overbought = _safe_float(row.get("overbought_score"))
    overbought_status = str(row.get("overbought_status", "") or "")
    overbought_status_cn = _status_label(overbought_status)
    cash_signal = bool(row.get("cash_signal", False))
    prior_high_gap = _safe_float(row.get("mom_distance_high", row.get("distance_to_prior_high_252")), default=np.nan)
    retest = _safe_float(row.get("retest_component"), default=np.nan)
    ma_gap = _safe_float(row.get("retest_distance_ma_pct"), default=np.nan)
    mom_return = _safe_float(row.get("mom_return"), default=np.nan)
    volume_exp = _safe_float(row.get("mom_volume_expansion"), default=np.nan)
    days_to_earnings = _safe_float(row.get("days_to_earnings"), default=np.nan)

    base = f"{action_cn}：目标权重 {target:.2%}，当前权重 {current:.2%}，需要调整 {delta:+.2%}。"
    if action == "EXIT":
        exit_reason = str(row.get("reason_for_exit", "") or "")
        source = "最新目标池不再包含该标的" if "Dropped from" in exit_reason else "目标仓位降为 0"
        base = f"{action_cn}：{source}，当前权重 {current:.2%}，退出或降到现金。"
    elif action == "REDUCE":
        base = f"{action_cn}：目标权重 {target:.2%} 低于当前 {current:.2%}，降低 {abs(delta):.2%}。"
    elif action in {"OPEN LONG", "OPEN SHORT"}:
        base = f"{action_cn}：从 0 建仓到目标权重 {target:.2%}。"

    score_line = ""
    if action != "EXIT" or final > 0 or tech > 0 or rs > 0:
        score_line = (
            f"综合分 {final:.1f}，技术 {tech:.1f}，相对强弱 {rs:.1f}，"
            f"基本面 {fundamental:.1f}，{_event_context_text(event, days_to_earnings)}。"
        )
    thesis = _decision_thesis(
        action=action,
        final=final,
        tech=tech,
        rs=rs,
        fundamental=fundamental,
        event=event,
        prior_high_gap=prior_high_gap,
        retest=retest,
        ma_gap=ma_gap,
        mom_return=mom_return,
        volume_exp=volume_exp,
        days_to_earnings=days_to_earnings,
    )
    risk_line = f"风控：过热分 {overbought:.0f}（{overbought_status_cn}），空仓信号 {'是' if cash_signal else '否'}。"
    detail = _score_decomposition_cn(row.get("score_decomposition", ""))
    return f"{base} {score_line} {thesis} {detail} {risk_line}".replace("  ", " ").strip()


def _decision_thesis(
    *,
    action: str,
    final: float,
    tech: float,
    rs: float,
    fundamental: float,
    event: float,
    prior_high_gap: float,
    retest: float,
    ma_gap: float,
    mom_return: float,
    volume_exp: float,
    days_to_earnings: float,
) -> str:
    """Synthesize a compact trading thesis from numeric evidence."""

    evidence: list[str] = []
    if np.isfinite(mom_return):
        if mom_return >= 0.25:
            evidence.append(f"中短期动量很强（动量收益 {mom_return:.1%}）")
        elif mom_return <= -0.10:
            evidence.append(f"动量已经转弱（动量收益 {mom_return:.1%}）")
    if np.isfinite(prior_high_gap):
        if prior_high_gap >= 0.02:
            evidence.append(f"价格已突破前高 {prior_high_gap:.1%}")
        elif prior_high_gap > -0.05:
            evidence.append(f"价格离前高很近（{prior_high_gap:.1%}）")
        else:
            evidence.append(f"价格仍低于前高 {abs(prior_high_gap):.1%}")
    if np.isfinite(retest) and retest >= 60:
        evidence.append(f"回踩质量 {retest:.0f}，接近均线 {ma_gap:.1%}")
    elif np.isfinite(ma_gap) and abs(ma_gap) <= 0.03:
        evidence.append(f"距离回踩均线 {ma_gap:.1%}，属于可观察区")
    if np.isfinite(volume_exp):
        if volume_exp >= 1.5:
            evidence.append(f"量能放大 {volume_exp:.2f} 倍")
        elif volume_exp < 0.8:
            evidence.append(f"量能偏弱 {volume_exp:.2f} 倍")
    if event >= 70:
        if np.isfinite(days_to_earnings) and days_to_earnings >= 0:
            evidence.append(f"财报倒计时 {days_to_earnings:.0f} 天，事件风险高")
        elif np.isfinite(days_to_earnings):
            evidence.append(f"财报后第 {abs(days_to_earnings):.0f} 天，适合观察反应延续而不是盲目追单")
    elif event > 0:
        evidence.append(f"事件风险可控但不为零（{event:.0f}）")
    if fundamental < 45:
        evidence.append("基本面分偏弱，仓位不宜只靠价格动量放大")
    elif fundamental >= 60:
        evidence.append("基本面分对多头有支撑")

    if not evidence:
        evidence.append("当前信号主要来自综合排名和目标仓位再平衡")
    stance = "核心判断"
    if action in {"OPEN LONG", "ADD"}:
        stance = "多头逻辑"
    elif action in {"REDUCE", "EXIT"}:
        stance = "降仓逻辑"
    elif action in {"OPEN SHORT", "ADD SHORT"}:
        stance = "空头逻辑"
    return f"{stance}：" + "；".join(evidence[:5]) + "。"


def _event_context_text(event: float, days_to_earnings: float) -> str:
    if np.isfinite(days_to_earnings):
        if days_to_earnings >= 0:
            return f"事件风险 {event:.1f}（距财报 {days_to_earnings:.0f} 天）"
        return f"事件风险 {event:.1f}（财报后 {abs(days_to_earnings):.0f} 天）"
    if event > 0:
        return f"事件风险 {event:.1f}"
    return "事件：无已知临近财报/重大事件"


def _event_table_value(row: pd.Series) -> str:
    event = _safe_float(row.get("event_risk_score"))
    days = _safe_float(row.get("days_to_earnings"), default=np.nan)
    if event == 0 and not np.isfinite(days):
        return "无已知临近"
    return _number(event, digits=1)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_decomposition_cn(value) -> str:
    text = "" if pd.isna(value) else str(value)
    if not text:
        return ""
    fields = {
        "return": "动量收益",
        "risk_adj": "风险调整",
        "near_high": "距滚动高点",
        "prior_high_gap": "距前高",
        "volume_exp": "量能倍数",
        "rs": "相对强弱",
        "retest": "回踩",
        "retest_score": "回踩分",
        "ma_gap": "距回踩均线",
        "price_vs_slow": "较慢均线",
        "adx": "ADX",
    }
    parts: list[str] = []
    for key, label in fields.items():
        match = re.search(rf"(?:^|[;\s|]){re.escape(key)}=([^;|]+)", text)
        if match:
            raw = match.group(1).strip()
            numeric = _parse_numeric_text(raw)
            if key in {"retest", "retest_score"} and numeric is not None and numeric <= 0:
                continue
            if key == "ma_gap" and numeric is not None and abs(numeric) > 5:
                continue
            parts.append(f"{label} {raw}")
    if not parts:
        return ""
    return "关键触发：" + "，".join(parts[:7]) + "。"


def _parse_numeric_text(value: str) -> float | None:
    try:
        return float(str(value).replace("%", "").replace("x", ""))
    except ValueError:
        return None


def _status_label(status: str) -> str:
    return {
        "normal": "正常",
        "warning": "偏热",
        "reduce": "降敞口",
        "cash": "空仓",
        "risk_on": "风险偏好",
        "risk_off": "风险规避",
    }.get(str(status), str(status) or "未知")


def _period_label(period: str) -> str:
    return {
        "full": "全样本",
        "train": "训练期",
        "validation": "验证期",
        "forward_current": "2026前推",
        "ai_cycle": "AI周期",
        "full_2020_2026": "全样本",
        "forward_2026": "2026前推",
    }.get(str(period), str(period))


def _side_label(side: str) -> str:
    return {"buy": "买入", "sell": "卖出"}.get(str(side).lower(), str(side))


def _trade_note_cn(row: pd.Series) -> str:
    side = _side_label(str(row.get("side", "")))
    quantity = _safe_float(row.get("quantity"))
    price = _safe_float(row.get("price"))
    target = _safe_float(row.get("target_weight"))
    final = _safe_float(row.get("final_score"))
    fundamental = _safe_float(row.get("fundamental_score"))
    event = _safe_float(row.get("event_risk_score"))
    days_to_earnings = _safe_float(row.get("days_to_earnings"), default=np.nan)
    exit_reason = str(row.get("reason_for_exit", "") or "")
    base = f"{side} {quantity:,.1f} 股，模拟成交价 {price:,.2f}，成交后目标权重 {target:.2%}。"
    if "Dropped from" in exit_reason:
        return base + " 退出原因：该标的从最新目标池移除。"
    if final > 0:
        return base + f" 综合分 {final:.1f}，基本面 {fundamental:.1f}，{_event_context_text(event, days_to_earnings)}。"
    return base


def save_report_charts(result, out: Path) -> dict[str, str]:
    """Save report charts as PNG files and return relative paths."""

    import matplotlib.pyplot as plt

    charts: dict[str, str] = {}
    equity = result.equity_curve.copy()
    if not equity.empty:
        equity["date"] = pd.to_datetime(equity["date"])
        equity = equity.sort_values("date")
        curve = equity["equity"].astype(float)
        drawdown = curve / curve.cummax() - 1.0

        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(equity["date"], curve, color="#2563eb", linewidth=1.8)
        ax.set_title("Equity Curve")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        charts["equity"] = "equity_curve.png"
        fig.savefig(out / charts["equity"], dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.fill_between(equity["date"], drawdown, 0, color="#dc2626", alpha=0.35)
        ax.plot(equity["date"], drawdown, color="#991b1b", linewidth=1.0)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        charts["drawdown"] = "drawdown.png"
        fig.savefig(out / charts["drawdown"], dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
        axes[0].plot(equity["date"], equity["gross_exposure"], label="Gross", color="#0f766e")
        axes[0].plot(equity["date"], equity["net_exposure"], label="Net", color="#7c3aed")
        axes[0].axhline(0, color="#64748b", linewidth=0.8)
        axes[0].set_title("Exposure")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.25)
        axes[1].bar(equity["date"], equity["turnover"], color="#f97316", width=1.0)
        axes[1].set_title("Daily Turnover")
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        charts["exposure"] = "exposure_turnover.png"
        fig.savefig(out / charts["exposure"], dpi=150)
        plt.close(fig)

        monthly = monthly_returns(equity)
        if not monthly.empty:
            monthly["month"] = pd.to_datetime(monthly["month"])
            monthly["year"] = monthly["month"].dt.year
            monthly["month_num"] = monthly["month"].dt.month
            pivot = monthly.pivot_table(index="year", columns="month_num", values="return", aggfunc="sum").fillna(0.0)
            fig, ax = plt.subplots(figsize=(11, max(2.6, 0.45 * len(pivot))))
            image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-0.10, vmax=0.10)
            ax.set_title("Monthly Returns")
            ax.set_xticks(range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            ax.set_yticks(range(len(pivot.index)), labels=[str(y) for y in pivot.index])
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.iloc[i, j]
                    ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=8, color="#111827")
            fig.colorbar(image, ax=ax, label="Return")
            fig.tight_layout()
            charts["monthly"] = "monthly_returns_heatmap.png"
            fig.savefig(out / charts["monthly"], dpi=150)
            plt.close(fig)

        yearly, monthly_diag = performance_breakdowns(equity)
        if not yearly.empty:
            fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
            for year, frame in equity.groupby(equity["date"].dt.year):
                frame = frame.sort_values("date").copy()
                x = np.arange(len(frame))
                normalized = frame["equity"].astype(float) / float(frame["equity"].iloc[0])
                dd = frame["equity"].astype(float) / frame["equity"].astype(float).cummax() - 1.0
                axes[0].plot(x, normalized, linewidth=1.4, label=str(year), alpha=0.9)
                axes[1].plot(x, dd, linewidth=1.2, label=str(year), alpha=0.9)
            axes[0].set_title("Yearly Normalized Equity Curves")
            axes[0].set_ylabel("Start of year = 1.0")
            axes[0].grid(True, alpha=0.25)
            axes[1].set_title("Yearly Drawdown Curves")
            axes[1].set_ylabel("Drawdown")
            axes[1].yaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
            axes[1].set_xlabel("Trading day within year")
            axes[1].grid(True, alpha=0.25)
            axes[0].legend(ncol=4, fontsize=8)
            fig.tight_layout()
            charts["yearly_equity_drawdown"] = "yearly_equity_drawdown.png"
            fig.savefig(out / charts["yearly_equity_drawdown"], dpi=150)
            plt.close(fig)

        if not monthly_diag.empty:
            monthly_diag["month_dt"] = pd.to_datetime(monthly_diag["month"])
            monthly_diag["year"] = monthly_diag["month_dt"].dt.year
            monthly_diag["month_num"] = monthly_diag["month_dt"].dt.month
            pivot = monthly_diag.pivot_table(index="year", columns="month_num", values="max_drawdown", aggfunc="min").fillna(0.0)
            fig, ax = plt.subplots(figsize=(11, max(2.6, 0.45 * len(pivot))))
            image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Reds_r", vmin=-0.20, vmax=0.0)
            ax.set_title("Monthly Max Drawdown")
            ax.set_xticks(range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            ax.set_yticks(range(len(pivot.index)), labels=[str(y) for y in pivot.index])
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.iloc[i, j]
                    ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=8, color="#111827")
            fig.colorbar(image, ax=ax, label="Max drawdown")
            fig.tight_layout()
            charts["monthly_drawdown"] = "monthly_drawdown_heatmap.png"
            fig.savefig(out / charts["monthly_drawdown"], dpi=150)
            plt.close(fig)

    positions = result.positions.copy()
    if not positions.empty:
        positions["date"] = pd.to_datetime(positions["date"])
        last_date = positions["date"].max()
        latest = positions[positions["date"] == last_date].copy()
        latest["abs_market_value"] = latest["market_value"].abs()
        latest = latest.sort_values("abs_market_value", ascending=False).head(20)
        if not latest.empty:
            fig, ax = plt.subplots(figsize=(11, 5.2))
            colors = np.where(latest["market_value"] >= 0, "#16a34a", "#dc2626")
            ax.bar(latest["symbol"], latest["market_value"], color=colors)
            ax.set_title(f"Top Positions by Market Value ({last_date.date()})")
            ax.tick_params(axis="x", rotation=45)
            ax.axhline(0, color="#64748b", linewidth=0.8)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            charts["positions"] = "top_positions.png"
            fig.savefig(out / charts["positions"], dpi=150)
            plt.close(fig)
    return charts


def save_manual_trading_chart(manual_blotter: pd.DataFrame, out: Path) -> str | None:
    """Save a horizontal bar chart of the largest latest manual actions."""

    if manual_blotter.empty or "estimated_trade_notional" not in manual_blotter.columns:
        return None
    data = manual_blotter.copy()
    data["estimated_trade_notional"] = pd.to_numeric(data["estimated_trade_notional"], errors="coerce")
    data = data.dropna(subset=["estimated_trade_notional"])
    data = data[data["estimated_trade_notional"].abs() > 0].copy()
    if data.empty:
        return None
    data["abs_notional"] = data["estimated_trade_notional"].abs()
    data = data.sort_values("abs_notional", ascending=True).tail(30)
    import matplotlib.pyplot as plt

    colors = np.where(data["estimated_trade_notional"] >= 0, "#16a34a", "#dc2626")
    labels = data["symbol"].astype(str) + " " + data["action"].astype(str)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.27 * len(data))))
    ax.barh(labels, data["estimated_trade_notional"], color=colors)
    ax.axvline(0, color="#64748b", linewidth=0.9)
    ax.set_title("Latest Manual Trading Actions by Estimated Notional")
    ax.set_xlabel("Estimated trade notional")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"${x/1000:,.0f}k")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    filename = "manual_trading_actions.png"
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    return filename


def render_html_report(
    metrics: dict[str, float],
    result,
    warnings: list[str],
    period_table: pd.DataFrame | None = None,
    charts: dict[str, str] | None = None,
    manual_blotter: pd.DataFrame | None = None,
    factor_ic: pd.DataFrame | None = None,
    family_ic: pd.DataFrame | None = None,
    factor_corr: pd.DataFrame | None = None,
    yearly_breakdown: pd.DataFrame | None = None,
    monthly_breakdown: pd.DataFrame | None = None,
    drawdown_periods: pd.DataFrame | None = None,
    drawdown_contributors: pd.DataFrame | None = None,
    theme_divergence: pd.DataFrame | None = None,
    options_overlay: pd.DataFrame | None = None,
    institutional_attribution: dict[str, pd.DataFrame] | None = None,
    exposure_decomposition: pd.DataFrame | None = None,
    trading_desk_tables: dict[str, pd.DataFrame] | None = None,
    data_quality_summary: pd.DataFrame | None = None,
    data_quality_detail: pd.DataFrame | None = None,
    portfolio_risk_detail: pd.DataFrame | None = None,
    portfolio_risk_summary: pd.DataFrame | None = None,
    portfolio_risk_buckets: pd.DataFrame | None = None,
    portfolio_risk_stress: pd.DataFrame | None = None,
) -> str:
    """Render a visual HTML report."""

    warning_html = "".join(f"<li>{warning}</li>" for warning in warnings)
    charts = charts or {}
    headline_scope = str(metrics.get("headline_metric_scope", "full_period"))
    headline_note = (
        f"<p class='note'><b>主指标口径：</b>{escape(headline_scope)}。"
        "Full-period 数字仍保留在“全部指标”里，但如果存在主题/股票池选择偏差，它只作为研究诊断，不作为可交易绩效。</p>"
    )
    metric_labels = {
        "total_return": "总收益",
        "trailing_one_year_return": "近一年收益",
        "cagr": "CAGR",
        "annualized_volatility": "年化波动",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "calmar": "Calmar",
        "max_drawdown": "最大回撤",
        "average_gross_exposure": "平均总敞口",
        "average_net_exposure": "平均净敞口",
        "average_turnover": "平均换手",
    }
    cards = ""
    for key in ["total_return", "trailing_one_year_return", "cagr", "annualized_volatility", "sharpe", "sortino", "calmar", "max_drawdown", "average_gross_exposure", "average_net_exposure", "average_turnover"]:
        value = metrics.get(f"headline_{key}", metrics.get(key, 0.0))
        formatted = f"{value:.2%}" if key not in {"sharpe", "sortino", "calmar"} else f"{value:.2f}"
        cards += f"<div class='metric'><span>{metric_labels[key]}</span><strong>{formatted}</strong></div>"
    metric_rows = "".join(f"<tr><td>{escape(str(key))}</td><td>{escape(_metric_value_text(value))}</td></tr>" for key, value in metrics.items())
    period_html = period_table.to_html(index=False) if period_table is not None and not period_table.empty else "<p>暂无分段指标。</p>"
    trades = result.trades.tail(50).copy()
    trade_rows = ""
    if not trades.empty:
        for _, row in trades.iterrows():
            trade_date = pd.to_datetime(row.get("date")).date() if pd.notna(row.get("date")) else ""
            trade_rows += (
                "<tr>"
                f"<td>{trade_date}</td><td>{escape(str(row.get('symbol', '')))}</td><td>{escape(_side_label(str(row.get('side', ''))))}</td>"
                f"<td>{_percent(row.get('target_weight'))}</td><td>{_number(row.get('final_score'), digits=1)}</td>"
                f"<td>{_number(row.get('fundamental_score'), digits=1)}</td><td>{escape(_event_table_value(row))}</td>"
                f"<td>{escape(_trade_note_cn(row))}</td>"
                "</tr>"
            )
    manual_blotter = manual_blotter if manual_blotter is not None else pd.DataFrame()
    risk_overlay = result.risk_overlay if isinstance(getattr(result, "risk_overlay", None), pd.DataFrame) else pd.DataFrame()
    risk_overlay_html = _risk_overlay_html(risk_overlay)
    intraday_exits_html = _intraday_exits_html(result.trades, result.equity_curve)
    capacity_html = _capacity_html(result.trades)
    theme_divergence_html = _theme_divergence_html(theme_divergence if theme_divergence is not None else pd.DataFrame())
    period_breakdown_html = _period_breakdown_html(yearly_breakdown if yearly_breakdown is not None else pd.DataFrame(), monthly_breakdown if monthly_breakdown is not None else pd.DataFrame())
    drawdown_analysis_html = _drawdown_analysis_html(drawdown_periods if drawdown_periods is not None else pd.DataFrame(), drawdown_contributors if drawdown_contributors is not None else pd.DataFrame())
    options_overlay_html = _options_overlay_html(options_overlay if options_overlay is not None else pd.DataFrame())
    institutional_attribution_html = _institutional_attribution_html(institutional_attribution or {})
    exposure_mode_html = _exposure_mode_html(exposure_decomposition if exposure_decomposition is not None else pd.DataFrame())
    trading_desk_html = _trading_desk_html(trading_desk_tables or {})
    data_quality_html = _data_quality_html(data_quality_summary if data_quality_summary is not None else pd.DataFrame(), data_quality_detail if data_quality_detail is not None else pd.DataFrame())
    portfolio_risk_html = _portfolio_risk_model_html(
        portfolio_risk_detail if portfolio_risk_detail is not None else pd.DataFrame(),
        portfolio_risk_summary if portfolio_risk_summary is not None else pd.DataFrame(),
        portfolio_risk_buckets if portfolio_risk_buckets is not None else pd.DataFrame(),
        portfolio_risk_stress if portfolio_risk_stress is not None else pd.DataFrame(),
    )
    manual_trading_html = _manual_trading_html(manual_blotter)
    attribution_html = _attribution_html(
        factor_ic if factor_ic is not None else pd.DataFrame(),
        family_ic if family_ic is not None else pd.DataFrame(),
        factor_corr if factor_corr is not None else pd.DataFrame(),
    )
    chart_html = ""
    for title, key in [
        ("净值曲线", "equity"),
        ("回撤曲线", "drawdown"),
        ("按年净值/回撤曲线", "yearly_equity_drawdown"),
        ("月度收益热力图", "monthly"),
        ("月度最大回撤热力图", "monthly_drawdown"),
        ("敞口与换手", "exposure"),
        ("当前持仓集中度", "positions"),
        ("最新手动交易规模", "manual_trading"),
    ]:
        if key in charts:
            chart_html += f"<section><h2>{title}</h2><img class='chart' src='{charts[key]}' alt='{title}'></section>"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>quant_system 回测报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 0; color: #172033; background: #f6f8fb; }}
    header {{ padding: 28px 36px; background: #111827; color: white; }}
    main {{ padding: 28px 36px 44px; max-width: 1320px; margin: 0 auto; }}
    h1 {{ margin: 0; font-size: 28px; }}
    h2 {{ margin: 24px 0 12px; font-size: 19px; }}
    h3 {{ margin: 16px 0 8px; font-size: 15px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; background: white; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f7; text-align: left; }}
    section {{ background: white; border: 1px solid #e5e7eb; padding: 18px; margin: 18px 0; }}
    .warn {{ background: #fff7ed; border: 1px solid #fdba74; padding: 12px 16px; margin: 18px 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: white; border: 1px solid #e5e7eb; padding: 14px; }}
    .metric span {{ display: block; color: #64748b; font-size: 12px; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 6px; }}
    .chart {{ width: 100%; max-width: 1120px; display: block; }}
    .note {{ color: #64748b; }}
    .pill {{ display: inline-block; background: #e0f2fe; color: #075985; padding: 5px 9px; margin: 4px 6px 4px 0; font-size: 12px; }}
    .trading-desk {{ border: 0; background: #0f172a; color: #e5eefb; padding: 0; overflow: hidden; }}
    .desk-head {{ padding: 22px 22px 14px; border-bottom: 1px solid rgba(148, 163, 184, 0.25); }}
    .desk-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }}
    .desk-title h2 {{ color: white; margin: 0; font-size: 22px; }}
    .desk-subtitle {{ color: #94a3b8; margin: 8px 0 0; }}
    .desk-grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-top: 16px; }}
    .desk-stat {{ background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.28); padding: 12px; }}
    .desk-stat span {{ display: block; color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
    .desk-stat strong {{ display: block; color: white; font-size: 20px; margin-top: 5px; }}
    .action-panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 18px 22px 6px; }}
    .action-panel {{ background: white; color: #172033; border: 1px solid #d8dee9; }}
    .action-panel h3 {{ margin: 0; padding: 12px 14px; font-size: 15px; border-bottom: 1px solid #e5e7eb; }}
    .action-list {{ padding: 8px 12px 12px; }}
    .action-item {{ display: grid; grid-template-columns: 76px 1fr auto; gap: 10px; align-items: center; padding: 8px 0; border-bottom: 1px solid #eef2f7; }}
    .action-item:last-child {{ border-bottom: 0; }}
    .ticker {{ font-weight: 750; letter-spacing: .02em; }}
    .action-meta {{ color: #64748b; font-size: 12px; }}
    .notional {{ font-variant-numeric: tabular-nums; font-weight: 700; }}
    .badge {{ display: inline-block; min-width: 62px; text-align: center; padding: 4px 7px; font-size: 11px; font-weight: 750; color: white; }}
    .badge-add, .badge-open-long {{ background: #16a34a; }}
    .badge-reduce, .badge-exit {{ background: #dc2626; }}
    .badge-hold {{ background: #64748b; }}
    .desk-table-wrap {{ padding: 14px 22px 22px; overflow-x: auto; }}
    .trading-table {{ min-width: 1180px; margin: 0; color: #172033; }}
    .trading-table th {{ position: sticky; top: 0; z-index: 1; background: #e2e8f0; white-space: nowrap; }}
    .trading-table td {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .trading-table .reason {{ min-width: 320px; white-space: normal; color: #475569; }}
    .row-add td, .row-open-long td {{ background: #f0fdf4; }}
    .row-exit td, .row-reduce td {{ background: #fff1f2; }}
    .row-hold td {{ background: #f8fafc; }}
    @media (max-width: 900px) {{
      .action-panels {{ grid-template-columns: 1fr; }}
      .desk-grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header><h1>quant_system 回测报告</h1><p class="note">日线收盘后生成信号，下一交易日模拟成交；包含交易成本、融券成本、止损、敞口和分段诊断。</p></header>
  <main>
	    <div class="warn"><b>风险提示与偏差检查</b><ul>{warning_html}</ul></div>
	    {headline_note}
	    <div class="metrics">{cards}</div>
    {chart_html}
    {period_breakdown_html}
    {drawdown_analysis_html}
    {risk_overlay_html}
    {intraday_exits_html}
    {capacity_html}
    {data_quality_html}
    {trading_desk_html}
	    {portfolio_risk_html}
	    {exposure_mode_html}
    {options_overlay_html}
    {theme_divergence_html}
    {institutional_attribution_html}
    {attribution_html}
    {manual_trading_html}
    <section>
      <h2>训练 / 验证 / 前推指标</h2>
      <p class="note">前推/当前阶段单独展示，默认不参与参数优化，避免把今年行情直接拿来调参。</p>
      {period_html}
    </section>
    <section>
      <h2>全部指标</h2>
      <table><tbody>{metric_rows}</tbody></table>
    </section>
    <section>
      <h2>近期成交与解释</h2>
      <table>
        <thead><tr><th>日期</th><th>股票</th><th>方向</th><th>目标权重</th><th>综合分</th><th>基本面</th><th>事件风险</th><th>成交解释</th></tr></thead>
        <tbody>{trade_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""


def _capacity_html(trades: pd.DataFrame) -> str:
    if trades.empty or "adv_participation" not in trades.columns:
        return """
    <section>
      <h2>容量与冲击成本</h2>
      <p class="note">本次回测没有可用 ADV 参与率字段，无法生成容量诊断。</p>
    </section>"""
    data = trades.copy()
    data["notional"] = pd.to_numeric(data.get("quantity"), errors="coerce").fillna(0) * pd.to_numeric(data.get("price"), errors="coerce").fillna(0)
    data["adv_participation"] = pd.to_numeric(data["adv_participation"], errors="coerce")
    data["execution_cost_bps"] = pd.to_numeric(data.get("execution_cost_bps"), errors="coerce")
    data["estimated_liquidity_cost"] = pd.to_numeric(data.get("estimated_liquidity_cost"), errors="coerce").fillna(0)
    valid = data[data["adv_participation"].notna()].copy()
    if valid.empty:
        return """
    <section>
      <h2>容量与冲击成本</h2>
      <p class="note">本次回测成交缺少有效 ADV，容量诊断只覆盖有流动性数据的交易。</p>
    </section>"""
    cards = [
        ("成交名义金额", _money(data["notional"].sum())),
        ("估算流动性成本", _money(data["estimated_liquidity_cost"].sum())),
        ("平均 ADV 参与率", _percent(valid["adv_participation"].mean())),
        ("95% ADV 参与率", _percent(valid["adv_participation"].quantile(0.95))),
        ("平均执行成本", f"{valid['execution_cost_bps'].mean():.1f} bps"),
        ("容量受限交易", str(int(data.get("capacity_limited", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()))),
    ]
    card_html = "".join(f"<div class='metric'><span>{label}</span><strong>{value}</strong></div>" for label, value in cards)
    top = valid.sort_values("adv_participation", ascending=False).head(15)
    rows = ""
    for _, row in top.iterrows():
        rows += (
            "<tr>"
            f"<td>{escape(str(row.get('date', '')))}</td>"
            f"<td>{escape(str(row.get('symbol', '')))}</td>"
            f"<td>{_money(row.get('notional'))}</td>"
            f"<td>{_percent(row.get('adv_participation'))}</td>"
            f"<td>{_number(row.get('execution_cost_bps'), digits=1)} bps</td>"
            f"<td>{_money(row.get('estimated_liquidity_cost'))}</td>"
            "</tr>"
        )
    return f"""
    <section>
      <h2>容量与冲击成本</h2>
      <p class="note">容量模型使用 ADV 参与率、半价差和非线性市场冲击估算单边成本；参与率越高，冲击成本按指数项上升。</p>
      <div class="metrics">{card_html}</div>
      <h3>最高 ADV 参与率成交</h3>
      <table><thead><tr><th>日期</th><th>股票</th><th>名义金额</th><th>ADV参与率</th><th>执行成本</th><th>估算成本</th></tr></thead><tbody>{rows}</tbody></table>
    </section>"""


def _exposure_mode_html(exposure_decomposition: pd.DataFrame) -> str:
    if exposure_decomposition.empty:
        return """
    <section>
      <h2>多空与 Beta 诊断</h2>
      <p class="note">本次报告没有可用的敞口模式分解。</p>
    </section>"""
    table = exposure_decomposition.to_html(index=False, border=0, classes="table", escape=True)
    return f"""
    <section>
      <h2>多空与 Beta 诊断</h2>
      <p class="note">这里把本次回测拆成“实际是否 long-only / 是否接近 market-neutral / short 是否真的贡献收益”。如果 short_contribution 为 0，报告不能证明做空 alpha。</p>
      {table}
    </section>"""


def _metric_value_text(value: object) -> str:
    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f"{float(value):.6f}"
    return str(value)


def _options_overlay_html(options_overlay: pd.DataFrame) -> str:
    if options_overlay.empty:
        return """
    <section>
      <h2>期权 Overlay 可执行合约</h2>
      <p class="note">本次没有触发期权 overlay 建议，或该功能未启用。</p>
    </section>"""
    cols = [
        col
        for col in [
            "symbol",
            "signal_date",
            "equity_action",
            "option_structure",
            "executable",
            "chain_validation_status",
            "contract_option_symbol",
            "contract_chain_quote_source",
            "contract_expiration",
            "contract_strike",
            "contract_mid",
            "contract_delta",
            "contract_iv",
            "contract_open_interest",
            "contract_volume",
            "contract_spread_pct_mid",
            "recommended_contracts",
            "estimated_option_premium",
            "premium_budget_utilization",
            "max_premium_budget",
            "final_score",
            "relative_strength_score",
            "overlay_note",
        ]
        if col in options_overlay.columns
    ]
    data = options_overlay[cols].copy()
    for column in ["contract_mid", "contract_delta", "contract_iv", "contract_spread_pct_mid", "max_premium_budget", "final_score", "relative_strength_score"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return f"""
    <section>
      <h2>期权 Overlay 可执行合约</h2>
      <p class="note">只有通过真实 option chain 的 DTE、delta、OI、成交量和 bid/ask spread 检查后，才会标记为 executable=true。否则只是方向性提示，不应直接下单。</p>
      {data.head(80).to_html(index=False, escape=False)}
    </section>"""


def _trading_desk_html(tables: dict[str, pd.DataFrame]) -> str:
    plan = tables.get("premarket_plan", pd.DataFrame())
    risk = tables.get("risk_state", pd.DataFrame())
    drift = tables.get("position_drift", pd.DataFrame())
    crowding = tables.get("crowding_watch", pd.DataFrame())
    if all(frame.empty for frame in [plan, risk, drift, crowding]):
        return ""
    risk_html = risk.to_html(index=False, escape=False) if not risk.empty else "<p class='note'>暂无风险状态。</p>"
    plan_cols = [col for col in ["symbol", "action_label", "delta_weight", "estimated_trade_notional", "priority_score", "execution_condition", "risk_instruction"] if col in plan.columns]
    plan_html = plan[plan_cols].head(30).to_html(index=False, escape=False) if plan_cols and not plan.empty else "<p class='note'>暂无盘前计划。</p>"
    drift_cols = [col for col in ["symbol", "action_label", "target_weight", "current_weight", "delta_weight", "abs_drift", "primary_theme"] if col in drift.columns]
    drift_html = drift[drift_cols].head(25).to_html(index=False, escape=False) if drift_cols and not drift.empty else "<p class='note'>暂无仓位偏离。</p>"
    crowd_cols = [col for col in ["bucket", "target_gross", "target_gross_change", "avg_relative_strength_score", "divergence_status", "suggested_tilt", "leaders"] if col in crowding.columns]
    crowd_html = crowding[crowd_cols].head(20).to_html(index=False, escape=False) if crowd_cols and not crowding.empty else "<p class='note'>暂无拥挤度表。</p>"
    return f"""
    <section>
      <h2>实时/盘中交易台</h2>
      <p class="note">把研究信号转成盘前计划、盘中风险状态、仓位偏离、主题拥挤和手动 override 记录模板。这里不是自动下单模块，而是实盘执行前的检查单。</p>
      <h3>盘中风险状态</h3>{risk_html}
      <h3>盘前执行计划</h3>{plan_html}
      <h3>当前持仓偏离</h3>{drift_html}
      <h3>行业/主题拥挤</h3>{crowd_html}
    </section>"""


def _data_quality_html(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    if summary.empty and detail.empty:
        return ""
    summary_html = summary.to_html(index=False, escape=False) if not summary.empty else "<p class='note'>暂无汇总。</p>"
    flagged = detail.copy()
    if not flagged.empty and "status" in flagged:
        flagged = flagged[flagged["status"].fillna("ok").ne("ok")]
    keep = [
        "symbol",
        "status",
        "latest_date",
        "stale_days_vs_global_latest",
        "missing_bar_count",
        "abnormal_jump_count",
        "split_adjustment_flag",
        "volume_anomaly_days",
        "broker_close_diff_pct",
        "earnings_calendar_status",
        "blocked_from_trading",
        "block_reason",
    ]
    detail_html = flagged[[col for col in keep if col in flagged.columns]].head(80).to_html(index=False, escape=False) if not flagged.empty else "<p class='note'>全部标的通过当前数据质量规则。</p>"
    return f"""
    <section>
      <h2>每日数据质量监控</h2>
      <p class="note">检查缺 bar、异常跳价、复权/分拆疑似错误、成交量异常、stale price、broker 对账差异和财报日历缺口。数据问题会直接影响信号可信度。</p>
      <h3>状态汇总</h3>{summary_html}
      <h3>异常标的</h3>{detail_html}
    </section>"""


def _portfolio_risk_model_html(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    buckets: pd.DataFrame | None = None,
    stress: pd.DataFrame | None = None,
) -> str:
    buckets = buckets if buckets is not None else pd.DataFrame()
    stress = stress if stress is not None else pd.DataFrame()
    if detail.empty and summary.empty and buckets.empty and stress.empty:
        return ""
    summary_html = summary.to_html(index=False, escape=False) if not summary.empty else "<p class='note'>暂无组合风险摘要。</p>"
    cols = [
        col
        for col in [
            "symbol",
            "weight",
            "annualized_vol_63d",
            "beta_to_benchmark_63d",
            "corr_to_benchmark_63d",
            "risk_contribution_pct",
            "primary_theme",
            "final_score",
            "relative_strength_score",
            "fundamental_score",
        ]
        if col in detail.columns
    ]
    data = detail[cols].head(30).copy() if cols and not detail.empty else pd.DataFrame()
    for column in ["weight", "annualized_vol_63d", "risk_contribution_pct"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    for column in ["beta_to_benchmark_63d", "corr_to_benchmark_63d", "final_score", "relative_strength_score", "fundamental_score"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
    detail_html = data.to_html(index=False, escape=False) if not data.empty else "<p class='note'>暂无风险贡献明细。</p>"
    bucket_data = buckets.copy()
    for column in ["gross_weight", "net_weight", "risk_contribution_pct"]:
        if column in bucket_data:
            bucket_data[column] = pd.to_numeric(bucket_data[column], errors="coerce").map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    for column in ["avg_beta_to_benchmark_63d"]:
        if column in bucket_data:
            bucket_data[column] = pd.to_numeric(bucket_data[column], errors="coerce").map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
    bucket_html = bucket_data.head(40).to_html(index=False, escape=False) if not bucket_data.empty else "<p class='note'>暂无主题/行业风险贡献。</p>"
    stress_data = stress.copy()
    if "estimated_pnl_pct_equity" in stress_data:
        stress_data["estimated_pnl_pct_equity"] = pd.to_numeric(stress_data["estimated_pnl_pct_equity"], errors="coerce").map(
            lambda value: f"{value:.2%}" if pd.notna(value) else ""
        )
    stress_html = stress_data.to_html(index=False, escape=False) if not stress_data.empty else "<p class='note'>暂无压力情景。</p>"
    return f"""
    <section>
      <h2>组合风险模型快照</h2>
      <p class="note">使用最近 63 个交易日估算组合波动、相对 benchmark beta/相关性、单票/主题/行业风险贡献和简化压力情景。这是交易前风险雷达，不是完整 Barra 风险模型。</p>
      <h3>组合摘要</h3>{summary_html}
      <h3>主题/行业风险贡献</h3>{bucket_html}
      <h3>压力情景</h3>{stress_html}
      <h3>单票风险贡献 Top 30</h3>{detail_html}
    </section>"""


def _institutional_attribution_html(tables: dict[str, pd.DataFrame]) -> str:
    if not tables:
        return ""
    sections = []
    labels = {
        "primary_theme": "按主题归因",
        "sector": "按行业归因",
        "industry": "按细分行业归因",
        "symbol": "按个股归因",
        "strategy_family": "按子策略归因",
        "entry_reason": "按入场理由归因",
        "factor_quintile": "按因子分位归因",
    }
    for name in ["primary_theme", "industry", "symbol", "strategy_family", "entry_reason", "factor_quintile"]:
        frame = tables.get(name, pd.DataFrame())
        if frame.empty:
            continue
        data = frame.head(25).copy()
        for column in ["avg_net_weight", "avg_gross_weight", "total_contribution", "avg_daily_contribution", "hit_rate", "worst_daily_contribution", "best_daily_contribution", "avg_weight"]:
            if column in data:
                data[column] = pd.to_numeric(data[column], errors="coerce").map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
        sections.append(f"<h3>{labels.get(name, name)}</h3>{data.to_html(index=False, escape=False)}")
    if not sections:
        return ""
    return f"""
    <section>
      <h2>机构级组合归因</h2>
      <p class="note">按主题、行业、个股、子策略、因子分位和入场理由拆解组合贡献。贡献为持仓权重乘以标的日收益的近似归因，用于定位收益/回撤来源。</p>
      {''.join(sections)}
    </section>"""


def _period_breakdown_html(yearly: pd.DataFrame, monthly: pd.DataFrame) -> str:
    if yearly.empty and monthly.empty:
        return ""
    yearly_html = "<p class='note'>暂无年度统计。</p>"
    if not yearly.empty:
        data = yearly.copy()
        for column in ["return", "max_drawdown", "volatility", "best_day", "worst_day", "avg_gross_exposure", "avg_turnover"]:
            if column in data:
                data[column] = data[column].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "")
        if "sharpe" in data:
            data["sharpe"] = data["sharpe"].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "")
        if "intraday_events" in data:
            data["intraday_events"] = data["intraday_events"].map(lambda value: f"{float(value):.0f}" if pd.notna(value) else "")
        labels = {
            "year": "年份",
            "return": "收益",
            "max_drawdown": "最大回撤",
            "volatility": "波动",
            "sharpe": "Sharpe",
            "best_day": "最佳日",
            "worst_day": "最差日",
            "avg_gross_exposure": "平均总敞口",
            "avg_turnover": "平均换手",
            "intraday_events": "分钟风控事件",
        }
        data = data.rename(columns=labels)
        yearly_html = data.to_html(index=False, escape=False)
    monthly_html = "<p class='note'>暂无月度统计。</p>"
    if not monthly.empty:
        data = monthly.copy().tail(36)
        for column in ["return", "max_drawdown", "best_day", "worst_day", "avg_gross_exposure", "avg_turnover"]:
            if column in data:
                data[column] = data[column].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "")
        if "intraday_events" in data:
            data["intraday_events"] = data["intraday_events"].map(lambda value: f"{float(value):.0f}" if pd.notna(value) else "")
        labels = {
            "month": "月份",
            "return": "收益",
            "max_drawdown": "月内最大回撤",
            "best_day": "最佳日",
            "worst_day": "最差日",
            "avg_gross_exposure": "平均总敞口",
            "avg_turnover": "平均换手",
            "intraday_events": "分钟风控事件",
        }
        monthly_html = data.rename(columns=labels).to_html(index=False, escape=False)
    return f"""
    <section>
      <h2>按年 / 按月表现拆解</h2>
      <p class="note">年度表用于看策略结构是否稳定；月度表重点显示最近 36 个月的收益、月内最大回撤、换手和分钟级风控频率。</p>
      <h3>年度表现</h3>
      {yearly_html}
      <h3>最近 36 个月</h3>
      {monthly_html}
    </section>"""


def _drawdown_analysis_html(periods: pd.DataFrame, contributors: pd.DataFrame) -> str:
    if periods.empty and contributors.empty:
        return """
    <section>
      <h2>回撤成分分析</h2>
      <p class="note">暂无足够数据生成回撤成分分析。</p>
    </section>"""
    period_html = "<p class='note'>暂无回撤区间。</p>"
    if not periods.empty:
        data = periods.head(8).copy()
        for column in ["start_date", "trough_date", "end_date"]:
            if column in data:
                data[column] = pd.to_datetime(data[column], errors="coerce").dt.date.astype(str).replace("NaT", "未修复")
        for column in ["max_drawdown", "return_start_to_trough"]:
            if column in data:
                data[column] = data[column].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "")
        labels = {
            "start_date": "起点",
            "trough_date": "谷底",
            "end_date": "修复日",
            "max_drawdown": "最大回撤",
            "days_to_trough": "到谷底天数",
            "recovery_days": "修复天数",
            "return_start_to_trough": "起点到谷底",
        }
        period_html = data.rename(columns=labels).to_html(index=False, escape=False)
    contributor_html = "<p class='note'>暂无单票贡献。</p>"
    if not contributors.empty:
        rows = ""
        data = contributors.copy()
        data["drawdown_date"] = pd.to_datetime(data["drawdown_date"]).dt.date.astype(str)
        for date, day in data.groupby("drawdown_date", sort=False):
            daily_return = _percent(day["daily_return"].iloc[0])
            intraday_events = _number(day["intraday_risk_event_count"].iloc[0], digits=0)
            exit_notional = _money(day["intraday_exit_notional"].iloc[0])
            rows += f"<tr><th colspan='7'>日期 {escape(date)}，组合日收益 {daily_return}，分钟风控事件 {intraday_events}，日内退出 {exit_notional}</th></tr>"
            for _, row in day.iterrows():
                rows += (
                    "<tr>"
                    f"<td>{escape(str(row.get('symbol', '')))}</td>"
                    f"<td>{_percent(row.get('previous_weight'))}</td>"
                    f"<td>{_percent(row.get('symbol_return'))}</td>"
                    f"<td>{_money(row.get('pnl'))}</td>"
                    f"<td>{_percent(row.get('portfolio_contribution'))}</td>"
                    f"<td>{_percent(row.get('gross_exposure'))}</td>"
                    f"<td>{_percent(row.get('turnover'))}</td>"
                    "</tr>"
                )
        contributor_html = f"<table><thead><tr><th>股票</th><th>前日权重</th><th>个股收益</th><th>PnL</th><th>组合贡献</th><th>总敞口</th><th>换手</th></tr></thead><tbody>{rows}</tbody></table>"
    return f"""
    <section>
      <h2>回撤成分分析</h2>
      <p class="note">先定位最大回撤区间，再拆解最差交易日的前日持仓贡献。若当天有分钟级风控成交，表头会显示事件数和退出名义金额。</p>
      <h3>最大回撤区间</h3>
      {period_html}
      <h3>最差交易日单票贡献</h3>
      {contributor_html}
    </section>"""
    data = trades.copy()
    data["notional"] = pd.to_numeric(data.get("quantity"), errors="coerce").fillna(0) * pd.to_numeric(data.get("price"), errors="coerce").fillna(0)
    data["adv_participation"] = pd.to_numeric(data["adv_participation"], errors="coerce")
    data["execution_cost_bps"] = pd.to_numeric(data.get("execution_cost_bps"), errors="coerce")
    data["estimated_liquidity_cost"] = pd.to_numeric(data.get("estimated_liquidity_cost"), errors="coerce").fillna(0)
    valid = data[data["adv_participation"].notna()].copy()
    if valid.empty:
        return """
    <section>
      <h2>容量与冲击成本</h2>
      <p class="note">本次回测成交缺少有效 ADV，容量诊断只覆盖有流动性数据的交易。</p>
    </section>"""
    cards = [
        ("成交名义金额", _money(data["notional"].sum())),
        ("估算流动性成本", _money(data["estimated_liquidity_cost"].sum())),
        ("平均 ADV 参与率", _percent(valid["adv_participation"].mean())),
        ("95% ADV 参与率", _percent(valid["adv_participation"].quantile(0.95))),
        ("平均执行成本", f"{valid['execution_cost_bps'].mean():.1f} bps"),
        ("容量受限交易", str(int(data.get("capacity_limited", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()))),
    ]
    card_html = "".join(f"<div class='metric'><span>{label}</span><strong>{value}</strong></div>" for label, value in cards)
    top = valid.sort_values("adv_participation", ascending=False).head(15)
    rows = ""
    for _, row in top.iterrows():
        rows += (
            "<tr>"
            f"<td>{escape(str(row.get('date', '')))}</td>"
            f"<td>{escape(str(row.get('symbol', '')))}</td>"
            f"<td>{_money(row.get('notional'))}</td>"
            f"<td>{_percent(row.get('adv_participation'))}</td>"
            f"<td>{_number(row.get('execution_cost_bps'), digits=1)} bps</td>"
            f"<td>{_money(row.get('estimated_liquidity_cost'))}</td>"
            "</tr>"
        )
    return f"""
    <section>
      <h2>容量与冲击成本</h2>
      <p class="note">容量模型使用 ADV 参与率、半价差和非线性市场冲击估算单边成本；参与率越高，冲击成本按指数项上升。</p>
      <div class="metrics">{card_html}</div>
      <h3>最高 ADV 参与率成交</h3>
      <table><thead><tr><th>日期</th><th>股票</th><th>名义金额</th><th>ADV参与率</th><th>执行成本</th><th>估算成本</th></tr></thead><tbody>{rows}</tbody></table>
    </section>"""


def _intraday_exits_html(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> str:
    if trades.empty or "intraday_risk_exit" not in trades.columns:
        return """
    <section>
      <h2>分钟级风控成交</h2>
      <p class="note">本次回测没有真实分钟级风控成交。若启用 intraday_enabled，需要先下载分钟/5分钟 OHLCV。</p>
    </section>"""
    intraday_flags = trades["intraday_risk_exit"].map(lambda value: bool(value) if pd.notna(value) else False)
    exits = trades[intraday_flags].copy()
    if exits.empty:
        return """
    <section>
      <h2>分钟级风控成交</h2>
      <p class="note">已接入分钟级风控引擎，但本次回测未触发日内减仓或止损。</p>
    </section>"""
    exits["notional"] = pd.to_numeric(exits["quantity"], errors="coerce").fillna(0) * pd.to_numeric(exits["price"], errors="coerce").fillna(0)
    summary = exits.groupby("date", as_index=False).agg(
        exits=("symbol", "count"),
        symbols=("symbol", "nunique"),
        exit_notional=("notional", "sum"),
    )
    if not equity_curve.empty and "intraday_risk_event_count" in equity_curve.columns:
        daily = equity_curve[["date", "intraday_risk_event_count", "intraday_exit_notional"]].copy()
        summary = summary.merge(daily, on="date", how="left")
    summary = summary.sort_values("date", ascending=False)
    rows = ""
    for _, row in summary.head(12).iterrows():
        rows += (
            "<tr>"
            f"<td>{escape(str(row.get('date', '')))}</td>"
            f"<td>{int(row.get('exits', 0))}</td>"
            f"<td>{int(row.get('symbols', 0))}</td>"
            f"<td>{_money(row.get('exit_notional'))}</td>"
            f"<td>{_number(row.get('intraday_risk_event_count'), digits=0)}</td>"
            "</tr>"
        )
    latest = exits.sort_values(["date", "datetime"] if "datetime" in exits.columns else ["date"], ascending=False).head(20)
    trade_rows = ""
    for _, row in latest.iterrows():
        trade_rows += (
            "<tr>"
            f"<td>{escape(str(row.get('datetime', row.get('date', ''))))}</td>"
            f"<td>{escape(str(row.get('symbol', '')))}</td>"
            f"<td>{escape(_side_label(str(row.get('side', ''))))}</td>"
            f"<td>{_money(row.get('notional'))}</td>"
            f"<td>{escape(str(row.get('reason_for_exit', '')))}</td>"
            "</tr>"
        )
    return f"""
    <section>
      <h2>分钟级风控成交</h2>
      <p class="note">这些成交使用真实分钟/5分钟 bar，在当天触发后按触发分钟 close 加滑点模拟成交，会直接改变当天 PnL。</p>
      <h3>触发日期汇总</h3>
      <table><thead><tr><th>日期</th><th>成交数</th><th>股票数</th><th>退出名义金额</th><th>日内事件数</th></tr></thead><tbody>{rows}</tbody></table>
      <h3>最近触发成交</h3>
      <table><thead><tr><th>时间</th><th>股票</th><th>方向</th><th>名义金额</th><th>原因</th></tr></thead><tbody>{trade_rows}</tbody></table>
    </section>"""


def _risk_overlay_html(risk_overlay: pd.DataFrame) -> str:
    if risk_overlay.empty:
        return """
    <section>
      <h2>风控层与空仓信号</h2>
      <p class="note">本次回测未启用或无法计算过热/空仓风控层。</p>
    </section>"""
    data = risk_overlay.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.date
    latest = data.tail(1).iloc[0]
    cash_signal = bool(latest.get("cash_signal", False)) or bool(latest.get("intraday_cash_signal", False))
    exposure_multiplier = min(
        float(latest.get("overbought_multiplier", 1.0) if pd.notna(latest.get("overbought_multiplier", 1.0)) else 1.0),
        float(latest.get("intraday_risk_multiplier", 1.0) if pd.notna(latest.get("intraday_risk_multiplier", 1.0)) else 1.0),
    )
    latest_cards = [
        ("日期", str(latest.get("date", ""))),
        ("过热分", f"{float(latest.get('overbought_score', 0.0)):.0f}"),
        ("状态", _status_label(str(latest.get("overbought_status", "")))),
        ("盘中风险分", f"{float(latest.get('intraday_risk_score', 0.0)):.0f}"),
        ("盘中状态", _status_label(str(latest.get("intraday_risk_status", "")))),
        ("空仓信号", "是" if cash_signal else "否"),
        ("敞口乘数", f"{exposure_multiplier:.2f}"),
        ("基准 RSI", f"{float(latest.get('rsi_14', 0.0)):.1f}"),
    ]
    cards = "".join(f"<div class='metric'><span>{label}</span><strong>{value}</strong></div>" for label, value in latest_cards)
    recent = data.tail(10).copy()
    for column in [
        "ret_5d",
        "ret_20d",
        "dist_ma20",
        "dist_ma50",
        "pct_rsi_over_70",
        "pct_z_over_2",
        "pct_near_high",
        "benchmark_intraday_return",
        "benchmark_gap_return",
        "benchmark_low_from_open",
        "benchmark_high_low_range",
        "pct_down_from_open",
        "pct_large_intraday_loss",
        "pct_gap_down",
        "pct_large_range",
        "pct_volume_expansion",
        "median_intraday_return",
    ]:
        if column in recent:
            recent[column] = recent[column].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "")
    for column in ["overbought_score", "rsi_14", "intraday_risk_score", "benchmark_volume_expansion", "median_volume_expansion"]:
        if column in recent:
            recent[column] = recent[column].map(lambda value: f"{float(value):.1f}" if pd.notna(value) else "")
    display_columns = [
        "date",
        "overbought_score",
        "overbought_status",
        "overbought_multiplier",
        "intraday_risk_score",
        "intraday_risk_status",
        "intraday_risk_multiplier",
        "benchmark_intraday_return",
        "benchmark_gap_return",
        "pct_down_from_open",
        "pct_large_intraday_loss",
        "pct_volume_expansion",
    ]
    recent = recent[[column for column in display_columns if column in recent.columns]]
    table = recent.to_html(index=False)
    return f"""
    <section>
      <h2>风控层、空仓与盘中代理风险</h2>
      <p class="note">过热与盘中代理风险都在收盘后评估，用于下一交易日降低敞口或触发普通策略空仓；当前盘中层使用日线 OHLCV 近似，接入分钟数据后可升级为真正实时风控。</p>
      <div class="metrics">{cards}</div>
      {table}
    </section>"""


def _theme_divergence_html(theme_divergence: pd.DataFrame) -> str:
    if theme_divergence.empty:
        return """
    <section>
      <h2>主题 / 板块分化</h2>
      <p class="note">本次回测没有可用主题或板块字段，无法生成分化视图。</p>
    </section>"""
    data = theme_divergence.copy().head(30)
    latest_date = ""
    if "date" in data:
        latest_date = str(pd.to_datetime(data["date"], format="mixed", errors="coerce").max().date())
    labels = {
        "bucket": "主题/板块",
        "divergence_status": "状态",
        "suggested_tilt": "操作倾向",
        "symbols": "标的数",
        "target_gross": "目标总敞口",
        "target_net": "目标净敞口",
        "target_gross_change": "敞口变化",
        "avg_final_score": "综合分",
        "avg_relative_strength_score": "相对强弱",
        "avg_technical_score": "技术分",
        "avg_fundamental_score": "基本面",
        "avg_ret_5d": "5日收益",
        "avg_ret_20d": "20日收益",
        "avg_ret_63d": "63日收益",
        "positive_20d_rate": "20日上涨占比",
        "near_high_rate": "近高位占比",
        "avg_volume_expansion": "量能倍数",
        "buy_notional": "买入额",
        "sell_notional": "卖出额",
        "net_trade_notional": "净流入",
        "leaders": "当前龙头",
    }
    keep = [
        "bucket",
        "divergence_status",
        "suggested_tilt",
        "symbols",
        "target_gross",
        "target_net",
        "target_gross_change",
        "avg_final_score",
        "avg_relative_strength_score",
        "avg_technical_score",
        "avg_fundamental_score",
        "avg_ret_5d",
        "avg_ret_20d",
        "avg_ret_63d",
        "positive_20d_rate",
        "near_high_rate",
        "avg_volume_expansion",
        "buy_notional",
        "sell_notional",
        "net_trade_notional",
        "leaders",
    ]
    keep = [column for column in keep if column in data.columns]
    rows = ""
    for _, row in data[keep].iterrows():
        cells = []
        status = str(row.get("divergence_status", ""))
        row_class = "theme-" + re.sub(r"[^a-z0-9]+", "-", status.lower()).strip("-")
        for column in keep:
            value = row.get(column, "")
            if column in {"target_gross", "target_net", "target_gross_change", "avg_ret_5d", "avg_ret_20d", "avg_ret_63d", "positive_20d_rate", "near_high_rate"}:
                formatted = _percent(value)
            elif column in {"buy_notional", "sell_notional", "net_trade_notional"}:
                formatted = _money(value)
            elif column in {"avg_final_score", "avg_relative_strength_score", "avg_technical_score", "avg_fundamental_score", "avg_volume_expansion"}:
                formatted = _number(value, digits=1)
            else:
                formatted = escape("" if pd.isna(value) else str(value))
            cells.append(f"<td>{formatted}</td>")
        rows += f"<tr class='{row_class}'>" + "".join(cells) + "</tr>"
    header = "".join(f"<th>{escape(labels.get(column, column))}</th>" for column in keep)
    return f"""
    <section>
      <h2>主题 / 板块分化</h2>
      <p class="note">截至 {escape(latest_date)}，此表按主题聚合目标敞口、强弱、短中期收益和调仓方向。它用于判断 AI 主线内部是扩散、轮动还是退潮。</p>
      <table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>
    </section>"""


def _attribution_html(factor_ic: pd.DataFrame, family_ic: pd.DataFrame, factor_corr: pd.DataFrame) -> str:
    if factor_ic.empty and family_ic.empty and factor_corr.empty:
        return """
    <section>
      <h2>因子与子策略归因</h2>
      <p class="note">本次回测没有足够数据计算横截面 IC 归因。</p>
    </section>"""
    factor_table = _ic_table_html(
        factor_ic,
        columns=["period", "horizon", "label", "mean_rank_ic", "ic_t_stat", "positive_ic_rate", "n_days"],
        top_per_group=8,
    )
    family_table = _ic_table_html(
        family_ic,
        columns=["period", "horizon", "label", "mean_rank_ic", "ic_t_stat", "positive_ic_rate", "n_days"],
        top_per_group=5,
    )
    corr_table = _factor_correlation_html(factor_corr)
    return f"""
    <section>
      <h2>因子与子策略归因</h2>
      <p class="note">IC 是每日横截面排名相关：信号日因子分数 vs 未来收益。forward/current 单独展示，避免把当前 AI 主线结果混进训练期。</p>
      <h3>子策略族群 IC</h3>
      {family_table}
      <h3>单因子 IC</h3>
      {factor_table}
      <h3>因子共线性诊断</h3>
      {corr_table}
    </section>"""


def _factor_correlation_html(data: pd.DataFrame) -> str:
    if data.empty:
        return "<p class='note'>暂无因子相关性数据。</p>"
    frame = data.copy().head(20)
    for column in ["rank_correlation", "abs_correlation"]:
        if column in frame:
            frame[column] = frame[column].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "")
    labels = {
        "factor_a": "因子 A",
        "factor_b": "因子 B",
        "rank_correlation": "排名相关",
        "abs_correlation": "绝对相关",
    }
    note = "<p class='note'>绝对相关高于 0.80 的因子应避免重复计权，优先保留 IC 更稳定且解释更直接的一项。</p>"
    return note + frame.rename(columns=labels).to_html(index=False, escape=False)


def _ic_table_html(data: pd.DataFrame, columns: list[str], top_per_group: int) -> str:
    if data.empty:
        return "<p class='note'>暂无归因数据。</p>"
    frame = data.copy()
    if "horizon_days" in frame.columns:
        frame = frame.sort_values(["period", "horizon_days", "mean_rank_ic"], ascending=[True, True, False])
    elif "mean_rank_ic" in frame.columns:
        frame = frame.sort_values("mean_rank_ic", ascending=False)
    group_keys = [key for key in ["period", "horizon"] if key in frame.columns]
    if group_keys:
        frame = frame.groupby(group_keys, group_keys=False).head(top_per_group)
    labels = {
        "period": "阶段",
        "horizon": "预测周期",
        "label": "因子",
        "family": "子策略族群",
        "mean_rank_ic": "平均IC",
        "ic_t_stat": "t值",
        "positive_ic_rate": "IC胜率",
        "n_days": "天数",
    }
    keep = [column for column in columns if column in frame.columns]
    header = "".join(f"<th>{escape(labels.get(column, column))}</th>" for column in keep)
    rows = ""
    for _, row in frame[keep].iterrows():
        cells = []
        for column in keep:
            value = row.get(column, "")
            if column == "period":
                formatted = escape(_period_label(str(value)))
            elif column == "mean_rank_ic":
                formatted = _number(value, digits=4)
            elif column == "ic_t_stat":
                formatted = _number(value, digits=2)
            elif column == "positive_ic_rate":
                formatted = _percent(value)
            else:
                formatted = escape("" if pd.isna(value) else str(value))
            cells.append(f"<td>{formatted}</td>")
        rows += "<tr>" + "".join(cells) + "</tr>"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"


def _manual_trading_html(manual_blotter: pd.DataFrame) -> str:
    """Render a trading-desk style manual blotter."""

    if manual_blotter.empty:
        return """
    <section class="trading-desk">
      <div class="desk-head">
        <div class="desk-title"><h2>手动交易视图</h2></div>
        <p class="desk-subtitle">最新信号日没有达到阈值的目标仓位变化。</p>
      </div>
    </section>"""
    data = manual_blotter.copy()
    signal_date = str(data.get("signal_date", pd.Series([""])).iloc[0])
    next_execution = str(data.get("next_execution", pd.Series(["下一交易日开盘"])).iloc[0])
    counts = data["action"].value_counts().to_dict() if "action" in data else {}
    net_notional = float(pd.to_numeric(data.get("estimated_trade_notional", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    gross_notional = float(pd.to_numeric(data.get("estimated_trade_notional", pd.Series(dtype=float)), errors="coerce").abs().fillna(0).sum())
    cash_signal = bool(data.get("cash_signal", pd.Series([False])).astype(str).str.lower().eq("true").any())
    overbought_score = pd.to_numeric(data.get("overbought_score", pd.Series(dtype=float)), errors="coerce").dropna()
    overbought_display = f"{overbought_score.iloc[-1]:.0f}" if not overbought_score.empty else "n/a"
    stats = [
        ("信号日", signal_date[:10]),
        ("计划成交", next_execution),
        ("空仓信号", "是" if cash_signal else "否"),
        ("过热分", overbought_display),
        ("新开/加仓", str(int(counts.get("OPEN LONG", 0) + counts.get("ADD", 0)))),
        ("清仓/减仓", str(int(counts.get("EXIT", 0) + counts.get("REDUCE", 0)))),
        ("总调仓额", _money(gross_notional)),
        ("净调仓额", _money(net_notional)),
    ]
    stat_html = "".join(f"<div class='desk-stat'><span>{escape(label)}</span><strong>{escape(value)}</strong></div>" for label, value in stats)
    buys = data[data["action"].isin(["OPEN LONG", "ADD"])].sort_values("estimated_trade_notional", ascending=False).head(10)
    sells = data[data["action"].isin(["EXIT", "REDUCE"])].sort_values("estimated_trade_notional", ascending=True).head(10)
    buy_html = _action_list_html(buys, "买入 / 加仓")
    sell_html = _action_list_html(sells, "卖出 / 减仓 / 清仓")
    table_html = _manual_table_html(data)
    return f"""
    <section class="trading-desk">
      <div class="desk-head">
        <div class="desk-title">
          <h2>手动交易视图</h2>
          <span class="pill">辅助决策，不是成交保证</span>
        </div>
        <p class="desk-subtitle">展示最新目标仓位与当前仓位的差异。信号在收盘后生成，计划用于下一交易日开盘模拟执行。</p>
        <div class="desk-grid">{stat_html}</div>
      </div>
      <div class="action-panels">{buy_html}{sell_html}</div>
      <div class="desk-table-wrap">{table_html}</div>
    </section>"""


def _action_list_html(frame: pd.DataFrame, title: str) -> str:
    rows = ""
    if frame.empty:
        rows = "<div class='action-item'><span class='badge badge-hold'>无</span><span>没有动作</span><span></span></div>"
    for _, row in frame.iterrows():
        action = str(row.get("action", ""))
        badge = _badge(action)
        symbol = escape(str(row.get("symbol", "")))
        target = _percent(row.get("target_weight"))
        delta = _percent(row.get("delta_weight"))
        score = _number(row.get("final_score"), digits=1)
        notional = _money(row.get("estimated_trade_notional"))
        rows += (
            "<div class='action-item'>"
            f"{badge}"
            f"<div><div class='ticker'>{symbol}</div><div class='action-meta'>目标 {target} / 变化 {delta} / 分数 {score}</div></div>"
            f"<div class='notional'>{notional}</div>"
            "</div>"
        )
    return f"<div class='action-panel'><h3>{escape(title)}</h3><div class='action-list'>{rows}</div></div>"


def _manual_table_html(data: pd.DataFrame) -> str:
    keep = [
        "action",
        "symbol",
        "target_weight",
        "current_weight",
        "delta_weight",
        "estimated_trade_notional",
        "estimated_shares",
        "close",
        "final_score",
        "technical_score",
        "relative_strength_score",
        "fundamental_score",
        "sector",
        "industry",
        "fundamental_data_age_days",
        "event_risk_score",
        "days_to_earnings",
        "next_earnings_date",
        "expected_move",
        "mom_return",
        "mom_distance_high",
        "mom_volume_expansion",
        "retest_component",
        "retest_distance_ma_pct",
        "overbought_score",
        "overbought_status",
        "cash_signal",
        "crowding_risk_score",
        "crowding_multiplier",
        "primary_theme",
        "overnight_gap_risk_score",
        "overnight_gap_risk_status",
        "minute_pretrade_risk_score",
        "minute_pretrade_risk_status",
        "decision_note",
    ]
    keep = [column for column in keep if column in data.columns]
    column_labels = {
        "action": "动作",
        "symbol": "股票",
        "target_weight": "目标权重",
        "current_weight": "当前权重",
        "delta_weight": "权重变化",
        "estimated_trade_notional": "估算金额",
        "estimated_shares": "估算股数",
        "close": "参考收盘价",
        "final_score": "综合分",
        "technical_score": "技术分",
        "relative_strength_score": "相对强弱",
        "fundamental_score": "基本面",
        "sector": "行业",
        "industry": "细分行业",
        "fundamental_data_age_days": "基本面数据年龄",
        "event_risk_score": "事件风险",
        "days_to_earnings": "距财报日",
        "next_earnings_date": "下次财报日",
        "expected_move": "预期波动",
        "mom_return": "动量收益",
        "mom_distance_high": "距前高",
        "mom_volume_expansion": "量能倍数",
        "retest_component": "回踩分",
        "retest_distance_ma_pct": "距回踩均线",
        "overbought_score": "过热分",
        "overbought_status": "过热状态",
        "cash_signal": "空仓信号",
        "crowding_risk_score": "拥挤风险",
        "crowding_multiplier": "拥挤乘数",
        "primary_theme": "主题",
        "overnight_gap_risk_score": "隔夜跳空风险",
        "overnight_gap_risk_status": "跳空状态",
        "minute_pretrade_risk_score": "分钟预算风险",
        "minute_pretrade_risk_status": "分钟预算状态",
        "decision_note": "交易解释",
        "reason_for_entry": "原始进场理由",
        "reason_for_exit": "原始退出理由",
    }
    header = "".join(f"<th>{escape(column_labels.get(column, column))}</th>" for column in keep)
    body = ""
    display = data.copy().head(160)
    for _, row in display[keep].iterrows():
        action = str(row.get("action", ""))
        row_class = "row-" + action.lower().replace(" ", "-")
        cells = []
        for column in keep:
            value = row.get(column, "")
            if column == "action":
                formatted = _badge(action)
            elif column == "symbol":
                formatted = f"<span class='ticker'>{escape(str(value))}</span>"
            elif column in {"target_weight", "current_weight", "delta_weight", "mom_return", "mom_distance_high", "retest_distance_ma_pct"}:
                formatted = _percent(value)
            elif column == "estimated_trade_notional":
                formatted = _money(value)
            elif column == "estimated_shares":
                formatted = _number(value, digits=1)
            elif column == "event_risk_score":
                event = _safe_float(value)
                days = _safe_float(row.get("days_to_earnings"), default=np.nan)
                formatted = "无已知临近事件" if event == 0 and not np.isfinite(days) else _number(value, digits=2)
            elif column == "retest_component":
                retest = _safe_float(value, default=np.nan)
                formatted = "无有效回踩" if not np.isfinite(retest) or retest <= 0 else _number(value, digits=1)
            elif column in {
                "close",
                "final_score",
                "technical_score",
                "relative_strength_score",
                "fundamental_score",
                "overbought_score",
                "days_to_earnings",
                "mom_volume_expansion",
                "crowding_risk_score",
                "crowding_multiplier",
                "overnight_gap_risk_score",
                "minute_pretrade_risk_score",
                "fundamental_data_age_days",
            }:
                formatted = _number(value, digits=2)
            elif column == "overbought_status":
                formatted = escape(_status_label(str(value)))
            elif column == "cash_signal":
                formatted = "是" if str(value).lower() == "true" else "否"
            elif column == "decision_note":
                formatted = escape("" if pd.isna(value) else str(value))
                cells.append(f"<td class='reason'>{formatted}</td>")
                continue
            else:
                formatted = escape("" if pd.isna(value) else str(value))
            cells.append(f"<td>{formatted}</td>")
        body += f"<tr class='{row_class}'>" + "".join(cells) + "</tr>"
    return f"<table class='trading-table'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _badge(action: str) -> str:
    cls = "badge-" + action.lower().replace(" ", "-")
    return f"<span class='badge {cls}'>{escape(_action_label(action))}</span>"


def _money(value) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _percent(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _number(value, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.{digits}f}"
