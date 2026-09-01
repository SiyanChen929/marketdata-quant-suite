"""Command-line interface for quant_system."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
import shutil
import time
import traceback
from datetime import datetime
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd
from quant_marketdata import MarketDataStore

from quant_system.backtest.reports import save_run
from quant_system.config import AppConfig, dump_config, load_config
from quant_system.data.marketdata_provider import MarketDataAPIProvider
from quant_system.data.intraday_provider import MarketDataIntradayProvider
from quant_system.data.lake import QuantSystemLake, duckdb_available
from quant_system.data.maintenance import (
    build_cleanup_plan,
    bytes_label,
    consolidate_daily_ohlcv,
    execute_cleanup,
    storage_inventory,
)
from quant_system.data.audit import run_data_audit
from quant_system.data.quality_monitor import run_daily_data_quality_monitor
from quant_system.data.sample_data import generate_sample_data
from quant_system.experiments.management import build_experiment_management_tables, write_experiment_management_report
from quant_system.experiments.registry import compare_runs, record_run
from quant_system.features.relative_strength import add_relative_strength
from quant_system.features.technical import add_technical_features
from quant_system.backtest.metrics import equity_metrics
from quant_system.live.monitor import run_live_monitor
from quant_system.live.runner import create_live_plan, submit_live_plan
from quant_system.optimization.activation_validator import save_activation_validation_report, selected_row_activation_summary
from quant_system.optimization.factor_fitting import fit_factor_model, save_factor_fit_report
from quant_system.optimization.factor_mining import add_activation_diagnostics, mine_factors, save_factor_mining_report
from quant_system.optimization.leader_substitution_audit import (
    build_leader_substitution_audit,
    save_leader_substitution_audit,
    summarize_leader_substitution_audit,
)
from quant_system.optimization.holding_timing_audit import (
    build_holding_timing_audit,
    save_holding_timing_audit,
    summarize_holding_timing_audit,
)
from quant_system.optimization.parameter_search import default_research_scenarios, objective_score
from quant_system.optimization.reports import save_optimization_report, save_topn_full_validation_report
from quant_system.optimization.robustness import deflated_sharpe_ratio, estimate_pbo, flag_overfit_risk, parameter_stability
from quant_system.optimization.walk_forward import make_walk_forward_splits
from quant_system.portfolio.construction import construct_target_weights
from quant_system.portfolio.crowding import apply_crowding_risk_to_targets
from quant_system.portfolio.gap_risk import apply_overnight_gap_risk_to_targets
from quant_system.research import (
    _attach_theme_memberships,
    build_features_and_context,
    generate_cached_ensemble_signals,
    prepare_research_bundle,
    prepare_research_data,
    run_research_backtest,
)
from quant_system.regime.detector import apply_regime_exposure_overlay
from quant_system.strategies.base import StrategyContext
from quant_system.trade_location import apply_trade_location_to_targets
from quant_system.universe.filters import build_asof_tradable_universe
from quant_system.universe.thematic import dynamic_theme_scores
from quant_system.utils.logging import setup_logging


_OPTIMIZATION_INPUT_CACHE: dict[tuple, tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame, object]] = {}
_SAME_THEME_SUBSTITUTION_MOMENTUM_KEYS = (
    "pullback_reset_same_theme_substitution_overlay",
    "pullback_reset_same_theme_substitution_min_relative_strength_edge",
    "pullback_reset_same_theme_substitution_min_theme_score_edge",
    "pullback_reset_same_theme_substitution_min_reset_score",
    "pullback_reset_same_theme_substitution_min_reclaim_score",
    "pullback_reset_same_theme_substitution_min_theme_strength_delta_score",
    "pullback_reset_same_theme_substitution_max_final_score_deficit",
    "pullback_reset_same_theme_substitution_max_score_rank_gap",
    "pullback_reset_same_theme_substitution_max_event_risk_score_circuit_breaker",
    "pullback_reset_same_theme_substitution_max_overnight_gap_risk_score_circuit_breaker",
    "pullback_reset_same_theme_substitution_max_vol_20d_circuit_breaker",
    "pullback_reset_same_theme_substitution_max_promotions_per_date",
    "pullback_reset_same_theme_substitution_require_theme_active",
    "pullback_reset_same_theme_substitution_require_activation_support",
    "pullback_reset_same_theme_substitution_require_theme_breadth_not_deteriorating",
    "pullback_reset_same_theme_substitution_theme_breadth_lookback_days",
    "pullback_reset_same_theme_substitution_theme_breadth_active_share_threshold",
    "pullback_reset_same_theme_substitution_theme_breadth_shortfall_threshold",
)


def main(argv: list[str] | None = None) -> None:
    matplotlib_cache = Path("data/cache/matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))

    parser = argparse.ArgumentParser(description="quant_system research CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "backtest", "optimize"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/default.yml")
        if name == "backtest":
            command.add_argument("--fast", action="store_true", help="Skip heavy CSV artifacts and chart rendering for iterative research.")
        if name == "optimize":
            command.add_argument("--profile", choices=["quick", "research", "aggressive", "cycle_return", "sharpe15"], default="quick")
            command.add_argument("--max-runs", type=int, default=None)
            command.add_argument("--jobs", type=int, default=1)
            command.add_argument("--second-stage-top-n", type=int, default=None)
            command.add_argument(
                "--scenario-contains",
                action="append",
                default=None,
                help="Only evaluate scenarios whose name, strategy profile, or constraint profile contains this text. Can be repeated.",
            )
    intraday = sub.add_parser("download-intraday")
    intraday.add_argument("--config", default="configs/default.yml")
    intraday.add_argument("--jobs", type=int, default=None)
    incremental = sub.add_parser("update-market-data")
    incremental.add_argument("--config", default="configs/default.yml")
    incremental.add_argument("--through", default=datetime.now().date().isoformat())
    incremental.add_argument("--jobs", type=int, default=4)
    incremental.add_argument("--report-out", default="runs/data_update/latest")
    lake = sub.add_parser("build-lake")
    lake.add_argument("--config", default="configs/default.yml")
    lake.add_argument("--daily-only", action="store_true")
    lake.add_argument("--intraday-only", action="store_true")
    audit = sub.add_parser("audit-data")
    audit.add_argument("--config", default="configs/default.yml")
    audit.add_argument("--out", default="runs/data_audit")
    quality = sub.add_parser("quality-monitor")
    quality.add_argument("--config", default="configs/default.yml")
    quality.add_argument("--out", default="runs/data_quality")
    quality.add_argument("--broker-prices", default=None)
    maintain = sub.add_parser("maintain-data")
    maintain.add_argument("--data-root", default="data")
    maintain.add_argument("--runs-root", default="runs")
    maintain.add_argument("--out", default="runs/data_maintenance")
    maintain.add_argument("--source", action="append", default=[], help="Daily CSV/Parquet source to consolidate; repeatable.")
    maintain.add_argument("--consolidated-out", default=None, help="Destination CSV/Parquet for merged OHLCV sources.")
    maintain.add_argument("--keep-signals", type=int, default=20)
    maintain.add_argument("--keep-bundles", type=int, default=5)
    maintain.add_argument("--min-age-days", type=int, default=14)
    maintain.add_argument("--apply-cleanup", action="store_true")
    maintain.add_argument("--confirm", action="store_true", help="Required with --apply-cleanup.")
    today = sub.add_parser("today")
    today.add_argument("--config", default="configs/default.yml")
    today.add_argument("--lookback-days", type=int, default=380, help="Recent trading sessions used for fast daily signal generation.")
    today.add_argument("--top", type=int, default=20, help="Number of non-zero target rows to print.")
    today.add_argument("--out", default="runs/today/latest")
    compare = sub.add_parser("compare-runs")
    compare.add_argument("--output-dir", default="runs")
    compare.add_argument("--limit", type=int, default=20)
    experiment = sub.add_parser("experiment-report")
    experiment.add_argument("--output-dir", default="runs")
    experiment.add_argument("--optimization-dir", default="runs/optimization")
    experiment.add_argument("--out", default="runs/experiment_management")
    report = sub.add_parser("report")
    report.add_argument("--run-id", default="latest")
    live_plan = sub.add_parser("live-plan")
    live_plan.add_argument("--config", default="configs/default.yml")
    live_plan.add_argument("--run-id", default="latest")
    live_submit = sub.add_parser("live-submit")
    live_submit.add_argument("--config", default="configs/default.yml")
    live_submit.add_argument("--plan-dir", required=True)
    live_submit.add_argument("--confirm-live", action="store_true")
    live_monitor = sub.add_parser("live-monitor")
    live_monitor.add_argument("--config", default="configs/default.yml")
    live_monitor.add_argument("--plan-dir", default="latest")
    live_monitor.add_argument("--out", default=None)
    live_monitor.add_argument("--interval-seconds", type=int, default=60)
    live_monitor.add_argument("--iterations", type=int, default=1, help="Use 0 for an infinite polling loop.")
    factor_mine = sub.add_parser("factor-mine")
    factor_mine.add_argument("--config", default="configs/default.yml")
    factor_mine.add_argument("--out", default=None)
    factor_mine.add_argument("--max-factors", type=int, default=80)
    factor_mine.add_argument("--min-names-per-date", type=int, default=20)
    factor_fit = sub.add_parser("factor-fit")
    factor_fit.add_argument("--config", default="configs/default.yml")
    factor_fit.add_argument("--out", default=None)
    factor_fit.add_argument("--horizon", type=int, default=5)
    factor_fit.add_argument("--max-factors", type=int, default=8)
    factor_fit.add_argument("--ridge-alpha", type=float, default=10.0)
    factor_fit.add_argument("--min-names-per-date", type=int, default=20)
    factor_fit.add_argument("--allow-negative-weights", action="store_true")
    activation_validate = sub.add_parser("activation-validate")
    activation_validate.add_argument("--config", default="configs/default.yml")
    activation_validate.add_argument("--profile", choices=["quick", "research", "aggressive", "cycle_return", "sharpe15"], default="cycle_return")
    activation_validate.add_argument("--out", default=None)
    activation_validate.add_argument("--max-scenarios", type=int, default=12)
    activation_validate.add_argument("--include-target-weights", action="store_true")
    activation_validate.add_argument("--start-date", default=None)
    activation_validate.add_argument("--end-date", default=None)
    activation_validate.add_argument("--max-dates", type=int, default=None)
    leader_substitution = sub.add_parser("leader-substitution-audit")
    leader_substitution.add_argument("--config", default="configs/default.yml")
    leader_substitution.add_argument("--out", default=None)
    leader_substitution.add_argument("--horizon", type=int, default=21)
    leader_substitution.add_argument("--max-rejected-per-theme", type=int, default=3)
    leader_substitution.add_argument("--start-date", default=None)
    leader_substitution.add_argument("--end-date", default=None)
    leader_substitution.add_argument("--max-dates", type=int, default=None)
    holding_timing = sub.add_parser("holding-timing-audit")
    holding_timing.add_argument("--config", default="configs/default.yml")
    holding_timing.add_argument("--out", default=None)
    holding_timing.add_argument("--horizon", type=int, default=21)
    holding_timing.add_argument("--min-weight-drop", type=float, default=0.005)
    holding_timing.add_argument("--start-date", default=None)
    holding_timing.add_argument("--end-date", default=None)
    holding_timing.add_argument("--max-dates", type=int, default=None)
    daily = sub.add_parser("daily-update")
    daily.add_argument("--config", default="configs/default.yml")
    daily.add_argument("--skip-intraday", action="store_true")
    daily.add_argument("--intraday-jobs", type=int, default=None)
    daily.add_argument("--skip-backtest", action="store_true")
    daily.add_argument("--full-report", action="store_true")
    daily.add_argument("--skip-live-plan", action="store_true")
    daily.add_argument("--broker-prices", default=None)
    daily.add_argument("--fit-factors", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    if args.command == "download":
        cmd_download(load_config(args.config))
    elif args.command == "update-market-data":
        cmd_update_market_data(load_config(args.config), through=args.through, jobs=args.jobs, report_out=args.report_out)
    elif args.command == "download-intraday":
        cmd_download_intraday(load_config(args.config), jobs=args.jobs)
    elif args.command == "build-lake":
        cmd_build_lake(load_config(args.config), daily_only=args.daily_only, intraday_only=args.intraday_only)
    elif args.command == "audit-data":
        cmd_audit_data(load_config(args.config), args.out)
    elif args.command == "quality-monitor":
        cmd_quality_monitor(load_config(args.config), args.out, args.broker_prices)
    elif args.command == "maintain-data":
        cmd_maintain_data(
            data_root=args.data_root,
            runs_root=args.runs_root,
            out_dir=args.out,
            sources=args.source,
            consolidated_out=args.consolidated_out,
            keep_signals=args.keep_signals,
            keep_bundles=args.keep_bundles,
            min_age_days=args.min_age_days,
            apply_cleanup=args.apply_cleanup,
            confirmed=args.confirm,
        )
    elif args.command == "today":
        cmd_today(load_config(args.config), lookback_days=args.lookback_days, top=args.top, out=args.out)
    elif args.command == "compare-runs":
        cmd_compare_runs(args.output_dir, args.limit)
    elif args.command == "experiment-report":
        cmd_experiment_report(args.output_dir, args.optimization_dir, args.out)
    elif args.command == "backtest":
        cmd_backtest(load_config(args.config), fast=args.fast)
    elif args.command == "optimize":
        cmd_optimize(
            load_config(args.config),
            profile=args.profile,
            max_runs=args.max_runs,
            jobs=args.jobs,
            second_stage_top_n=args.second_stage_top_n,
            scenario_contains=args.scenario_contains,
        )
    elif args.command == "report":
        cmd_report(args.run_id)
    elif args.command == "live-plan":
        cmd_live_plan(load_config(args.config), args.run_id)
    elif args.command == "live-submit":
        cmd_live_submit(load_config(args.config), args.plan_dir, args.confirm_live)
    elif args.command == "live-monitor":
        cmd_live_monitor(load_config(args.config), args.plan_dir, args.out, args.interval_seconds, args.iterations)
    elif args.command == "factor-mine":
        cmd_factor_mine(load_config(args.config), args.out, args.max_factors, args.min_names_per_date)
    elif args.command == "factor-fit":
        cmd_factor_fit(load_config(args.config), args.out, args.horizon, args.max_factors, args.ridge_alpha, args.min_names_per_date, args.allow_negative_weights)
    elif args.command == "activation-validate":
        cmd_activation_validate(
            load_config(args.config),
            args.profile,
            args.out,
            args.max_scenarios,
            args.include_target_weights,
            args.start_date,
            args.end_date,
            args.max_dates,
        )
    elif args.command == "leader-substitution-audit":
        cmd_leader_substitution_audit(
            load_config(args.config),
            out=args.out,
            horizon=args.horizon,
            max_rejected_per_theme=args.max_rejected_per_theme,
            start_date=args.start_date,
            end_date=args.end_date,
            max_dates=args.max_dates,
        )
    elif args.command == "holding-timing-audit":
        cmd_holding_timing_audit(
            load_config(args.config),
            out=args.out,
            horizon=args.horizon,
            min_weight_drop=args.min_weight_drop,
            start_date=args.start_date,
            end_date=args.end_date,
            max_dates=args.max_dates,
        )
    elif args.command == "daily-update":
        cmd_daily_update(
            load_config(args.config),
            skip_intraday=args.skip_intraday,
            intraday_jobs=args.intraday_jobs,
            skip_backtest=args.skip_backtest,
            full_report=args.full_report,
            skip_live_plan=args.skip_live_plan,
            broker_prices=args.broker_prices,
            fit_factors=args.fit_factors,
        )


def cmd_today(config: AppConfig, *, lookback_days: int = 380, top: int = 20, out: str = "runs/today/latest") -> None:
    """Generate a fast latest-session target list without full backtest/report work."""

    started = time.perf_counter()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prices, freshness = _load_recent_cached_prices(config, lookback_days=lookback_days)
    candidate, features = _build_today_features(config, prices)
    warnings: list[str] = []
    context = _build_today_context(config, features)
    signal_features = features[features.get("tradable", False).fillna(False)].copy() if "tradable" in features.columns else features
    signals = generate_cached_ensemble_signals(config, signal_features, context, warnings)
    if signals.empty:
        raise ValueError("No signals generated for the latest cached session.")
    signals["date"] = pd.to_datetime(signals["date"], format="mixed", errors="coerce").dt.normalize()
    latest_date = signals["date"].max()
    latest_signals = signals[signals["date"].eq(latest_date)].copy()
    targets = construct_target_weights(latest_signals, config.portfolio)
    targets = apply_crowding_risk_to_targets(targets, config.regime, config.universe.theme_baskets_path)
    targets = apply_overnight_gap_risk_to_targets(targets, features, config.regime)
    targets = apply_trade_location_to_targets(targets, config.portfolio)
    targets = apply_regime_exposure_overlay(targets, features, config.universe.benchmark, config.regime)
    targets = _attach_today_data_quality(targets, freshness)
    summary = _today_summary(targets, latest_date, freshness, started)
    targets.to_csv(out_dir / "today_targets.csv", index=False)
    latest_signals.to_csv(out_dir / "today_signals.csv", index=False)
    candidate_latest = candidate[pd.to_datetime(candidate["date"], format="mixed", errors="coerce").dt.normalize().eq(latest_date)].copy()
    candidate_latest.to_csv(out_dir / "today_candidate_universe.csv", index=False)
    (out_dir / "today_manifest.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _print_today_signal_summary(targets, summary, top=top)
    print(f"Saved today signal files to {out_dir}")


def _load_recent_cached_prices(config: AppConfig, *, lookback_days: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load only recent OHLCV rows from the local cache for fast daily signals."""

    symbols = sorted(set(str(symbol).upper() for symbol in (config.universe.custom_symbols or config.universe.symbols)))
    if config.data_provider == "marketdata":
        prices = MarketDataStore().read_bars(
            symbols=symbols,
            start=config.start_date,
            end=config.end_date,
            finality="confirmed",
        )
        prices = _normalize_ohlcv_cache(prices)
        _require_confirmed_marketdata_cache(prices)
    else:
        path = Path(config.data_path)
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Price cache not found: {path}. Run download first.")
        prices = pd.read_csv(path)
    required = {"symbol", "date", "open", "high", "low", "close", "adj_close", "volume"}
    missing_columns = sorted(required - set(prices.columns))
    if missing_columns:
        raise ValueError(f"Price cache missing columns: {missing_columns}")
    prices["symbol"] = prices["symbol"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"], format="mixed", errors="coerce").dt.normalize()
    if symbols:
        prices = prices[prices["symbol"].isin(symbols)].copy()
    all_latest = prices.groupby("symbol")["date"].max()
    global_latest = all_latest.max()
    recent_dates = sorted(prices["date"].dropna().unique())[-int(lookback_days) :]
    prices = prices[prices["date"].isin(recent_dates)].copy()
    latest_count = int((all_latest == global_latest).sum()) if pd.notna(global_latest) else 0
    stale_gt_1d = all_latest[all_latest < global_latest - pd.Timedelta(days=1)] if pd.notna(global_latest) else pd.Series(dtype="datetime64[ns]")
    stale_gt_3d = all_latest[all_latest < global_latest - pd.Timedelta(days=3)] if pd.notna(global_latest) else pd.Series(dtype="datetime64[ns]")
    return prices.sort_values(["symbol", "date"]).reset_index(drop=True), {
        "configured_symbols": len(symbols),
        "cached_symbols": int(all_latest.size),
        "global_latest_date": str(global_latest.date()) if pd.notna(global_latest) else "",
        "symbols_on_latest_date": latest_count,
        "stale_gt_1d": int(stale_gt_1d.size),
        "stale_gt_3d": int(stale_gt_3d.size),
        "stale_gt_3d_symbols": stale_gt_3d.sort_values().head(20).index.tolist(),
        "lookback_days": int(lookback_days),
    }


def _build_today_features(config: AppConfig, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build point-in-time tradable flags and technical features on recent data."""

    tradable = build_asof_tradable_universe(prices, config.universe)
    candidate = tradable[tradable["tradable"]].copy()
    asof_cols = [
        "date",
        "symbol",
        "tradable",
        "limited_history_flag",
        "avg_dollar_volume_20d",
        "avg_volume_20d",
        "history_days",
        "quality_adjusted",
        "quality_adjustment_reason",
        "quality_position_multiplier",
        "quality_history_start",
        "filter_reason",
        "candidate_reason",
    ]
    merged = prices.merge(tradable[[col for col in asof_cols if col in tradable.columns]], on=["date", "symbol"], how="left")
    merged["tradable"] = merged["tradable"].fillna(False).astype(bool)
    if "quality_position_multiplier" in merged:
        merged["quality_position_multiplier"] = merged["quality_position_multiplier"].fillna(1.0)
    features = add_relative_strength(add_technical_features(merged), config.universe.benchmark)
    return candidate, features


def _build_today_context(config: AppConfig, features: pd.DataFrame) -> StrategyContext:
    """Create a fast context with dynamic theme scores and no slow external data calls."""

    theme_scores = dynamic_theme_scores(features, config.universe.benchmark, threshold=config.regime.theme_activation_threshold)
    theme_scores = _attach_theme_memberships(theme_scores, config.universe.theme_baskets_path)
    return StrategyContext(config.universe.benchmark, fundamentals=pd.DataFrame(), events=pd.DataFrame(), theme_scores=theme_scores)


def _attach_today_data_quality(targets: pd.DataFrame, freshness: dict[str, object]) -> pd.DataFrame:
    """Attach a compact freshness warning to today's target table."""

    out = targets.copy()
    out["fast_signal_mode"] = True
    out["data_latest_date"] = freshness.get("global_latest_date", "")
    out["data_freshness_note"] = (
        f"{freshness.get('symbols_on_latest_date', 0)}/{freshness.get('cached_symbols', 0)} symbols on latest date; "
        f"{freshness.get('stale_gt_1d', 0)} stale >1d; {freshness.get('stale_gt_3d', 0)} stale >3d"
    )
    return out


def _today_summary(targets: pd.DataFrame, latest_date: pd.Timestamp, freshness: dict[str, object], started: float) -> dict[str, object]:
    """Build a small manifest for the fast signal run."""

    nonzero = targets[targets.get("target_weight", pd.Series(dtype=float)).abs() > 1e-9] if not targets.empty else targets
    return {
        "signal_date": str(latest_date.date()) if pd.notna(latest_date) else "",
        "target_rows": int(len(targets)),
        "nonzero_targets": int(len(nonzero)),
        "gross_exposure": float(nonzero["target_weight"].abs().sum()) if not nonzero.empty and "target_weight" in nonzero else 0.0,
        "net_exposure": float(nonzero["target_weight"].sum()) if not nonzero.empty and "target_weight" in nonzero else 0.0,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "freshness": freshness,
        "mode": "fast_today_no_full_backtest_no_external_fundamentals_events",
    }


def _print_today_signal_summary(targets: pd.DataFrame, summary: dict[str, object], *, top: int) -> None:
    """Print a compact Chinese-friendly signal table."""

    print(
        f"Today signal date={summary['signal_date']} | nonzero={summary['nonzero_targets']} | "
        f"gross={summary['gross_exposure']:.2%} | net={summary['net_exposure']:.2%} | runtime={summary['runtime_seconds']}s"
    )
    freshness = summary.get("freshness", {})
    print(
        "Data freshness: "
        f"latest={freshness.get('global_latest_date', '')}, "
        f"latest coverage={freshness.get('symbols_on_latest_date', 0)}/{freshness.get('cached_symbols', 0)}, "
        f"stale>3d={freshness.get('stale_gt_3d', 0)} {freshness.get('stale_gt_3d_symbols', [])}"
    )
    if targets.empty or "target_weight" not in targets:
        print("No target rows.")
        return
    table = targets[targets["target_weight"].abs() > 1e-9].copy().sort_values("target_weight", ascending=False).head(int(top))
    display_cols = [
        "symbol",
        "target_weight",
        "final_score",
        "technical_score",
        "relative_strength_score",
        "theme_score",
        "entry_quality_score",
        "reward_risk_estimate",
        "trade_location_type",
        "trade_action_intent",
        "primary_theme",
        "overnight_gap_risk_score",
        "reason_for_entry",
    ]
    display_cols = [col for col in display_cols if col in table.columns]
    if "reason_for_entry" in table.columns:
        table["reason_for_entry"] = table["reason_for_entry"].astype(str).str.slice(0, 180)
    print(table[display_cols].to_string(index=False))


def cmd_leader_substitution_audit(
    config: AppConfig,
    *,
    out: str | None = None,
    horizon: int = 21,
    max_rejected_per_theme: int = 3,
    start_date: str | None = None,
    end_date: str | None = None,
    max_dates: int | None = None,
) -> None:
    """Audit whether rejected same-theme leaders outperform selected longs."""

    prices, warnings, _, features, context = prepare_research_bundle(config)
    signals = generate_cached_ensemble_signals(config, features, context, warnings)
    audit = build_leader_substitution_audit(
        signals,
        prices,
        config,
        horizon=horizon,
        max_rejected_per_theme=max_rejected_per_theme,
        start_date=start_date,
        end_date=end_date,
        max_dates=max_dates,
    )
    summary = summarize_leader_substitution_audit(audit, config)
    out_dir = Path(out) if out else Path(config.output_dir) / "leader_substitution_audit" / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_leader_substitution_audit(audit, summary, out_dir)
    print(f"Leader substitution audit saved to {out_dir}")
    print(f"Report: {out_dir / 'leader_substitution_audit.html'}")
    if start_date or end_date or max_dates:
        print("This audit used a bounded date window; treat it as diagnostic before a full walk-forward mutation.")
    if not summary.empty:
        print(summary.to_string(index=False))
    if warnings:
        print(f"Warnings: {len(warnings)}; first warning: {warnings[0]}")


def cmd_holding_timing_audit(
    config: AppConfig,
    *,
    out: str | None = None,
    horizon: int = 21,
    min_weight_drop: float = 0.005,
    start_date: str | None = None,
    end_date: str | None = None,
    max_dates: int | None = None,
) -> None:
    """Audit whether long exits/cuts are followed by missed upside."""

    prices, warnings, _, features, context = prepare_research_bundle(config)
    signals = generate_cached_ensemble_signals(config, features, context, warnings)
    audit = build_holding_timing_audit(
        signals,
        prices,
        config,
        horizon=horizon,
        min_weight_drop=min_weight_drop,
        start_date=start_date,
        end_date=end_date,
        max_dates=max_dates,
    )
    summary = summarize_holding_timing_audit(audit, config)
    out_dir = Path(out) if out else Path(config.output_dir) / "holding_timing_audit" / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_holding_timing_audit(audit, summary, out_dir)
    print(f"Holding timing audit saved to {out_dir}")
    print(f"Report: {out_dir / 'holding_timing_audit.html'}")
    if start_date or end_date or max_dates:
        print("This audit used a bounded date window; treat it as diagnostic before a full walk-forward mutation.")
    if not summary.empty:
        print(summary.to_string(index=False))
    if warnings:
        print(f"Warnings: {len(warnings)}; first warning: {warnings[0]}")


def cmd_activation_validate(
    config: AppConfig,
    profile: str = "cycle_return",
    out: str | None = None,
    max_scenarios: int = 12,
    include_target_weights: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    max_dates: int | None = None,
) -> None:
    """Quickly test whether scenarios alter selected names or target weights."""

    _, warnings, _, features, context = prepare_research_bundle(config)
    baseline_signals = generate_cached_ensemble_signals(config, features, context, warnings)
    scenarios = default_research_scenarios(profile)[: max(1, max_scenarios)]
    rows = []
    for idx, scenario in enumerate(scenarios, start=1):
        print(f"[{idx}/{len(scenarios)}] activation: {scenario['scenario']}", flush=True)
        scenario_config = _config_with_scenario(config, scenario)
        candidate_signals = generate_cached_ensemble_signals(scenario_config, features, context, warnings)
        summary = selected_row_activation_summary(
            baseline_signals,
            candidate_signals,
            scenario_config,
            label=scenario["scenario"],
            include_target_weights=include_target_weights,
            start_date=start_date,
            end_date=end_date,
            max_dates=max_dates,
        )
        for key in ("rebalance_profile", "strategy_profile", "constraint_profile", "event_profile"):
            summary[key] = scenario.get(key, "")
        rows.append(summary)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_dir = Path(out) if out else Path(config.output_dir) / "activation_validation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_activation_validation_report(frame, out_dir)
    validation = frame[frame["period"].eq("validation")].copy() if not frame.empty else pd.DataFrame()
    active = validation[validation["activation_pass"].astype(bool)] if not validation.empty and "activation_pass" in validation else pd.DataFrame()
    print(f"Activation validation saved to {out_dir}")
    print(f"Report: {out_dir / 'activation_validation.html'}")
    if not include_target_weights:
        print("Target-weight diff skipped for speed; rerun with --include-target-weights for exact target checks.")
    elif start_date or end_date or max_dates:
        print("Target-weight diff used a bounded date window; daily-throttled carry state is diagnostic, not a full replay.")
    print(f"Validation-active scenarios: {len(active)}/{len(validation)}")
    if not active.empty:
        cols = [
            "label",
            "selected_changed_date_rate",
            "mean_changed_names",
            "target_changed_date_rate",
            "mean_abs_target_weight_diff",
            "max_abs_target_weight_diff",
        ]
        print(active[cols].sort_values(["target_changed_date_rate", "mean_changed_names"], ascending=False).head(10).to_string(index=False))


def cmd_download(config: AppConfig) -> None:
    """Download data into the suite's canonical external MarketData store."""

    symbols = list(config.universe.custom_symbols or config.universe.symbols)
    print(f"Preparing to download {len(symbols)} symbols.")
    if config.data_provider == "csv":
        paths = generate_sample_data(symbols, config.start_date, config.end_date)
        print(f"Generated sample data: {paths[0]}, {paths[1]}, {paths[2]}")
        return
    if config.data_provider != "marketdata":
        raise ValueError("download supports only csv demo generation or the shared marketdata provider")
    provider = MarketDataAPIProvider()
    requested_symbols = symbols
    if os.getenv("MARKETDATA_DOWNLOAD_ONLY_MISSING", "").lower() in {"1", "true", "yes"}:
        completed_end = config.end_date or _latest_completed_us_equity_date().date().isoformat()
        existing = provider.store.read_bars(
            symbols=symbols,
            start=config.start_date,
            end=completed_end,
            finality="confirmed",
        )
        present = set(existing.get("symbol", pd.Series(dtype=str)).astype(str).str.upper())
        requested_symbols = [symbol for symbol in symbols if str(symbol).upper() not in present]
        print(f"Downloading only missing symbols: {len(requested_symbols)}/{len(symbols)}.")
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for idx, symbol in enumerate(requested_symbols, start=1):
        try:
            frame = provider.get_ohlcv(symbol, config.start_date, config.end_date, config.timeframe)
        except Exception as exc:  # pragma: no cover - real provider resilience
            errors[symbol] = str(exc)
            frame = pd.DataFrame()
        if not frame.empty:
            frames.append(frame)
        if idx == 1 or idx % 25 == 0 or idx == len(requested_symbols):
            print(f"Downloaded daily OHLCV {idx}/{len(requested_symbols)} symbols; errors={len(errors)}", flush=True)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    completed_end = config.end_date or _latest_completed_us_equity_date().date().isoformat()
    available = provider.store.read_bars(
        symbols=symbols,
        start=config.start_date,
        end=completed_end,
        finality="confirmed",
    )
    fetched_symbols = available["symbol"].nunique() if "symbol" in available.columns else 0
    print(f"Canonical MarketData store: {provider.store.root}")
    print(f"Fetched {len(data)} rows; canonical coverage is {fetched_symbols}/{len(symbols)} configured symbols.")
    if fetched_symbols < len(symbols):
        missing = sorted(set(symbols) - set(available.get("symbol", pd.Series(dtype=str)).astype(str).str.upper()))
        print(f"Warning: missing {len(missing)} symbols: {missing[:30]}")
    if errors:
        error_path = Path(config.output_dir) / "data_download" / "marketdata_errors.csv"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"symbol": symbol, "error": error} for symbol, error in sorted(errors.items())]).to_csv(error_path, index=False)
        print(f"Warning: {len(errors)} provider errors saved to {error_path}")


def cmd_update_market_data(config: AppConfig, *, through: str, jobs: int = 4, report_out: str = "runs/data_update/latest") -> None:
    """Incrementally update the canonical external store through a target date."""

    if config.data_provider != "marketdata":
        raise ValueError("update-market-data requires data_provider: marketdata")
    provider = MarketDataAPIProvider()
    requested_end = pd.Timestamp(through).normalize()
    if pd.isna(requested_end):
        raise ValueError(f"Invalid --through date: {through}")
    completed_market_date = _latest_completed_us_equity_date()
    end = min(requested_end, completed_market_date)
    symbols = sorted({str(symbol).upper() for symbol in (config.universe.custom_symbols or config.universe.symbols)})
    existing = provider.store.read_bars(
        symbols=symbols,
        start=config.start_date,
        end=end,
        finality="confirmed",
    )
    existing = _normalize_ohlcv_cache(existing)
    _require_confirmed_marketdata_cache(existing)
    existing_dates = pd.to_datetime(existing["date"], format="mixed", errors="coerce")
    existing = existing.assign(_parsed_date=existing_dates)
    existing = existing[existing["_parsed_date"].le(end)].copy()
    latest = existing.groupby("symbol")["_parsed_date"].max()
    if end < requested_end:
        print(
            f"Requested through {requested_end.date()}, but the latest completed US equity session is "
            f"{end.date()}; excluding any in-progress daily bar.",
            flush=True,
        )
    requests_to_make: list[tuple[str, str]] = []
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        last = latest.get(symbol, pd.NaT)
        start = pd.Timestamp(config.start_date).normalize() if pd.isna(last) else pd.Timestamp(last).normalize() + pd.Timedelta(days=1)
        if start > end:
            rows.append({"symbol": symbol, "requested_from": start.date(), "requested_through": end.date(), "rows_fetched": 0, "status": "already_current"})
        else:
            requests_to_make.append((symbol, start.date().isoformat()))

    def fetch(item: tuple[str, str]) -> tuple[str, str, pd.DataFrame, str]:
        symbol, start = item
        try:
            frame = provider.get_ohlcv(symbol, start, end.date().isoformat(), config.timeframe)
            return symbol, start, frame, ""
        except Exception as exc:  # pragma: no cover - live provider path
            return symbol, start, pd.DataFrame(), str(exc)

    frames: list[pd.DataFrame] = []
    completed_count = 0
    worker_count = max(1, int(jobs))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(fetch, item): item for item in requests_to_make}
        for future in as_completed(futures):
            symbol, start, frame, error = future.result()
            completed_count += 1
            if not frame.empty:
                frames.append(frame)
            fetched_latest = pd.to_datetime(frame.get("date"), errors="coerce").max() if not frame.empty else pd.NaT
            rows.append(
                {
                    "symbol": symbol,
                    "requested_from": start,
                    "requested_through": end.date(),
                    "rows_fetched": len(frame),
                    "latest_fetched": fetched_latest,
                    "status": "error" if error else ("updated" if not frame.empty else "no_data"),
                    "error": error,
                }
            )
            if completed_count == 1 or completed_count % 25 == 0 or completed_count == len(requests_to_make):
                print(
                    f"Incremental daily update {completed_count}/{len(requests_to_make)}; "
                    f"rows={sum(len(part) for part in frames)}; errors={sum(bool(row.get('error')) for row in rows)}",
                    flush=True,
                )

    downloaded = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    report_dir = Path(report_out)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    report.to_csv(report_dir / "market_data_update.csv", index=False)
    active = pd.concat([existing.drop(columns="_parsed_date", errors="ignore"), downloaded], ignore_index=True)
    active = _normalize_ohlcv_cache(active)
    active = active.drop_duplicates(["symbol", "date"], keep="last")
    merged_dates = pd.to_datetime(active["date"], errors="coerce")
    active = active.assign(_date=merged_dates)
    active = active[active["symbol"].isin(symbols)]
    latest_after = active.groupby("symbol")["_date"].max()
    summary = {
        "through": requested_end.date().isoformat(),
        "effective_through": end.date().isoformat(),
        "excluded_incomplete_session": bool(end < requested_end),
        "configured_symbols": len(symbols),
        "requested_symbols": len(requests_to_make),
        "rows_fetched": len(downloaded),
        "errors": int(report.get("status", pd.Series(dtype=str)).eq("error").sum()),
        "latest_market_date": latest_after.max().date().isoformat() if latest_after.notna().any() else None,
        "symbols_at_latest_market_date": int(latest_after.eq(latest_after.max()).sum()) if latest_after.notna().any() else 0,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Persisted {len(downloaded)} new rows to canonical store {provider.store.root}")
    print(f"Update report: {report_dir / 'market_data_update.csv'}")
    print(json.dumps(summary, indent=2))


def _symbols_missing_from_cache(path: Path, symbols: list[str]) -> list[str]:
    """Return configured symbols not already present in the OHLCV cache."""

    if not path.exists() or path.stat().st_size == 0:
        return symbols
    try:
        cached = pd.read_csv(path, usecols=["symbol"])["symbol"].astype(str).str.upper().unique()
    except Exception:
        return symbols
    cached_set = set(cached)
    return [symbol for symbol in symbols if str(symbol).upper() not in cached_set]


def cmd_live_plan(config: AppConfig, run_id: str = "latest") -> None:
    """Create live/paper order tickets from a saved research run."""

    out = create_live_plan(config, run_id=run_id)
    print(f"Live trading plan saved to {out}")
    print(f"Open: {out / 'live_plan.html'}")
    print("Review preflight_orders.csv and approved_orders.csv before any submission.")


def cmd_live_submit(config: AppConfig, plan_dir: str, confirm_live: bool = False) -> None:
    """Submit approved order tickets to the configured broker."""

    out = submit_live_plan(config, plan_dir=plan_dir, confirm_live=confirm_live)
    print(f"Submitted order statuses saved to {out}")


def cmd_live_monitor(config: AppConfig, plan_dir: str = "latest", out: str | None = None, interval_seconds: int = 60, iterations: int = 1) -> None:
    """Run the live monitor once or as a polling loop."""

    monitor_dir = run_live_monitor(config, plan_dir=plan_dir, out_dir=out, interval_seconds=interval_seconds, iterations=iterations)
    print(f"Live monitor saved to {monitor_dir}")
    print(f"Open: {monitor_dir / 'live_monitor.html'}")


def cmd_factor_mine(config: AppConfig, out: str | None = None, max_factors: int = 80, min_names_per_date: int = 20) -> None:
    """Mine candidate predictive factors from the current research bundle."""

    _, warnings, _, features, context = prepare_research_bundle(config)
    tables = mine_factors(features, context, config, max_factors=max_factors, min_names_per_date=min_names_per_date)
    signals = generate_cached_ensemble_signals(config, features, context, warnings)
    tables = add_activation_diagnostics(tables, signals, config, horizon=5, top_n=20, min_names_per_date=max(10, min_names_per_date // 2))
    out_dir = Path(out) if out else Path(config.output_dir) / "factor_mining" / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_factor_mining_report(tables, out_dir)
    selected = tables.get("selected", pd.DataFrame())
    summary = tables.get("summary", pd.DataFrame())
    activation = tables.get("activation", pd.DataFrame())
    print(f"Factor mining report saved to {out_dir}")
    print(f"Report: {out_dir / 'factor_mining_report.html'}")
    print(f"Factors analyzed: {summary['factor'].nunique() if not summary.empty and 'factor' in summary else 0}")
    print(f"Candidate factors: {len(selected)}")
    if not selected.empty:
        cols = [col for col in ["factor", "direction", "selection_score", "validation_mean_ic_aligned", "overfit_flags"] if col in selected]
        print(selected[cols].head(10).to_string(index=False))
    if not activation.empty:
        print("Activation diagnostics:")
        cols = [
            col
            for col in [
                "period",
                "factor",
                "activation_score",
                "mean_forward_return_lift",
                "mean_changed_names",
                "positive_lift_rate",
            ]
            if col in activation
        ]
        print(activation[activation["period"].eq("validation")][cols].head(10).to_string(index=False))
    if warnings:
        print(f"Warnings: {len(warnings)}; first warning: {warnings[0]}")


def cmd_factor_fit(
    config: AppConfig,
    out: str | None = None,
    horizon: int = 5,
    max_factors: int = 8,
    ridge_alpha: float = 10.0,
    min_names_per_date: int = 20,
    allow_negative_weights: bool = False,
) -> None:
    """Fit a regularized composite factor using train-only weights."""

    _, warnings, _, features, context = prepare_research_bundle(config)
    tables = fit_factor_model(
        features,
        context,
        config,
        horizon=horizon,
        max_factors=max_factors,
        ridge_alpha=ridge_alpha,
        min_names_per_date=min_names_per_date,
        allow_negative_weights=allow_negative_weights,
    )
    out_dir = Path(out) if out else Path(config.output_dir) / "factor_fit" / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_factor_fit_report(tables, out_dir)
    weights = tables.get("weights", pd.DataFrame())
    performance = tables.get("performance", pd.DataFrame())
    print(f"Factor fit report saved to {out_dir}")
    print(f"Report: {out_dir / 'factor_fit_report.html'}")
    if not performance.empty:
        print(performance.to_string(index=False))
    if not weights.empty:
        print(weights[[col for col in ["factor", "fitted_weight", "direction", "selection_score"] if col in weights]].head(12).to_string(index=False))
    if warnings:
        print(f"Warnings: {len(warnings)}; first warning: {warnings[0]}")


def cmd_daily_update(
    config: AppConfig,
    *,
    skip_intraday: bool = False,
    intraday_jobs: int | None = None,
    skip_backtest: bool = False,
    full_report: bool = False,
    skip_live_plan: bool = False,
    broker_prices: str | None = None,
    fit_factors: bool = False,
) -> None:
    """Run the daily production update pipeline with per-step logs."""

    started = datetime.now()
    out = Path(config.output_dir) / "daily_update" / started.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "started_at": started.isoformat(timespec="seconds"),
        "config_data_path": config.data_path,
        "intraday_data_path": config.intraday_data_path,
        "steps": [],
        "status": "running",
    }
    effective_through = config.end_date or _latest_completed_us_equity_date().date().isoformat()
    daily_jobs = max(1, int(os.getenv("MARKETDATA_DAILY_JOBS", "4")))
    if config.data_provider == "marketdata":
        daily_step = lambda: cmd_update_market_data(
            config,
            through=effective_through,
            jobs=daily_jobs,
            report_out=str(out / "data_update"),
        )
    else:
        daily_step = lambda: cmd_download(config)
    steps: list[tuple[str, callable]] = [("download_daily", daily_step)]
    if not skip_intraday:
        steps.append(("download_intraday", lambda: cmd_download_intraday(config, jobs=intraday_jobs)))
    steps.extend(
        [
            ("build_lake", lambda: cmd_build_lake(config)),
            ("audit_data", lambda: cmd_audit_data(config, str(out / "data_audit"))),
            ("quality_monitor", lambda: cmd_quality_monitor(config, str(out / "data_quality"), broker_prices)),
        ]
    )
    if not skip_backtest:
        steps.append(("backtest", lambda: cmd_backtest(config, fast=not full_report)))
    if fit_factors:
        steps.append(("factor_fit", lambda: cmd_factor_fit(config, out=str(out / "factor_fit"))))
    if not skip_live_plan:
        steps.append(("live_plan", lambda: cmd_live_plan(config, run_id="latest")))

    try:
        for step_name, step in steps:
            _run_daily_update_step(step_name, step, out, manifest)
        manifest["status"] = "ok"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print(f"Daily update failed at {manifest['steps'][-1]['step'] if manifest.get('steps') else 'startup'}")
        print(f"Manifest: {out / 'manifest.json'}")
        raise
    finally:
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["duration_seconds"] = round((datetime.now() - started).total_seconds(), 2)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Daily update completed: {out}")
    print(f"Manifest: {out / 'manifest.json'}")
    print(f"Latest report: {Path(config.output_dir) / 'latest' / 'report.html'}")


def _run_daily_update_step(step_name: str, step: Callable[[], None], out: Path, manifest: dict[str, object]) -> None:
    """Run one daily update step with stdout/stderr captured to a log."""

    log_path = out / f"{step_name}.log"
    started = time.monotonic()
    entry = {"step": step_name, "status": "running", "log": str(log_path)}
    manifest.setdefault("steps", []).append(entry)
    print(f"[daily-update] starting {step_name}")
    try:
        with log_path.open("w", encoding="utf-8") as handle, redirect_stdout(handle), redirect_stderr(handle):
            step()
    except Exception:
        entry["status"] = "failed"
        entry["duration_seconds"] = round(time.monotonic() - started, 2)
        entry["traceback"] = traceback.format_exc()
        raise
    entry["status"] = "ok"
    entry["duration_seconds"] = round(time.monotonic() - started, 2)
    print(f"[daily-update] finished {step_name} in {entry['duration_seconds']}s")


def cmd_build_lake(config: AppConfig, daily_only: bool = False, intraday_only: bool = False) -> None:
    """Materialize DuckDB-readable Parquet datasets for the hot research path."""

    lake = QuantSystemLake()
    built: list[str] = []
    if not intraday_only:
        if config.data_provider == "marketdata":
            print("Daily MarketData already uses the shared canonical Parquet lake; no second lake was built.")
        else:
            daily_source = Path(config.data_path)
            if daily_source.exists() and daily_source.stat().st_size > 0:
                path = lake.materialize_daily_from_csv(daily_source)
                built.append(f"daily_prices -> {path}")
            else:
                print(f"Daily source not found or empty: {daily_source}")
    if not daily_only:
        if config.data_provider == "marketdata":
            print("Intraday MarketData already uses the shared provisional Parquet lake; no second lake was built.")
        else:
            intraday_source = Path(config.intraday_data_path)
            if intraday_source.exists() and intraday_source.stat().st_size > 0:
                path = lake.materialize_intraday_from_csv(intraday_source, config.risk.intraday_timeframe)
                built.append(f"intraday_{config.risk.intraday_timeframe} -> {path}")
                summary_path = lake.materialize_intraday_summary_from_csv(intraday_source, config.risk.intraday_timeframe)
                built.append(f"intraday_summary_{config.risk.intraday_timeframe} -> {summary_path}")
            else:
                print(f"Intraday source not found or empty: {intraday_source}")
    print("Built lake tables:")
    for item in built:
        print(f"- {item}")
    print(f"Manifest: {lake.manifest_path}")
    print(f"DuckDB available: {duckdb_available()} (Parquet fallback works with pyarrow when DuckDB is absent)")


def cmd_audit_data(config: AppConfig, out_dir: str) -> None:
    """Run data quality audits and write CSV/HTML reports."""

    reports = run_data_audit(config, out_dir)
    print(f"Data audit saved to {Path(out_dir) / 'data_audit.html'}")
    for name, frame in reports.items():
        bad = int(frame["status"].fillna("unknown").ne("ok").sum()) if "status" in frame else 0
        print(f"{name}: {len(frame)} rows, {bad} non-ok")


def cmd_quality_monitor(config: AppConfig, out_dir: str, broker_prices: str | None = None) -> None:
    """Run daily production-style data quality checks."""

    detail, summary = run_daily_data_quality_monitor(config, broker_prices_path=broker_prices)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "daily_data_quality_detail.csv", index=False)
    summary.to_csv(out / "daily_data_quality_summary.csv", index=False)
    html = (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>Daily Data Quality</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:7px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}</style></head><body>"
        "<h1>每日数据质量监控</h1><h2>状态汇总</h2>"
        f"{summary.to_html(index=False, border=0) if not summary.empty else '<p>暂无汇总。</p>'}"
        "<h2>标的明细</h2>"
        f"{detail.head(300).to_html(index=False, border=0) if not detail.empty else '<p>暂无明细。</p>'}"
        "</body></html>"
    )
    (out / "daily_data_quality.html").write_text(html, encoding="utf-8")
    bad = int(detail["status"].fillna("unknown").ne("ok").sum()) if "status" in detail else 0
    print(f"Daily data quality saved to {out / 'daily_data_quality.html'}")
    print(f"Symbols: {len(detail)}, non-ok: {bad}")


def cmd_maintain_data(
    *,
    data_root: str,
    runs_root: str,
    out_dir: str,
    sources: list[str],
    consolidated_out: str | None,
    keep_signals: int,
    keep_bundles: int,
    min_age_days: int,
    apply_cleanup: bool,
    confirmed: bool,
) -> None:
    """Inventory storage, optionally consolidate OHLCV, and clean safe caches."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory = storage_inventory(data_root, runs_root)
    inventory.to_csv(out / "storage_inventory.csv", index=False)
    cache_root = Path(data_root) / "cache"
    plan = build_cleanup_plan(
        cache_root,
        keep_signals=keep_signals,
        keep_bundles=keep_bundles,
        min_age_days=min_age_days,
    )
    plan.to_csv(out / "cleanup_plan.csv", index=False)
    reclaimable = int(plan["size_bytes"].sum()) if not plan.empty else 0
    print(f"Storage inventory saved to {out / 'storage_inventory.csv'}")
    print(f"Cleanup plan: {len(plan)} files, {bytes_label(reclaimable)} reclaimable")

    if sources or consolidated_out:
        if not sources or not consolidated_out:
            raise ValueError("Both --source and --consolidated-out are required for consolidation")
        combined, audit = consolidate_daily_ohlcv(sources, consolidated_out)
        audit.to_csv(out / "consolidation_audit.csv", index=False)
        print(f"Consolidated {len(combined)} OHLCV rows to {consolidated_out}")

    if apply_cleanup:
        if not confirmed:
            raise ValueError("--apply-cleanup requires --confirm")
        result = execute_cleanup(plan, cache_root, confirmed=True)
        result.to_csv(out / "cleanup_result.csv", index=False)
        reclaimed = int(result.loc[result["status"].eq("deleted"), "size_bytes"].sum()) if not result.empty else 0
        print(f"Deleted {int(result['status'].eq('deleted').sum())} files; reclaimed {bytes_label(reclaimed)}")
    else:
        print("Dry run only. Add --apply-cleanup --confirm to delete planned rebuildable caches.")


def cmd_compare_runs(output_dir: str, limit: int) -> None:
    """Print recent run registry comparison."""

    table = compare_runs(output_dir, limit)
    if table.empty:
        print("No run registry found.")
        return
    cols = [
        "run_id",
        "fast",
        "cagr",
        "sharpe",
        "max_drawdown",
        "trailing_one_year_return",
        "average_turnover",
        "average_gross_exposure",
    ]
    print(table[[col for col in cols if col in table.columns]].to_string(index=False))


def cmd_experiment_report(output_dir: str, optimization_dir: str, out_dir: str) -> None:
    """Write experiment-management diagnostics."""

    tables = build_experiment_management_tables(output_dir, optimization_dir)
    path = write_experiment_management_report(tables, out_dir)
    print(f"Experiment management report saved to {path}")
    for name, frame in tables.items():
        print(f"{name}: {len(frame)} rows")


def _merge_downloaded_ohlcv(
    path: Path,
    downloaded: pd.DataFrame,
    *,
    maximum_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Merge fresh OHLCV with an existing cache without letting data go backward."""

    fresh = _normalize_ohlcv_cache(downloaded) if not downloaded.empty else downloaded.copy()
    existing = _normalize_ohlcv_cache(pd.read_csv(path)) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    if not fresh.empty:
        _require_confirmed_marketdata_cache(fresh)
    if not existing.empty:
        _require_confirmed_marketdata_cache(existing)
    if maximum_date is not None:
        cutoff = pd.Timestamp(maximum_date).normalize()
        if not fresh.empty:
            fresh = fresh[pd.to_datetime(fresh["date"], format="mixed", errors="coerce").le(cutoff)].copy()
        if not existing.empty:
            existing = existing[pd.to_datetime(existing["date"], format="mixed", errors="coerce").le(cutoff)].copy()
    if fresh.empty:
        if not existing.empty:
            print("Warning: provider returned no completed-session rows; preserving filtered market data cache.")
            return _normalize_ohlcv_cache(existing)
        return fresh
    if existing.empty:
        return fresh
    old_max = pd.to_datetime(existing["date"], format="mixed", errors="coerce").max()
    new_max = pd.to_datetime(fresh["date"], format="mixed", errors="coerce").max()
    if pd.notna(old_max) and pd.notna(new_max) and new_max < old_max:
        print(f"Warning: provider latest date {new_max.date()} is older than existing cache {old_max.date()}; merging and preserving newer rows.")
    merged = pd.concat([existing, fresh], ignore_index=True)
    merged = merged.drop_duplicates(["symbol", "date"], keep="last")
    return _normalize_ohlcv_cache(merged)


def _latest_completed_us_equity_date(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the latest date eligible for official daily-bar research.

    MarketData can expose today's still-forming daily candle during US trading
    hours. A 15-minute close buffer keeps that partial candle out of signals,
    backtests, and the canonical daily cache.
    """

    current = now if now is not None else pd.Timestamp.now(tz="America/New_York")
    if current.tzinfo is None:
        current = current.tz_localize("America/New_York")
    else:
        current = current.tz_convert("America/New_York")
    session_date = current.tz_localize(None).normalize()
    close_buffer = current.normalize() + pd.Timedelta(hours=16, minutes=15)
    return session_date if current >= close_buffer else session_date - pd.Timedelta(days=1)


def _normalize_ohlcv_cache(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV cache schema for deterministic CSV output."""

    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    columns = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    lineage = [column for column in ("source", "finality") if column in out.columns]
    for column in lineage:
        out[column] = out[column].astype("string").str.strip().str.lower()
    return out[[*columns, *lineage]].dropna(subset=["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)


def _require_confirmed_marketdata_cache(frame: pd.DataFrame) -> None:
    """Reject unlabeled or provisional rows before a formal cache merge."""

    missing = [column for column in ("source", "finality") if column not in frame.columns]
    if missing:
        raise ValueError(
            "Existing MarketData cache lacks source/finality lineage; migrate it with "
            "the suite importer and an explicit completed-session attestation before updating"
        )
    if frame[["source", "finality"]].isna().any().any():
        raise ValueError("MarketData cache contains missing source/finality lineage")
    if not frame["source"].eq("marketdata.app").all():
        raise ValueError("MarketData cache contains a non-MarketData source")
    if not frame["finality"].eq("confirmed").all():
        raise ValueError("MarketData cache contains provisional rows")


def cmd_download_intraday(config: AppConfig, jobs: int | None = None) -> None:
    """Download provisional intraday bars into the shared external store."""

    if config.data_provider != "marketdata":
        raise ValueError("download-intraday requires data_provider: marketdata")
    symbols = list(config.universe.custom_symbols or config.universe.symbols)
    start = config.risk.intraday_start_date or config.start_date
    jobs = jobs or int(os.getenv("MARKETDATA_INTRADAY_JOBS", "8"))
    provider = MarketDataIntradayProvider(chunk_days=int(os.getenv("MARKETDATA_INTRADAY_CHUNK_DAYS", "60")))
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    if jobs <= 1:
        for idx, symbol in enumerate(symbols, start=1):
            try:
                frame = provider.get_intraday(symbol, start, config.end_date, config.risk.intraday_timeframe)
            except Exception as exc:
                errors[symbol] = str(exc)
                frame = pd.DataFrame()
            if not frame.empty:
                frames.append(frame)
            if idx % 10 == 0:
                print(f"Downloaded intraday {idx}/{len(symbols)} symbols", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(provider.get_intraday, symbol, start, config.end_date, config.risk.intraday_timeframe): symbol
                for symbol in symbols
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                symbol = futures[future]
                try:
                    frame = future.result()
                except Exception as exc:
                    errors[symbol] = str(exc)
                    frame = pd.DataFrame()
                if not frame.empty:
                    frames.append(frame)
                if idx % 10 == 0:
                    print(f"Downloaded intraday {idx}/{len(symbols)} symbols", flush=True)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not data.empty and {"symbol", "datetime"}.issubset(data.columns):
        data = data.drop_duplicates(["symbol", "datetime"], keep="last")
    fetched_symbols = data["symbol"].nunique() if "symbol" in data.columns and not data.empty else 0
    print(f"Canonical provisional store: {provider.store.root}")
    print(f"Fetched {len(data)} rows for {fetched_symbols}/{len(symbols)} symbols at {config.risk.intraday_timeframe}.")
    if errors:
        sample = list(errors.items())[:10]
        print(f"Warning: {len(errors)} symbols failed: {sample}")


def cmd_backtest(config: AppConfig, fast: bool = False) -> None:
    """Run a configured backtest and save artifacts."""

    result, warnings, candidate = run_research_backtest(config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / run_id
    metrics = save_run(result, config, run_dir, warnings, full_report=not fast, write_heavy_artifacts=not fast)
    registry_path = record_run(run_dir, config, metrics, fast=fast)
    if fast:
        latest_candidate = candidate.copy()
        if "date" in latest_candidate.columns and not latest_candidate.empty:
            latest_candidate["date"] = pd.to_datetime(latest_candidate["date"], format="mixed", errors="coerce")
            latest_candidate = latest_candidate[latest_candidate["date"].eq(latest_candidate["date"].max())]
        latest_candidate.to_csv(run_dir / "candidate_universe_latest.csv", index=False)
    else:
        candidate.to_csv(run_dir / "candidate_universe.csv", index=False)
    universe_audit = candidate.attrs.get("universe_audit")
    if not fast and isinstance(universe_audit, pd.DataFrame) and not universe_audit.empty:
        universe_audit.to_csv(run_dir / "universe_audit.csv", index=False)
    latest = Path(config.output_dir) / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(run_dir, latest)
    print(f"Backtest saved to {run_dir}")
    print(f"Latest report: {latest / 'report.html'}")
    print(f"Run registry: {registry_path}")
    if fast:
        print("Fast mode: skipped heavy signals/targets/trades CSVs, full attribution charts, and full candidate history.")
    print(f"CAGR={metrics.get('cagr', 0):.2%} MaxDD={metrics.get('max_drawdown', 0):.2%} Sharpe={metrics.get('sharpe', 0):.2f}")


def cmd_optimize(
    config: AppConfig,
    profile: str = "quick",
    max_runs: int | None = None,
    jobs: int = 1,
    second_stage_top_n: int | None = None,
    scenario_contains: list[str] | None = None,
) -> None:
    """Run a coarse walk-forward scenario search."""

    date_config = replace(config, end_date=config.regime.validation_end)
    prices_for_dates, warnings, _candidate = prepare_research_data(date_config)
    eligible_dates = prices_for_dates[
        (pd.to_datetime(prices_for_dates["date"]) >= pd.Timestamp(config.regime.train_start))
        & (pd.to_datetime(prices_for_dates["date"]) <= pd.Timestamp(config.regime.validation_end))
    ]["date"]
    splits = make_walk_forward_splits(eligible_dates, train_years=3, test_months=6)
    out = Path(config.output_dir) / "optimization"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split.__dict__ for split in splits]).to_csv(out / "walk_forward_splits.csv", index=False)
    warnings = list(warnings)
    if not config.regime.allow_current_year_optimization:
        warnings.append("Forward/current-year data is excluded from parameter optimization.")
    train_start = pd.Timestamp(config.regime.train_start)
    train_end = pd.Timestamp(config.regime.train_end)
    validation_start = pd.Timestamp(config.regime.validation_start)
    validation_end = pd.Timestamp(config.regime.validation_end)
    if prices_for_dates["date"].min() > train_start or prices_for_dates["date"].max() < validation_end:
        warnings.append("Configured train/validation windows extend beyond available data; results may be sample-limited.")
    train_days = prices_for_dates[(prices_for_dates["date"] >= train_start) & (prices_for_dates["date"] <= train_end)]["date"].nunique()
    if train_days < 504:
        warnings.append("Training sample has fewer than roughly two trading years; overfit risk is high.")
    scenarios = default_research_scenarios(profile)
    if scenario_contains:
        scenarios = _filter_scenarios(scenarios, scenario_contains)
        warnings.append(f"Scenario filter retained {len(scenarios)} scenario(s): {', '.join(scenario_contains)}.")
        if not scenarios:
            raise ValueError(f"No scenarios matched --scenario-contains={scenario_contains}")
    if max_runs is not None:
        scenarios = scenarios[: max(1, max_runs)]
    warnings.append(f"Optimization profile '{profile}' evaluates {len(scenarios)} coarse scenarios across {len(splits)} walk-forward splits.")
    warnings.append("Each split rebuilds the as-of universe and feature/candidate set before validation scoring.")
    warnings.append(
        "Optimizer uses fast daily mode: true 5-minute intraday execution controls are disabled during grid search. "
        "Fundamental/event provider I/O is also disabled during the first-stage grid. "
        "Only configs that pass validation are promoted to best_config.yml; otherwise inspect candidate_fast_config.yml and run a normal backtest."
    )
    if jobs > 1:
        warnings.append(f"Optimization runs scenarios with {jobs} worker threads.")
    detail_rows = []
    tasks = [(split_idx, split, scenario) for split_idx, split in enumerate(splits, start=1) for scenario in scenarios]
    if jobs <= 1:
        for idx, (split_idx, split, scenario) in enumerate(tasks, start=1):
            print(f"[{idx}/{len(tasks)}] split {split_idx}: {scenario['scenario']}", flush=True)
            detail_rows.append(_evaluate_optimization_scenario(config, scenario, split_idx, split, len(scenarios)))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_evaluate_optimization_scenario, config, scenario, split_idx, split, len(scenarios)): (idx, split_idx, scenario)
                for idx, (split_idx, split, scenario) in enumerate(tasks, start=1)
            }
            for future in as_completed(futures):
                idx, split_idx, scenario = futures[future]
                print(f"[{idx}/{len(tasks)}] split {split_idx}: {scenario['scenario']}", flush=True)
                detail_rows.append(future.result())
    detail_frame = pd.DataFrame(detail_rows)
    detail_frame.to_csv(out / "walk_forward_detail.csv", index=False)
    selected_frame = _selected_by_train_results(detail_frame)
    selected_frame.to_csv(out / "walk_forward_selected_by_train.csv", index=False)
    if not selected_frame.empty:
        warnings.append(
            "walk_forward_selected_by_train.csv records the OOS validation legs from the parameter set selected inside each training window."
        )
    results_frame = _aggregate_walk_forward_results(detail_frame)
    pbo_summary = estimate_pbo(detail_frame, scenario_col="scenario", split_col="split_id", score_col="validation_objective")
    results_frame = flag_overfit_risk(results_frame, "validation_objective")
    if "objective_gap" in results_frame:
        results_frame["overfit_risk"] = (
            (results_frame["objective_gap"] > 0.10)
            | ((results_frame["train_objective"] > 0) & (results_frame["validation_objective"] < 0))
        )
    if "validation_dsr" in results_frame:
        results_frame["overfit_risk"] = results_frame.get("overfit_risk", False) | pd.to_numeric(
            results_frame["validation_dsr"], errors="coerce"
        ).lt(0.50).fillna(False)
    if pbo_summary.get("status") == "ok":
        warnings.append(
            f"PBO parameter-selection diagnostic: pbo={float(pbo_summary.get('pbo', 0.0)):.2f} "
            f"over {int(pbo_summary.get('combinations', 0))} CSCV combinations."
        )
        if float(pbo_summary.get("pbo", 0.0)) > 0.50:
            warnings.append("PBO is above 0.50; parameter selection is likely unstable/overfit.")
    pbo_blocks_promotion = pbo_summary.get("status") == "ok" and float(pbo_summary.get("pbo", 0.0)) > 0.50
    stability = parameter_stability(
        results_frame,
        [
            "rebalance",
            "strategy_profile",
            "constraint_profile",
            "lookback_returns",
            "skip_recent_days",
            "long_quantile",
            "short_quantile",
            "breakout_window",
            "daily_turnover_cap",
            "target_gross_exposure",
            "min_holding_days",
        ],
    )
    save_optimization_report(results_frame, stability, warnings, out, pbo_summary=pbo_summary)
    candidate_config = _best_config_from_results(config, results_frame, require_validation_pass=False)
    second_stage = _run_second_stage_full_validation(config, results_frame, out, warnings, top_n=second_stage_top_n)
    if not second_stage.empty:
        second_stage = _apply_return_promotion_guards(second_stage, warnings, profile=profile, initial_capital=config.portfolio.initial_capital)
        second_stage.to_csv(out / "second_stage_full_validation.csv", index=False)
        save_topn_full_validation_report(
            second_stage,
            out,
            top_n=second_stage_top_n or int(os.getenv("QS_OPTIMIZE_SECOND_STAGE_TOP_N", "5")),
        )
        best_config = _best_config_from_results(config, second_stage, require_validation_pass=True)
    else:
        best_config = _best_config_from_results(config, results_frame, require_validation_pass=True)
    if pbo_blocks_promotion:
        best_config = None
        warnings.append("High PBO blocks best_config.yml promotion; inspect candidate_fast_config.yml and run shadow validation.")
    if candidate_config is not None:
        dump_config(candidate_config, out / "candidate_fast_config.yml")
    if best_config is not None:
        dump_config(best_config, out / "best_config.yml")
    elif candidate_config is not None:
        stale_best = out / "best_config.yml"
        if stale_best.exists():
            stale_best.unlink()
        warnings.append("No scenario passed the validation gate, so no best_config.yml was promoted from fast-mode optimization.")
    save_optimization_report(results_frame, stability, warnings, out, pbo_summary=pbo_summary)
    print(f"Saved optimization report to {out / 'robustness_report.html'}")
    if not second_stage.empty:
        print(f"Top-N full validation report: {out / 'topn_full_validation_report.html'}")
    if best_config is not None:
        print(f"Best validation config: {out / 'best_config.yml'}")
    elif candidate_config is not None:
        print(f"No validation-passing best config; fast candidate saved to {out / 'candidate_fast_config.yml'}")


def _run_second_stage_full_validation(
    config: AppConfig,
    fast_results: pd.DataFrame,
    out: Path,
    warnings: list[str],
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Rerun top fast-screened scenarios with full risk/fundamental/event settings."""

    if fast_results.empty:
        return pd.DataFrame()
    top_n = int(top_n or os.getenv("QS_OPTIMIZE_SECOND_STAGE_TOP_N", "5"))
    if top_n <= 0:
        warnings.append("Second-stage full validation disabled by QS_OPTIMIZE_SECOND_STAGE_TOP_N<=0.")
        return pd.DataFrame()
    pool = fast_results.copy()
    if "validation_pass" in pool:
        pool["_pass_rank"] = pool["validation_pass"].astype(bool).astype(int)
    else:
        pool["_pass_rank"] = 0
    sort_cols = ["_pass_rank", "validation_objective", "train_objective"]
    if config.universe.strict_backtest:
        sort_cols = ["_pass_rank", "train_objective", "validation_objective"]
    pool = pool.sort_values(sort_cols, ascending=[False, False, False]).head(top_n)
    rows: list[dict] = []
    warnings.append(f"Second-stage full validation reruns top {len(pool)} fast scenarios with intraday/fundamental/event settings enabled.")
    for idx, (_, row) in enumerate(pool.iterrows(), start=1):
        scenario_config = _best_config_from_results(config, pd.DataFrame([row]), require_validation_pass=False)
        if scenario_config is None:
            continue
        scenario_config = replace(
            scenario_config,
            start_date=config.regime.train_start,
            end_date=config.regime.validation_end,
        )
        print(f"[second-stage {idx}/{len(pool)}] {row.get('scenario', 'scenario')}", flush=True)
        result, _scenario_warnings, _candidate = run_research_backtest(scenario_config)
        curve = result.equity_curve.copy()
        curve["date"] = pd.to_datetime(curve["date"])
        full_metrics = equity_metrics(curve)
        train_curve = curve[(curve["date"] >= config.regime.train_start) & (curve["date"] <= config.regime.train_end)]
        validation_curve = curve[(curve["date"] >= config.regime.validation_start) & (curve["date"] <= config.regime.validation_end)]
        train_metrics = equity_metrics(train_curve)
        validation_metrics = equity_metrics(validation_curve)
        objective_params = {
            "max_drawdown_penalty": float(row.get("objective_max_drawdown_penalty", 0.5) or 0.5),
            "turnover_penalty": float(row.get("objective_turnover_penalty", 0.25) or 0.25),
            "cagr_weight": float(row.get("objective_cagr_weight", 1.0) or 1.0),
            "total_return_weight": float(row.get("objective_total_return_weight", 0.0) or 0.0),
            "trailing_one_year_weight": float(row.get("objective_trailing_one_year_weight", 0.0) or 0.0),
        }
        out_row = row.to_dict()
        train_objective = objective_score(train_metrics, **objective_params)
        validation_objective = objective_score(validation_metrics, **objective_params)
        train_dsr = deflated_sharpe_ratio(
            train_curve["equity"].pct_change(),
            observed_sharpe=train_metrics.get("sharpe"),
            trials=max(len(pool), 1),
        )
        validation_dsr = deflated_sharpe_ratio(
            validation_curve["equity"].pct_change(),
            observed_sharpe=validation_metrics.get("sharpe"),
            trials=max(len(pool), 1),
        )
        out_row.update(
            {
                "optimization_mode": "second_stage_full_intraday_fundamental_event_validation",
                "full_total_return": full_metrics.get("total_return", 0.0),
                "full_trailing_one_year_return": full_metrics.get("trailing_one_year_return", 0.0),
                "full_cagr": full_metrics.get("cagr", 0.0),
                "full_max_drawdown": full_metrics.get("max_drawdown", 0.0),
                "full_sharpe": full_metrics.get("sharpe", 0.0),
                "full_turnover": full_metrics.get("average_turnover", 0.0),
                "full_long_contribution": full_metrics.get("long_contribution", 0.0),
                "full_short_contribution": full_metrics.get("short_contribution", 0.0),
                "train_cagr": train_metrics.get("cagr", 0.0),
                "train_max_drawdown": train_metrics.get("max_drawdown", 0.0),
                "train_sharpe": train_metrics.get("sharpe", 0.0),
                "train_turnover": train_metrics.get("average_turnover", 0.0),
                "train_objective": train_objective,
                "train_dsr": train_dsr.get("dsr"),
                "train_dsr_status": train_dsr.get("status"),
                "validation_cagr": validation_metrics.get("cagr", 0.0),
                "validation_total_return": validation_metrics.get("total_return", 0.0),
                "validation_max_drawdown": validation_metrics.get("max_drawdown", 0.0),
                "validation_sharpe": validation_metrics.get("sharpe", 0.0),
                "validation_turnover": validation_metrics.get("average_turnover", 0.0),
                "validation_objective": validation_objective,
                "validation_dsr": validation_dsr.get("dsr"),
                "validation_dsr_status": validation_dsr.get("status"),
                "objective_gap": train_objective - validation_objective,
                "trade_count": len(result.trades),
            }
        )
        out_row["validation_pass"] = (
            validation_objective > 0
            and validation_metrics.get("max_drawdown", 0.0) > -0.25
            and validation_metrics.get("cagr", 0.0) > 0
        )
        rows.append(out_row)
    second_stage = pd.DataFrame(rows)
    if not second_stage.empty:
        warnings.append(f"Second-stage full validation saved to {out / 'second_stage_full_validation.csv'}.")
    return second_stage


def _filter_scenarios(scenarios: list[dict], needles: list[str]) -> list[dict]:
    """Return scenarios matching any requested text fragment."""

    lowered = [needle.lower() for needle in needles if str(needle).strip()]
    if not lowered:
        return scenarios
    out = []
    for scenario in scenarios:
        haystack = " ".join(
            str(scenario.get(key, ""))
            for key in ("scenario", "strategy_profile", "constraint_profile", "rebalance_profile", "event_profile")
        ).lower()
        if any(needle in haystack for needle in lowered):
            out.append(scenario)
    return out


def _apply_return_promotion_guards(
    results: pd.DataFrame,
    warnings: list[str],
    *,
    profile: str,
    initial_capital: float,
) -> pd.DataFrame:
    """Add conservative promotion gates for return-max profiles.

    The gate only controls optimizer promotion. It does not change strategy
    logic or production configs; failed candidates remain available as shadow
    rows in the optimization artifacts.
    """

    if results.empty or profile != "cycle_return":
        return results
    guarded = results.copy()
    min_validation_total_return = float(os.getenv("QS_CYCLE_RETURN_MIN_PROMOTION_VALIDATION_TOTAL_RETURN", "0.0"))
    min_validation_sharpe = float(os.getenv("QS_CYCLE_RETURN_MIN_PROMOTION_VALIDATION_SHARPE", "0.50"))
    min_validation_cagr = float(os.getenv("QS_CYCLE_RETURN_MIN_PROMOTION_VALIDATION_CAGR", "0.10"))
    max_short_drag_pct = float(os.getenv("QS_CYCLE_RETURN_MAX_PROMOTION_SHORT_DRAG_PCT", "0.20"))

    validation_total = _numeric_result_series(guarded, "validation_total_return", fallback_col="validation_cagr", default=-1.0)
    validation_sharpe = _numeric_result_series(guarded, "validation_sharpe", fallback_col="full_sharpe", default=-99.0)
    validation_cagr = _numeric_result_series(guarded, "validation_cagr", default=-1.0)
    short_contribution = _numeric_result_series(guarded, "full_short_contribution", default=0.0)
    short_drag_pct = (-short_contribution.clip(upper=0.0)) / max(float(initial_capital), 1.0)

    guard_pass = (
        validation_total.ge(min_validation_total_return)
        & validation_sharpe.ge(min_validation_sharpe)
        & validation_cagr.ge(min_validation_cagr)
        & short_drag_pct.le(max_short_drag_pct)
    )
    guarded["promotion_guard_pass"] = guard_pass
    guarded["promotion_guard_reason"] = ""
    guarded.loc[validation_total.lt(min_validation_total_return), "promotion_guard_reason"] += "validation_total_return_below_guard;"
    guarded.loc[validation_sharpe.lt(min_validation_sharpe), "promotion_guard_reason"] += "validation_sharpe_below_guard;"
    guarded.loc[validation_cagr.lt(min_validation_cagr), "promotion_guard_reason"] += "validation_cagr_below_guard;"
    guarded.loc[short_drag_pct.gt(max_short_drag_pct), "promotion_guard_reason"] += "short_drag_above_guard;"
    if "validation_pass" in guarded:
        guarded["validation_pass_before_promotion_guard"] = guarded["validation_pass"].astype(bool)
        guarded["validation_pass"] = guarded["validation_pass"].astype(bool) & guarded["promotion_guard_pass"].astype(bool)
    blocked = int((~guarded["promotion_guard_pass"]).sum())
    if blocked:
        warnings.append(
            "Cycle-return promotion guard blocked "
            f"{blocked} second-stage candidate(s): min_validation_total_return={min_validation_total_return:.2f}, "
            f"min_validation_sharpe={min_validation_sharpe:.2f}, min_validation_cagr={min_validation_cagr:.2f}, "
            f"max_short_drag_pct={max_short_drag_pct:.2f}."
        )
    return guarded


def _numeric_result_series(frame: pd.DataFrame, column: str, *, fallback_col: str | None = None, default: float = 0.0) -> pd.Series:
    """Return a numeric result column with row-wise fallback/default handling."""

    if column in frame:
        series = pd.to_numeric(frame[column], errors="coerce")
    elif fallback_col and fallback_col in frame:
        series = pd.to_numeric(frame[fallback_col], errors="coerce")
    else:
        series = pd.Series(default, index=frame.index, dtype=float)
    if fallback_col and fallback_col in frame:
        series = series.fillna(pd.to_numeric(frame[fallback_col], errors="coerce"))
    return series.fillna(default)


def _config_with_scenario(config: AppConfig, scenario: dict) -> AppConfig:
    """Overlay one optimization scenario onto an AppConfig."""

    portfolio_updates = dict(scenario.get("portfolio", {}))
    portfolio = replace(config.portfolio, **portfolio_updates)
    execution_updates = dict(scenario.get("execution", {}))
    execution = replace(config.execution, **execution_updates)
    risk_updates = dict(scenario.get("risk", {}))
    risk = replace(config.risk, **risk_updates)
    events_updates = dict(scenario.get("events", {}))
    events = replace(config.events, **events_updates)
    regime_updates = dict(scenario.get("regime", {}))
    regime = replace(config.regime, **regime_updates)
    strategies = {name: dict(spec) for name, spec in config.strategies.items()}
    for name, updates in scenario.get("strategies", {}).items():
        strategies.setdefault(name, {})
        strategies[name].update(updates)
        strategies[name]["enabled"] = bool(strategies[name].get("enabled", True))
    return replace(config, execution=execution, portfolio=portfolio, risk=risk, events=events, regime=regime, strategies=strategies)


def _evaluate_optimization_scenario(
    config: AppConfig,
    scenario: dict,
    split_idx: int,
    split,
    trial_count: int = 1,
) -> dict:
    """Run one optimization scenario on one independently rebuilt split."""

    trial_config = _config_with_scenario(config, scenario)
    trial_config = replace(
        trial_config,
        risk=replace(trial_config.risk, intraday_enabled=False),
        fundamentals=replace(trial_config.fundamentals, enabled=False),
        events=replace(trial_config.events, enabled=False),
        regime=replace(
            trial_config.regime,
            intraday_risk_overlay=False,
            minute_pretrade_budget_overlay=False,
        ),
    )
    trial_config = replace(
        trial_config,
        start_date=pd.Timestamp(split.train_start).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(split.test_end).strftime("%Y-%m-%d"),
    )
    prices, trial_warnings, trial_candidate, features, context = _optimization_inputs_for_config(trial_config)
    trial_result, _trial_warnings, _trial_candidate = run_research_backtest(
        trial_config,
        prices=prices,
        features=features,
        context=context,
        candidate=trial_candidate,
        warnings=trial_warnings,
    )
    curve = trial_result.equity_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    full_metrics = equity_metrics(curve)
    train_curve = curve[(curve["date"] >= split.train_start) & (curve["date"] <= split.train_end)]
    validation_curve = curve[(curve["date"] >= split.test_start) & (curve["date"] <= split.test_end)]
    train_metrics = equity_metrics(train_curve)
    validation_metrics = equity_metrics(validation_curve)
    train_dsr = deflated_sharpe_ratio(
        train_curve["equity"].pct_change(),
        observed_sharpe=train_metrics.get("sharpe"),
        trials=trial_count,
    )
    validation_dsr = deflated_sharpe_ratio(
        validation_curve["equity"].pct_change(),
        observed_sharpe=validation_metrics.get("sharpe"),
        trials=trial_count,
    )
    objective_params = dict(scenario.get("objective", {}))
    train_objective = objective_score(train_metrics, **objective_params)
    validation_objective = objective_score(validation_metrics, **objective_params)
    row = {
        "split_id": split_idx,
        "split_train_start": split.train_start,
        "split_train_end": split.train_end,
        "split_validation_start": split.test_start,
        "split_validation_end": split.test_end,
        "scenario": scenario["scenario"],
        "optimization_mode": "fast_daily_no_intraday_no_fundamental_event_io",
        "rebalance_profile": scenario["rebalance_profile"],
        "strategy_profile": scenario["strategy_profile"],
        "constraint_profile": scenario.get("constraint_profile", "base"),
        "event_profile": scenario.get("event_profile", "event_config"),
        "rebalance": trial_config.portfolio.rebalance,
        "construction": trial_config.portfolio.construction,
        "throttle_min_weight_change": trial_config.portfolio.throttle_min_weight_change,
        "throttle_min_score_change": trial_config.portfolio.throttle_min_score_change,
        "throttle_min_new_weight": trial_config.portfolio.throttle_min_new_weight,
        "target_gross_exposure": trial_config.portfolio.target_gross_exposure,
        "target_net_exposure": trial_config.portfolio.target_net_exposure,
        "max_total_positions": trial_config.portfolio.max_total_positions,
        "min_target_weight": trial_config.portfolio.min_target_weight,
        "leader_addon_overlay": trial_config.portfolio.leader_addon_overlay,
        "leader_addon_multiplier": trial_config.portfolio.leader_addon_multiplier,
        "leader_addon_min_final_score": trial_config.portfolio.leader_addon_min_final_score,
        "leader_addon_min_relative_strength_score": trial_config.portfolio.leader_addon_min_relative_strength_score,
        "leader_addon_min_theme_score": trial_config.portfolio.leader_addon_min_theme_score,
        "leader_addon_min_mom_return": trial_config.portfolio.leader_addon_min_mom_return,
        "leader_addon_max_names_per_date": trial_config.portfolio.leader_addon_max_names_per_date,
        "leader_addon_require_benchmark_risk_on": trial_config.portfolio.leader_addon_require_benchmark_risk_on,
        "leader_addon_require_controlled_pullback": trial_config.portfolio.leader_addon_require_controlled_pullback,
        "leader_addon_require_same_theme_peer_support": trial_config.portfolio.leader_addon_require_same_theme_peer_support,
        "leader_addon_same_theme_peer_min_count": trial_config.portfolio.leader_addon_same_theme_peer_min_count,
        "leader_addon_same_theme_peer_min_share": trial_config.portfolio.leader_addon_same_theme_peer_min_share,
        "leader_addon_min_drawdown_from_high": trial_config.portfolio.leader_addon_min_drawdown_from_high,
        "leader_addon_max_drawdown_from_high": trial_config.portfolio.leader_addon_max_drawdown_from_high,
        "leader_addon_max_above_ma20_pct_circuit_breaker": trial_config.portfolio.leader_addon_max_above_ma20_pct_circuit_breaker,
        "leader_addon_max_event_risk_score_circuit_breaker": trial_config.portfolio.leader_addon_max_event_risk_score_circuit_breaker,
        "leader_addon_max_overnight_gap_risk_score_circuit_breaker": trial_config.portfolio.leader_addon_max_overnight_gap_risk_score_circuit_breaker,
        "leader_hold_buffer_overlay": trial_config.portfolio.leader_hold_buffer_overlay,
        "leader_hold_buffer_min_final_score": trial_config.portfolio.leader_hold_buffer_min_final_score,
        "leader_hold_buffer_min_relative_strength_score": trial_config.portfolio.leader_hold_buffer_min_relative_strength_score,
        "leader_hold_buffer_min_theme_score": trial_config.portfolio.leader_hold_buffer_min_theme_score,
        "leader_hold_buffer_weight_fraction": trial_config.portfolio.leader_hold_buffer_weight_fraction,
        "leader_hold_buffer_max_weight": trial_config.portfolio.leader_hold_buffer_max_weight,
        "leader_hold_buffer_max_days": trial_config.portfolio.leader_hold_buffer_max_days,
        "leader_hold_buffer_require_benchmark_risk_on": trial_config.portfolio.leader_hold_buffer_require_benchmark_risk_on,
        "leader_hold_buffer_max_event_risk_score_circuit_breaker": trial_config.portfolio.leader_hold_buffer_max_event_risk_score_circuit_breaker,
        "leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker": trial_config.portfolio.leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker,
        "leader_delayed_exit_overlay": trial_config.portfolio.leader_delayed_exit_overlay,
        "leader_delayed_exit_min_final_score": trial_config.portfolio.leader_delayed_exit_min_final_score,
        "leader_delayed_exit_min_relative_strength_score": trial_config.portfolio.leader_delayed_exit_min_relative_strength_score,
        "leader_delayed_exit_min_theme_score": trial_config.portfolio.leader_delayed_exit_min_theme_score,
        "leader_delayed_exit_min_mom_return": trial_config.portfolio.leader_delayed_exit_min_mom_return,
        "leader_delayed_exit_max_drawdown_from_high": trial_config.portfolio.leader_delayed_exit_max_drawdown_from_high,
        "leader_delayed_exit_retain_fraction": trial_config.portfolio.leader_delayed_exit_retain_fraction,
        "leader_delayed_exit_max_weight": trial_config.portfolio.leader_delayed_exit_max_weight,
        "leader_delayed_exit_max_days": trial_config.portfolio.leader_delayed_exit_max_days,
        "leader_delayed_exit_require_benchmark_risk_on": trial_config.portfolio.leader_delayed_exit_require_benchmark_risk_on,
        "leader_delayed_exit_require_theme_active": trial_config.portfolio.leader_delayed_exit_require_theme_active,
        "leader_delayed_exit_max_event_risk_score_circuit_breaker": trial_config.portfolio.leader_delayed_exit_max_event_risk_score_circuit_breaker,
        "leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker": trial_config.portfolio.leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker,
        "leader_delayed_exit_stability_overlay": trial_config.portfolio.leader_delayed_exit_stability_overlay,
        "leader_delayed_exit_max_final_score_drop": trial_config.portfolio.leader_delayed_exit_max_final_score_drop,
        "leader_delayed_exit_max_relative_strength_drop": trial_config.portfolio.leader_delayed_exit_max_relative_strength_drop,
        "leader_persistence_overlay": trial_config.portfolio.leader_persistence_overlay,
        "leader_persistence_min_final_score": trial_config.portfolio.leader_persistence_min_final_score,
        "leader_persistence_min_relative_strength_score": trial_config.portfolio.leader_persistence_min_relative_strength_score,
        "leader_persistence_min_theme_score": trial_config.portfolio.leader_persistence_min_theme_score,
        "leader_persistence_min_mom_return": trial_config.portfolio.leader_persistence_min_mom_return,
        "leader_persistence_min_theme_peer_count": trial_config.portfolio.leader_persistence_min_theme_peer_count,
        "leader_persistence_min_theme_peer_share": trial_config.portfolio.leader_persistence_min_theme_peer_share,
        "leader_persistence_retain_fraction_of_cut": trial_config.portfolio.leader_persistence_retain_fraction_of_cut,
        "leader_persistence_max_weight_bonus": trial_config.portfolio.leader_persistence_max_weight_bonus,
        "leader_persistence_require_theme_active": trial_config.portfolio.leader_persistence_require_theme_active,
        "leader_persistence_require_benchmark_risk_on": trial_config.portfolio.leader_persistence_require_benchmark_risk_on,
        "leader_persistence_max_event_risk_score_circuit_breaker": trial_config.portfolio.leader_persistence_max_event_risk_score_circuit_breaker,
        "leader_persistence_max_overnight_gap_risk_score_circuit_breaker": trial_config.portfolio.leader_persistence_max_overnight_gap_risk_score_circuit_breaker,
        "leader_persistence_stability_overlay": trial_config.portfolio.leader_persistence_stability_overlay,
        "leader_persistence_min_desired_weight_fraction": trial_config.portfolio.leader_persistence_min_desired_weight_fraction,
        "leader_persistence_max_final_score_drop": trial_config.portfolio.leader_persistence_max_final_score_drop,
        "leader_persistence_max_relative_strength_drop": trial_config.portfolio.leader_persistence_max_relative_strength_drop,
        "leader_persistence_peer_strength_reward_overlay": trial_config.portfolio.leader_persistence_peer_strength_reward_overlay,
        "leader_persistence_peer_strength_reward_min_theme_peer_count": trial_config.portfolio.leader_persistence_peer_strength_reward_min_theme_peer_count,
        "leader_persistence_peer_strength_reward_min_theme_peer_share": trial_config.portfolio.leader_persistence_peer_strength_reward_min_theme_peer_share,
        "leader_persistence_peer_strength_reward_min_relative_strength_score": trial_config.portfolio.leader_persistence_peer_strength_reward_min_relative_strength_score,
        "leader_persistence_peer_strength_reward_retain_fraction_boost": trial_config.portfolio.leader_persistence_peer_strength_reward_retain_fraction_boost,
        "leader_persistence_peer_strength_reward_max_weight_bonus": trial_config.portfolio.leader_persistence_peer_strength_reward_max_weight_bonus,
        "leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker": trial_config.portfolio.leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker,
        "conviction_sizing_overlay": trial_config.portfolio.conviction_sizing_overlay,
        "conviction_sizing_power": trial_config.portfolio.conviction_sizing_power,
        "conviction_sizing_max_multiplier": trial_config.portfolio.conviction_sizing_max_multiplier,
        "conviction_sizing_min_names": trial_config.portfolio.conviction_sizing_min_names,
        "conviction_sizing_min_final_score": trial_config.portfolio.conviction_sizing_min_final_score,
        "conviction_sizing_min_relative_strength_score": trial_config.portfolio.conviction_sizing_min_relative_strength_score,
        "conviction_sizing_min_theme_score": trial_config.portfolio.conviction_sizing_min_theme_score,
        "conviction_sizing_max_event_risk_score_circuit_breaker": trial_config.portfolio.conviction_sizing_max_event_risk_score_circuit_breaker,
        "conviction_sizing_max_overnight_gap_risk_score_circuit_breaker": trial_config.portfolio.conviction_sizing_max_overnight_gap_risk_score_circuit_breaker,
        "conviction_sizing_require_benchmark_risk_on": trial_config.portfolio.conviction_sizing_require_benchmark_risk_on,
        "conviction_sizing_require_same_theme_peer_support": trial_config.portfolio.conviction_sizing_require_same_theme_peer_support,
        "conviction_sizing_allowed_themes": ",".join(trial_config.portfolio.conviction_sizing_allowed_themes),
        "conviction_sizing_same_theme_peer_min_count": trial_config.portfolio.conviction_sizing_same_theme_peer_min_count,
        "conviction_sizing_same_theme_peer_min_share": trial_config.portfolio.conviction_sizing_same_theme_peer_min_share,
        "conviction_sizing_same_theme_peer_min_avg_relative_strength_score": trial_config.portfolio.conviction_sizing_same_theme_peer_min_avg_relative_strength_score,
        "conviction_sizing_same_theme_peer_min_avg_mom_return": trial_config.portfolio.conviction_sizing_same_theme_peer_min_avg_mom_return,
        "daily_turnover_cap": trial_config.execution.daily_turnover_cap,
        "max_adv_participation": trial_config.execution.max_adv_participation,
        "market_impact_bps_per_1pct_adv": trial_config.execution.market_impact_bps_per_1pct_adv,
        "min_trade_notional": trial_config.execution.min_trade_notional,
        "min_holding_days": trial_config.risk.min_holding_days,
        "max_drawdown_reduce_exposure": trial_config.risk.max_drawdown_reduce_exposure,
        "max_drawdown_cash_mode": trial_config.risk.max_drawdown_cash_mode,
        "drawdown_reduction_multiplier": trial_config.risk.drawdown_reduction_multiplier,
        "drawdown_reset": trial_config.risk.drawdown_reset,
        "event_block_high_risk": trial_config.events.block_high_event_risk,
        "event_earnings_risk_multiplier": trial_config.events.earnings_risk_multiplier,
        "event_max_risk_score": trial_config.events.max_event_risk_score_for_normal_strategies,
        "regime_exposure_overlay": trial_config.regime.exposure_overlay,
        "risk_off_exposure_multiplier": trial_config.regime.risk_off_exposure_multiplier,
        "partial_risk_on_exposure_multiplier": trial_config.regime.partial_risk_on_exposure_multiplier,
        "partial_exposure_regime_score": trial_config.regime.partial_exposure_regime_score,
        "full_exposure_regime_score": trial_config.regime.full_exposure_regime_score,
        "analog_momentum_overlay": trial_config.regime.analog_momentum_overlay,
        "analog_inactive_exposure_multiplier": trial_config.regime.analog_inactive_exposure_multiplier,
        "analog_active_exposure_multiplier": trial_config.regime.analog_active_exposure_multiplier,
        "analog_quantile": trial_config.regime.analog_quantile,
        "leader_crowding_relief_overlay": trial_config.regime.leader_crowding_relief_overlay,
        "leader_crowding_relief_min_relative_strength_score": trial_config.regime.leader_crowding_relief_min_relative_strength_score,
        "leader_crowding_relief_min_theme_score": trial_config.regime.leader_crowding_relief_min_theme_score,
        "leader_crowding_relief_weight": trial_config.regime.leader_crowding_relief_weight,
        "leader_crowding_relief_max_names_per_bucket": trial_config.regime.leader_crowding_relief_max_names_per_bucket,
        "leader_crowding_relief_max_weight_per_symbol": trial_config.regime.leader_crowding_relief_max_weight_per_symbol,
        "leader_crowding_relief_max_bucket_weight_restore": trial_config.regime.leader_crowding_relief_max_bucket_weight_restore,
        "leader_crowding_relief_require_benchmark_risk_on": trial_config.regime.leader_crowding_relief_require_benchmark_risk_on,
        "objective_max_drawdown_penalty": objective_params.get("max_drawdown_penalty", 0.5),
        "objective_turnover_penalty": objective_params.get("turnover_penalty", 0.25),
        "objective_cagr_weight": objective_params.get("cagr_weight", 1.0),
        "objective_total_return_weight": objective_params.get("total_return_weight", 0.0),
        "objective_trailing_one_year_weight": objective_params.get("trailing_one_year_weight", 0.0),
        "momentum_weight": trial_config.strategies.get("momentum", {}).get("weight"),
        "breakout_weight": trial_config.strategies.get("breakout", {}).get("weight"),
        "trend_weight": trial_config.strategies.get("trend", {}).get("weight"),
        "lookback_returns": trial_config.strategies.get("momentum", {}).get("lookback_returns"),
        "skip_recent_days": trial_config.strategies.get("momentum", {}).get("skip_recent_days"),
        "long_quantile": trial_config.strategies.get("momentum", {}).get("long_quantile"),
        "short_quantile": trial_config.strategies.get("momentum", {}).get("short_quantile"),
        "momentum_max_positions": trial_config.strategies.get("momentum", {}).get("max_positions"),
        "momentum_custom_final_score_weights": trial_config.strategies.get("momentum", {}).get("momentum_custom_final_score_weights"),
        "momentum_final_score_technical_weight": trial_config.strategies.get("momentum", {}).get("momentum_final_score_technical_weight"),
        "momentum_final_score_relative_strength_weight": trial_config.strategies.get("momentum", {}).get("momentum_final_score_relative_strength_weight"),
        "momentum_final_score_theme_weight": trial_config.strategies.get("momentum", {}).get("momentum_final_score_theme_weight"),
        "momentum_final_score_fundamental_weight": trial_config.strategies.get("momentum", {}).get("momentum_final_score_fundamental_weight"),
        "momentum_final_score_event_risk_penalty": trial_config.strategies.get("momentum", {}).get("momentum_final_score_event_risk_penalty"),
        "momentum_custom_final_score_blend": trial_config.strategies.get("momentum", {}).get("momentum_custom_final_score_blend"),
        "momentum_custom_score_controlled_entry_overlay": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_controlled_entry_overlay"),
        "momentum_custom_score_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_min_relative_strength_score"),
        "momentum_custom_score_min_theme_score": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_min_theme_score"),
        "momentum_custom_score_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_max_above_ma20_pct"),
        "momentum_custom_score_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_max_event_risk_score_circuit_breaker"),
        "momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker"),
        "momentum_custom_score_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_min_benchmark_ret63d_circuit_breaker"),
        "momentum_custom_score_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_require_benchmark_risk_on"),
        "momentum_custom_score_require_theme_active": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_require_theme_active"),
        "momentum_custom_score_require_price_above_ma50": trial_config.strategies.get("momentum", {}).get("momentum_custom_score_require_price_above_ma50"),
        "theme_strength_delta_overlay": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_overlay"),
        "theme_strength_delta_lookback_days": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_lookback_days"),
        "theme_strength_delta_min_peer_count": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_min_peer_count"),
        "theme_strength_delta_min_theme_score": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_min_theme_score"),
        "theme_strength_delta_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_min_relative_strength_score"),
        "theme_strength_delta_min_theme_prior": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_min_theme_prior"),
        "theme_strength_delta_min_peer_rs_prior": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_min_peer_rs_prior"),
        "theme_strength_delta_min_combined_delta": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_min_combined_delta"),
        "theme_strength_delta_max_score_boost": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_max_score_boost"),
        "theme_strength_delta_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_require_benchmark_risk_on"),
        "theme_strength_delta_require_theme_active": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_require_theme_active"),
        "theme_strength_delta_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_max_event_risk_score_circuit_breaker"),
        "theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker"),
        "theme_strength_delta_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("theme_strength_delta_min_benchmark_ret63d_circuit_breaker"),
        "theme_breadth_acceleration_overlay": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_overlay"),
        "theme_breadth_acceleration_lookback_days": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_lookback_days"),
        "theme_breadth_acceleration_min_active_share": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_min_active_share"),
        "theme_breadth_acceleration_min_change": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_min_change"),
        "theme_breadth_acceleration_min_theme_peer_count": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_min_theme_peer_count"),
        "theme_breadth_acceleration_min_theme_score": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_min_theme_score"),
        "theme_breadth_acceleration_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_min_relative_strength_score"),
        "theme_breadth_acceleration_max_score_boost": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_max_score_boost"),
        "theme_breadth_acceleration_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_require_benchmark_risk_on"),
        "theme_breadth_acceleration_require_theme_active": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_require_theme_active"),
        "theme_breadth_acceleration_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_max_event_risk_score_circuit_breaker"),
        "theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker"),
        "theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker"),
        "gap_adjusted_continuation_overlay": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_overlay"),
        "gap_adjusted_continuation_min_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_min_drawdown_from_high"),
        "gap_adjusted_continuation_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_max_drawdown_from_high"),
        "gap_adjusted_continuation_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_max_above_ma20_pct"),
        "gap_adjusted_continuation_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_min_relative_strength_score"),
        "gap_adjusted_continuation_min_theme_score": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_min_theme_score"),
        "gap_adjusted_continuation_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_min_adx_circuit_breaker"),
        "gap_adjusted_continuation_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_max_event_risk_score_circuit_breaker"),
        "gap_adjusted_continuation_max_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_max_gap_risk_score_circuit_breaker"),
        "gap_adjusted_continuation_benign_gap_risk_score": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_benign_gap_risk_score"),
        "gap_adjusted_continuation_gap_risk_lookback_days": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_gap_risk_lookback_days"),
        "gap_adjusted_continuation_min_gap_risk_improvement": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_min_gap_risk_improvement"),
        "gap_adjusted_continuation_min_volume_expansion": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_min_volume_expansion"),
        "gap_adjusted_continuation_max_volume_expansion": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_max_volume_expansion"),
        "gap_adjusted_continuation_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_min_benchmark_ret63d_circuit_breaker"),
        "gap_adjusted_continuation_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_require_benchmark_risk_on"),
        "gap_adjusted_continuation_require_theme_active": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_require_theme_active"),
        "gap_adjusted_continuation_same_theme_peer_quality_overlay": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_same_theme_peer_quality_overlay"),
        "gap_adjusted_continuation_same_theme_peer_min_count": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_same_theme_peer_min_count"),
        "gap_adjusted_continuation_same_theme_peer_min_share": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_same_theme_peer_min_share"),
        "gap_adjusted_continuation_same_theme_peer_min_avg_rs": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_same_theme_peer_min_avg_rs"),
        "gap_adjusted_continuation_max_score_boost": trial_config.strategies.get("momentum", {}).get("gap_adjusted_continuation_max_score_boost"),
        "compound_leader_score_credit_overlay": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_overlay"),
        "compound_leader_score_credit_min_126d_voladj_rank": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_126d_voladj_rank"),
        "compound_leader_score_credit_min_252d_voladj_rank": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_252d_voladj_rank"),
        "compound_leader_score_credit_min_final_score": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_final_score"),
        "compound_leader_score_credit_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_relative_strength_score"),
        "compound_leader_score_credit_min_theme_score": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_theme_score"),
        "compound_leader_score_credit_min_mom_return": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_mom_return"),
        "compound_leader_score_credit_min_theme_peer_count": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_theme_peer_count"),
        "compound_leader_score_credit_max_ret_10d": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_max_ret_10d"),
        "compound_leader_score_credit_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_max_drawdown_from_high"),
        "compound_leader_score_credit_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_max_above_ma20_pct"),
        "compound_leader_score_credit_boundary_overlay": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_boundary_overlay"),
        "compound_leader_score_credit_rank_buffer_below": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_rank_buffer_below"),
        "compound_leader_score_credit_rank_buffer_above": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_rank_buffer_above"),
        "compound_leader_score_credit_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_require_benchmark_risk_on"),
        "compound_leader_score_credit_require_theme_active": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_require_theme_active"),
        "compound_leader_score_credit_require_price_above_ma50": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_require_price_above_ma50"),
        "compound_leader_score_credit_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_max_event_risk_score_circuit_breaker"),
        "compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker"),
        "compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker"),
        "compound_leader_score_credit_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_adx_circuit_breaker"),
        "compound_leader_score_credit_max_score_boost": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_max_score_boost"),
        "compound_leader_score_credit_volume_confirmation_overlay": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_volume_confirmation_overlay"),
        "compound_leader_score_credit_min_ret_100d_rank": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_ret_100d_rank"),
        "compound_leader_score_credit_min_volume_expansion": trial_config.strategies.get("momentum", {}).get("compound_leader_score_credit_min_volume_expansion"),
        "medium_term_leader_pullback_overlay": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_overlay"),
        "medium_term_leader_pullback_min_ret_100d_rank": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_min_ret_100d_rank"),
        "medium_term_leader_pullback_max_ret_5d": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_max_ret_5d"),
        "medium_term_leader_pullback_max_ret_10d": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_max_ret_10d"),
        "medium_term_leader_pullback_min_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_min_drawdown_from_high"),
        "medium_term_leader_pullback_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_max_drawdown_from_high"),
        "medium_term_leader_pullback_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_max_above_ma20_pct"),
        "medium_term_leader_pullback_max_volume_expansion": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_max_volume_expansion"),
        "medium_term_leader_pullback_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_min_relative_strength_score"),
        "medium_term_leader_pullback_min_theme_score": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_min_theme_score"),
        "medium_term_leader_pullback_min_mom_return": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_min_mom_return"),
        "medium_term_leader_pullback_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_require_benchmark_risk_on"),
        "medium_term_leader_pullback_require_theme_active": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_require_theme_active"),
        "medium_term_leader_pullback_require_price_above_ma50": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_require_price_above_ma50"),
        "medium_term_leader_pullback_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_max_event_risk_score_circuit_breaker"),
        "medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker"),
        "medium_term_leader_pullback_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_min_benchmark_ret63d_circuit_breaker"),
        "medium_term_leader_pullback_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_min_adx_circuit_breaker"),
        "medium_term_leader_pullback_max_score_boost": trial_config.strategies.get("momentum", {}).get("medium_term_leader_pullback_max_score_boost"),
        "boundary_rank_promotion_overlay": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_overlay"),
        "boundary_rank_promotion_rank_buffer_below": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_rank_buffer_below"),
        "boundary_rank_promotion_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_min_relative_strength_score"),
        "boundary_rank_promotion_min_theme_score": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_min_theme_score"),
        "boundary_rank_promotion_min_technical_score": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_min_technical_score"),
        "boundary_rank_promotion_min_mom_return": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_min_mom_return"),
        "boundary_rank_promotion_min_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_min_drawdown_from_high"),
        "boundary_rank_promotion_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_max_drawdown_from_high"),
        "boundary_rank_promotion_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_max_above_ma20_pct"),
        "boundary_rank_promotion_max_final_score_deficit": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_max_final_score_deficit"),
        "boundary_rank_promotion_max_promotions_per_date": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_max_promotions_per_date"),
        "boundary_rank_promotion_promoted_score_step": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_promoted_score_step"),
        "boundary_rank_promotion_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_require_benchmark_risk_on"),
        "boundary_rank_promotion_require_theme_active": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_require_theme_active"),
        "boundary_rank_promotion_require_price_above_ma50": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_require_price_above_ma50"),
        "boundary_rank_promotion_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_max_event_risk_score_circuit_breaker"),
        "boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker"),
        "boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker"),
        "boundary_rank_promotion_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_min_adx_circuit_breaker"),
        "boundary_rank_promotion_compound_pullback_overlay": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_compound_pullback_overlay"),
        "boundary_rank_promotion_compound_pullback_min_ret_100d_rank": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_compound_pullback_min_ret_100d_rank"),
        "boundary_rank_promotion_compound_pullback_min_252d_voladj_rank": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_compound_pullback_min_252d_voladj_rank"),
        "boundary_rank_promotion_compound_pullback_max_ret_5d": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_compound_pullback_max_ret_5d"),
        "boundary_rank_promotion_compound_pullback_max_ret_10d": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_compound_pullback_max_ret_10d"),
        "boundary_rank_promotion_compound_pullback_max_volume_expansion": trial_config.strategies.get("momentum", {}).get("boundary_rank_promotion_compound_pullback_max_volume_expansion"),
        "long_pullback_entry_overlay": trial_config.strategies.get("momentum", {}).get("long_pullback_entry_overlay"),
        "pullback_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("pullback_min_relative_strength_score"),
        "pullback_min_theme_score": trial_config.strategies.get("momentum", {}).get("pullback_min_theme_score"),
        "pullback_min_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("pullback_min_drawdown_from_high"),
        "pullback_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("pullback_max_drawdown_from_high"),
        "pullback_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("pullback_max_above_ma20_pct"),
        "pullback_max_score_boost": trial_config.strategies.get("momentum", {}).get("pullback_max_score_boost"),
        "long_pullback_volume_contraction_overlay": trial_config.strategies.get("momentum", {}).get("long_pullback_volume_contraction_overlay"),
        "pullback_volume_contraction_max_volume_expansion": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_max_volume_expansion"),
        "pullback_volume_contraction_min_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_min_drawdown_from_high"),
        "pullback_volume_contraction_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_max_drawdown_from_high"),
        "pullback_volume_contraction_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_max_above_ma20_pct"),
        "pullback_volume_contraction_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_min_relative_strength_score"),
        "pullback_volume_contraction_min_theme_score": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_min_theme_score"),
        "pullback_volume_contraction_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_require_benchmark_risk_on"),
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker"),
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker"),
        "pullback_volume_contraction_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_min_adx_circuit_breaker"),
        "pullback_volume_contraction_max_score_boost": trial_config.strategies.get("momentum", {}).get("pullback_volume_contraction_max_score_boost"),
        "pullback_short_term_reset_overlay": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_overlay"),
        "pullback_short_term_reset_min_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_min_drawdown_from_high"),
        "pullback_short_term_reset_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_max_drawdown_from_high"),
        "pullback_short_term_reset_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_max_above_ma20_pct"),
        "pullback_short_term_reset_max_ret_5d": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_max_ret_5d"),
        "pullback_short_term_reset_max_ret_10d": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_max_ret_10d"),
        "pullback_short_term_reset_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_min_relative_strength_score"),
        "pullback_short_term_reset_min_theme_score": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_min_theme_score"),
        "pullback_short_term_reset_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_require_benchmark_risk_on"),
        "pullback_short_term_reset_require_theme_active": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_require_theme_active"),
        "pullback_short_term_reset_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_max_event_risk_score_circuit_breaker"),
        "pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker"),
        "pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker"),
        "pullback_short_term_reset_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_min_adx_circuit_breaker"),
        "pullback_short_term_reset_max_score_boost": trial_config.strategies.get("momentum", {}).get("pullback_short_term_reset_max_score_boost"),
        "short_term_volume_tilt_overlay": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_overlay"),
        "short_term_volume_tilt_max_ret_5d": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_max_ret_5d"),
        "short_term_volume_tilt_min_volume_expansion": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_min_volume_expansion"),
        "short_term_volume_tilt_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_min_relative_strength_score"),
        "short_term_volume_tilt_min_theme_score": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_min_theme_score"),
        "short_term_volume_tilt_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_max_above_ma20_pct"),
        "short_term_volume_tilt_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_require_benchmark_risk_on"),
        "short_term_volume_tilt_require_theme_active": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_require_theme_active"),
        "short_term_volume_tilt_require_price_above_ma50": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_require_price_above_ma50"),
        "short_term_volume_tilt_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_max_event_risk_score_circuit_breaker"),
        "short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker"),
        "short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker"),
        "short_term_volume_tilt_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_min_adx_circuit_breaker"),
        "short_term_volume_tilt_max_score_boost": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_max_score_boost"),
        "short_term_volume_tilt_same_theme_peer_confirmation_overlay": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_same_theme_peer_confirmation_overlay"),
        "short_term_volume_tilt_same_theme_peer_min_count": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_same_theme_peer_min_count"),
        "short_term_volume_tilt_same_theme_peer_min_share": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_same_theme_peer_min_share"),
        "short_term_volume_tilt_same_theme_peer_min_avg_mom_return": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_same_theme_peer_min_avg_mom_return"),
        "short_term_volume_tilt_same_theme_peer_volume_substitution_overlay": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_same_theme_peer_volume_substitution_overlay"),
        "short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion"),
        "short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall"),
        "short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale"),
        "short_term_volume_tilt_non_risk_on_exception_overlay": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_non_risk_on_exception_overlay"),
        "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count"),
        "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share"),
        "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return"),
        "short_term_volume_tilt_non_risk_on_exception_score_boost_scale": trial_config.strategies.get("momentum", {}).get("short_term_volume_tilt_non_risk_on_exception_score_boost_scale"),
        "boundary_rs_theme_credit_overlay": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_overlay"),
        "boundary_rs_theme_credit_rank_buffer_below": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_rank_buffer_below"),
        "boundary_rs_theme_credit_rank_buffer_above": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_rank_buffer_above"),
        "boundary_rs_theme_credit_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_min_relative_strength_score"),
        "boundary_rs_theme_credit_min_theme_score": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_min_theme_score"),
        "boundary_rs_theme_credit_min_technical_score": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_min_technical_score"),
        "boundary_rs_theme_credit_min_mom_return": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_min_mom_return"),
        "boundary_rs_theme_credit_min_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_min_drawdown_from_high"),
        "boundary_rs_theme_credit_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_max_drawdown_from_high"),
        "boundary_rs_theme_credit_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_max_above_ma20_pct"),
        "boundary_rs_theme_credit_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_require_benchmark_risk_on"),
        "boundary_rs_theme_credit_require_theme_active": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_require_theme_active"),
        "boundary_rs_theme_credit_require_price_above_ma50": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_require_price_above_ma50"),
        "boundary_rs_theme_credit_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_max_event_risk_score_circuit_breaker"),
        "boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker"),
        "boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker"),
        "boundary_rs_theme_credit_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_min_adx_circuit_breaker"),
        "boundary_rs_theme_credit_max_score_boost": trial_config.strategies.get("momentum", {}).get("boundary_rs_theme_credit_max_score_boost"),
        "exit_quality_rank_credit_overlay": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_overlay"),
        "exit_quality_rank_credit_rank_buffer_below": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_rank_buffer_below"),
        "exit_quality_rank_credit_rank_buffer_above": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_rank_buffer_above"),
        "exit_quality_rank_credit_min_final_score": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_min_final_score"),
        "exit_quality_rank_credit_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_min_relative_strength_score"),
        "exit_quality_rank_credit_min_theme_score": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_min_theme_score"),
        "exit_quality_rank_credit_min_technical_score": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_min_technical_score"),
        "exit_quality_rank_credit_min_mom_return": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_min_mom_return"),
        "exit_quality_rank_credit_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_max_drawdown_from_high"),
        "exit_quality_rank_credit_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_max_above_ma20_pct"),
        "exit_quality_rank_credit_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_require_benchmark_risk_on"),
        "exit_quality_rank_credit_require_theme_active": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_require_theme_active"),
        "exit_quality_rank_credit_require_price_above_ma50": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_require_price_above_ma50"),
        "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_max_event_risk_score_circuit_breaker"),
        "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker"),
        "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker"),
        "exit_quality_rank_credit_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_min_adx_circuit_breaker"),
        "exit_quality_rank_credit_max_score_rank_credit": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_max_score_rank_credit"),
        "exit_quality_rank_credit_theme_support_overlay": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_theme_support_overlay"),
        "exit_quality_rank_credit_theme_support_lookback_days": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_theme_support_lookback_days"),
        "exit_quality_rank_credit_theme_support_min_peer_count": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_theme_support_min_peer_count"),
        "exit_quality_rank_credit_theme_support_min_theme_score_prior": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_theme_support_min_theme_score_prior"),
        "exit_quality_rank_credit_theme_support_min_peer_rs_prior": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_theme_support_min_peer_rs_prior"),
        "exit_quality_rank_credit_theme_support_min_combined_delta": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_theme_support_min_combined_delta"),
        "exit_quality_rank_credit_reset_support_overlay": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_overlay"),
        "exit_quality_rank_credit_reset_support_min_reset_score": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_min_reset_score"),
        "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_theme_breadth_lookback_days"),
        "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold"),
        "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold"),
        "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_theme_breadth_min_active_share"),
        "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_require_theme_breadth_expansion"),
        "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold"),
        "exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay"),
        "exit_quality_rank_credit_reset_support_same_theme_peer_min_count": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_peer_min_count"),
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay"),
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share"),
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count"),
        "exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay"),
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs"),
        "exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay"),
        "exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct"),
        "exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap"),
        "exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count": trial_config.strategies.get("momentum", {}).get("exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count"),
        "long_pullback_reclaim_overlay": trial_config.strategies.get("momentum", {}).get("long_pullback_reclaim_overlay"),
        "pullback_reclaim_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_min_relative_strength_score"),
        "pullback_reclaim_min_theme_score": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_min_theme_score"),
        "pullback_reclaim_min_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_min_drawdown_from_high"),
        "pullback_reclaim_max_drawdown_from_high": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_max_drawdown_from_high"),
        "pullback_reclaim_recent_below_ma20_lookback_days": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_recent_below_ma20_lookback_days"),
        "pullback_reclaim_min_recent_below_ma20_pct": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_min_recent_below_ma20_pct"),
        "pullback_reclaim_max_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_max_above_ma20_pct"),
        "pullback_reclaim_min_volume_expansion": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_min_volume_expansion"),
        "pullback_reclaim_require_benchmark_risk_on": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_require_benchmark_risk_on"),
        "pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker"),
        "pullback_reclaim_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_min_benchmark_ret63d_circuit_breaker"),
        "pullback_reclaim_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_min_adx_circuit_breaker"),
        "pullback_reclaim_max_score_boost": trial_config.strategies.get("momentum", {}).get("pullback_reclaim_max_score_boost"),
        "long_pullback_reset_inclusion_overlay": trial_config.strategies.get("momentum", {}).get("long_pullback_reset_inclusion_overlay"),
        "pullback_reset_inclusion_min_score": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_min_score"),
        "pullback_reset_inclusion_max_names_per_date": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_max_names_per_date"),
        "pullback_reset_inclusion_min_final_score": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_min_final_score"),
        "pullback_reset_inclusion_min_score_rank": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_min_score_rank"),
        "pullback_reset_inclusion_activation_support_overlay": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_activation_support_overlay"),
        "pullback_reset_inclusion_activation_min_score": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_activation_min_score"),
        "pullback_reset_inclusion_activation_score_rank_credit": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_activation_score_rank_credit"),
        "pullback_reset_inclusion_reclaim_support_overlay": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_reclaim_support_overlay"),
        "pullback_reset_inclusion_min_reclaim_score": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_min_reclaim_score"),
        "pullback_reset_inclusion_reclaim_score_rank_credit": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_reclaim_score_rank_credit"),
        "pullback_reset_inclusion_require_theme_breadth_not_deteriorating": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_require_theme_breadth_not_deteriorating"),
        "pullback_reset_inclusion_theme_breadth_active_share_threshold": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_theme_breadth_active_share_threshold"),
        "pullback_reset_inclusion_theme_breadth_shortfall_threshold": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_theme_breadth_shortfall_threshold"),
        "pullback_reset_inclusion_require_theme_breadth_expansion": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_require_theme_breadth_expansion"),
        "pullback_reset_inclusion_theme_breadth_lookback_days": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_theme_breadth_lookback_days"),
        "pullback_reset_inclusion_theme_breadth_min_active_share": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_theme_breadth_min_active_share"),
        "pullback_reset_inclusion_theme_breadth_expansion_threshold": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_theme_breadth_expansion_threshold"),
        "pullback_reset_inclusion_require_rs_acceleration": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_require_rs_acceleration"),
        "pullback_reset_inclusion_rs_acceleration_lookback_days": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_rs_acceleration_lookback_days"),
        "pullback_reset_inclusion_min_current_rs_score": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_min_current_rs_score"),
        "pullback_reset_inclusion_min_rs_acceleration": trial_config.strategies.get("momentum", {}).get("pullback_reset_inclusion_min_rs_acceleration"),
        "long_reentry_discipline_overlay": trial_config.strategies.get("momentum", {}).get("long_reentry_discipline_overlay"),
        "reentry_discipline_near_high_threshold": trial_config.strategies.get("momentum", {}).get("reentry_discipline_near_high_threshold"),
        "reentry_discipline_min_above_ma20_pct": trial_config.strategies.get("momentum", {}).get("reentry_discipline_min_above_ma20_pct"),
        "reentry_discipline_max_volume_expansion": trial_config.strategies.get("momentum", {}).get("reentry_discipline_max_volume_expansion"),
        "reentry_discipline_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("reentry_discipline_min_adx_circuit_breaker"),
        "reentry_discipline_max_score_penalty": trial_config.strategies.get("momentum", {}).get("reentry_discipline_max_score_penalty"),
        "reentry_discipline_min_pullback_volume_reset_score_exemption": trial_config.strategies.get("momentum", {}).get("reentry_discipline_min_pullback_volume_reset_score_exemption"),
        "reentry_discipline_min_relative_strength_score_exemption": trial_config.strategies.get("momentum", {}).get("reentry_discipline_min_relative_strength_score_exemption"),
        "reentry_discipline_min_theme_score_exemption": trial_config.strategies.get("momentum", {}).get("reentry_discipline_min_theme_score_exemption"),
        "reentry_discipline_require_theme_deterioration": trial_config.strategies.get("momentum", {}).get("reentry_discipline_require_theme_deterioration"),
        "reentry_discipline_theme_score_deterioration_threshold": trial_config.strategies.get("momentum", {}).get("reentry_discipline_theme_score_deterioration_threshold"),
        "reentry_discipline_penalize_theme_inactive": trial_config.strategies.get("momentum", {}).get("reentry_discipline_penalize_theme_inactive"),
        "reentry_discipline_require_theme_breadth_deterioration": trial_config.strategies.get("momentum", {}).get("reentry_discipline_require_theme_breadth_deterioration"),
        "reentry_discipline_theme_breadth_lookback_days": trial_config.strategies.get("momentum", {}).get("reentry_discipline_theme_breadth_lookback_days"),
        "reentry_discipline_theme_breadth_active_share_threshold": trial_config.strategies.get("momentum", {}).get("reentry_discipline_theme_breadth_active_share_threshold"),
        "reentry_discipline_theme_breadth_shortfall_threshold": trial_config.strategies.get("momentum", {}).get("reentry_discipline_theme_breadth_shortfall_threshold"),
        "short_rebound_avoidance_overlay": trial_config.strategies.get("momentum", {}).get("short_rebound_avoidance_overlay"),
        "short_rebound_avoidance_min_below_ma20_pct": trial_config.strategies.get("momentum", {}).get("short_rebound_avoidance_min_below_ma20_pct"),
        "short_rebound_avoidance_min_distance_from_high": trial_config.strategies.get("momentum", {}).get("short_rebound_avoidance_min_distance_from_high"),
        "short_rebound_avoidance_min_volume_expansion": trial_config.strategies.get("momentum", {}).get("short_rebound_avoidance_min_volume_expansion"),
        "short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker": trial_config.strategies.get("momentum", {}).get("short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker"),
        "short_rebound_avoidance_min_adx_circuit_breaker": trial_config.strategies.get("momentum", {}).get("short_rebound_avoidance_min_adx_circuit_breaker"),
        "long_min_final_score": trial_config.strategies.get("momentum", {}).get("long_min_final_score"),
        "long_min_technical_score": trial_config.strategies.get("momentum", {}).get("long_min_technical_score"),
        "long_min_relative_strength_score": trial_config.strategies.get("momentum", {}).get("long_min_relative_strength_score"),
        "long_min_theme_score": trial_config.strategies.get("momentum", {}).get("long_min_theme_score"),
        "require_long_theme_active": trial_config.strategies.get("momentum", {}).get("require_long_theme_active"),
        "long_require_price_above_ma50": trial_config.strategies.get("momentum", {}).get("long_require_price_above_ma50"),
        "long_require_price_above_ma200": trial_config.strategies.get("momentum", {}).get("long_require_price_above_ma200"),
        "short_min_weakness_conditions": trial_config.strategies.get("momentum", {}).get("short_min_weakness_conditions"),
        "short_only_when_benchmark_risk_off": trial_config.strategies.get("momentum", {}).get("short_only_when_benchmark_risk_off"),
        "short_max_final_score": trial_config.strategies.get("momentum", {}).get("short_max_final_score"),
        "breakout_window": trial_config.strategies.get("breakout", {}).get("breakout_window"),
        "breakout_max_holding_days": trial_config.strategies.get("breakout", {}).get("max_holding_days"),
        "trend_ma_fast": trial_config.strategies.get("trend", {}).get("ma_fast"),
        "trend_ma_slow": trial_config.strategies.get("trend", {}).get("ma_slow"),
        "full_cagr": full_metrics.get("cagr", 0.0),
        "full_max_drawdown": full_metrics.get("max_drawdown", 0.0),
        "full_sharpe": full_metrics.get("sharpe", 0.0),
        "full_turnover": full_metrics.get("average_turnover", 0.0),
        "train_cagr": train_metrics.get("cagr", 0.0),
        "train_max_drawdown": train_metrics.get("max_drawdown", 0.0),
        "train_sharpe": train_metrics.get("sharpe", 0.0),
        "train_turnover": train_metrics.get("average_turnover", 0.0),
        "train_objective": train_objective,
        "train_dsr": train_dsr.get("dsr"),
        "train_dsr_status": train_dsr.get("status"),
        "validation_cagr": validation_metrics.get("cagr", 0.0),
        "validation_total_return": validation_metrics.get("total_return", 0.0),
        "validation_max_drawdown": validation_metrics.get("max_drawdown", 0.0),
        "validation_sharpe": validation_metrics.get("sharpe", 0.0),
        "validation_turnover": validation_metrics.get("average_turnover", 0.0),
        "validation_objective": validation_objective,
        "validation_dsr": validation_dsr.get("dsr"),
        "validation_dsr_status": validation_dsr.get("status"),
        "forward_cagr": pd.NA,
        "forward_max_drawdown": pd.NA,
        "forward_sharpe": pd.NA,
        "forward_turnover": pd.NA,
        "forward_total_return": pd.NA,
        "objective_gap": train_objective - validation_objective,
        "trade_count": len(trial_result.trades),
    }
    momentum_config = trial_config.strategies.get("momentum", {})
    for key in _SAME_THEME_SUBSTITUTION_MOMENTUM_KEYS:
        row[key] = momentum_config.get(key)
    row["validation_pass"] = (
        validation_objective > 0
        and validation_metrics.get("max_drawdown", 0.0) > -0.25
        and validation_metrics.get("cagr", 0.0) > 0
    )
    return row


def _optimization_inputs_for_config(config: AppConfig) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame, object]:
    """Cache expensive split-level research data during grid search."""

    key = (
        str(config.data_path),
        str(config.fundamentals_path),
        str(config.events_path),
        str(config.start_date),
        str(config.end_date),
        str(config.universe.custom_symbols),
        str(config.universe.theme_baskets_path),
        bool(config.fundamentals.enabled),
        str(config.fundamentals.provider),
        bool(config.events.enabled),
        str(config.events.provider),
    )
    cached = _OPTIMIZATION_INPUT_CACHE.get(key)
    if cached is not None:
        return cached
    prices, warnings, candidate = prepare_research_data(config)
    features, context = build_features_and_context(config, prices, candidate, warnings)
    cached = (prices, warnings, candidate, features, context)
    _OPTIMIZATION_INPUT_CACHE[key] = cached
    return cached


def _aggregate_walk_forward_results(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-level optimization results by scenario."""

    if detail.empty:
        return detail
    detail = detail.copy()
    if "validation_total_return" not in detail:
        detail["validation_total_return"] = pd.to_numeric(detail.get("validation_cagr", 0.0), errors="coerce")
    parameter_cols = [
        "scenario",
        "rebalance_profile",
        "strategy_profile",
        "constraint_profile",
        "event_profile",
        "rebalance",
        "construction",
        "throttle_min_weight_change",
        "throttle_min_score_change",
        "throttle_min_new_weight",
        "target_gross_exposure",
        "target_net_exposure",
        "max_total_positions",
        "min_target_weight",
        "leader_addon_overlay",
        "leader_addon_multiplier",
        "leader_addon_min_final_score",
        "leader_addon_min_relative_strength_score",
        "leader_addon_min_theme_score",
        "leader_addon_min_mom_return",
        "leader_addon_max_names_per_date",
        "leader_addon_require_benchmark_risk_on",
        "leader_addon_require_controlled_pullback",
        "leader_addon_require_same_theme_peer_support",
        "leader_addon_same_theme_peer_min_count",
        "leader_addon_same_theme_peer_min_share",
        "leader_addon_min_drawdown_from_high",
        "leader_addon_max_drawdown_from_high",
        "leader_addon_max_above_ma20_pct_circuit_breaker",
        "leader_addon_max_event_risk_score_circuit_breaker",
        "leader_addon_max_overnight_gap_risk_score_circuit_breaker",
        "leader_hold_buffer_overlay",
        "leader_hold_buffer_min_final_score",
        "leader_hold_buffer_min_relative_strength_score",
        "leader_hold_buffer_min_theme_score",
        "leader_hold_buffer_weight_fraction",
        "leader_hold_buffer_max_weight",
        "leader_hold_buffer_max_days",
        "leader_hold_buffer_require_benchmark_risk_on",
        "leader_hold_buffer_max_event_risk_score_circuit_breaker",
        "leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker",
        "leader_delayed_exit_overlay",
        "leader_delayed_exit_min_final_score",
        "leader_delayed_exit_min_relative_strength_score",
        "leader_delayed_exit_min_theme_score",
        "leader_delayed_exit_min_mom_return",
        "leader_delayed_exit_max_drawdown_from_high",
        "leader_delayed_exit_retain_fraction",
        "leader_delayed_exit_max_weight",
        "leader_delayed_exit_max_days",
        "leader_delayed_exit_require_benchmark_risk_on",
        "leader_delayed_exit_require_theme_active",
        "leader_delayed_exit_max_event_risk_score_circuit_breaker",
        "leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker",
        "leader_delayed_exit_stability_overlay",
        "leader_delayed_exit_max_final_score_drop",
        "leader_delayed_exit_max_relative_strength_drop",
        "leader_persistence_overlay",
        "leader_persistence_min_final_score",
        "leader_persistence_min_relative_strength_score",
        "leader_persistence_min_theme_score",
        "leader_persistence_min_mom_return",
        "leader_persistence_min_theme_peer_count",
        "leader_persistence_min_theme_peer_share",
        "leader_persistence_retain_fraction_of_cut",
        "leader_persistence_max_weight_bonus",
        "leader_persistence_require_theme_active",
        "leader_persistence_require_benchmark_risk_on",
        "leader_persistence_max_event_risk_score_circuit_breaker",
        "leader_persistence_max_overnight_gap_risk_score_circuit_breaker",
        "leader_persistence_stability_overlay",
        "leader_persistence_min_desired_weight_fraction",
        "leader_persistence_max_final_score_drop",
        "leader_persistence_max_relative_strength_drop",
        "leader_persistence_peer_strength_reward_overlay",
        "leader_persistence_peer_strength_reward_min_theme_peer_count",
        "leader_persistence_peer_strength_reward_min_theme_peer_share",
        "leader_persistence_peer_strength_reward_min_relative_strength_score",
        "leader_persistence_peer_strength_reward_retain_fraction_boost",
        "leader_persistence_peer_strength_reward_max_weight_bonus",
        "leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker",
        "daily_turnover_cap",
        "max_adv_participation",
        "market_impact_bps_per_1pct_adv",
        "min_trade_notional",
        "min_holding_days",
        "max_drawdown_reduce_exposure",
        "max_drawdown_cash_mode",
        "drawdown_reduction_multiplier",
        "drawdown_reset",
        "event_block_high_risk",
        "event_earnings_risk_multiplier",
        "event_max_risk_score",
        "regime_exposure_overlay",
        "risk_off_exposure_multiplier",
        "partial_risk_on_exposure_multiplier",
        "partial_exposure_regime_score",
        "full_exposure_regime_score",
        "analog_momentum_overlay",
        "analog_inactive_exposure_multiplier",
        "analog_active_exposure_multiplier",
        "analog_quantile",
        "objective_max_drawdown_penalty",
        "objective_turnover_penalty",
        "objective_cagr_weight",
        "objective_total_return_weight",
        "objective_trailing_one_year_weight",
        "momentum_weight",
        "breakout_weight",
        "trend_weight",
        "lookback_returns",
        "skip_recent_days",
        "long_quantile",
        "short_quantile",
        "momentum_max_positions",
        "momentum_custom_final_score_weights",
        "momentum_final_score_technical_weight",
        "momentum_final_score_relative_strength_weight",
        "momentum_final_score_theme_weight",
        "momentum_final_score_fundamental_weight",
        "momentum_final_score_event_risk_penalty",
        "momentum_custom_final_score_blend",
        "momentum_custom_score_controlled_entry_overlay",
        "momentum_custom_score_min_relative_strength_score",
        "momentum_custom_score_min_theme_score",
        "momentum_custom_score_max_above_ma20_pct",
        "momentum_custom_score_max_event_risk_score_circuit_breaker",
        "momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker",
        "momentum_custom_score_min_benchmark_ret63d_circuit_breaker",
        "momentum_custom_score_require_benchmark_risk_on",
        "momentum_custom_score_require_theme_active",
        "momentum_custom_score_require_price_above_ma50",
        "theme_strength_delta_overlay",
        "theme_strength_delta_lookback_days",
        "theme_strength_delta_min_peer_count",
        "theme_strength_delta_min_theme_score",
        "theme_strength_delta_min_relative_strength_score",
        "theme_strength_delta_min_theme_prior",
        "theme_strength_delta_min_peer_rs_prior",
        "theme_strength_delta_min_combined_delta",
        "theme_strength_delta_max_score_boost",
        "theme_strength_delta_require_benchmark_risk_on",
        "theme_strength_delta_require_theme_active",
        "theme_strength_delta_max_event_risk_score_circuit_breaker",
        "theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker",
        "theme_strength_delta_min_benchmark_ret63d_circuit_breaker",
        "theme_breadth_acceleration_overlay",
        "theme_breadth_acceleration_lookback_days",
        "theme_breadth_acceleration_min_active_share",
        "theme_breadth_acceleration_min_change",
        "theme_breadth_acceleration_min_theme_peer_count",
        "theme_breadth_acceleration_min_theme_score",
        "theme_breadth_acceleration_min_relative_strength_score",
        "theme_breadth_acceleration_max_score_boost",
        "theme_breadth_acceleration_require_benchmark_risk_on",
        "theme_breadth_acceleration_require_theme_active",
        "theme_breadth_acceleration_max_event_risk_score_circuit_breaker",
        "theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker",
        "theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker",
        "gap_adjusted_continuation_overlay",
        "gap_adjusted_continuation_min_drawdown_from_high",
        "gap_adjusted_continuation_max_drawdown_from_high",
        "gap_adjusted_continuation_max_above_ma20_pct",
        "gap_adjusted_continuation_min_relative_strength_score",
        "gap_adjusted_continuation_min_theme_score",
        "gap_adjusted_continuation_min_adx_circuit_breaker",
        "gap_adjusted_continuation_max_event_risk_score_circuit_breaker",
        "gap_adjusted_continuation_max_gap_risk_score_circuit_breaker",
        "gap_adjusted_continuation_benign_gap_risk_score",
        "gap_adjusted_continuation_gap_risk_lookback_days",
        "gap_adjusted_continuation_min_gap_risk_improvement",
        "gap_adjusted_continuation_min_volume_expansion",
        "gap_adjusted_continuation_max_volume_expansion",
        "gap_adjusted_continuation_min_benchmark_ret63d_circuit_breaker",
        "gap_adjusted_continuation_require_benchmark_risk_on",
        "gap_adjusted_continuation_require_theme_active",
        "gap_adjusted_continuation_same_theme_peer_quality_overlay",
        "gap_adjusted_continuation_same_theme_peer_min_count",
        "gap_adjusted_continuation_same_theme_peer_min_share",
        "gap_adjusted_continuation_same_theme_peer_min_avg_rs",
        "gap_adjusted_continuation_max_score_boost",
        "compound_leader_score_credit_overlay",
        "compound_leader_score_credit_min_126d_voladj_rank",
        "compound_leader_score_credit_min_252d_voladj_rank",
        "compound_leader_score_credit_min_final_score",
        "compound_leader_score_credit_min_relative_strength_score",
        "compound_leader_score_credit_min_theme_score",
        "compound_leader_score_credit_min_mom_return",
        "compound_leader_score_credit_min_theme_peer_count",
        "compound_leader_score_credit_max_ret_10d",
        "compound_leader_score_credit_max_drawdown_from_high",
        "compound_leader_score_credit_max_above_ma20_pct",
        "compound_leader_score_credit_boundary_overlay",
        "compound_leader_score_credit_rank_buffer_below",
        "compound_leader_score_credit_rank_buffer_above",
        "compound_leader_score_credit_require_benchmark_risk_on",
        "compound_leader_score_credit_require_theme_active",
        "compound_leader_score_credit_require_price_above_ma50",
        "compound_leader_score_credit_max_event_risk_score_circuit_breaker",
        "compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker",
        "compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker",
        "compound_leader_score_credit_min_adx_circuit_breaker",
        "compound_leader_score_credit_max_score_boost",
        "compound_leader_score_credit_volume_confirmation_overlay",
        "compound_leader_score_credit_min_ret_100d_rank",
        "compound_leader_score_credit_min_volume_expansion",
        "medium_term_leader_pullback_overlay",
        "medium_term_leader_pullback_min_ret_100d_rank",
        "medium_term_leader_pullback_max_ret_5d",
        "medium_term_leader_pullback_max_ret_10d",
        "medium_term_leader_pullback_min_drawdown_from_high",
        "medium_term_leader_pullback_max_drawdown_from_high",
        "medium_term_leader_pullback_max_above_ma20_pct",
        "medium_term_leader_pullback_max_volume_expansion",
        "medium_term_leader_pullback_min_relative_strength_score",
        "medium_term_leader_pullback_min_theme_score",
        "medium_term_leader_pullback_min_mom_return",
        "medium_term_leader_pullback_require_benchmark_risk_on",
        "medium_term_leader_pullback_require_theme_active",
        "medium_term_leader_pullback_require_price_above_ma50",
        "medium_term_leader_pullback_max_event_risk_score_circuit_breaker",
        "medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker",
        "medium_term_leader_pullback_min_benchmark_ret63d_circuit_breaker",
        "medium_term_leader_pullback_min_adx_circuit_breaker",
        "medium_term_leader_pullback_max_score_boost",
        "boundary_rank_promotion_overlay",
        "boundary_rank_promotion_rank_buffer_below",
        "boundary_rank_promotion_min_relative_strength_score",
        "boundary_rank_promotion_min_theme_score",
        "boundary_rank_promotion_min_technical_score",
        "boundary_rank_promotion_min_mom_return",
        "boundary_rank_promotion_min_drawdown_from_high",
        "boundary_rank_promotion_max_drawdown_from_high",
        "boundary_rank_promotion_max_above_ma20_pct",
        "boundary_rank_promotion_max_final_score_deficit",
        "boundary_rank_promotion_max_promotions_per_date",
        "boundary_rank_promotion_promoted_score_step",
        "boundary_rank_promotion_require_benchmark_risk_on",
        "boundary_rank_promotion_require_theme_active",
        "boundary_rank_promotion_require_price_above_ma50",
        "boundary_rank_promotion_max_event_risk_score_circuit_breaker",
        "boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker",
        "boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker",
        "boundary_rank_promotion_min_adx_circuit_breaker",
        "boundary_rank_promotion_compound_pullback_overlay",
        "boundary_rank_promotion_compound_pullback_min_ret_100d_rank",
        "boundary_rank_promotion_compound_pullback_min_252d_voladj_rank",
        "boundary_rank_promotion_compound_pullback_max_ret_5d",
        "boundary_rank_promotion_compound_pullback_max_ret_10d",
        "boundary_rank_promotion_compound_pullback_max_volume_expansion",
        "long_pullback_entry_overlay",
        "pullback_min_relative_strength_score",
        "pullback_min_theme_score",
        "pullback_min_drawdown_from_high",
        "pullback_max_drawdown_from_high",
        "pullback_max_above_ma20_pct",
        "pullback_max_score_boost",
        "long_pullback_volume_contraction_overlay",
        "pullback_volume_contraction_max_volume_expansion",
        "pullback_volume_contraction_min_drawdown_from_high",
        "pullback_volume_contraction_max_drawdown_from_high",
        "pullback_volume_contraction_max_above_ma20_pct",
        "pullback_volume_contraction_min_relative_strength_score",
        "pullback_volume_contraction_min_theme_score",
        "pullback_volume_contraction_require_benchmark_risk_on",
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker",
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker",
        "pullback_volume_contraction_min_adx_circuit_breaker",
        "pullback_volume_contraction_max_score_boost",
        "pullback_short_term_reset_overlay",
        "pullback_short_term_reset_min_drawdown_from_high",
        "pullback_short_term_reset_max_drawdown_from_high",
        "pullback_short_term_reset_max_above_ma20_pct",
        "pullback_short_term_reset_max_ret_5d",
        "pullback_short_term_reset_max_ret_10d",
        "pullback_short_term_reset_min_relative_strength_score",
        "pullback_short_term_reset_min_theme_score",
        "pullback_short_term_reset_require_benchmark_risk_on",
        "pullback_short_term_reset_require_theme_active",
        "pullback_short_term_reset_max_event_risk_score_circuit_breaker",
        "pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker",
        "pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker",
        "pullback_short_term_reset_min_adx_circuit_breaker",
        "pullback_short_term_reset_max_score_boost",
        "short_term_volume_tilt_overlay",
        "short_term_volume_tilt_max_ret_5d",
        "short_term_volume_tilt_min_volume_expansion",
        "short_term_volume_tilt_min_relative_strength_score",
        "short_term_volume_tilt_min_theme_score",
        "short_term_volume_tilt_max_above_ma20_pct",
        "short_term_volume_tilt_require_benchmark_risk_on",
        "short_term_volume_tilt_require_theme_active",
        "short_term_volume_tilt_require_price_above_ma50",
        "short_term_volume_tilt_max_event_risk_score_circuit_breaker",
        "short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker",
        "short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker",
        "short_term_volume_tilt_min_adx_circuit_breaker",
        "short_term_volume_tilt_max_score_boost",
        "short_term_volume_tilt_same_theme_peer_confirmation_overlay",
        "short_term_volume_tilt_same_theme_peer_min_count",
        "short_term_volume_tilt_same_theme_peer_min_share",
        "short_term_volume_tilt_same_theme_peer_min_avg_mom_return",
        "short_term_volume_tilt_same_theme_peer_volume_substitution_overlay",
        "short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion",
        "short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall",
        "short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale",
        "short_term_volume_tilt_non_risk_on_exception_overlay",
        "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count",
        "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share",
        "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return",
        "short_term_volume_tilt_non_risk_on_exception_score_boost_scale",
        "boundary_rs_theme_credit_overlay",
        "boundary_rs_theme_credit_rank_buffer_below",
        "boundary_rs_theme_credit_rank_buffer_above",
        "boundary_rs_theme_credit_min_relative_strength_score",
        "boundary_rs_theme_credit_min_theme_score",
        "boundary_rs_theme_credit_min_technical_score",
        "boundary_rs_theme_credit_min_mom_return",
        "boundary_rs_theme_credit_min_drawdown_from_high",
        "boundary_rs_theme_credit_max_drawdown_from_high",
        "boundary_rs_theme_credit_max_above_ma20_pct",
        "boundary_rs_theme_credit_require_benchmark_risk_on",
        "boundary_rs_theme_credit_require_theme_active",
        "boundary_rs_theme_credit_require_price_above_ma50",
        "boundary_rs_theme_credit_max_event_risk_score_circuit_breaker",
        "boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker",
        "boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker",
        "boundary_rs_theme_credit_min_adx_circuit_breaker",
        "boundary_rs_theme_credit_max_score_boost",
        "exit_quality_rank_credit_overlay",
        "exit_quality_rank_credit_rank_buffer_below",
        "exit_quality_rank_credit_rank_buffer_above",
        "exit_quality_rank_credit_min_final_score",
        "exit_quality_rank_credit_min_relative_strength_score",
        "exit_quality_rank_credit_min_theme_score",
        "exit_quality_rank_credit_min_technical_score",
        "exit_quality_rank_credit_min_mom_return",
        "exit_quality_rank_credit_max_drawdown_from_high",
        "exit_quality_rank_credit_max_above_ma20_pct",
        "exit_quality_rank_credit_require_benchmark_risk_on",
        "exit_quality_rank_credit_require_theme_active",
        "exit_quality_rank_credit_require_price_above_ma50",
        "exit_quality_rank_credit_max_event_risk_score_circuit_breaker",
        "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker",
        "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker",
        "exit_quality_rank_credit_min_adx_circuit_breaker",
        "exit_quality_rank_credit_max_score_rank_credit",
        "exit_quality_rank_credit_theme_support_overlay",
        "exit_quality_rank_credit_theme_support_lookback_days",
        "exit_quality_rank_credit_theme_support_min_peer_count",
        "exit_quality_rank_credit_theme_support_min_theme_score_prior",
        "exit_quality_rank_credit_theme_support_min_peer_rs_prior",
        "exit_quality_rank_credit_theme_support_min_combined_delta",
        "exit_quality_rank_credit_reset_support_overlay",
        "exit_quality_rank_credit_reset_support_min_reset_score",
        "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days",
        "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold",
        "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold",
        "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share",
        "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion",
        "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold",
        "exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay",
        "exit_quality_rank_credit_reset_support_same_theme_peer_min_count",
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay",
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share",
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count",
        "exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay",
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs",
        "exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay",
        "exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct",
        "exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap",
        "exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count",
        "long_pullback_reclaim_overlay",
        "pullback_reclaim_min_relative_strength_score",
        "pullback_reclaim_min_theme_score",
        "pullback_reclaim_min_drawdown_from_high",
        "pullback_reclaim_max_drawdown_from_high",
        "pullback_reclaim_recent_below_ma20_lookback_days",
        "pullback_reclaim_min_recent_below_ma20_pct",
        "pullback_reclaim_max_above_ma20_pct",
        "pullback_reclaim_min_volume_expansion",
        "pullback_reclaim_require_benchmark_risk_on",
        "pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker",
        "pullback_reclaim_min_benchmark_ret63d_circuit_breaker",
        "pullback_reclaim_min_adx_circuit_breaker",
        "pullback_reclaim_max_score_boost",
        "long_pullback_reset_inclusion_overlay",
        "pullback_reset_inclusion_min_score",
        "pullback_reset_inclusion_max_names_per_date",
        "pullback_reset_inclusion_min_final_score",
        "pullback_reset_inclusion_min_score_rank",
        "pullback_reset_inclusion_activation_support_overlay",
        "pullback_reset_inclusion_activation_min_score",
        "pullback_reset_inclusion_activation_score_rank_credit",
        "pullback_reset_inclusion_reclaim_support_overlay",
        "pullback_reset_inclusion_min_reclaim_score",
        "pullback_reset_inclusion_reclaim_score_rank_credit",
        "pullback_reset_inclusion_require_theme_breadth_not_deteriorating",
        "pullback_reset_inclusion_theme_breadth_active_share_threshold",
        "pullback_reset_inclusion_theme_breadth_shortfall_threshold",
        "pullback_reset_inclusion_require_theme_breadth_expansion",
        "pullback_reset_inclusion_theme_breadth_lookback_days",
        "pullback_reset_inclusion_theme_breadth_min_active_share",
        "pullback_reset_inclusion_theme_breadth_expansion_threshold",
        "pullback_reset_inclusion_require_rs_acceleration",
        "pullback_reset_inclusion_rs_acceleration_lookback_days",
        "pullback_reset_inclusion_min_current_rs_score",
        "pullback_reset_inclusion_min_rs_acceleration",
        *_SAME_THEME_SUBSTITUTION_MOMENTUM_KEYS,
        "long_reentry_discipline_overlay",
        "reentry_discipline_near_high_threshold",
        "reentry_discipline_min_above_ma20_pct",
        "reentry_discipline_max_volume_expansion",
        "reentry_discipline_min_adx_circuit_breaker",
        "reentry_discipline_max_score_penalty",
        "reentry_discipline_min_pullback_volume_reset_score_exemption",
        "reentry_discipline_min_relative_strength_score_exemption",
        "reentry_discipline_min_theme_score_exemption",
        "reentry_discipline_require_theme_deterioration",
        "reentry_discipline_theme_score_deterioration_threshold",
        "reentry_discipline_penalize_theme_inactive",
        "reentry_discipline_require_theme_breadth_deterioration",
        "reentry_discipline_theme_breadth_lookback_days",
        "reentry_discipline_theme_breadth_active_share_threshold",
        "reentry_discipline_theme_breadth_shortfall_threshold",
        "short_rebound_avoidance_overlay",
        "short_rebound_avoidance_min_below_ma20_pct",
        "short_rebound_avoidance_min_distance_from_high",
        "short_rebound_avoidance_min_volume_expansion",
        "short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker",
        "short_rebound_avoidance_min_adx_circuit_breaker",
        "long_min_final_score",
        "long_min_technical_score",
        "long_min_relative_strength_score",
        "long_min_theme_score",
        "require_long_theme_active",
        "long_require_price_above_ma50",
        "long_require_price_above_ma200",
        "short_min_weakness_conditions",
        "short_only_when_benchmark_risk_off",
        "short_max_final_score",
        "breakout_window",
        "breakout_max_holding_days",
        "trend_ma_fast",
        "trend_ma_slow",
    ]
    parameter_cols = [col for col in parameter_cols if col in detail.columns]
    group_cols = ["scenario"] if "scenario" in detail.columns else parameter_cols
    ordered = detail.sort_values(group_cols + (["split_id"] if "split_id" in detail.columns else []))
    params = ordered.drop_duplicates(group_cols)[parameter_cols].copy()
    metrics = detail.groupby(group_cols, dropna=False, as_index=False).agg(
        full_cagr=("full_cagr", "mean"),
        full_max_drawdown=("full_max_drawdown", "min"),
        full_sharpe=("full_sharpe", "mean"),
        full_turnover=("full_turnover", "mean"),
        train_cagr=("train_cagr", "mean"),
        train_max_drawdown=("train_max_drawdown", "min"),
        train_sharpe=("train_sharpe", "mean"),
        train_turnover=("train_turnover", "mean"),
        train_objective=("train_objective", "mean"),
        train_dsr=("train_dsr", "mean"),
        validation_cagr=("validation_cagr", "mean"),
        validation_total_return=("validation_total_return", "mean"),
        validation_max_drawdown=("validation_max_drawdown", "min"),
        validation_sharpe=("validation_sharpe", "mean"),
        validation_turnover=("validation_turnover", "mean"),
        validation_objective=("validation_objective", "mean"),
        validation_dsr=("validation_dsr", "mean"),
        objective_gap=("objective_gap", "mean"),
        trade_count=("trade_count", "mean"),
        n_splits=("split_id", "nunique"),
        validation_pass_rate=("validation_pass", "mean"),
    ).copy()
    out = params.merge(metrics, on=group_cols, how="inner")
    out["validation_pass"] = out["validation_pass_rate"] >= 0.60
    return out.sort_values(["train_objective", "validation_objective"], ascending=False).reset_index(drop=True)


def _selected_by_train_results(detail: pd.DataFrame) -> pd.DataFrame:
    """Return the validation leg selected solely by train-window objective for each split."""

    if detail.empty or not {"split_id", "train_objective"}.issubset(detail.columns):
        return pd.DataFrame()
    ordered = detail.sort_values(["split_id", "train_objective", "validation_objective"], ascending=[True, False, False])
    selected = ordered.groupby("split_id", as_index=False).head(1).reset_index(drop=True)
    selected["selection_rule"] = "highest_train_objective_with_validation_reported_oos"
    return selected


def _parse_csv_tuple(value: object) -> tuple[str, ...]:
    """Parse optimizer row values that may store a tuple/list as CSV text."""

    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    try:
        return tuple(str(part).strip() for part in value if str(part).strip())  # type: ignore[union-attr]
    except TypeError:
        text = str(value).strip()
        return (text,) if text else ()


def _best_config_from_results(config: AppConfig, results: pd.DataFrame, require_validation_pass: bool = False) -> AppConfig | None:
    """Build a runnable config from the best validation row."""

    if results.empty:
        return None
    eligible = results[results.get("validation_pass", False).astype(bool)] if "validation_pass" in results else results
    if require_validation_pass and eligible.empty:
        return None
    pool = eligible if not eligible.empty else results
    sort_cols = ["validation_objective", "train_objective"]
    if config.universe.strict_backtest:
        sort_cols = ["train_objective", "validation_objective"]
    best = pool.sort_values(sort_cols, ascending=False).iloc[0]
    portfolio = replace(
        config.portfolio,
        rebalance=str(best["rebalance"]),
        construction=str(best.get("construction", config.portfolio.construction)),
        throttle_min_weight_change=float(best["throttle_min_weight_change"]),
        throttle_min_score_change=float(best["throttle_min_score_change"]),
        throttle_min_new_weight=float(best["throttle_min_new_weight"]),
        target_gross_exposure=float(best.get("target_gross_exposure", config.portfolio.target_gross_exposure)),
        target_net_exposure=float(best.get("target_net_exposure", config.portfolio.target_net_exposure)),
        max_total_positions=int(best.get("max_total_positions", config.portfolio.max_total_positions)),
        min_target_weight=float(best.get("min_target_weight", config.portfolio.min_target_weight)),
        leader_addon_overlay=bool(best.get("leader_addon_overlay", config.portfolio.leader_addon_overlay)),
        leader_addon_multiplier=float(best.get("leader_addon_multiplier", config.portfolio.leader_addon_multiplier)),
        leader_addon_min_final_score=float(best.get("leader_addon_min_final_score", config.portfolio.leader_addon_min_final_score)),
        leader_addon_min_relative_strength_score=float(
            best.get("leader_addon_min_relative_strength_score", config.portfolio.leader_addon_min_relative_strength_score)
        ),
        leader_addon_min_theme_score=float(best.get("leader_addon_min_theme_score", config.portfolio.leader_addon_min_theme_score)),
        leader_addon_min_mom_return=float(best.get("leader_addon_min_mom_return", config.portfolio.leader_addon_min_mom_return)),
        leader_addon_max_names_per_date=int(best.get("leader_addon_max_names_per_date", config.portfolio.leader_addon_max_names_per_date)),
        leader_addon_require_benchmark_risk_on=bool(
            best.get("leader_addon_require_benchmark_risk_on", config.portfolio.leader_addon_require_benchmark_risk_on)
        ),
        leader_addon_require_controlled_pullback=bool(
            best.get("leader_addon_require_controlled_pullback", config.portfolio.leader_addon_require_controlled_pullback)
        ),
        leader_addon_require_same_theme_peer_support=bool(
            best.get(
                "leader_addon_require_same_theme_peer_support",
                config.portfolio.leader_addon_require_same_theme_peer_support,
            )
        ),
        leader_addon_same_theme_peer_min_count=int(
            best.get("leader_addon_same_theme_peer_min_count", config.portfolio.leader_addon_same_theme_peer_min_count)
        ),
        leader_addon_same_theme_peer_min_share=float(
            best.get("leader_addon_same_theme_peer_min_share", config.portfolio.leader_addon_same_theme_peer_min_share)
        ),
        leader_addon_min_drawdown_from_high=float(
            best.get("leader_addon_min_drawdown_from_high", config.portfolio.leader_addon_min_drawdown_from_high)
        ),
        leader_addon_max_drawdown_from_high=float(
            best.get("leader_addon_max_drawdown_from_high", config.portfolio.leader_addon_max_drawdown_from_high)
        ),
        leader_addon_max_above_ma20_pct_circuit_breaker=float(
            best.get(
                "leader_addon_max_above_ma20_pct_circuit_breaker",
                config.portfolio.leader_addon_max_above_ma20_pct_circuit_breaker,
            )
        ),
        leader_addon_max_event_risk_score_circuit_breaker=float(
            best.get(
                "leader_addon_max_event_risk_score_circuit_breaker",
                config.portfolio.leader_addon_max_event_risk_score_circuit_breaker,
            )
        ),
        leader_addon_max_overnight_gap_risk_score_circuit_breaker=float(
            best.get(
                "leader_addon_max_overnight_gap_risk_score_circuit_breaker",
                config.portfolio.leader_addon_max_overnight_gap_risk_score_circuit_breaker,
            )
        ),
        leader_hold_buffer_overlay=bool(best.get("leader_hold_buffer_overlay", config.portfolio.leader_hold_buffer_overlay)),
        leader_hold_buffer_min_final_score=float(
            best.get("leader_hold_buffer_min_final_score", config.portfolio.leader_hold_buffer_min_final_score)
        ),
        leader_hold_buffer_min_relative_strength_score=float(
            best.get(
                "leader_hold_buffer_min_relative_strength_score",
                config.portfolio.leader_hold_buffer_min_relative_strength_score,
            )
        ),
        leader_hold_buffer_min_theme_score=float(
            best.get("leader_hold_buffer_min_theme_score", config.portfolio.leader_hold_buffer_min_theme_score)
        ),
        leader_hold_buffer_weight_fraction=float(
            best.get("leader_hold_buffer_weight_fraction", config.portfolio.leader_hold_buffer_weight_fraction)
        ),
        leader_hold_buffer_max_weight=float(best.get("leader_hold_buffer_max_weight", config.portfolio.leader_hold_buffer_max_weight)),
        leader_hold_buffer_max_days=int(best.get("leader_hold_buffer_max_days", config.portfolio.leader_hold_buffer_max_days)),
        leader_hold_buffer_require_benchmark_risk_on=bool(
            best.get(
                "leader_hold_buffer_require_benchmark_risk_on",
                config.portfolio.leader_hold_buffer_require_benchmark_risk_on,
            )
        ),
        leader_hold_buffer_max_event_risk_score_circuit_breaker=float(
            best.get(
                "leader_hold_buffer_max_event_risk_score_circuit_breaker",
                config.portfolio.leader_hold_buffer_max_event_risk_score_circuit_breaker,
            )
        ),
        leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker=float(
            best.get(
                "leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker",
                config.portfolio.leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker,
            )
        ),
        leader_delayed_exit_overlay=bool(best.get("leader_delayed_exit_overlay", config.portfolio.leader_delayed_exit_overlay)),
        leader_delayed_exit_min_final_score=float(
            best.get("leader_delayed_exit_min_final_score", config.portfolio.leader_delayed_exit_min_final_score)
        ),
        leader_delayed_exit_min_relative_strength_score=float(
            best.get(
                "leader_delayed_exit_min_relative_strength_score",
                config.portfolio.leader_delayed_exit_min_relative_strength_score,
            )
        ),
        leader_delayed_exit_min_theme_score=float(
            best.get("leader_delayed_exit_min_theme_score", config.portfolio.leader_delayed_exit_min_theme_score)
        ),
        leader_delayed_exit_min_mom_return=float(
            best.get("leader_delayed_exit_min_mom_return", config.portfolio.leader_delayed_exit_min_mom_return)
        ),
        leader_delayed_exit_max_drawdown_from_high=float(
            best.get(
                "leader_delayed_exit_max_drawdown_from_high",
                config.portfolio.leader_delayed_exit_max_drawdown_from_high,
            )
        ),
        leader_delayed_exit_retain_fraction=float(
            best.get("leader_delayed_exit_retain_fraction", config.portfolio.leader_delayed_exit_retain_fraction)
        ),
        leader_delayed_exit_max_weight=float(
            best.get("leader_delayed_exit_max_weight", config.portfolio.leader_delayed_exit_max_weight)
        ),
        leader_delayed_exit_max_days=int(best.get("leader_delayed_exit_max_days", config.portfolio.leader_delayed_exit_max_days)),
        leader_delayed_exit_require_benchmark_risk_on=bool(
            best.get(
                "leader_delayed_exit_require_benchmark_risk_on",
                config.portfolio.leader_delayed_exit_require_benchmark_risk_on,
            )
        ),
        leader_delayed_exit_require_theme_active=bool(
            best.get("leader_delayed_exit_require_theme_active", config.portfolio.leader_delayed_exit_require_theme_active)
        ),
        leader_delayed_exit_max_event_risk_score_circuit_breaker=float(
            best.get(
                "leader_delayed_exit_max_event_risk_score_circuit_breaker",
                config.portfolio.leader_delayed_exit_max_event_risk_score_circuit_breaker,
            )
        ),
        leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker=float(
            best.get(
                "leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker",
                config.portfolio.leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker,
            )
        ),
        leader_delayed_exit_stability_overlay=bool(
            best.get("leader_delayed_exit_stability_overlay", config.portfolio.leader_delayed_exit_stability_overlay)
        ),
        leader_delayed_exit_max_final_score_drop=float(
            best.get("leader_delayed_exit_max_final_score_drop", config.portfolio.leader_delayed_exit_max_final_score_drop)
        ),
        leader_delayed_exit_max_relative_strength_drop=float(
            best.get(
                "leader_delayed_exit_max_relative_strength_drop",
                config.portfolio.leader_delayed_exit_max_relative_strength_drop,
            )
        ),
        leader_persistence_overlay=bool(best.get("leader_persistence_overlay", config.portfolio.leader_persistence_overlay)),
        leader_persistence_min_final_score=float(
            best.get("leader_persistence_min_final_score", config.portfolio.leader_persistence_min_final_score)
        ),
        leader_persistence_min_relative_strength_score=float(
            best.get(
                "leader_persistence_min_relative_strength_score",
                config.portfolio.leader_persistence_min_relative_strength_score,
            )
        ),
        leader_persistence_min_theme_score=float(
            best.get("leader_persistence_min_theme_score", config.portfolio.leader_persistence_min_theme_score)
        ),
        leader_persistence_min_mom_return=float(
            best.get("leader_persistence_min_mom_return", config.portfolio.leader_persistence_min_mom_return)
        ),
        leader_persistence_min_theme_peer_count=int(
            best.get("leader_persistence_min_theme_peer_count", config.portfolio.leader_persistence_min_theme_peer_count)
        ),
        leader_persistence_min_theme_peer_share=float(
            best.get("leader_persistence_min_theme_peer_share", config.portfolio.leader_persistence_min_theme_peer_share)
        ),
        leader_persistence_retain_fraction_of_cut=float(
            best.get(
                "leader_persistence_retain_fraction_of_cut",
                config.portfolio.leader_persistence_retain_fraction_of_cut,
            )
        ),
        leader_persistence_max_weight_bonus=float(
            best.get("leader_persistence_max_weight_bonus", config.portfolio.leader_persistence_max_weight_bonus)
        ),
        leader_persistence_require_theme_active=bool(
            best.get("leader_persistence_require_theme_active", config.portfolio.leader_persistence_require_theme_active)
        ),
        leader_persistence_require_benchmark_risk_on=bool(
            best.get(
                "leader_persistence_require_benchmark_risk_on",
                config.portfolio.leader_persistence_require_benchmark_risk_on,
            )
        ),
        leader_persistence_max_event_risk_score_circuit_breaker=float(
            best.get(
                "leader_persistence_max_event_risk_score_circuit_breaker",
                config.portfolio.leader_persistence_max_event_risk_score_circuit_breaker,
            )
        ),
        leader_persistence_max_overnight_gap_risk_score_circuit_breaker=float(
            best.get(
                "leader_persistence_max_overnight_gap_risk_score_circuit_breaker",
                config.portfolio.leader_persistence_max_overnight_gap_risk_score_circuit_breaker,
            )
        ),
        leader_persistence_stability_overlay=bool(
            best.get("leader_persistence_stability_overlay", config.portfolio.leader_persistence_stability_overlay)
        ),
        leader_persistence_min_desired_weight_fraction=float(
            best.get(
                "leader_persistence_min_desired_weight_fraction",
                config.portfolio.leader_persistence_min_desired_weight_fraction,
            )
        ),
        leader_persistence_max_final_score_drop=float(
            best.get("leader_persistence_max_final_score_drop", config.portfolio.leader_persistence_max_final_score_drop)
        ),
        leader_persistence_max_relative_strength_drop=float(
            best.get(
                "leader_persistence_max_relative_strength_drop",
                config.portfolio.leader_persistence_max_relative_strength_drop,
            )
        ),
        leader_persistence_peer_strength_reward_overlay=bool(
            best.get(
                "leader_persistence_peer_strength_reward_overlay",
                config.portfolio.leader_persistence_peer_strength_reward_overlay,
            )
        ),
        leader_persistence_peer_strength_reward_min_theme_peer_count=int(
            best.get(
                "leader_persistence_peer_strength_reward_min_theme_peer_count",
                config.portfolio.leader_persistence_peer_strength_reward_min_theme_peer_count,
            )
        ),
        leader_persistence_peer_strength_reward_min_theme_peer_share=float(
            best.get(
                "leader_persistence_peer_strength_reward_min_theme_peer_share",
                config.portfolio.leader_persistence_peer_strength_reward_min_theme_peer_share,
            )
        ),
        leader_persistence_peer_strength_reward_min_relative_strength_score=float(
            best.get(
                "leader_persistence_peer_strength_reward_min_relative_strength_score",
                config.portfolio.leader_persistence_peer_strength_reward_min_relative_strength_score,
            )
        ),
        leader_persistence_peer_strength_reward_retain_fraction_boost=float(
            best.get(
                "leader_persistence_peer_strength_reward_retain_fraction_boost",
                config.portfolio.leader_persistence_peer_strength_reward_retain_fraction_boost,
            )
        ),
        leader_persistence_peer_strength_reward_max_weight_bonus=float(
            best.get(
                "leader_persistence_peer_strength_reward_max_weight_bonus",
                config.portfolio.leader_persistence_peer_strength_reward_max_weight_bonus,
            )
        ),
        leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker=float(
            best.get(
                "leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker",
                config.portfolio.leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker,
            )
        ),
        conviction_sizing_overlay=bool(best.get("conviction_sizing_overlay", config.portfolio.conviction_sizing_overlay)),
        conviction_sizing_power=float(best.get("conviction_sizing_power", config.portfolio.conviction_sizing_power)),
        conviction_sizing_max_multiplier=float(
            best.get("conviction_sizing_max_multiplier", config.portfolio.conviction_sizing_max_multiplier)
        ),
        conviction_sizing_min_names=int(best.get("conviction_sizing_min_names", config.portfolio.conviction_sizing_min_names)),
        conviction_sizing_min_final_score=float(
            best.get("conviction_sizing_min_final_score", config.portfolio.conviction_sizing_min_final_score)
        ),
        conviction_sizing_min_relative_strength_score=float(
            best.get(
                "conviction_sizing_min_relative_strength_score",
                config.portfolio.conviction_sizing_min_relative_strength_score,
            )
        ),
        conviction_sizing_min_theme_score=float(
            best.get("conviction_sizing_min_theme_score", config.portfolio.conviction_sizing_min_theme_score)
        ),
        conviction_sizing_max_event_risk_score_circuit_breaker=float(
            best.get(
                "conviction_sizing_max_event_risk_score_circuit_breaker",
                config.portfolio.conviction_sizing_max_event_risk_score_circuit_breaker,
            )
        ),
        conviction_sizing_max_overnight_gap_risk_score_circuit_breaker=float(
            best.get(
                "conviction_sizing_max_overnight_gap_risk_score_circuit_breaker",
                config.portfolio.conviction_sizing_max_overnight_gap_risk_score_circuit_breaker,
            )
        ),
        conviction_sizing_require_benchmark_risk_on=bool(
            best.get("conviction_sizing_require_benchmark_risk_on", config.portfolio.conviction_sizing_require_benchmark_risk_on)
        ),
        conviction_sizing_require_same_theme_peer_support=bool(
            best.get(
                "conviction_sizing_require_same_theme_peer_support",
                config.portfolio.conviction_sizing_require_same_theme_peer_support,
            )
        ),
        conviction_sizing_allowed_themes=_parse_csv_tuple(
            best.get("conviction_sizing_allowed_themes", config.portfolio.conviction_sizing_allowed_themes)
        ),
        conviction_sizing_same_theme_peer_min_count=int(
            best.get("conviction_sizing_same_theme_peer_min_count", config.portfolio.conviction_sizing_same_theme_peer_min_count)
        ),
        conviction_sizing_same_theme_peer_min_share=float(
            best.get("conviction_sizing_same_theme_peer_min_share", config.portfolio.conviction_sizing_same_theme_peer_min_share)
        ),
        conviction_sizing_same_theme_peer_min_avg_relative_strength_score=float(
            best.get(
                "conviction_sizing_same_theme_peer_min_avg_relative_strength_score",
                config.portfolio.conviction_sizing_same_theme_peer_min_avg_relative_strength_score,
            )
        ),
        conviction_sizing_same_theme_peer_min_avg_mom_return=float(
            best.get(
                "conviction_sizing_same_theme_peer_min_avg_mom_return",
                config.portfolio.conviction_sizing_same_theme_peer_min_avg_mom_return,
            )
        ),
    )
    execution = replace(
        config.execution,
        daily_turnover_cap=float(best.get("daily_turnover_cap", config.execution.daily_turnover_cap)),
        max_adv_participation=float(best.get("max_adv_participation", config.execution.max_adv_participation)),
        market_impact_bps_per_1pct_adv=float(best.get("market_impact_bps_per_1pct_adv", config.execution.market_impact_bps_per_1pct_adv)),
        min_trade_notional=float(best.get("min_trade_notional", config.execution.min_trade_notional)),
    )
    risk = replace(
        config.risk,
        min_holding_days=int(best.get("min_holding_days", config.risk.min_holding_days)),
        max_drawdown_reduce_exposure=float(best.get("max_drawdown_reduce_exposure", config.risk.max_drawdown_reduce_exposure)),
        max_drawdown_cash_mode=float(best.get("max_drawdown_cash_mode", config.risk.max_drawdown_cash_mode)),
        drawdown_reduction_multiplier=float(best.get("drawdown_reduction_multiplier", config.risk.drawdown_reduction_multiplier)),
        drawdown_reset=str(best.get("drawdown_reset", config.risk.drawdown_reset)),
    )
    regime = replace(
        config.regime,
        exposure_overlay=bool(best.get("regime_exposure_overlay", config.regime.exposure_overlay)),
        risk_off_exposure_multiplier=float(best.get("risk_off_exposure_multiplier", config.regime.risk_off_exposure_multiplier)),
        partial_risk_on_exposure_multiplier=float(best.get("partial_risk_on_exposure_multiplier", config.regime.partial_risk_on_exposure_multiplier)),
        partial_exposure_regime_score=float(best.get("partial_exposure_regime_score", config.regime.partial_exposure_regime_score)),
        full_exposure_regime_score=float(best.get("full_exposure_regime_score", config.regime.full_exposure_regime_score)),
        analog_momentum_overlay=bool(best.get("analog_momentum_overlay", config.regime.analog_momentum_overlay)),
        analog_inactive_exposure_multiplier=float(best.get("analog_inactive_exposure_multiplier", config.regime.analog_inactive_exposure_multiplier)),
        analog_active_exposure_multiplier=float(best.get("analog_active_exposure_multiplier", config.regime.analog_active_exposure_multiplier)),
        analog_quantile=float(best.get("analog_quantile", config.regime.analog_quantile)),
        leader_crowding_relief_overlay=bool(best.get("leader_crowding_relief_overlay", config.regime.leader_crowding_relief_overlay)),
        leader_crowding_relief_min_relative_strength_score=float(
            best.get(
                "leader_crowding_relief_min_relative_strength_score",
                config.regime.leader_crowding_relief_min_relative_strength_score,
            )
        ),
        leader_crowding_relief_min_theme_score=float(
            best.get("leader_crowding_relief_min_theme_score", config.regime.leader_crowding_relief_min_theme_score)
        ),
        leader_crowding_relief_weight=float(best.get("leader_crowding_relief_weight", config.regime.leader_crowding_relief_weight)),
        leader_crowding_relief_max_names_per_bucket=int(
            best.get("leader_crowding_relief_max_names_per_bucket", config.regime.leader_crowding_relief_max_names_per_bucket)
        ),
        leader_crowding_relief_max_weight_per_symbol=float(
            best.get("leader_crowding_relief_max_weight_per_symbol", config.regime.leader_crowding_relief_max_weight_per_symbol)
        ),
        leader_crowding_relief_max_bucket_weight_restore=float(
            best.get("leader_crowding_relief_max_bucket_weight_restore", config.regime.leader_crowding_relief_max_bucket_weight_restore)
        ),
        leader_crowding_relief_require_benchmark_risk_on=bool(
            best.get(
                "leader_crowding_relief_require_benchmark_risk_on",
                config.regime.leader_crowding_relief_require_benchmark_risk_on,
            )
        ),
        leader_crowding_relief_controlled_pullback_overlay=bool(
            best.get(
                "leader_crowding_relief_controlled_pullback_overlay",
                config.regime.leader_crowding_relief_controlled_pullback_overlay,
            )
        ),
        leader_crowding_relief_controlled_pullback_min_final_score=float(
            best.get(
                "leader_crowding_relief_controlled_pullback_min_final_score",
                config.regime.leader_crowding_relief_controlled_pullback_min_final_score,
            )
        ),
        leader_crowding_relief_controlled_pullback_min_distance_from_high=float(
            best.get(
                "leader_crowding_relief_controlled_pullback_min_distance_from_high",
                config.regime.leader_crowding_relief_controlled_pullback_min_distance_from_high,
            )
        ),
        leader_crowding_relief_controlled_pullback_max_distance_from_high=float(
            best.get(
                "leader_crowding_relief_controlled_pullback_max_distance_from_high",
                config.regime.leader_crowding_relief_controlled_pullback_max_distance_from_high,
            )
        ),
        leader_crowding_relief_controlled_pullback_max_above_ma20_pct=float(
            best.get(
                "leader_crowding_relief_controlled_pullback_max_above_ma20_pct",
                config.regime.leader_crowding_relief_controlled_pullback_max_above_ma20_pct,
            )
        ),
        leader_crowding_relief_controlled_pullback_max_mom_return=float(
            best.get(
                "leader_crowding_relief_controlled_pullback_max_mom_return",
                config.regime.leader_crowding_relief_controlled_pullback_max_mom_return,
            )
        ),
    )
    events = replace(
        config.events,
        block_high_event_risk=bool(best.get("event_block_high_risk", config.events.block_high_event_risk)),
        earnings_risk_multiplier=float(best.get("event_earnings_risk_multiplier", config.events.earnings_risk_multiplier)),
        max_event_risk_score_for_normal_strategies=float(best.get("event_max_risk_score", config.events.max_event_risk_score_for_normal_strategies)),
    )
    strategies = {name: dict(spec) for name, spec in config.strategies.items()}
    strategies.setdefault("momentum", {})
    momentum_updates = {
        "enabled": True,
        "weight": float(best["momentum_weight"]),
        "lookback_returns": int(best["lookback_returns"]),
        "skip_recent_days": int(best["skip_recent_days"]),
        "long_quantile": float(best["long_quantile"]),
        "short_quantile": float(best["short_quantile"]),
        "max_positions": int(best["momentum_max_positions"]),
        "momentum_custom_final_score_weights": (
            bool(best.get("momentum_custom_final_score_weights", False))
            if pd.notna(best.get("momentum_custom_final_score_weights"))
            else False
        ),
        "momentum_final_score_technical_weight": float(best["momentum_final_score_technical_weight"]) if pd.notna(best.get("momentum_final_score_technical_weight")) else None,
        "momentum_final_score_relative_strength_weight": float(best["momentum_final_score_relative_strength_weight"]) if pd.notna(best.get("momentum_final_score_relative_strength_weight")) else None,
        "momentum_final_score_theme_weight": float(best["momentum_final_score_theme_weight"]) if pd.notna(best.get("momentum_final_score_theme_weight")) else None,
        "momentum_final_score_fundamental_weight": float(best["momentum_final_score_fundamental_weight"]) if pd.notna(best.get("momentum_final_score_fundamental_weight")) else None,
        "momentum_final_score_event_risk_penalty": float(best["momentum_final_score_event_risk_penalty"]) if pd.notna(best.get("momentum_final_score_event_risk_penalty")) else None,
        "momentum_custom_final_score_blend": float(best["momentum_custom_final_score_blend"]) if pd.notna(best.get("momentum_custom_final_score_blend")) else None,
        "momentum_custom_score_controlled_entry_overlay": bool(best.get("momentum_custom_score_controlled_entry_overlay", False)) if pd.notna(best.get("momentum_custom_score_controlled_entry_overlay")) else False,
        "momentum_custom_score_min_relative_strength_score": float(best["momentum_custom_score_min_relative_strength_score"]) if pd.notna(best.get("momentum_custom_score_min_relative_strength_score")) else None,
        "momentum_custom_score_min_theme_score": float(best["momentum_custom_score_min_theme_score"]) if pd.notna(best.get("momentum_custom_score_min_theme_score")) else None,
        "momentum_custom_score_max_above_ma20_pct": float(best["momentum_custom_score_max_above_ma20_pct"]) if pd.notna(best.get("momentum_custom_score_max_above_ma20_pct")) else None,
        "momentum_custom_score_max_event_risk_score_circuit_breaker": float(best["momentum_custom_score_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("momentum_custom_score_max_event_risk_score_circuit_breaker")) else None,
        "momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker": float(best["momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("momentum_custom_score_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "momentum_custom_score_min_benchmark_ret63d_circuit_breaker": float(best["momentum_custom_score_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("momentum_custom_score_min_benchmark_ret63d_circuit_breaker")) else None,
        "momentum_custom_score_require_benchmark_risk_on": bool(best.get("momentum_custom_score_require_benchmark_risk_on", False)) if pd.notna(best.get("momentum_custom_score_require_benchmark_risk_on")) else False,
        "momentum_custom_score_require_theme_active": bool(best.get("momentum_custom_score_require_theme_active", False)) if pd.notna(best.get("momentum_custom_score_require_theme_active")) else False,
        "momentum_custom_score_require_price_above_ma50": bool(best.get("momentum_custom_score_require_price_above_ma50", False)) if pd.notna(best.get("momentum_custom_score_require_price_above_ma50")) else False,
        "theme_strength_delta_overlay": bool(best.get("theme_strength_delta_overlay", False)) if pd.notna(best.get("theme_strength_delta_overlay")) else False,
        "theme_strength_delta_lookback_days": int(best["theme_strength_delta_lookback_days"]) if pd.notna(best.get("theme_strength_delta_lookback_days")) else None,
        "theme_strength_delta_min_peer_count": int(best["theme_strength_delta_min_peer_count"]) if pd.notna(best.get("theme_strength_delta_min_peer_count")) else None,
        "theme_strength_delta_min_theme_score": float(best["theme_strength_delta_min_theme_score"]) if pd.notna(best.get("theme_strength_delta_min_theme_score")) else None,
        "theme_strength_delta_min_relative_strength_score": float(best["theme_strength_delta_min_relative_strength_score"]) if pd.notna(best.get("theme_strength_delta_min_relative_strength_score")) else None,
        "theme_strength_delta_min_theme_prior": float(best["theme_strength_delta_min_theme_prior"]) if pd.notna(best.get("theme_strength_delta_min_theme_prior")) else None,
        "theme_strength_delta_min_peer_rs_prior": float(best["theme_strength_delta_min_peer_rs_prior"]) if pd.notna(best.get("theme_strength_delta_min_peer_rs_prior")) else None,
        "theme_strength_delta_min_combined_delta": float(best["theme_strength_delta_min_combined_delta"]) if pd.notna(best.get("theme_strength_delta_min_combined_delta")) else None,
        "theme_strength_delta_max_score_boost": float(best["theme_strength_delta_max_score_boost"]) if pd.notna(best.get("theme_strength_delta_max_score_boost")) else None,
        "theme_strength_delta_require_benchmark_risk_on": bool(best.get("theme_strength_delta_require_benchmark_risk_on", False)) if pd.notna(best.get("theme_strength_delta_require_benchmark_risk_on")) else False,
        "theme_strength_delta_require_theme_active": bool(best.get("theme_strength_delta_require_theme_active", False)) if pd.notna(best.get("theme_strength_delta_require_theme_active")) else False,
        "theme_strength_delta_max_event_risk_score_circuit_breaker": float(best["theme_strength_delta_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("theme_strength_delta_max_event_risk_score_circuit_breaker")) else None,
        "theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker": float(best["theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("theme_strength_delta_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "theme_strength_delta_min_benchmark_ret63d_circuit_breaker": float(best["theme_strength_delta_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("theme_strength_delta_min_benchmark_ret63d_circuit_breaker")) else None,
        "theme_breadth_acceleration_overlay": bool(best.get("theme_breadth_acceleration_overlay", False)) if pd.notna(best.get("theme_breadth_acceleration_overlay")) else False,
        "theme_breadth_acceleration_lookback_days": int(best["theme_breadth_acceleration_lookback_days"]) if pd.notna(best.get("theme_breadth_acceleration_lookback_days")) else None,
        "theme_breadth_acceleration_min_active_share": float(best["theme_breadth_acceleration_min_active_share"]) if pd.notna(best.get("theme_breadth_acceleration_min_active_share")) else None,
        "theme_breadth_acceleration_min_change": float(best["theme_breadth_acceleration_min_change"]) if pd.notna(best.get("theme_breadth_acceleration_min_change")) else None,
        "theme_breadth_acceleration_min_theme_peer_count": int(best["theme_breadth_acceleration_min_theme_peer_count"]) if pd.notna(best.get("theme_breadth_acceleration_min_theme_peer_count")) else None,
        "theme_breadth_acceleration_min_theme_score": float(best["theme_breadth_acceleration_min_theme_score"]) if pd.notna(best.get("theme_breadth_acceleration_min_theme_score")) else None,
        "theme_breadth_acceleration_min_relative_strength_score": float(best["theme_breadth_acceleration_min_relative_strength_score"]) if pd.notna(best.get("theme_breadth_acceleration_min_relative_strength_score")) else None,
        "theme_breadth_acceleration_max_score_boost": float(best["theme_breadth_acceleration_max_score_boost"]) if pd.notna(best.get("theme_breadth_acceleration_max_score_boost")) else None,
        "theme_breadth_acceleration_require_benchmark_risk_on": bool(best.get("theme_breadth_acceleration_require_benchmark_risk_on", False)) if pd.notna(best.get("theme_breadth_acceleration_require_benchmark_risk_on")) else False,
        "theme_breadth_acceleration_require_theme_active": bool(best.get("theme_breadth_acceleration_require_theme_active", False)) if pd.notna(best.get("theme_breadth_acceleration_require_theme_active")) else False,
        "theme_breadth_acceleration_max_event_risk_score_circuit_breaker": float(best["theme_breadth_acceleration_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("theme_breadth_acceleration_max_event_risk_score_circuit_breaker")) else None,
        "theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker": float(best["theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("theme_breadth_acceleration_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker": float(best["theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("theme_breadth_acceleration_min_benchmark_ret63d_circuit_breaker")) else None,
        "gap_adjusted_continuation_overlay": bool(best.get("gap_adjusted_continuation_overlay", False)) if pd.notna(best.get("gap_adjusted_continuation_overlay")) else False,
        "gap_adjusted_continuation_min_drawdown_from_high": float(best["gap_adjusted_continuation_min_drawdown_from_high"]) if pd.notna(best.get("gap_adjusted_continuation_min_drawdown_from_high")) else None,
        "gap_adjusted_continuation_max_drawdown_from_high": float(best["gap_adjusted_continuation_max_drawdown_from_high"]) if pd.notna(best.get("gap_adjusted_continuation_max_drawdown_from_high")) else None,
        "gap_adjusted_continuation_max_above_ma20_pct": float(best["gap_adjusted_continuation_max_above_ma20_pct"]) if pd.notna(best.get("gap_adjusted_continuation_max_above_ma20_pct")) else None,
        "gap_adjusted_continuation_min_relative_strength_score": float(best["gap_adjusted_continuation_min_relative_strength_score"]) if pd.notna(best.get("gap_adjusted_continuation_min_relative_strength_score")) else None,
        "gap_adjusted_continuation_min_theme_score": float(best["gap_adjusted_continuation_min_theme_score"]) if pd.notna(best.get("gap_adjusted_continuation_min_theme_score")) else None,
        "gap_adjusted_continuation_min_adx_circuit_breaker": float(best["gap_adjusted_continuation_min_adx_circuit_breaker"]) if pd.notna(best.get("gap_adjusted_continuation_min_adx_circuit_breaker")) else None,
        "gap_adjusted_continuation_max_event_risk_score_circuit_breaker": float(best["gap_adjusted_continuation_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("gap_adjusted_continuation_max_event_risk_score_circuit_breaker")) else None,
        "gap_adjusted_continuation_max_gap_risk_score_circuit_breaker": float(best["gap_adjusted_continuation_max_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("gap_adjusted_continuation_max_gap_risk_score_circuit_breaker")) else None,
        "gap_adjusted_continuation_benign_gap_risk_score": float(best["gap_adjusted_continuation_benign_gap_risk_score"]) if pd.notna(best.get("gap_adjusted_continuation_benign_gap_risk_score")) else None,
        "gap_adjusted_continuation_gap_risk_lookback_days": int(best["gap_adjusted_continuation_gap_risk_lookback_days"]) if pd.notna(best.get("gap_adjusted_continuation_gap_risk_lookback_days")) else None,
        "gap_adjusted_continuation_min_gap_risk_improvement": float(best["gap_adjusted_continuation_min_gap_risk_improvement"]) if pd.notna(best.get("gap_adjusted_continuation_min_gap_risk_improvement")) else None,
        "gap_adjusted_continuation_min_volume_expansion": float(best["gap_adjusted_continuation_min_volume_expansion"]) if pd.notna(best.get("gap_adjusted_continuation_min_volume_expansion")) else None,
        "gap_adjusted_continuation_max_volume_expansion": float(best["gap_adjusted_continuation_max_volume_expansion"]) if pd.notna(best.get("gap_adjusted_continuation_max_volume_expansion")) else None,
        "gap_adjusted_continuation_min_benchmark_ret63d_circuit_breaker": float(best["gap_adjusted_continuation_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("gap_adjusted_continuation_min_benchmark_ret63d_circuit_breaker")) else None,
        "gap_adjusted_continuation_require_benchmark_risk_on": bool(best.get("gap_adjusted_continuation_require_benchmark_risk_on", False)) if pd.notna(best.get("gap_adjusted_continuation_require_benchmark_risk_on")) else False,
        "gap_adjusted_continuation_require_theme_active": bool(best.get("gap_adjusted_continuation_require_theme_active", False)) if pd.notna(best.get("gap_adjusted_continuation_require_theme_active")) else False,
        "gap_adjusted_continuation_same_theme_peer_quality_overlay": bool(best.get("gap_adjusted_continuation_same_theme_peer_quality_overlay", False)) if pd.notna(best.get("gap_adjusted_continuation_same_theme_peer_quality_overlay")) else False,
        "gap_adjusted_continuation_same_theme_peer_min_count": int(best["gap_adjusted_continuation_same_theme_peer_min_count"]) if pd.notna(best.get("gap_adjusted_continuation_same_theme_peer_min_count")) else None,
        "gap_adjusted_continuation_same_theme_peer_min_share": float(best["gap_adjusted_continuation_same_theme_peer_min_share"]) if pd.notna(best.get("gap_adjusted_continuation_same_theme_peer_min_share")) else None,
        "gap_adjusted_continuation_same_theme_peer_min_avg_rs": float(best["gap_adjusted_continuation_same_theme_peer_min_avg_rs"]) if pd.notna(best.get("gap_adjusted_continuation_same_theme_peer_min_avg_rs")) else None,
        "gap_adjusted_continuation_max_score_boost": float(best["gap_adjusted_continuation_max_score_boost"]) if pd.notna(best.get("gap_adjusted_continuation_max_score_boost")) else None,
        "compound_leader_score_credit_overlay": bool(best.get("compound_leader_score_credit_overlay", False)) if pd.notna(best.get("compound_leader_score_credit_overlay")) else False,
        "compound_leader_score_credit_min_126d_voladj_rank": float(best["compound_leader_score_credit_min_126d_voladj_rank"]) if pd.notna(best.get("compound_leader_score_credit_min_126d_voladj_rank")) else None,
        "compound_leader_score_credit_min_252d_voladj_rank": float(best["compound_leader_score_credit_min_252d_voladj_rank"]) if pd.notna(best.get("compound_leader_score_credit_min_252d_voladj_rank")) else None,
        "compound_leader_score_credit_min_final_score": float(best["compound_leader_score_credit_min_final_score"]) if pd.notna(best.get("compound_leader_score_credit_min_final_score")) else None,
        "compound_leader_score_credit_min_relative_strength_score": float(best["compound_leader_score_credit_min_relative_strength_score"]) if pd.notna(best.get("compound_leader_score_credit_min_relative_strength_score")) else None,
        "compound_leader_score_credit_min_theme_score": float(best["compound_leader_score_credit_min_theme_score"]) if pd.notna(best.get("compound_leader_score_credit_min_theme_score")) else None,
        "compound_leader_score_credit_min_mom_return": float(best["compound_leader_score_credit_min_mom_return"]) if pd.notna(best.get("compound_leader_score_credit_min_mom_return")) else None,
        "compound_leader_score_credit_min_theme_peer_count": int(best["compound_leader_score_credit_min_theme_peer_count"]) if pd.notna(best.get("compound_leader_score_credit_min_theme_peer_count")) else None,
        "compound_leader_score_credit_max_ret_10d": float(best["compound_leader_score_credit_max_ret_10d"]) if pd.notna(best.get("compound_leader_score_credit_max_ret_10d")) else None,
        "compound_leader_score_credit_max_drawdown_from_high": float(best["compound_leader_score_credit_max_drawdown_from_high"]) if pd.notna(best.get("compound_leader_score_credit_max_drawdown_from_high")) else None,
        "compound_leader_score_credit_max_above_ma20_pct": float(best["compound_leader_score_credit_max_above_ma20_pct"]) if pd.notna(best.get("compound_leader_score_credit_max_above_ma20_pct")) else None,
        "compound_leader_score_credit_boundary_overlay": bool(best.get("compound_leader_score_credit_boundary_overlay", False)) if pd.notna(best.get("compound_leader_score_credit_boundary_overlay")) else False,
        "compound_leader_score_credit_rank_buffer_below": float(best["compound_leader_score_credit_rank_buffer_below"]) if pd.notna(best.get("compound_leader_score_credit_rank_buffer_below")) else None,
        "compound_leader_score_credit_rank_buffer_above": float(best["compound_leader_score_credit_rank_buffer_above"]) if pd.notna(best.get("compound_leader_score_credit_rank_buffer_above")) else None,
        "compound_leader_score_credit_require_benchmark_risk_on": bool(best.get("compound_leader_score_credit_require_benchmark_risk_on", False)) if pd.notna(best.get("compound_leader_score_credit_require_benchmark_risk_on")) else False,
        "compound_leader_score_credit_require_theme_active": bool(best.get("compound_leader_score_credit_require_theme_active", False)) if pd.notna(best.get("compound_leader_score_credit_require_theme_active")) else False,
        "compound_leader_score_credit_require_price_above_ma50": bool(best.get("compound_leader_score_credit_require_price_above_ma50", False)) if pd.notna(best.get("compound_leader_score_credit_require_price_above_ma50")) else False,
        "compound_leader_score_credit_max_event_risk_score_circuit_breaker": float(best["compound_leader_score_credit_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("compound_leader_score_credit_max_event_risk_score_circuit_breaker")) else None,
        "compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker": float(best["compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("compound_leader_score_credit_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker": float(best["compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("compound_leader_score_credit_min_benchmark_ret63d_circuit_breaker")) else None,
        "compound_leader_score_credit_min_adx_circuit_breaker": float(best["compound_leader_score_credit_min_adx_circuit_breaker"]) if pd.notna(best.get("compound_leader_score_credit_min_adx_circuit_breaker")) else None,
        "compound_leader_score_credit_max_score_boost": float(best["compound_leader_score_credit_max_score_boost"]) if pd.notna(best.get("compound_leader_score_credit_max_score_boost")) else None,
        "compound_leader_score_credit_volume_confirmation_overlay": bool(best.get("compound_leader_score_credit_volume_confirmation_overlay", False)) if pd.notna(best.get("compound_leader_score_credit_volume_confirmation_overlay")) else False,
        "compound_leader_score_credit_min_ret_100d_rank": float(best["compound_leader_score_credit_min_ret_100d_rank"]) if pd.notna(best.get("compound_leader_score_credit_min_ret_100d_rank")) else None,
        "compound_leader_score_credit_min_volume_expansion": float(best["compound_leader_score_credit_min_volume_expansion"]) if pd.notna(best.get("compound_leader_score_credit_min_volume_expansion")) else None,
        "medium_term_leader_pullback_overlay": bool(best.get("medium_term_leader_pullback_overlay", False)) if pd.notna(best.get("medium_term_leader_pullback_overlay")) else False,
        "medium_term_leader_pullback_min_ret_100d_rank": float(best["medium_term_leader_pullback_min_ret_100d_rank"]) if pd.notna(best.get("medium_term_leader_pullback_min_ret_100d_rank")) else None,
        "medium_term_leader_pullback_max_ret_5d": float(best["medium_term_leader_pullback_max_ret_5d"]) if pd.notna(best.get("medium_term_leader_pullback_max_ret_5d")) else None,
        "medium_term_leader_pullback_max_ret_10d": float(best["medium_term_leader_pullback_max_ret_10d"]) if pd.notna(best.get("medium_term_leader_pullback_max_ret_10d")) else None,
        "medium_term_leader_pullback_min_drawdown_from_high": float(best["medium_term_leader_pullback_min_drawdown_from_high"]) if pd.notna(best.get("medium_term_leader_pullback_min_drawdown_from_high")) else None,
        "medium_term_leader_pullback_max_drawdown_from_high": float(best["medium_term_leader_pullback_max_drawdown_from_high"]) if pd.notna(best.get("medium_term_leader_pullback_max_drawdown_from_high")) else None,
        "medium_term_leader_pullback_max_above_ma20_pct": float(best["medium_term_leader_pullback_max_above_ma20_pct"]) if pd.notna(best.get("medium_term_leader_pullback_max_above_ma20_pct")) else None,
        "medium_term_leader_pullback_max_volume_expansion": float(best["medium_term_leader_pullback_max_volume_expansion"]) if pd.notna(best.get("medium_term_leader_pullback_max_volume_expansion")) else None,
        "medium_term_leader_pullback_min_relative_strength_score": float(best["medium_term_leader_pullback_min_relative_strength_score"]) if pd.notna(best.get("medium_term_leader_pullback_min_relative_strength_score")) else None,
        "medium_term_leader_pullback_min_theme_score": float(best["medium_term_leader_pullback_min_theme_score"]) if pd.notna(best.get("medium_term_leader_pullback_min_theme_score")) else None,
        "medium_term_leader_pullback_min_mom_return": float(best["medium_term_leader_pullback_min_mom_return"]) if pd.notna(best.get("medium_term_leader_pullback_min_mom_return")) else None,
        "medium_term_leader_pullback_require_benchmark_risk_on": bool(best.get("medium_term_leader_pullback_require_benchmark_risk_on", False)) if pd.notna(best.get("medium_term_leader_pullback_require_benchmark_risk_on")) else False,
        "medium_term_leader_pullback_require_theme_active": bool(best.get("medium_term_leader_pullback_require_theme_active", False)) if pd.notna(best.get("medium_term_leader_pullback_require_theme_active")) else False,
        "medium_term_leader_pullback_require_price_above_ma50": bool(best.get("medium_term_leader_pullback_require_price_above_ma50", False)) if pd.notna(best.get("medium_term_leader_pullback_require_price_above_ma50")) else False,
        "medium_term_leader_pullback_max_event_risk_score_circuit_breaker": float(best["medium_term_leader_pullback_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("medium_term_leader_pullback_max_event_risk_score_circuit_breaker")) else None,
        "medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker": float(best["medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("medium_term_leader_pullback_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "medium_term_leader_pullback_min_benchmark_ret63d_circuit_breaker": float(best["medium_term_leader_pullback_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("medium_term_leader_pullback_min_benchmark_ret63d_circuit_breaker")) else None,
        "medium_term_leader_pullback_min_adx_circuit_breaker": float(best["medium_term_leader_pullback_min_adx_circuit_breaker"]) if pd.notna(best.get("medium_term_leader_pullback_min_adx_circuit_breaker")) else None,
        "medium_term_leader_pullback_max_score_boost": float(best["medium_term_leader_pullback_max_score_boost"]) if pd.notna(best.get("medium_term_leader_pullback_max_score_boost")) else None,
        "boundary_rank_promotion_overlay": bool(best.get("boundary_rank_promotion_overlay", False)) if pd.notna(best.get("boundary_rank_promotion_overlay")) else False,
        "boundary_rank_promotion_rank_buffer_below": float(best["boundary_rank_promotion_rank_buffer_below"]) if pd.notna(best.get("boundary_rank_promotion_rank_buffer_below")) else None,
        "boundary_rank_promotion_min_relative_strength_score": float(best["boundary_rank_promotion_min_relative_strength_score"]) if pd.notna(best.get("boundary_rank_promotion_min_relative_strength_score")) else None,
        "boundary_rank_promotion_min_theme_score": float(best["boundary_rank_promotion_min_theme_score"]) if pd.notna(best.get("boundary_rank_promotion_min_theme_score")) else None,
        "boundary_rank_promotion_min_technical_score": float(best["boundary_rank_promotion_min_technical_score"]) if pd.notna(best.get("boundary_rank_promotion_min_technical_score")) else None,
        "boundary_rank_promotion_min_mom_return": float(best["boundary_rank_promotion_min_mom_return"]) if pd.notna(best.get("boundary_rank_promotion_min_mom_return")) else None,
        "boundary_rank_promotion_min_drawdown_from_high": float(best["boundary_rank_promotion_min_drawdown_from_high"]) if pd.notna(best.get("boundary_rank_promotion_min_drawdown_from_high")) else None,
        "boundary_rank_promotion_max_drawdown_from_high": float(best["boundary_rank_promotion_max_drawdown_from_high"]) if pd.notna(best.get("boundary_rank_promotion_max_drawdown_from_high")) else None,
        "boundary_rank_promotion_max_above_ma20_pct": float(best["boundary_rank_promotion_max_above_ma20_pct"]) if pd.notna(best.get("boundary_rank_promotion_max_above_ma20_pct")) else None,
        "boundary_rank_promotion_max_final_score_deficit": float(best["boundary_rank_promotion_max_final_score_deficit"]) if pd.notna(best.get("boundary_rank_promotion_max_final_score_deficit")) else None,
        "boundary_rank_promotion_max_promotions_per_date": int(best["boundary_rank_promotion_max_promotions_per_date"]) if pd.notna(best.get("boundary_rank_promotion_max_promotions_per_date")) else None,
        "boundary_rank_promotion_promoted_score_step": float(best["boundary_rank_promotion_promoted_score_step"]) if pd.notna(best.get("boundary_rank_promotion_promoted_score_step")) else None,
        "boundary_rank_promotion_require_benchmark_risk_on": bool(best.get("boundary_rank_promotion_require_benchmark_risk_on", False)) if pd.notna(best.get("boundary_rank_promotion_require_benchmark_risk_on")) else False,
        "boundary_rank_promotion_require_theme_active": bool(best.get("boundary_rank_promotion_require_theme_active", False)) if pd.notna(best.get("boundary_rank_promotion_require_theme_active")) else False,
        "boundary_rank_promotion_require_price_above_ma50": bool(best.get("boundary_rank_promotion_require_price_above_ma50", False)) if pd.notna(best.get("boundary_rank_promotion_require_price_above_ma50")) else False,
        "boundary_rank_promotion_max_event_risk_score_circuit_breaker": float(best["boundary_rank_promotion_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("boundary_rank_promotion_max_event_risk_score_circuit_breaker")) else None,
        "boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker": float(best["boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("boundary_rank_promotion_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker": float(best["boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("boundary_rank_promotion_min_benchmark_ret63d_circuit_breaker")) else None,
        "boundary_rank_promotion_min_adx_circuit_breaker": float(best["boundary_rank_promotion_min_adx_circuit_breaker"]) if pd.notna(best.get("boundary_rank_promotion_min_adx_circuit_breaker")) else None,
        "boundary_rank_promotion_compound_pullback_overlay": bool(best.get("boundary_rank_promotion_compound_pullback_overlay", False)) if pd.notna(best.get("boundary_rank_promotion_compound_pullback_overlay")) else False,
        "boundary_rank_promotion_compound_pullback_min_ret_100d_rank": float(best["boundary_rank_promotion_compound_pullback_min_ret_100d_rank"]) if pd.notna(best.get("boundary_rank_promotion_compound_pullback_min_ret_100d_rank")) else None,
        "boundary_rank_promotion_compound_pullback_min_252d_voladj_rank": float(best["boundary_rank_promotion_compound_pullback_min_252d_voladj_rank"]) if pd.notna(best.get("boundary_rank_promotion_compound_pullback_min_252d_voladj_rank")) else None,
        "boundary_rank_promotion_compound_pullback_max_ret_5d": float(best["boundary_rank_promotion_compound_pullback_max_ret_5d"]) if pd.notna(best.get("boundary_rank_promotion_compound_pullback_max_ret_5d")) else None,
        "boundary_rank_promotion_compound_pullback_max_ret_10d": float(best["boundary_rank_promotion_compound_pullback_max_ret_10d"]) if pd.notna(best.get("boundary_rank_promotion_compound_pullback_max_ret_10d")) else None,
        "boundary_rank_promotion_compound_pullback_max_volume_expansion": float(best["boundary_rank_promotion_compound_pullback_max_volume_expansion"]) if pd.notna(best.get("boundary_rank_promotion_compound_pullback_max_volume_expansion")) else None,
        "long_pullback_entry_overlay": bool(best.get("long_pullback_entry_overlay", False)),
        "pullback_min_relative_strength_score": float(best["pullback_min_relative_strength_score"]) if pd.notna(best.get("pullback_min_relative_strength_score")) else None,
        "pullback_min_theme_score": float(best["pullback_min_theme_score"]) if pd.notna(best.get("pullback_min_theme_score")) else None,
        "pullback_min_drawdown_from_high": float(best["pullback_min_drawdown_from_high"]) if pd.notna(best.get("pullback_min_drawdown_from_high")) else None,
        "pullback_max_drawdown_from_high": float(best["pullback_max_drawdown_from_high"]) if pd.notna(best.get("pullback_max_drawdown_from_high")) else None,
        "pullback_max_above_ma20_pct": float(best["pullback_max_above_ma20_pct"]) if pd.notna(best.get("pullback_max_above_ma20_pct")) else None,
        "pullback_max_score_boost": float(best["pullback_max_score_boost"]) if pd.notna(best.get("pullback_max_score_boost")) else None,
        "long_pullback_volume_contraction_overlay": bool(best.get("long_pullback_volume_contraction_overlay", False)),
        "pullback_volume_contraction_max_volume_expansion": float(best["pullback_volume_contraction_max_volume_expansion"]) if pd.notna(best.get("pullback_volume_contraction_max_volume_expansion")) else None,
        "pullback_volume_contraction_min_drawdown_from_high": float(best["pullback_volume_contraction_min_drawdown_from_high"]) if pd.notna(best.get("pullback_volume_contraction_min_drawdown_from_high")) else None,
        "pullback_volume_contraction_max_drawdown_from_high": float(best["pullback_volume_contraction_max_drawdown_from_high"]) if pd.notna(best.get("pullback_volume_contraction_max_drawdown_from_high")) else None,
        "pullback_volume_contraction_max_above_ma20_pct": float(best["pullback_volume_contraction_max_above_ma20_pct"]) if pd.notna(best.get("pullback_volume_contraction_max_above_ma20_pct")) else None,
        "pullback_volume_contraction_min_relative_strength_score": float(best["pullback_volume_contraction_min_relative_strength_score"]) if pd.notna(best.get("pullback_volume_contraction_min_relative_strength_score")) else None,
        "pullback_volume_contraction_min_theme_score": float(best["pullback_volume_contraction_min_theme_score"]) if pd.notna(best.get("pullback_volume_contraction_min_theme_score")) else None,
        "pullback_volume_contraction_require_benchmark_risk_on": bool(best.get("pullback_volume_contraction_require_benchmark_risk_on", False)),
        "pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker": float(best["pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("pullback_volume_contraction_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker": float(best["pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("pullback_volume_contraction_min_benchmark_ret63d_circuit_breaker")) else None,
        "pullback_volume_contraction_min_adx_circuit_breaker": float(best["pullback_volume_contraction_min_adx_circuit_breaker"]) if pd.notna(best.get("pullback_volume_contraction_min_adx_circuit_breaker")) else None,
        "pullback_volume_contraction_max_score_boost": float(best["pullback_volume_contraction_max_score_boost"]) if pd.notna(best.get("pullback_volume_contraction_max_score_boost")) else None,
        "pullback_short_term_reset_overlay": bool(best.get("pullback_short_term_reset_overlay", False)) if pd.notna(best.get("pullback_short_term_reset_overlay")) else False,
        "pullback_short_term_reset_min_drawdown_from_high": float(best["pullback_short_term_reset_min_drawdown_from_high"]) if pd.notna(best.get("pullback_short_term_reset_min_drawdown_from_high")) else None,
        "pullback_short_term_reset_max_drawdown_from_high": float(best["pullback_short_term_reset_max_drawdown_from_high"]) if pd.notna(best.get("pullback_short_term_reset_max_drawdown_from_high")) else None,
        "pullback_short_term_reset_max_above_ma20_pct": float(best["pullback_short_term_reset_max_above_ma20_pct"]) if pd.notna(best.get("pullback_short_term_reset_max_above_ma20_pct")) else None,
        "pullback_short_term_reset_max_ret_5d": float(best["pullback_short_term_reset_max_ret_5d"]) if pd.notna(best.get("pullback_short_term_reset_max_ret_5d")) else None,
        "pullback_short_term_reset_max_ret_10d": float(best["pullback_short_term_reset_max_ret_10d"]) if pd.notna(best.get("pullback_short_term_reset_max_ret_10d")) else None,
        "pullback_short_term_reset_min_relative_strength_score": float(best["pullback_short_term_reset_min_relative_strength_score"]) if pd.notna(best.get("pullback_short_term_reset_min_relative_strength_score")) else None,
        "pullback_short_term_reset_min_theme_score": float(best["pullback_short_term_reset_min_theme_score"]) if pd.notna(best.get("pullback_short_term_reset_min_theme_score")) else None,
        "pullback_short_term_reset_require_benchmark_risk_on": bool(best.get("pullback_short_term_reset_require_benchmark_risk_on", False)) if pd.notna(best.get("pullback_short_term_reset_require_benchmark_risk_on")) else False,
        "pullback_short_term_reset_require_theme_active": bool(best.get("pullback_short_term_reset_require_theme_active", False)) if pd.notna(best.get("pullback_short_term_reset_require_theme_active")) else False,
        "pullback_short_term_reset_max_event_risk_score_circuit_breaker": float(best["pullback_short_term_reset_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("pullback_short_term_reset_max_event_risk_score_circuit_breaker")) else None,
        "pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker": float(best["pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("pullback_short_term_reset_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker": float(best["pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("pullback_short_term_reset_min_benchmark_ret63d_circuit_breaker")) else None,
        "pullback_short_term_reset_min_adx_circuit_breaker": float(best["pullback_short_term_reset_min_adx_circuit_breaker"]) if pd.notna(best.get("pullback_short_term_reset_min_adx_circuit_breaker")) else None,
        "pullback_short_term_reset_max_score_boost": float(best["pullback_short_term_reset_max_score_boost"]) if pd.notna(best.get("pullback_short_term_reset_max_score_boost")) else None,
        "short_term_volume_tilt_overlay": bool(best.get("short_term_volume_tilt_overlay", False)) if pd.notna(best.get("short_term_volume_tilt_overlay")) else False,
        "short_term_volume_tilt_max_ret_5d": float(best["short_term_volume_tilt_max_ret_5d"]) if pd.notna(best.get("short_term_volume_tilt_max_ret_5d")) else None,
        "short_term_volume_tilt_min_volume_expansion": float(best["short_term_volume_tilt_min_volume_expansion"]) if pd.notna(best.get("short_term_volume_tilt_min_volume_expansion")) else None,
        "short_term_volume_tilt_min_relative_strength_score": float(best["short_term_volume_tilt_min_relative_strength_score"]) if pd.notna(best.get("short_term_volume_tilt_min_relative_strength_score")) else None,
        "short_term_volume_tilt_min_theme_score": float(best["short_term_volume_tilt_min_theme_score"]) if pd.notna(best.get("short_term_volume_tilt_min_theme_score")) else None,
        "short_term_volume_tilt_max_above_ma20_pct": float(best["short_term_volume_tilt_max_above_ma20_pct"]) if pd.notna(best.get("short_term_volume_tilt_max_above_ma20_pct")) else None,
        "short_term_volume_tilt_require_benchmark_risk_on": bool(best.get("short_term_volume_tilt_require_benchmark_risk_on", False)) if pd.notna(best.get("short_term_volume_tilt_require_benchmark_risk_on")) else False,
        "short_term_volume_tilt_require_theme_active": bool(best.get("short_term_volume_tilt_require_theme_active", False)) if pd.notna(best.get("short_term_volume_tilt_require_theme_active")) else False,
        "short_term_volume_tilt_require_price_above_ma50": bool(best.get("short_term_volume_tilt_require_price_above_ma50", False)) if pd.notna(best.get("short_term_volume_tilt_require_price_above_ma50")) else False,
        "short_term_volume_tilt_max_event_risk_score_circuit_breaker": float(best["short_term_volume_tilt_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("short_term_volume_tilt_max_event_risk_score_circuit_breaker")) else None,
        "short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker": float(best["short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("short_term_volume_tilt_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker": float(best["short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("short_term_volume_tilt_min_benchmark_ret63d_circuit_breaker")) else None,
        "short_term_volume_tilt_min_adx_circuit_breaker": float(best["short_term_volume_tilt_min_adx_circuit_breaker"]) if pd.notna(best.get("short_term_volume_tilt_min_adx_circuit_breaker")) else None,
        "short_term_volume_tilt_max_score_boost": float(best["short_term_volume_tilt_max_score_boost"]) if pd.notna(best.get("short_term_volume_tilt_max_score_boost")) else None,
        "short_term_volume_tilt_same_theme_peer_confirmation_overlay": bool(best.get("short_term_volume_tilt_same_theme_peer_confirmation_overlay", False)) if pd.notna(best.get("short_term_volume_tilt_same_theme_peer_confirmation_overlay")) else False,
        "short_term_volume_tilt_same_theme_peer_min_count": int(best["short_term_volume_tilt_same_theme_peer_min_count"]) if pd.notna(best.get("short_term_volume_tilt_same_theme_peer_min_count")) else None,
        "short_term_volume_tilt_same_theme_peer_min_share": float(best["short_term_volume_tilt_same_theme_peer_min_share"]) if pd.notna(best.get("short_term_volume_tilt_same_theme_peer_min_share")) else None,
        "short_term_volume_tilt_same_theme_peer_min_avg_mom_return": float(best["short_term_volume_tilt_same_theme_peer_min_avg_mom_return"]) if pd.notna(best.get("short_term_volume_tilt_same_theme_peer_min_avg_mom_return")) else None,
        "short_term_volume_tilt_same_theme_peer_volume_substitution_overlay": bool(best.get("short_term_volume_tilt_same_theme_peer_volume_substitution_overlay", False)) if pd.notna(best.get("short_term_volume_tilt_same_theme_peer_volume_substitution_overlay")) else False,
        "short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion": float(best["short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion"]) if pd.notna(best.get("short_term_volume_tilt_same_theme_peer_substitute_min_volume_expansion")) else None,
        "short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall": float(best["short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall"]) if pd.notna(best.get("short_term_volume_tilt_same_theme_peer_substitute_max_volume_shortfall")) else None,
        "short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale": float(best["short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale"]) if pd.notna(best.get("short_term_volume_tilt_same_theme_peer_substitute_score_boost_scale")) else None,
        "short_term_volume_tilt_non_risk_on_exception_overlay": bool(best.get("short_term_volume_tilt_non_risk_on_exception_overlay", False)) if pd.notna(best.get("short_term_volume_tilt_non_risk_on_exception_overlay")) else False,
        "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count": int(best["short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count"]) if pd.notna(best.get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_count")) else None,
        "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share": float(best["short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share"]) if pd.notna(best.get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_share")) else None,
        "short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return": float(best["short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return"]) if pd.notna(best.get("short_term_volume_tilt_non_risk_on_exception_min_same_theme_peer_avg_mom_return")) else None,
        "short_term_volume_tilt_non_risk_on_exception_score_boost_scale": float(best["short_term_volume_tilt_non_risk_on_exception_score_boost_scale"]) if pd.notna(best.get("short_term_volume_tilt_non_risk_on_exception_score_boost_scale")) else None,
        "boundary_rs_theme_credit_overlay": bool(best.get("boundary_rs_theme_credit_overlay", False)) if pd.notna(best.get("boundary_rs_theme_credit_overlay")) else False,
        "boundary_rs_theme_credit_rank_buffer_below": float(best["boundary_rs_theme_credit_rank_buffer_below"]) if pd.notna(best.get("boundary_rs_theme_credit_rank_buffer_below")) else None,
        "boundary_rs_theme_credit_rank_buffer_above": float(best["boundary_rs_theme_credit_rank_buffer_above"]) if pd.notna(best.get("boundary_rs_theme_credit_rank_buffer_above")) else None,
        "boundary_rs_theme_credit_min_relative_strength_score": float(best["boundary_rs_theme_credit_min_relative_strength_score"]) if pd.notna(best.get("boundary_rs_theme_credit_min_relative_strength_score")) else None,
        "boundary_rs_theme_credit_min_theme_score": float(best["boundary_rs_theme_credit_min_theme_score"]) if pd.notna(best.get("boundary_rs_theme_credit_min_theme_score")) else None,
        "boundary_rs_theme_credit_min_technical_score": float(best["boundary_rs_theme_credit_min_technical_score"]) if pd.notna(best.get("boundary_rs_theme_credit_min_technical_score")) else None,
        "boundary_rs_theme_credit_min_mom_return": float(best["boundary_rs_theme_credit_min_mom_return"]) if pd.notna(best.get("boundary_rs_theme_credit_min_mom_return")) else None,
        "boundary_rs_theme_credit_min_drawdown_from_high": float(best["boundary_rs_theme_credit_min_drawdown_from_high"]) if pd.notna(best.get("boundary_rs_theme_credit_min_drawdown_from_high")) else None,
        "boundary_rs_theme_credit_max_drawdown_from_high": float(best["boundary_rs_theme_credit_max_drawdown_from_high"]) if pd.notna(best.get("boundary_rs_theme_credit_max_drawdown_from_high")) else None,
        "boundary_rs_theme_credit_max_above_ma20_pct": float(best["boundary_rs_theme_credit_max_above_ma20_pct"]) if pd.notna(best.get("boundary_rs_theme_credit_max_above_ma20_pct")) else None,
        "boundary_rs_theme_credit_require_benchmark_risk_on": bool(best.get("boundary_rs_theme_credit_require_benchmark_risk_on", False)) if pd.notna(best.get("boundary_rs_theme_credit_require_benchmark_risk_on")) else False,
        "boundary_rs_theme_credit_require_theme_active": bool(best.get("boundary_rs_theme_credit_require_theme_active", False)) if pd.notna(best.get("boundary_rs_theme_credit_require_theme_active")) else False,
        "boundary_rs_theme_credit_require_price_above_ma50": bool(best.get("boundary_rs_theme_credit_require_price_above_ma50", False)) if pd.notna(best.get("boundary_rs_theme_credit_require_price_above_ma50")) else False,
        "boundary_rs_theme_credit_max_event_risk_score_circuit_breaker": float(best["boundary_rs_theme_credit_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("boundary_rs_theme_credit_max_event_risk_score_circuit_breaker")) else None,
        "boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker": float(best["boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("boundary_rs_theme_credit_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker": float(best["boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("boundary_rs_theme_credit_min_benchmark_ret63d_circuit_breaker")) else None,
        "boundary_rs_theme_credit_min_adx_circuit_breaker": float(best["boundary_rs_theme_credit_min_adx_circuit_breaker"]) if pd.notna(best.get("boundary_rs_theme_credit_min_adx_circuit_breaker")) else None,
        "boundary_rs_theme_credit_max_score_boost": float(best["boundary_rs_theme_credit_max_score_boost"]) if pd.notna(best.get("boundary_rs_theme_credit_max_score_boost")) else None,
        "exit_quality_rank_credit_overlay": bool(best.get("exit_quality_rank_credit_overlay", False)) if pd.notna(best.get("exit_quality_rank_credit_overlay")) else False,
        "exit_quality_rank_credit_rank_buffer_below": float(best["exit_quality_rank_credit_rank_buffer_below"]) if pd.notna(best.get("exit_quality_rank_credit_rank_buffer_below")) else None,
        "exit_quality_rank_credit_rank_buffer_above": float(best["exit_quality_rank_credit_rank_buffer_above"]) if pd.notna(best.get("exit_quality_rank_credit_rank_buffer_above")) else None,
        "exit_quality_rank_credit_min_final_score": float(best["exit_quality_rank_credit_min_final_score"]) if pd.notna(best.get("exit_quality_rank_credit_min_final_score")) else None,
        "exit_quality_rank_credit_min_relative_strength_score": float(best["exit_quality_rank_credit_min_relative_strength_score"]) if pd.notna(best.get("exit_quality_rank_credit_min_relative_strength_score")) else None,
        "exit_quality_rank_credit_min_theme_score": float(best["exit_quality_rank_credit_min_theme_score"]) if pd.notna(best.get("exit_quality_rank_credit_min_theme_score")) else None,
        "exit_quality_rank_credit_min_technical_score": float(best["exit_quality_rank_credit_min_technical_score"]) if pd.notna(best.get("exit_quality_rank_credit_min_technical_score")) else None,
        "exit_quality_rank_credit_min_mom_return": float(best["exit_quality_rank_credit_min_mom_return"]) if pd.notna(best.get("exit_quality_rank_credit_min_mom_return")) else None,
        "exit_quality_rank_credit_max_drawdown_from_high": float(best["exit_quality_rank_credit_max_drawdown_from_high"]) if pd.notna(best.get("exit_quality_rank_credit_max_drawdown_from_high")) else None,
        "exit_quality_rank_credit_max_above_ma20_pct": float(best["exit_quality_rank_credit_max_above_ma20_pct"]) if pd.notna(best.get("exit_quality_rank_credit_max_above_ma20_pct")) else None,
        "exit_quality_rank_credit_require_benchmark_risk_on": bool(best.get("exit_quality_rank_credit_require_benchmark_risk_on", False)) if pd.notna(best.get("exit_quality_rank_credit_require_benchmark_risk_on")) else False,
        "exit_quality_rank_credit_require_theme_active": bool(best.get("exit_quality_rank_credit_require_theme_active", False)) if pd.notna(best.get("exit_quality_rank_credit_require_theme_active")) else False,
        "exit_quality_rank_credit_require_price_above_ma50": bool(best.get("exit_quality_rank_credit_require_price_above_ma50", False)) if pd.notna(best.get("exit_quality_rank_credit_require_price_above_ma50")) else False,
        "exit_quality_rank_credit_max_event_risk_score_circuit_breaker": float(best["exit_quality_rank_credit_max_event_risk_score_circuit_breaker"]) if pd.notna(best.get("exit_quality_rank_credit_max_event_risk_score_circuit_breaker")) else None,
        "exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker": float(best["exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("exit_quality_rank_credit_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker": float(best["exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("exit_quality_rank_credit_min_benchmark_ret63d_circuit_breaker")) else None,
        "exit_quality_rank_credit_min_adx_circuit_breaker": float(best["exit_quality_rank_credit_min_adx_circuit_breaker"]) if pd.notna(best.get("exit_quality_rank_credit_min_adx_circuit_breaker")) else None,
        "exit_quality_rank_credit_max_score_rank_credit": float(best["exit_quality_rank_credit_max_score_rank_credit"]) if pd.notna(best.get("exit_quality_rank_credit_max_score_rank_credit")) else None,
        "exit_quality_rank_credit_theme_support_overlay": bool(best.get("exit_quality_rank_credit_theme_support_overlay", False)) if pd.notna(best.get("exit_quality_rank_credit_theme_support_overlay")) else False,
        "exit_quality_rank_credit_theme_support_lookback_days": int(best["exit_quality_rank_credit_theme_support_lookback_days"]) if pd.notna(best.get("exit_quality_rank_credit_theme_support_lookback_days")) else None,
        "exit_quality_rank_credit_theme_support_min_peer_count": int(best["exit_quality_rank_credit_theme_support_min_peer_count"]) if pd.notna(best.get("exit_quality_rank_credit_theme_support_min_peer_count")) else None,
        "exit_quality_rank_credit_theme_support_min_theme_score_prior": float(best["exit_quality_rank_credit_theme_support_min_theme_score_prior"]) if pd.notna(best.get("exit_quality_rank_credit_theme_support_min_theme_score_prior")) else None,
        "exit_quality_rank_credit_theme_support_min_peer_rs_prior": float(best["exit_quality_rank_credit_theme_support_min_peer_rs_prior"]) if pd.notna(best.get("exit_quality_rank_credit_theme_support_min_peer_rs_prior")) else None,
        "exit_quality_rank_credit_theme_support_min_combined_delta": float(best["exit_quality_rank_credit_theme_support_min_combined_delta"]) if pd.notna(best.get("exit_quality_rank_credit_theme_support_min_combined_delta")) else None,
        "exit_quality_rank_credit_reset_support_overlay": bool(best.get("exit_quality_rank_credit_reset_support_overlay", False)) if pd.notna(best.get("exit_quality_rank_credit_reset_support_overlay")) else False,
        "exit_quality_rank_credit_reset_support_min_reset_score": float(best["exit_quality_rank_credit_reset_support_min_reset_score"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_min_reset_score")) else None,
        "exit_quality_rank_credit_reset_support_theme_breadth_lookback_days": int(best["exit_quality_rank_credit_reset_support_theme_breadth_lookback_days"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_theme_breadth_lookback_days")) else None,
        "exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold": float(best["exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_theme_breadth_active_share_threshold")) else None,
        "exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold": float(best["exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_theme_breadth_shortfall_threshold")) else None,
        "exit_quality_rank_credit_reset_support_theme_breadth_min_active_share": float(best["exit_quality_rank_credit_reset_support_theme_breadth_min_active_share"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_theme_breadth_min_active_share")) else None,
        "exit_quality_rank_credit_reset_support_require_theme_breadth_expansion": bool(best.get("exit_quality_rank_credit_reset_support_require_theme_breadth_expansion", False)) if pd.notna(best.get("exit_quality_rank_credit_reset_support_require_theme_breadth_expansion")) else False,
        "exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold": float(best["exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_theme_breadth_expansion_threshold")) else None,
        "exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay": bool(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay", False)) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_breadth_overlay")) else False,
        "exit_quality_rank_credit_reset_support_same_theme_peer_min_count": int(best["exit_quality_rank_credit_reset_support_same_theme_peer_min_count"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_min_count")) else None,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay": bool(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay", False)) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_breadth_overlay")) else False,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share": float(best["exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_share")) else None,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count": int(best["exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_count")) else None,
        "exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay": bool(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay", False)) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_quality_overlay")) else False,
        "exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs": float(best["exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_peer_reset_min_avg_rs")) else None,
        "exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay": bool(best.get("exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay", False)) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_leader_guard_overlay")) else False,
        "exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct": float(best["exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_leader_min_rank_pct")) else None,
        "exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap": float(best["exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_leader_max_final_score_gap")) else None,
        "exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count": int(best["exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count"]) if pd.notna(best.get("exit_quality_rank_credit_reset_support_same_theme_leader_min_peer_count")) else None,
        "long_pullback_reclaim_overlay": bool(best.get("long_pullback_reclaim_overlay", False)) if pd.notna(best.get("long_pullback_reclaim_overlay")) else False,
        "pullback_reclaim_min_relative_strength_score": float(best["pullback_reclaim_min_relative_strength_score"]) if pd.notna(best.get("pullback_reclaim_min_relative_strength_score")) else None,
        "pullback_reclaim_min_theme_score": float(best["pullback_reclaim_min_theme_score"]) if pd.notna(best.get("pullback_reclaim_min_theme_score")) else None,
        "pullback_reclaim_min_drawdown_from_high": float(best["pullback_reclaim_min_drawdown_from_high"]) if pd.notna(best.get("pullback_reclaim_min_drawdown_from_high")) else None,
        "pullback_reclaim_max_drawdown_from_high": float(best["pullback_reclaim_max_drawdown_from_high"]) if pd.notna(best.get("pullback_reclaim_max_drawdown_from_high")) else None,
        "pullback_reclaim_recent_below_ma20_lookback_days": int(best["pullback_reclaim_recent_below_ma20_lookback_days"]) if pd.notna(best.get("pullback_reclaim_recent_below_ma20_lookback_days")) else None,
        "pullback_reclaim_min_recent_below_ma20_pct": float(best["pullback_reclaim_min_recent_below_ma20_pct"]) if pd.notna(best.get("pullback_reclaim_min_recent_below_ma20_pct")) else None,
        "pullback_reclaim_max_above_ma20_pct": float(best["pullback_reclaim_max_above_ma20_pct"]) if pd.notna(best.get("pullback_reclaim_max_above_ma20_pct")) else None,
        "pullback_reclaim_min_volume_expansion": float(best["pullback_reclaim_min_volume_expansion"]) if pd.notna(best.get("pullback_reclaim_min_volume_expansion")) else None,
        "pullback_reclaim_require_benchmark_risk_on": bool(best.get("pullback_reclaim_require_benchmark_risk_on", False)) if pd.notna(best.get("pullback_reclaim_require_benchmark_risk_on")) else False,
        "pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker": float(best["pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker"]) if pd.notna(best.get("pullback_reclaim_max_overnight_gap_risk_score_circuit_breaker")) else None,
        "pullback_reclaim_min_benchmark_ret63d_circuit_breaker": float(best["pullback_reclaim_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("pullback_reclaim_min_benchmark_ret63d_circuit_breaker")) else None,
        "pullback_reclaim_min_adx_circuit_breaker": float(best["pullback_reclaim_min_adx_circuit_breaker"]) if pd.notna(best.get("pullback_reclaim_min_adx_circuit_breaker")) else None,
        "pullback_reclaim_max_score_boost": float(best["pullback_reclaim_max_score_boost"]) if pd.notna(best.get("pullback_reclaim_max_score_boost")) else None,
        "long_pullback_reset_inclusion_overlay": bool(best.get("long_pullback_reset_inclusion_overlay", False)),
        "pullback_reset_inclusion_min_score": float(best["pullback_reset_inclusion_min_score"]) if pd.notna(best.get("pullback_reset_inclusion_min_score")) else None,
        "pullback_reset_inclusion_max_names_per_date": int(best["pullback_reset_inclusion_max_names_per_date"]) if pd.notna(best.get("pullback_reset_inclusion_max_names_per_date")) else None,
        "pullback_reset_inclusion_min_final_score": float(best["pullback_reset_inclusion_min_final_score"]) if pd.notna(best.get("pullback_reset_inclusion_min_final_score")) else None,
        "pullback_reset_inclusion_min_score_rank": float(best["pullback_reset_inclusion_min_score_rank"]) if pd.notna(best.get("pullback_reset_inclusion_min_score_rank")) else None,
        "pullback_reset_inclusion_activation_support_overlay": bool(best.get("pullback_reset_inclusion_activation_support_overlay", False)) if pd.notna(best.get("pullback_reset_inclusion_activation_support_overlay")) else False,
        "pullback_reset_inclusion_activation_min_score": float(best["pullback_reset_inclusion_activation_min_score"]) if pd.notna(best.get("pullback_reset_inclusion_activation_min_score")) else None,
        "pullback_reset_inclusion_activation_score_rank_credit": float(best["pullback_reset_inclusion_activation_score_rank_credit"]) if pd.notna(best.get("pullback_reset_inclusion_activation_score_rank_credit")) else None,
        "pullback_reset_inclusion_reclaim_support_overlay": bool(best.get("pullback_reset_inclusion_reclaim_support_overlay", False)) if pd.notna(best.get("pullback_reset_inclusion_reclaim_support_overlay")) else False,
        "pullback_reset_inclusion_min_reclaim_score": float(best["pullback_reset_inclusion_min_reclaim_score"]) if pd.notna(best.get("pullback_reset_inclusion_min_reclaim_score")) else None,
        "pullback_reset_inclusion_reclaim_score_rank_credit": float(best["pullback_reset_inclusion_reclaim_score_rank_credit"]) if pd.notna(best.get("pullback_reset_inclusion_reclaim_score_rank_credit")) else None,
        "pullback_reset_inclusion_require_theme_breadth_not_deteriorating": bool(best.get("pullback_reset_inclusion_require_theme_breadth_not_deteriorating", False)) if pd.notna(best.get("pullback_reset_inclusion_require_theme_breadth_not_deteriorating")) else False,
        "pullback_reset_inclusion_theme_breadth_active_share_threshold": float(best["pullback_reset_inclusion_theme_breadth_active_share_threshold"]) if pd.notna(best.get("pullback_reset_inclusion_theme_breadth_active_share_threshold")) else None,
        "pullback_reset_inclusion_theme_breadth_shortfall_threshold": float(best["pullback_reset_inclusion_theme_breadth_shortfall_threshold"]) if pd.notna(best.get("pullback_reset_inclusion_theme_breadth_shortfall_threshold")) else None,
        "pullback_reset_inclusion_require_theme_breadth_expansion": bool(best.get("pullback_reset_inclusion_require_theme_breadth_expansion", False)),
        "pullback_reset_inclusion_theme_breadth_lookback_days": int(best["pullback_reset_inclusion_theme_breadth_lookback_days"]) if pd.notna(best.get("pullback_reset_inclusion_theme_breadth_lookback_days")) else None,
        "pullback_reset_inclusion_theme_breadth_min_active_share": float(best["pullback_reset_inclusion_theme_breadth_min_active_share"]) if pd.notna(best.get("pullback_reset_inclusion_theme_breadth_min_active_share")) else None,
        "pullback_reset_inclusion_theme_breadth_expansion_threshold": float(best["pullback_reset_inclusion_theme_breadth_expansion_threshold"]) if pd.notna(best.get("pullback_reset_inclusion_theme_breadth_expansion_threshold")) else None,
        "pullback_reset_inclusion_require_rs_acceleration": bool(best.get("pullback_reset_inclusion_require_rs_acceleration", False)),
        "pullback_reset_inclusion_rs_acceleration_lookback_days": int(best["pullback_reset_inclusion_rs_acceleration_lookback_days"]) if pd.notna(best.get("pullback_reset_inclusion_rs_acceleration_lookback_days")) else None,
        "pullback_reset_inclusion_min_current_rs_score": float(best["pullback_reset_inclusion_min_current_rs_score"]) if pd.notna(best.get("pullback_reset_inclusion_min_current_rs_score")) else None,
        "pullback_reset_inclusion_min_rs_acceleration": float(best["pullback_reset_inclusion_min_rs_acceleration"]) if pd.notna(best.get("pullback_reset_inclusion_min_rs_acceleration")) else None,
        "long_reentry_discipline_overlay": bool(best.get("long_reentry_discipline_overlay", False)),
        "reentry_discipline_near_high_threshold": float(best["reentry_discipline_near_high_threshold"]) if pd.notna(best.get("reentry_discipline_near_high_threshold")) else None,
        "reentry_discipline_min_above_ma20_pct": float(best["reentry_discipline_min_above_ma20_pct"]) if pd.notna(best.get("reentry_discipline_min_above_ma20_pct")) else None,
        "reentry_discipline_max_volume_expansion": float(best["reentry_discipline_max_volume_expansion"]) if pd.notna(best.get("reentry_discipline_max_volume_expansion")) else None,
        "reentry_discipline_min_adx_circuit_breaker": float(best["reentry_discipline_min_adx_circuit_breaker"]) if pd.notna(best.get("reentry_discipline_min_adx_circuit_breaker")) else None,
        "reentry_discipline_max_score_penalty": float(best["reentry_discipline_max_score_penalty"]) if pd.notna(best.get("reentry_discipline_max_score_penalty")) else None,
        "reentry_discipline_min_pullback_volume_reset_score_exemption": float(best["reentry_discipline_min_pullback_volume_reset_score_exemption"]) if pd.notna(best.get("reentry_discipline_min_pullback_volume_reset_score_exemption")) else None,
        "reentry_discipline_min_relative_strength_score_exemption": float(best["reentry_discipline_min_relative_strength_score_exemption"]) if pd.notna(best.get("reentry_discipline_min_relative_strength_score_exemption")) else None,
        "reentry_discipline_min_theme_score_exemption": float(best["reentry_discipline_min_theme_score_exemption"]) if pd.notna(best.get("reentry_discipline_min_theme_score_exemption")) else None,
        "reentry_discipline_require_theme_deterioration": bool(best.get("reentry_discipline_require_theme_deterioration", False)),
        "reentry_discipline_theme_score_deterioration_threshold": float(best["reentry_discipline_theme_score_deterioration_threshold"]) if pd.notna(best.get("reentry_discipline_theme_score_deterioration_threshold")) else None,
        "reentry_discipline_penalize_theme_inactive": bool(best.get("reentry_discipline_penalize_theme_inactive", True)),
        "reentry_discipline_require_theme_breadth_deterioration": bool(best.get("reentry_discipline_require_theme_breadth_deterioration", False)),
        "reentry_discipline_theme_breadth_lookback_days": int(best["reentry_discipline_theme_breadth_lookback_days"]) if pd.notna(best.get("reentry_discipline_theme_breadth_lookback_days")) else None,
        "reentry_discipline_theme_breadth_active_share_threshold": float(best["reentry_discipline_theme_breadth_active_share_threshold"]) if pd.notna(best.get("reentry_discipline_theme_breadth_active_share_threshold")) else None,
        "reentry_discipline_theme_breadth_shortfall_threshold": float(best["reentry_discipline_theme_breadth_shortfall_threshold"]) if pd.notna(best.get("reentry_discipline_theme_breadth_shortfall_threshold")) else None,
        "short_rebound_avoidance_overlay": bool(best.get("short_rebound_avoidance_overlay", False)),
        "short_rebound_avoidance_min_below_ma20_pct": float(best["short_rebound_avoidance_min_below_ma20_pct"]) if pd.notna(best.get("short_rebound_avoidance_min_below_ma20_pct")) else None,
        "short_rebound_avoidance_min_distance_from_high": float(best["short_rebound_avoidance_min_distance_from_high"]) if pd.notna(best.get("short_rebound_avoidance_min_distance_from_high")) else None,
        "short_rebound_avoidance_min_volume_expansion": float(best["short_rebound_avoidance_min_volume_expansion"]) if pd.notna(best.get("short_rebound_avoidance_min_volume_expansion")) else None,
        "short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker": float(best["short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker"]) if pd.notna(best.get("short_rebound_avoidance_min_benchmark_ret63d_circuit_breaker")) else None,
        "short_rebound_avoidance_min_adx_circuit_breaker": float(best["short_rebound_avoidance_min_adx_circuit_breaker"]) if pd.notna(best.get("short_rebound_avoidance_min_adx_circuit_breaker")) else None,
        "long_min_final_score": float(best["long_min_final_score"]) if pd.notna(best.get("long_min_final_score")) else None,
        "long_min_technical_score": float(best["long_min_technical_score"]) if pd.notna(best.get("long_min_technical_score")) else None,
        "long_min_relative_strength_score": float(best["long_min_relative_strength_score"]) if pd.notna(best.get("long_min_relative_strength_score")) else None,
        "long_min_theme_score": float(best["long_min_theme_score"]) if pd.notna(best.get("long_min_theme_score")) else None,
        "require_long_theme_active": bool(best.get("require_long_theme_active", False)),
        "long_require_price_above_ma50": bool(best.get("long_require_price_above_ma50", False)),
        "long_require_price_above_ma200": bool(best.get("long_require_price_above_ma200", False)),
        "short_min_weakness_conditions": int(best["short_min_weakness_conditions"]) if pd.notna(best.get("short_min_weakness_conditions")) else 0,
        "short_only_when_benchmark_risk_off": bool(best.get("short_only_when_benchmark_risk_off", False)),
        "short_max_final_score": float(best["short_max_final_score"]) if pd.notna(best.get("short_max_final_score")) else None,
    }
    for key in _SAME_THEME_SUBSTITUTION_MOMENTUM_KEYS:
        value = _coerce_same_theme_substitution_value(best, key)
        if value is not None:
            momentum_updates[key] = value
    strategies["momentum"].update({key: value for key, value in momentum_updates.items() if value is not None})
    strategies.setdefault("breakout", {})
    strategies["breakout"].update(
        {
            "enabled": True,
            "weight": float(best["breakout_weight"]),
            "breakout_window": int(best["breakout_window"]),
            "max_holding_days": int(best["breakout_max_holding_days"]),
        }
    )
    strategies.setdefault("trend", {})
    strategies["trend"].update(
        {
            "enabled": True,
            "weight": float(best["trend_weight"]),
            "ma_fast": int(best["trend_ma_fast"]),
            "ma_slow": int(best["trend_ma_slow"]),
        }
    )
    return replace(config, execution=execution, portfolio=portfolio, risk=risk, events=events, regime=regime, strategies=strategies)


def _coerce_same_theme_substitution_value(row: pd.Series, key: str):
    """Recover same-theme substitution parameters from optimizer result rows."""

    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    if (
        key.endswith("_overlay")
        or key.endswith("_require_theme_active")
        or key.endswith("_require_activation_support")
        or key.endswith("_require_theme_breadth_not_deteriorating")
    ):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if key.endswith("_max_promotions_per_date"):
        return int(value)
    return float(value)


def cmd_report(run_id: str) -> None:
    """Print the report path for an existing run."""

    run_dir = Path("runs") / run_id
    if run_id == "latest":
        run_dir = Path("runs/latest")
    report = run_dir / "report.html"
    if not report.exists():
        raise FileNotFoundError(report)
    print(report)
if __name__ == "__main__":
    main()
