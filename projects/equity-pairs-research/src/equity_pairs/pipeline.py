"""End-to-end orchestration for the equity-pairs research package."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import (
    backtest_selected_pairs,
    build_portfolios,
    cost_sensitivity,
)
from .config import PipelineConfig
from .data import (
    FRED_OBSERVATIONS_URL,
    SP500_WIKIPEDIA_URL,
    download_confirmed_close,
    download_fred_observations,
    fetch_sp500_constituents,
)
from .macro import derive_macro_variables, regime_attribution
from .reporting import (
    plot_cointegration_heatmap,
    plot_macro_history,
    plot_pair_dashboard,
    plot_portfolio_curves,
    plot_sector_performance,
    render_html_report,
)
from .statistics import (
    PairScreenSettings,
    descriptive_top_pairs,
    screen_sector_pairs,
    select_top_pairs,
)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    raw: Path
    processed: Path
    charts: Path

    @classmethod
    def create(cls, root: str | Path) -> "RunPaths":
        root_path = Path(root).resolve()
        paths = cls(
            root=root_path,
            raw=root_path / "data" / "raw",
            processed=root_path / "data" / "processed",
            charts=root_path / "charts",
        )
        for directory in (paths.root, paths.raw, paths.processed, paths.charts):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    names = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "quant-marketdata",
        "matplotlib",
        "requests",
        "jinja2",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _load_or_fetch_universe(paths: RunPaths, refresh_data: bool) -> pd.DataFrame:
    path = paths.raw / "sp500_current_constituents.csv"
    if path.exists() and not refresh_data:
        log("Loading cached current S&P 500 universe")
        return pd.read_csv(path, dtype={"cik": "string"})
    log("Downloading current S&P 500 constituent/GICS snapshot")
    universe = fetch_sp500_constituents()
    universe.to_csv(path, index=False)
    _write_json(
        paths.raw / "sp500_current_constituents.metadata.json",
        {
            "source": SP500_WIKIPEDIA_URL,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "universe_contract": "current_constituents_current_gics",
            "security_count": int(len(universe)),
            "issuer_count_by_cik": int(universe["cik"].nunique()),
            "sector_count": int(universe["sector"].nunique()),
        },
    )
    return universe


def _coverage(prices: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    formation = prices.loc[prices.index < test_start]
    test = prices.loc[prices.index >= test_start]
    rows: list[dict[str, Any]] = []
    for ticker in prices:
        series = prices[ticker]
        valid = series.dropna()
        rows.append(
            {
                "ticker": ticker,
                "first_observation": valid.index.min() if not valid.empty else pd.NaT,
                "last_observation": valid.index.max() if not valid.empty else pd.NaT,
                "observations": int(series.notna().sum()),
                "full_history_ratio": float(series.notna().mean()),
                "formation_observations": int(formation[ticker].notna().sum()),
                "formation_history_ratio": float(formation[ticker].notna().mean()) if len(formation) else 0.0,
                "test_observations": int(test[ticker].notna().sum()),
                "test_history_ratio": float(test[ticker].notna().mean()) if len(test) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _same_issuer_pair_count(frame: pd.DataFrame, universe: pd.DataFrame) -> int:
    if frame.empty or "cik" not in universe:
        return 0
    issuer = universe.drop_duplicates("ticker").set_index("ticker")["cik"].astype("string").to_dict()
    return int(
        sum(
            pd.notna(issuer.get(row.ticker_a))
            and issuer.get(row.ticker_a) == issuer.get(row.ticker_b)
            for row in frame.itertuples(index=False)
        )
    )


def _load_or_screen(
    name: str,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    settings: PairScreenSettings,
    paths: RunPaths,
    reuse_computations: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics_path = paths.processed / f"{name}_all_pair_diagnostics.csv.gz"
    integration_path = paths.processed / f"{name}_ticker_integration_diagnostics.csv"
    if reuse_computations and diagnostics_path.exists() and integration_path.exists():
        log(f"Loading cached {name.replace('_', ' ')} cointegration screen")
        return pd.read_csv(diagnostics_path), pd.read_csv(integration_path)
    log(
        f"Screening every different-issuer within-sector pair on {name.replace('_', ' ')} "
        f"({len(prices):,} sessions, {settings.workers} workers)"
    )
    started = time.monotonic()
    diagnostics, integration = screen_sector_pairs(prices, universe, settings)
    diagnostics.to_csv(diagnostics_path, index=False, compression="gzip")
    integration.to_csv(integration_path, index=False)
    log(f"Completed {name.replace('_', ' ')} screen: {len(diagnostics):,} pairs in {(time.monotonic() - started) / 60:.1f} minutes")
    return diagnostics, integration


def _rank_best(metrics: pd.DataFrame, minimum_trades: int, top_n: int = 10) -> pd.DataFrame:
    successful = metrics[
        metrics["backtest_status"].eq("ok")
        & metrics["net_cagr"].notna()
        & metrics["trade_count"].ge(minimum_trades)
    ].copy()
    successful = successful.sort_values(
        ["net_cagr", "sharpe", "pair"], ascending=[False, False, True], na_position="last"
    ).head(top_n)
    successful.insert(0, "rank", np.arange(1, len(successful) + 1))
    return successful.reset_index(drop=True)


def _quality_checks(
    *,
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    coverage: pd.DataFrame,
    full_diagnostics: pd.DataFrame,
    formation_selected: pd.DataFrame,
    metrics: pd.DataFrame,
    best: pd.DataFrame,
    fred: pd.DataFrame,
    expected_top_per_sector: int,
) -> dict[str, Any]:
    sector_counts = formation_selected.groupby("sector")["pair"].count()
    same_issuer = _same_issuer_pair_count(formation_selected, universe)
    checks: dict[str, Any] = {
        "universe_contract": "current_constituents_current_gics_survivorship_biased",
        "universe_security_count": int(len(universe)),
        "universe_issuer_count": int(universe["cik"].nunique()) if "cik" in universe else None,
        "sector_count": int(universe["sector"].nunique()),
        "price_rows": int(len(prices)),
        "tickers_with_any_price": int((coverage["observations"] > 0).sum()),
        "tickers_with_at_least_90pct_formation_history": int((coverage["formation_history_ratio"] >= 0.90).sum()),
        "full_window_pairs_evaluated": int(len(full_diagnostics)),
        "formation_pairs_selected": int(len(formation_selected)),
        "strict_formation_selections": int(formation_selected["eligible"].fillna(False).sum()),
        "fallback_formation_selections": int((~formation_selected["eligible"].fillna(False)).sum()),
        "formation_selection_counts_by_sector": {str(key): int(value) for key, value in sector_counts.items()},
        "same_issuer_selected_pairs": same_issuer,
        "successful_pair_backtests": int(metrics["backtest_status"].eq("ok").sum()),
        "failed_pair_backtests": int(metrics["backtest_status"].eq("failed").sum()),
        "best_pair_count": int(len(best)),
        "fred_series_count": int(len(fred.columns)),
        "fred_observation_rows": int(len(fred)),
        "credentials_serialized": False,
        "full_window_ranking_used_for_backtest_selection": False,
    }
    critical = [
        checks["sector_count"] == 11,
        len(sector_counts) == 11,
        bool((sector_counts == expected_top_per_sector).all()),
        same_issuer == 0,
        checks["successful_pair_backtests"] >= 100,
        checks["best_pair_count"] == 10,
        checks["fred_series_count"] >= 5,
    ]
    checks["pipeline_integrity_status"] = "passed" if all(critical) else "failed"
    evidence_warnings: list[str] = []
    if checks["fallback_formation_selections"]:
        evidence_warnings.append(
            "not every requested sector slot passed the predeclared FDR/I(1)/mean-reversion gate"
        )
    evidence_warnings.append(
        "current constituents/current GICS create survivorship and historical-classification bias"
    )
    evidence_warnings.append(
        "the winning ten are an ex-post test-period ranking, not a deployable selection rule"
    )
    checks["research_evidence_status"] = "warning" if evidence_warnings else "passed"
    checks["research_evidence_warnings"] = evidence_warnings
    checks["status"] = (
        "failed"
        if checks["pipeline_integrity_status"] == "failed"
        else checks["research_evidence_status"]
    )
    return checks


def _relative_chart(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def run_pipeline(
    *,
    as_of: str | pd.Timestamp,
    output_dir: str | Path,
    config: PipelineConfig,
    refresh_data: bool = False,
    reuse_computations: bool = True,
) -> Path:
    """Run the complete data, screen, held-out backtest, chart and report workflow."""
    as_of_ts = pd.Timestamp(as_of).normalize()
    start = as_of_ts - pd.DateOffset(years=config.research.years)
    end_exclusive = as_of_ts + pd.Timedelta(days=1)
    test_start = start + pd.DateOffset(years=config.research.formation_years)
    paths = RunPaths.create(output_dir)
    log(f"Research window {start.date()} through {as_of_ts.date()}; held-out test begins {test_start.date()}")

    universe = _load_or_fetch_universe(paths, refresh_data)
    tickers = universe["ticker"].astype(str).tolist()
    log(f"Downloading/loading confirmed MarketData closes for {len(tickers)} securities")
    prices = download_confirmed_close(
        tickers,
        start,
        end_exclusive,
        refresh=refresh_data,
    )
    prices = prices.sort_index()
    coverage = _coverage(prices, test_start)
    coverage.to_csv(paths.processed / "price_coverage.csv", index=False)
    failed_tickers = coverage.loc[coverage["observations"].eq(0), "ticker"].tolist()
    log(
        "Confirmed MarketData coverage: "
        f"{len(tickers) - len(failed_tickers)}/{len(tickers)} securities have data"
    )

    settings = PairScreenSettings(
        minimum_observations=config.research.minimum_formation_observations,
        minimum_history_ratio=config.research.minimum_history_ratio,
        alpha=config.research.cointegration_alpha,
        maximum_half_life_days=config.research.maximum_half_life_days,
        adf_maxlag=config.research.adf_maxlag,
        exclude_same_issuer=config.research.exclude_same_issuer,
        workers=config.research.workers,
    )
    formation_prices = prices.loc[prices.index < test_start]
    formation_diagnostics, _ = _load_or_screen(
        "formation_window",
        formation_prices,
        universe,
        settings,
        paths,
        reuse_computations,
    )
    formation_selected = select_top_pairs(
        formation_diagnostics, config.research.top_pairs_per_sector
    )
    formation_selected.to_csv(paths.processed / "formation_selected_top10_by_sector.csv", index=False)

    full_diagnostics, _ = _load_or_screen(
        "full_8y_window",
        prices,
        universe,
        settings,
        paths,
        reuse_computations,
    )
    full_top = descriptive_top_pairs(full_diagnostics, config.research.top_pairs_per_sector)
    full_top.to_csv(paths.processed / "full_8y_top10_cointegrated_by_sector.csv", index=False)

    log(f"Backtesting {len(formation_selected)} formation-selected pairs on the held-out window")
    metrics, signals, trades, pair_returns = backtest_selected_pairs(
        formation_selected,
        prices,
        test_start,
        config.strategy,
    )
    metrics.to_csv(paths.processed / "pair_backtest_metrics.csv", index=False)
    signals.to_csv(paths.processed / "pair_daily_signals.csv.gz", index=False, compression="gzip")
    trades.to_csv(paths.processed / "pair_trade_ledger.csv", index=False)
    pair_returns.to_csv(paths.processed / "pair_net_returns.csv")
    best = _rank_best(
        metrics,
        minimum_trades=config.strategy.minimum_trades_for_ranking,
        top_n=10,
    )
    selection_audit_columns = [
        "pair",
        "selection_status",
        "eligible",
        "pair_pvalue",
        "fdr_qvalue",
        "cointegration_rank_in_sector",
    ]
    selection_audit = formation_selected[selection_audit_columns].rename(
        columns={
            "pair_pvalue": "formation_pair_pvalue",
            "fdr_qvalue": "formation_fdr_qvalue",
            "cointegration_rank_in_sector": "formation_cointegration_rank_in_sector",
        }
    )
    best = best.merge(selection_audit, on="pair", how="left", validate="one_to_one")
    best.to_csv(paths.processed / "best_10_pairs_by_heldout_net_cagr.csv", index=False)
    best_names = best["pair"].tolist()
    portfolio_curves, portfolio_metrics = build_portfolios(
        pair_returns,
        formation_selected,
        best_names,
    )
    portfolio_curves.to_csv(paths.processed / "portfolio_equity_curves.csv")
    portfolio_metrics.to_csv(paths.processed / "portfolio_metrics.csv", index=False)
    pair_cost_sensitivity, portfolio_cost_sensitivity = cost_sensitivity(
        signals,
        formation_selected,
        best_names,
        config.strategy.transaction_cost_scenarios_bps,
    )
    pair_cost_sensitivity.to_csv(paths.processed / "pair_cost_sensitivity.csv", index=False)
    portfolio_cost_sensitivity.to_csv(paths.processed / "portfolio_cost_sensitivity.csv", index=False)

    log("Downloading/loading FRED macro series for regime attribution")
    fred_ids = {series_id: series_id for series_id in config.macro.series}
    fred_raw = download_fred_observations(
        fred_ids,
        start,
        end_exclusive,
        cache_path=paths.raw / "fred_observations.csv",
        refresh=refresh_data,
    )
    fred = derive_macro_variables(fred_raw)
    fred.to_csv(paths.processed / "fred_macro_and_derived.csv")
    return_columns = [column for column in portfolio_curves if not column.endswith("_equity")]
    regimes = regime_attribution(portfolio_curves[return_columns], fred)
    regimes.to_csv(paths.processed / "portfolio_fred_regime_attribution.csv", index=False)

    log("Rendering charts and pair dashboards")
    cointegration_chart = plot_cointegration_heatmap(
        full_top, paths.charts / "full_8y_cointegration_leaders.png"
    )
    sector_chart = plot_sector_performance(
        metrics, paths.charts / "heldout_sector_performance.png"
    )
    portfolio_chart = plot_portfolio_curves(
        portfolio_curves, paths.charts / "heldout_portfolio_equity.png"
    )
    macro_chart = plot_macro_history(fred, paths.charts / "fred_macro_regimes.png")
    dashboards: list[str] = []
    for row in best.to_dict(orient="records"):
        pair = str(row["pair"])
        pair_frame = signals[signals["pair"].eq(pair)].copy()
        dashboard_path = plot_pair_dashboard(
            pair_frame,
            row,
            paths.charts / f"pair_{pair.lower()}_dashboard.png",
        )
        dashboards.append(_relative_chart(dashboard_path, paths.root))

    quality = _quality_checks(
        universe=universe,
        prices=prices,
        coverage=coverage,
        full_diagnostics=full_diagnostics,
        formation_selected=formation_selected,
        metrics=metrics,
        best=best,
        fred=fred_raw,
        expected_top_per_sector=config.research.top_pairs_per_sector,
    )
    _write_json(paths.root / "quality_checks.json", quality)

    test_years = config.research.years - config.research.formation_years
    context = {
        "as_of": str(as_of_ts.date()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "formation_years": config.research.formation_years,
        "test_years": test_years,
        "transaction_cost_bps": config.strategy.transaction_cost_bps,
        "borrow_bps": config.strategy.annual_short_borrow_bps,
        "strict_selection_count": int(formation_selected["eligible"].fillna(False).sum()),
        "fallback_selection_count": int((~formation_selected["eligible"].fillna(False)).sum()),
        "formation_selection_count": int(len(formation_selected)),
        "config_json": json.dumps(config.to_dict(), indent=2, sort_keys=True),
        "cards": [
            {"label": "Current securities", "value": f"{len(universe):,}"},
            {"label": "GICS sectors", "value": f"{universe['sector'].nunique():,}"},
            {"label": "Full-window pairs", "value": f"{len(full_diagnostics):,}"},
            {"label": "Held-out pair tests", "value": f"{metrics['backtest_status'].eq('ok').sum():,}"},
            {"label": "Modeled cost", "value": f"{config.strategy.transaction_cost_bps:g} bps"},
            {"label": "Quality status", "value": str(quality["status"]).upper()},
        ],
        "sources": [
            {"label": "Gatev, Goetzmann & Rouwenhorst — Pairs Trading", "url": "https://www.nber.org/papers/w7032"},
            {"label": "Engle & Granger — Cointegration and Error Correction", "url": "https://ideas.repec.org/a/ecm/emetrp/v55y1987i2p251-76.html"},
            {"label": "Krauss — Pairs Trading Review", "url": "https://doi.org/10.1111/joes.12153"},
            {"label": "Avellaneda & Lee — Statistical Arbitrage", "url": "https://doi.org/10.1080/14697680903124632"},
            {"label": "statsmodels statistical tests", "url": "https://www.statsmodels.org/stable/api.html"},
            {"label": "MarketData API documentation", "url": "https://www.marketdata.app/docs/api/"},
            {"label": "FRED observations API", "url": "https://fred.stlouisfed.org/docs/api/fred/series_observations.html"},
            {"label": "S&P U.S. Indices methodology", "url": "https://www.spglobal.com/spdji/en/methodology/article/sp-us-indices-methodology/"},
        ],
    }
    chart_paths = {
        "cointegration": _relative_chart(cointegration_chart, paths.root),
        "sectors": _relative_chart(sector_chart, paths.root),
        "portfolios": _relative_chart(portfolio_chart, paths.root),
        "macro": _relative_chart(macro_chart, paths.root),
        "dashboards": dashboards,
    }
    report = render_html_report(
        paths.root / "report.html",
        context=context,
        best_pairs=best,
        full_top_pairs=full_top,
        formation_pairs=formation_selected,
        portfolio_metrics=portfolio_metrics,
        portfolio_cost_sensitivity=portfolio_cost_sensitivity,
        regime_attribution=regimes,
        quality_checks=quality,
        chart_paths=chart_paths,
    )

    files = {}
    for path in sorted(paths.root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(paths.root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    manifest = {
        "project": "equity-pairs-research",
        "version": "0.1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": str(as_of_ts.date()),
        "research_start": str(start.date()),
        "formation_end_exclusive": str(test_start.date()),
        "test_start": str(test_start.date()),
        "universe_contract": "current_constituents_current_gics_survivorship_biased",
        "sources": {
            "constituents": SP500_WIKIPEDIA_URL,
            "prices": "MarketData confirmed daily bars via quant_marketdata",
            "macro": FRED_OBSERVATIONS_URL,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "config": config.to_dict(),
        "credentials_included": False,
        "files": files,
    }
    _write_json(paths.root / "manifest.json", manifest)
    log(f"Research package complete: {report}")
    return report
