"""Reusable research pipeline for backtests and optimization."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from quant_marketdata import MarketDataStore

from quant_system.backtest.engine import BacktestResult, DailyBacktestEngine
from quant_system.config import AppConfig
from quant_system.data.csv_provider import CSVMarketDataProvider
from quant_system.data.cache import attach_reference_cache_manifests, reference_cache_manifests
from quant_system.data.intraday_provider import CSVIntradayDataProvider, MarketDataIntradayProvider
from quant_system.data.lake import QuantSystemLake
from quant_system.data.lake_provider import LakeMarketDataProvider
from quant_system.data.marketdata_earnings import MarketDataEarningsClient
from quant_system.data.marketdata_store_provider import MarketDataStoreProvider
from quant_system.data.quality_monitor import run_daily_data_quality_monitor
from quant_system.events.earnings import align_earnings_to_dates
from quant_system.events.event_risk import score_event_risk
from quant_system.events.provider import CSVEventDataProvider, MarketDataEventDataProvider, earnings_to_event_calendar
from quant_system.features.relative_strength import add_relative_strength
from quant_system.features.technical import add_technical_features
from quant_system.fundamentals.provider import CSVFundamentalDataProvider, MarketDataFundamentalDataProvider, earnings_to_fundamentals
from quant_system.fundamentals.scoring import align_fundamentals_to_prices, score_fundamentals
from quant_system.portfolio.construction import construct_target_weights, rebalance_dates
from quant_system.portfolio.crowding import apply_crowding_risk_to_targets
from quant_system.portfolio.data_quality_gate import apply_latest_data_quality_gate_to_targets
from quant_system.portfolio.gap_risk import apply_overnight_gap_risk_to_targets, detect_overnight_gap_risk
from quant_system.portfolio.risk import apply_event_risk_to_targets
from quant_system.regime.detector import apply_regime_exposure_overlay, detect_overbought_risk
from quant_system.regime.intraday_risk import (
    apply_intraday_risk_overlay,
    apply_minute_pretrade_budget_to_targets,
    detect_intraday_proxy_risk,
    detect_minute_pretrade_risk,
)
from quant_system.strategies.base import StrategyContext
from quant_system.strategies.breakout_retest import BreakoutRetestTrendSystem
from quant_system.strategies.ensemble import combine_strategy_signals
from quant_system.strategies.mean_reversion import MeanReversionShortTerm
from quant_system.strategies.momentum_longs_short import CrossSectionalMomentumLongShort
from quant_system.strategies.trend_following import TrendFollowingWithRegimeFilter
from quant_system.trade_location import apply_trade_location_to_targets
from quant_system.universe.base import build_base_universe
from quant_system.universe.filters import build_asof_tradable_universe
from quant_system.universe.thematic import dynamic_theme_scores, load_thematic_baskets


_INTRADAY_DATA_CACHE: dict[tuple, pd.DataFrame] = {}
_INTRADAY_SUMMARY_CACHE: dict[tuple, pd.DataFrame] = {}
_MARKETDATA_EARNINGS_CACHE: dict[tuple, pd.DataFrame | None] = {}
_RESEARCH_BUNDLE_CACHE_VERSION = "research_bundle_v4_derived_only_20260901"
_SIGNAL_CACHE_VERSION = "signal_cache_v2_exit_quality_theme_support_20260517"


def provider_for(config: AppConfig):
    """Return the configured market data provider."""

    if config.data_provider == "lake":
        return LakeMarketDataProvider(QuantSystemLake())
    if config.data_provider == "csv":
        return CSVMarketDataProvider(config.data_path)
    if config.data_provider == "marketdata":
        # Research is read-only against the canonical confirmed lake. Ingestion
        # is an explicit CLI step, so a backtest never performs a hidden full-
        # history network fetch or substitutes an unlabeled CSV.
        return MarketDataStoreProvider()
    raise ValueError(f"Unsupported data provider: {config.data_provider!r}")


def prepare_research_data(config: AppConfig) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load prices and apply base/tradable/candidate universe layers."""

    if config.universe.strict_backtest and config.universe.custom_only:
        raise ValueError(
            "Strict backtest mode forbids universe.custom_only=true. Use a broad base universe/security master and keep themes as overlays."
        )
    provider = provider_for(config)
    provider_symbols: list[str] = []
    provider_universe_warnings: list[str] = []
    if not config.universe.custom_only:
        try:
            provider_symbols = provider.get_universe(config.universe.base)
        except Exception as exc:
            provider_universe_warnings.append(f"Provider base universe unavailable; falling back to configured symbols: {exc}")
    universe = build_base_universe(config.universe, provider_symbols)
    universe.warnings.extend(provider_universe_warnings)
    if config.universe.strict_backtest:
        universe.warnings.append(
            "Strict research mode: headline metrics use validation/forward slices; full-period metrics are exploratory diagnostics."
        )
    prices = provider.get_bulk_ohlcv(universe.symbols, config.start_date, config.end_date, config.timeframe)
    price_snapshot_attrs = dict(prices.attrs)
    tradable = build_asof_tradable_universe(prices, config.universe)
    latest_tradable = (
        tradable.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1).reset_index(drop=True)
        if not tradable.empty
        else tradable
    )
    rejected = latest_tradable[~latest_tradable["tradable"]].copy() if "tradable" in latest_tradable else pd.DataFrame()
    if not rejected.empty:
        quality_rejected = rejected[rejected["filter_reason"].str.contains("stale_price|possible_unhandled_corporate_action", na=False)]
        if not quality_rejected.empty:
            universe.warnings.append(
                "Data quality filter excluded "
                f"{len(quality_rejected)} symbols after quarantine because stale prices or possible unhandled corporate actions remain: "
                    + ", ".join(quality_rejected["symbol"].astype(str).head(20))
            )
    candidate = tradable[tradable["tradable"]].copy()
    candidate.attrs["universe_audit"] = latest_tradable.copy()
    if candidate.empty:
        raise ValueError("No tradable symbols after filters. Lower sample thresholds or provide broader data.")
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
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["symbol"] = prices["symbol"].astype(str).str.upper()
    prices = prices.merge(tradable[[col for col in asof_cols if col in tradable.columns]], on=["date", "symbol"], how="left")
    prices["tradable"] = prices["tradable"].fillna(False).astype(bool)
    prices["quality_position_multiplier"] = prices["quality_position_multiplier"].fillna(1.0)
    prices.attrs.update(price_snapshot_attrs)
    return prices, universe.warnings, candidate


def load_fundamentals(
    config: AppConfig,
    symbols: list[str],
    prices: pd.DataFrame,
    warnings: list[str],
    marketdata_earnings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load and as-of align fundamental scores."""

    if not config.fundamentals.enabled:
        warnings.append("Fundamental filter disabled by config.")
        return pd.DataFrame()
    raw = pd.DataFrame()
    provider_name = config.fundamentals.provider.lower()
    if marketdata_earnings is not None and provider_name in {"marketdata", "marketdata_or_csv", "marketdata_earnings"}:
        raw = earnings_to_fundamentals(marketdata_earnings)
        if not raw.empty:
            warnings.append("MarketData fundamentals enabled: using cached real earnings/EPS-derived factors shared with event risk.")
    elif provider_name in {"marketdata", "marketdata_or_csv", "marketdata_earnings"}:
        try:
            raw = MarketDataFundamentalDataProvider().get_bulk_fundamentals(symbols, config.start_date, config.end_date)
            warnings.append("MarketData fundamentals enabled: using real earnings/EPS-derived factors; full statements still require a financial-statement provider or CSV.")
        except Exception as exc:
            fallback = " CSV fallback is enabled." if provider_name in {"marketdata_or_csv", "marketdata_earnings"} else ""
            warnings.append(f"MarketData fundamentals unavailable.{fallback} Error: {exc}")
    if raw.empty and provider_name in {"csv", "marketdata_or_csv", "marketdata_earnings"}:
        provider = CSVFundamentalDataProvider(config.fundamentals_path)
        raw = provider.get_bulk_fundamentals(symbols, config.start_date, config.end_date)
    if raw.empty:
        warnings.append("Fundamental scoring disabled because point-in-time fundamentals data unavailable.")
        return attach_reference_cache_manifests(pd.DataFrame(), raw, marketdata_earnings)
    aligned = align_fundamentals_to_prices(score_fundamentals(raw, config.fundamentals), prices)
    return attach_reference_cache_manifests(aligned, raw)


def load_events(
    config: AppConfig,
    symbols: list[str],
    prices: pd.DataFrame,
    warnings: list[str],
    marketdata_earnings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load and score event risk."""

    if not config.events.enabled:
        warnings.append("Event risk filter disabled by config.")
        return pd.DataFrame()
    earnings = pd.DataFrame()
    provider_name = config.events.provider.lower()
    if marketdata_earnings is not None and provider_name in {"marketdata", "marketdata_or_csv", "marketdata_earnings"}:
        earnings = earnings_to_event_calendar(marketdata_earnings)
        if not earnings.empty:
            warnings.append("MarketData earnings/events enabled from the shared earnings cache. Historical earnings dates use known_date=event_date unless a point-in-time calendar is supplied.")
    elif provider_name in {"marketdata", "marketdata_or_csv", "marketdata_earnings"}:
        try:
            earnings = MarketDataEventDataProvider().get_earnings_calendar(symbols, config.start_date, config.end_date)
            warnings.append("MarketData earnings/events enabled. Historical earnings dates are treated conservatively with known_date=event_date unless a point-in-time calendar is supplied.")
        except Exception as exc:
            fallback = " CSV fallback is enabled." if provider_name in {"marketdata_or_csv", "marketdata_earnings"} else ""
            warnings.append(f"MarketData event data unavailable.{fallback} Error: {exc}")
    if provider_name in {"csv", "marketdata_or_csv", "marketdata_earnings"}:
        provider = CSVEventDataProvider(config.events_path)
        csv_earnings = provider.get_earnings_calendar(symbols, config.start_date, config.end_date)
        if not csv_earnings.empty:
            if earnings.empty:
                earnings = csv_earnings
            else:
                frames = [frame.dropna(axis=1, how="all") for frame in (earnings, csv_earnings) if not frame.empty]
                earnings = (
                    pd.concat(frames, ignore_index=True)
                    .sort_values(["symbol", "event_date", "known_date"])
                    .drop_duplicates(["symbol", "event_date", "event_type"], keep="last")
                )
    if earnings.empty:
        warnings.append("Event risk filter disabled because earnings/event data unavailable.")
        return attach_reference_cache_manifests(pd.DataFrame(), earnings, marketdata_earnings)
    scored = score_event_risk(align_earnings_to_dates(prices, earnings))
    return attach_reference_cache_manifests(scored, earnings)


def build_features_and_context(
    config: AppConfig,
    prices: pd.DataFrame,
    candidate: pd.DataFrame,
    warnings: list[str],
) -> tuple[pd.DataFrame, StrategyContext]:
    """Compute features and context scores once for a research run."""

    price_snapshot_attrs = dict(prices.attrs)
    features = add_relative_strength(add_technical_features(prices), config.universe.benchmark)
    features.attrs.update(price_snapshot_attrs)
    theme_scores = dynamic_theme_scores(features, config.universe.benchmark, threshold=config.regime.theme_activation_threshold)
    theme_scores = _attach_theme_memberships(theme_scores, config.universe.theme_baskets_path)
    symbols = sorted(candidate["symbol"].astype(str).str.upper().unique().tolist())
    marketdata_earnings = _load_shared_marketdata_earnings(config, symbols, warnings)
    fundamentals = load_fundamentals(config, symbols, prices, warnings, marketdata_earnings=marketdata_earnings)
    events = load_events(config, symbols, prices, warnings, marketdata_earnings=marketdata_earnings)
    reference_snapshots = reference_cache_manifests(marketdata_earnings, fundamentals, events)
    if reference_snapshots:
        features.attrs["reference_data_snapshots"] = reference_snapshots
    return features, StrategyContext(config.universe.benchmark, fundamentals=fundamentals, events=events, theme_scores=theme_scores)


def _attach_theme_memberships(theme_scores: pd.DataFrame, theme_baskets_path: str | Path) -> pd.DataFrame:
    """Attach stable thematic basket labels to point-in-time theme scores."""

    if theme_scores.empty:
        return theme_scores
    baskets = load_thematic_baskets(theme_baskets_path)
    if not baskets:
        return theme_scores
    membership: dict[str, list[str]] = {}
    for theme, symbols in baskets.items():
        for symbol in symbols:
            membership.setdefault(str(symbol).upper(), []).append(str(theme))
    out = theme_scores.copy()
    symbols = out["symbol"].astype(str).str.upper()
    out["theme_memberships"] = symbols.map(lambda symbol: ",".join(membership.get(symbol, [])))
    out["primary_theme"] = out["theme_memberships"].str.split(",").str[0].replace("", "unclassified")
    return out


def prepare_research_bundle(config: AppConfig) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame, StrategyContext]:
    """Load or build the reusable price/features/context bundle for one config.

    This cache deliberately stops before strategy signal generation, because
    optimization changes strategy parameters frequently while prices, technical
    factors, theme scores, and point-in-time context usually stay fixed.
    """

    cache_path = _research_bundle_cache_path(config)
    use_persistent_bundle_cache = config.data_provider != "marketdata"
    if use_persistent_bundle_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            candidate = cached["candidate"]
            features = cached["features"]
            prices = _prices_from_features(features)
            context = cached["context"]
            warnings = list(cached.get("warnings", []))
            warnings.append(f"Reused research bundle cache: {cache_path}")
            return prices, warnings, candidate, features, context
        except Exception:
            cache_path.unlink(missing_ok=True)
    warnings: list[str] = []
    prices, universe_warnings, candidate = prepare_research_data(config)
    warnings.extend(universe_warnings)
    features, context = build_features_and_context(config, prices, candidate, warnings)
    if use_persistent_bundle_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "warnings": warnings,
                    "candidate": candidate,
                    "features": features,
                    "context": context,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        warnings.append(f"Built derived-only research bundle cache: {cache_path}")
    else:
        warnings.append(
            "Persistent research-bundle caching is disabled for MarketData runs; "
            "confirmed bars remain only in the canonical shared store."
        )
    return prices, warnings, candidate, features, context


def _prices_from_features(features: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the compact price view needed by research diagnostics."""

    price_columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
        "finality",
    ]
    available = [column for column in price_columns if column in features.columns]
    prices = features[available].copy()
    prices.attrs.update(features.attrs)
    return prices


def _research_bundle_cache_path(config: AppConfig) -> Path:
    payload = {
        "version": _RESEARCH_BUNDLE_CACHE_VERSION,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "timeframe": config.timeframe,
        "data_provider": config.data_provider,
        "data_path": str(config.data_path),
        "universe": asdict(config.universe),
        "fundamentals": asdict(config.fundamentals),
        "events": asdict(config.events),
        "regime": {
            "theme_activation_threshold": config.regime.theme_activation_threshold,
            "train_start": config.regime.train_start,
            "train_end": config.regime.train_end,
            "validation_start": config.regime.validation_start,
            "validation_end": config.regime.validation_end,
            "forward_start": config.regime.forward_start,
        },
        "files": {
            "data": _configured_marketdata_fingerprint(config, "confirmed"),
            "fundamentals": _file_fingerprint(config.fundamentals_path),
            "events": _file_fingerprint(config.events_path),
            "themes": _file_fingerprint(config.universe.theme_baskets_path),
        },
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:20]
    return Path("data/cache/research_bundles") / f"{digest}.pkl"


def _file_fingerprint(path_like: str | Path) -> dict[str, object]:
    path = Path(path_like)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _configured_marketdata_fingerprint(config: AppConfig, finality: str) -> dict[str, object]:
    """Fingerprint the authoritative manifest for shared MarketData stores."""

    if config.data_provider != "marketdata":
        path = config.data_path if finality == "confirmed" else config.intraday_data_path
        return _file_fingerprint(path)
    try:
        store = MarketDataStore()
    except Exception:
        return {"provider": "marketdata", "finality": finality, "exists": False}
    manifest = store.root / "manifests" / "marketdata" / f"{finality}.json"
    return {
        "provider": "marketdata",
        "finality": finality,
        **_file_fingerprint(manifest),
    }


def load_intraday_data(config: AppConfig, symbols: list[str], warnings: list[str]) -> pd.DataFrame:
    """Load true intraday bars for same-day execution risk controls."""

    if not config.risk.intraday_enabled or not symbols:
        return pd.DataFrame()
    start = config.risk.intraday_start_date or config.start_date
    path = Path(config.intraday_data_path)
    data_fingerprint = _configured_marketdata_fingerprint(config, "provisional")
    cache_key = (
        str(path),
        json.dumps(data_fingerprint, sort_keys=True, default=str),
        str(start),
        str(config.end_date),
        str(config.risk.intraday_timeframe),
        tuple(sorted(str(symbol).upper() for symbol in symbols)),
    )
    cached = _INTRADAY_DATA_CACHE.get(cache_key)
    if cached is not None:
        warnings.append(
            f"Intraday risk engine reused cached true bars from {config.intraday_data_path}: "
            f"{len(cached)} rows, {cached['symbol'].nunique() if 'symbol' in cached else 0} symbols."
        )
        return cached
    if config.data_provider == "marketdata":
        provider = MarketDataIntradayProvider()
        data = provider.read_stored_intraday(
            symbols,
            start,
            config.end_date,
            config.risk.intraday_timeframe,
        )
        if data.empty and config.risk.intraday_fetch_from_provider:
            data = provider.get_bulk_intraday(symbols, start, config.end_date, config.risk.intraday_timeframe)
        if not data.empty:
            _INTRADAY_DATA_CACHE[cache_key] = data
            warnings.append(
                f"Intraday risk engine loaded provisional MarketData bars from {provider.store.root}: "
                f"{len(data)} rows, {data['symbol'].nunique()} symbols."
            )
            return data
        warnings.append("Intraday risk engine requested but no provisional MarketData bars were available.")
        return pd.DataFrame()
    provider = CSVIntradayDataProvider(config.intraday_data_path)
    lake = QuantSystemLake()
    intraday_source = str(config.intraday_data_path)
    if path.exists() and lake.has_fresh_intraday(path, config.risk.intraday_timeframe):
        data = lake.read_intraday(symbols, start, config.end_date, config.risk.intraday_timeframe)
        intraday_source = str(lake.intraday_path(config.risk.intraday_timeframe))
    else:
        data = provider.get_bulk_intraday(symbols, start, config.end_date, config.risk.intraday_timeframe)
    if not data.empty:
        _INTRADAY_DATA_CACHE[cache_key] = data
        warnings.append(
            f"Intraday risk engine enabled with true bars from {intraday_source}: "
            f"{len(data)} rows, {data['symbol'].nunique()} symbols."
        )
        return data
    warnings.append("Intraday risk engine requested but no minute/5-minute data was available; same-day intraday controls were disabled.")
    return pd.DataFrame()


def load_intraday_summary(config: AppConfig, symbols: list[str], warnings: list[str]) -> pd.DataFrame:
    """Load precomputed symbol/date intraday summaries for pretrade risk scoring."""

    if not config.risk.intraday_enabled or not config.regime.minute_pretrade_budget_overlay or not symbols:
        return pd.DataFrame()
    if config.data_provider == "marketdata":
        # The shared store owns raw provisional bars. The caller will derive
        # the daily risk summary in memory, avoiding a second Parquet lake.
        return pd.DataFrame()
    start = config.risk.intraday_start_date or config.start_date
    path = Path(config.intraday_data_path)
    lake = QuantSystemLake()
    mtime = path.stat().st_mtime if path.exists() else None
    cache_key = (
        "summary",
        str(path),
        mtime,
        str(start),
        str(config.end_date),
        str(config.risk.intraday_timeframe),
        tuple(sorted(str(symbol).upper() for symbol in symbols)),
    )
    cached = _INTRADAY_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        warnings.append(
            f"Intraday pretrade risk reused cached daily summaries: {len(cached)} rows, "
            f"{cached['symbol'].nunique() if 'symbol' in cached else 0} symbols."
        )
        return cached
    if path.exists() and lake.has_fresh_intraday_summary(path, config.risk.intraday_timeframe):
        summary = lake.read_intraday_summary(symbols, start, config.end_date, config.risk.intraday_timeframe)
        if not summary.empty:
            _INTRADAY_SUMMARY_CACHE[cache_key] = summary
            warnings.append(
                f"Intraday pretrade risk loaded daily summaries from {lake.intraday_summary_path(config.risk.intraday_timeframe)}: "
                f"{len(summary)} rows, {summary['symbol'].nunique()} symbols."
            )
            return summary
    return pd.DataFrame()


def _load_shared_marketdata_earnings(config: AppConfig, symbols: list[str], warnings: list[str]) -> pd.DataFrame | None:
    """Fetch MarketData earnings once when fundamentals/events both can reuse it."""

    providers = {
        config.fundamentals.provider.lower() if config.fundamentals.enabled else "",
        config.events.provider.lower() if config.events.enabled else "",
    }
    if not providers.intersection({"marketdata", "marketdata_or_csv", "marketdata_earnings"}):
        return None
    cache_key = (tuple(sorted(str(symbol).upper() for symbol in symbols)), str(config.start_date), str(config.end_date))
    if cache_key in _MARKETDATA_EARNINGS_CACHE:
        cached = _MARKETDATA_EARNINGS_CACHE[cache_key]
        if cached is not None and not cached.empty:
            warnings.append(f"Reused cached MarketData earnings rows: {len(cached)} rows across {cached['symbol'].nunique()} symbols.")
        return cached
    try:
        client = MarketDataEarningsClient()
        earnings = client.get_bulk_earnings(symbols, config.start_date, config.end_date)
    except Exception as exc:
        warnings.append(f"MarketData earnings unavailable, falling back to CSV if available: {exc}")
        _MARKETDATA_EARNINGS_CACHE[cache_key] = None
        return None
    if client.last_errors:
        sample = list(client.last_errors)[:8]
        warnings.append(f"MarketData earnings unavailable for {len(client.last_errors)} symbols; using available rows and CSV fallback where possible. Sample: {sample}")
    if earnings.empty:
        warnings.append("MarketData earnings returned no usable rows.")
    else:
        warnings.append(f"Loaded MarketData earnings rows once for fundamentals/events: {len(earnings)} rows across {earnings['symbol'].nunique()} symbols.")
    _MARKETDATA_EARNINGS_CACHE[cache_key] = earnings
    return earnings


def generate_ensemble_signals(config: AppConfig, features: pd.DataFrame, context: StrategyContext) -> pd.DataFrame:
    """Generate configured strategy signals and combine them."""

    strategy_signals: dict[str, pd.DataFrame] = {}
    if config.strategies.get("momentum", {}).get("enabled", True):
        params = dict(config.strategies.get("momentum", {}))
        params.setdefault("max_event_risk_score", config.events.max_event_risk_score_for_normal_strategies)
        params.setdefault("block_high_event_risk", config.events.block_high_event_risk)
        strategy_signals["momentum"] = CrossSectionalMomentumLongShort(params).generate_signals(features, context)
    if config.strategies.get("breakout", {}).get("enabled", False):
        params = dict(config.strategies.get("breakout", {}))
        params.setdefault("max_event_risk_score", config.events.max_event_risk_score_for_normal_strategies)
        params.setdefault("block_high_event_risk", config.events.block_high_event_risk)
        strategy_signals["breakout"] = BreakoutRetestTrendSystem(params).generate_signals(features, context)
    if config.strategies.get("trend", {}).get("enabled", False):
        params = dict(config.strategies.get("trend", {}))
        params.setdefault("max_event_risk_score", config.events.max_event_risk_score_for_normal_strategies)
        params.setdefault("block_high_event_risk", config.events.block_high_event_risk)
        strategy_signals["trend"] = TrendFollowingWithRegimeFilter(params).generate_signals(features, context)
    if config.strategies.get("mean_reversion", {}).get("enabled", False):
        params = dict(config.strategies.get("mean_reversion", {}))
        params.setdefault("max_event_risk_score", min(config.events.max_event_risk_score_for_normal_strategies, 60))
        params.setdefault("block_high_event_risk", config.events.block_high_event_risk)
        strategy_signals["mean_reversion"] = MeanReversionShortTerm(params).generate_signals(features, context)
    weights = {name: float(spec.get("weight", 0.0)) for name, spec in config.strategies.items()}
    return combine_strategy_signals(strategy_signals, weights or {"momentum": 1.0})


def generate_cached_ensemble_signals(config: AppConfig, features: pd.DataFrame, context: StrategyContext, warnings: list[str]) -> pd.DataFrame:
    """Generate strategy signals with a persistent cache for iterative runs."""

    cache_path = _signal_cache_path(config, features)
    if cache_path.exists():
        try:
            signals = pd.read_pickle(cache_path)
            warnings.append(f"Reused strategy signal cache: {cache_path}")
            return signals
        except Exception:
            cache_path.unlink(missing_ok=True)
    signals = generate_ensemble_signals(config, features, context)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_pickle(cache_path)
    warnings.append(f"Built strategy signal cache: {cache_path}")
    return signals


def _signal_cache_path(config: AppConfig, features: pd.DataFrame) -> Path:
    dates = pd.to_datetime(features["date"], format="mixed", errors="coerce") if "date" in features else pd.Series(dtype="datetime64[ns]")
    payload = {
        "version": _SIGNAL_CACHE_VERSION,
        "signal_code_fingerprint": _signal_code_fingerprint(),
        "start_date": config.start_date,
        "end_date": config.end_date,
        "data_path": str(config.data_path),
        "data_fingerprint": _configured_marketdata_fingerprint(config, "confirmed"),
        "fundamentals_fingerprint": _file_fingerprint(config.fundamentals_path),
        "events_fingerprint": _file_fingerprint(config.events_path),
        "features_rows": int(len(features)),
        "features_symbols": int(features["symbol"].nunique()) if "symbol" in features else 0,
        "features_min_date": str(dates.min().date()) if not dates.empty and pd.notna(dates.min()) else "",
        "features_max_date": str(dates.max().date()) if not dates.empty and pd.notna(dates.max()) else "",
        "strategies": config.strategies,
        "events": asdict(config.events),
        "regime_signal_inputs": {
            "theme_activation_threshold": config.regime.theme_activation_threshold,
        },
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:20]
    return Path("data/cache/signals") / f"{digest}.pkl"


def _signal_code_fingerprint() -> dict[str, object]:
    """Fingerprint signal-generation source files for automatic cache invalidation."""

    root = Path(__file__).resolve().parent
    files = [
        Path(__file__).resolve(),
        root / "strategies" / "base.py",
        root / "strategies" / "ensemble.py",
        root / "strategies" / "momentum_longs_short.py",
        root / "strategies" / "breakout_retest.py",
        root / "strategies" / "trend_following.py",
        root / "strategies" / "mean_reversion.py",
    ]
    rows: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for path in files:
        item: dict[str, object] = {"path": str(path)}
        if not path.exists():
            item["exists"] = False
            rows.append(item)
            digest.update(f"{path}:missing".encode("utf-8"))
            continue
        data = path.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()
        item.update({"exists": True, "sha256": file_hash, "bytes": len(data)})
        rows.append(item)
        digest.update(str(path).encode("utf-8"))
        digest.update(file_hash.encode("utf-8"))
    return {"version": 1, "sha256": digest.hexdigest(), "files": rows}


def run_research_backtest(
    config: AppConfig,
    prices: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    context: StrategyContext | None = None,
    candidate: pd.DataFrame | None = None,
    warnings: list[str] | None = None,
) -> tuple[BacktestResult, list[str], pd.DataFrame]:
    """Run the configured research backtest."""

    warnings = list(warnings or [])
    if prices is None and candidate is None and features is None and context is None:
        prices, bundle_warnings, candidate, features, context = prepare_research_bundle(config)
        warnings.extend(bundle_warnings)
    else:
        if prices is None or candidate is None:
            prices, universe_warnings, candidate = prepare_research_data(config)
            warnings.extend(universe_warnings)
        if features is None or context is None:
            features, context = build_features_and_context(config, prices, candidate, warnings)
    signal_features = features[features.get("tradable", False).fillna(False)].copy() if "tradable" in features.columns else features
    signals = generate_cached_ensemble_signals(config, signal_features, context, warnings)
    quality_cols = [
        "date",
        "symbol",
        "quality_adjusted",
        "quality_adjustment_reason",
        "quality_position_multiplier",
        "quality_history_start",
        "limited_history_flag",
        "filter_reason",
        "candidate_reason",
    ]
    available_quality_cols = [col for col in quality_cols if col in candidate.columns]
    if not signals.empty and {"date", "symbol"}.issubset(available_quality_cols):
        signals = signals.merge(candidate[available_quality_cols], on=["date", "symbol"], how="left")
        signals["quality_position_multiplier"] = signals["quality_position_multiplier"].fillna(1.0)
    dates = rebalance_dates(signals["date"], config.portfolio.rebalance) if not signals.empty else set()
    rebalanced_signals = signals[signals["date"].isin(dates)].copy() if not signals.empty else signals
    targets = construct_target_weights(rebalanced_signals, config.portfolio)
    targets = apply_event_risk_to_targets(targets, config.events)
    targets = apply_crowding_risk_to_targets(targets, config.regime, config.universe.theme_baskets_path)
    targets = apply_overnight_gap_risk_to_targets(targets, features, config.regime)
    targets = apply_trade_location_to_targets(targets, config.portfolio)
    targets = apply_regime_exposure_overlay(targets, features, config.universe.benchmark, config.regime)
    targets = apply_intraday_risk_overlay(targets, features, config.universe.benchmark, config.regime)
    candidate_symbols = sorted(candidate["symbol"].astype(str).str.upper().unique().tolist())
    intraday_summary = load_intraday_summary(config, candidate_symbols, warnings)
    fallback_intraday_for_pretrade = pd.DataFrame()
    if intraday_summary.empty and config.regime.minute_pretrade_budget_overlay:
        fallback_intraday_for_pretrade = load_intraday_data(config, candidate_symbols, warnings)
    minute_pretrade_source = intraday_summary if not intraday_summary.empty else fallback_intraday_for_pretrade
    targets = apply_minute_pretrade_budget_to_targets(targets, minute_pretrade_source, config.universe.benchmark, config.regime)
    if _live_data_quality_gate_enabled(config):
        data_quality_detail, _ = run_daily_data_quality_monitor(config)
        targets, data_quality_warnings = apply_latest_data_quality_gate_to_targets(targets, data_quality_detail, config)
        warnings.extend(data_quality_warnings)
    target_symbols = (
        sorted(targets.loc[targets["target_weight"].abs() > 1e-12, "symbol"].astype(str).str.upper().unique().tolist())
        if not targets.empty and {"symbol", "target_weight"}.issubset(targets.columns)
        else []
    )
    intraday_prices = (
        fallback_intraday_for_pretrade
        if not fallback_intraday_for_pretrade.empty
        else load_intraday_data(config, target_symbols, warnings)
    )
    result = DailyBacktestEngine(features, config, intraday_prices=intraday_prices).run(targets, rebalanced_signals)
    result.targets = targets
    result.prices = features
    result.risk_overlay = build_risk_overlay_report(features, config, intraday_prices=minute_pretrade_source)
    return result, warnings, candidate


def _live_data_quality_gate_enabled(config: AppConfig) -> bool:
    quality = config.data_quality
    return bool(quality.block_on_stale_price or quality.block_on_abnormal_jump or quality.block_on_broker_price_diff)


def build_risk_overlay_report(features: pd.DataFrame, config: AppConfig, intraday_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Combine enabled close-time risk overlays into one report table."""

    frames: list[pd.DataFrame] = []
    if config.regime.overbought_overlay:
        frames.append(detect_overbought_risk(features, config.universe.benchmark, config.regime))
    if config.regime.intraday_risk_overlay:
        frames.append(detect_intraday_proxy_risk(features, config.universe.benchmark, config.regime))
    if config.regime.overnight_gap_risk_overlay:
        gap = detect_overnight_gap_risk(features, config.regime)
        if not gap.empty:
            frames.append(
                gap.groupby("date", as_index=False).agg(
                    avg_overnight_gap_risk_score=("overnight_gap_risk_score", "mean"),
                    max_overnight_gap_risk_score=("overnight_gap_risk_score", "max"),
                    pct_overnight_gap_reduce=("overnight_gap_multiplier", lambda s: pd.to_numeric(s, errors="coerce").lt(1).mean()),
                )
            )
    if config.regime.minute_pretrade_budget_overlay and intraday_prices is not None and not intraday_prices.empty:
        frames.append(detect_minute_pretrade_risk(intraday_prices, config.universe.benchmark, config.regime))
    frames = [frame.copy() for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for frame in frames[1:]:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        out = out.merge(frame, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def config_with_strategy_params(config: AppConfig, strategy: str, params: dict[str, Any]) -> AppConfig:
    """Return a config copy with strategy params overlaid."""

    strategies = {name: dict(spec) for name, spec in config.strategies.items()}
    strategies.setdefault(strategy, {})
    strategies[strategy].update(params)
    strategies[strategy]["enabled"] = True
    return replace(config, strategies=strategies)
