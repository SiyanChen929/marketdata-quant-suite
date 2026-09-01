"""Daily data quality monitor for production-style research runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from quant_marketdata import MarketDataStore

from quant_system.config import AppConfig


_QUALITY_MONITOR_CACHE: dict[tuple, tuple[pd.DataFrame, pd.DataFrame]] = {}


def run_daily_data_quality_monitor(
    config: AppConfig,
    *,
    broker_prices_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return symbol-level and summary-level daily data quality checks."""

    cache_key = _cache_key(config, broker_prices_path)
    cached = _QUALITY_MONITOR_CACHE.get(cache_key)
    if cached is not None:
        detail, summary = cached
        return detail.copy(), summary.copy()
    symbols = sorted(set(config.universe.custom_symbols or config.universe.symbols))
    prices = _load_configured_prices(config, symbols)
    if prices.empty:
        detail = pd.DataFrame([{"symbol": symbol, "status": "missing_price_source"} for symbol in symbols])
        summary = _summary(detail)
        _QUALITY_MONITOR_CACHE[cache_key] = (detail.copy(), summary.copy())
        return detail, summary
    if symbols:
        prices = prices[prices["symbol"].isin(symbols)].copy()
    broker_path = broker_prices_path or config.data_quality.broker_prices_path
    broker = _load_prices(broker_path) if broker_path else pd.DataFrame()
    detail = _symbol_quality(prices, symbols or sorted(prices["symbol"].unique()), config, broker)
    summary = _summary(detail)
    _QUALITY_MONITOR_CACHE[cache_key] = (detail.copy(), summary.copy())
    return detail, summary


def _cache_key(config: AppConfig, broker_prices_path: str | Path | None) -> tuple:
    broker_path = broker_prices_path or config.data_quality.broker_prices_path
    symbols = tuple(sorted(set(config.universe.custom_symbols or config.universe.symbols)))
    quality = config.data_quality
    universe = config.universe
    return (
        _configured_price_state(config),
        _file_state(config.events_path),
        _file_state(broker_path),
        symbols,
        universe.max_stale_price_days,
        quality.broker_close_diff_threshold,
        quality.block_on_broker_price_diff,
        quality.block_on_stale_price,
        quality.block_on_abnormal_jump,
        quality.missing_bar_warning_threshold,
    )


def _file_state(path_like: str | Path | None) -> tuple:
    if not path_like:
        return ("", False, 0, 0)
    path = Path(path_like)
    if not path.exists():
        return (str(path), False, 0, 0)
    stat = path.stat()
    return (str(path.resolve()), True, stat.st_size, stat.st_mtime_ns)


def _configured_price_state(config: AppConfig) -> tuple:
    if config.data_provider != "marketdata":
        return _file_state(config.data_path)
    try:
        store = MarketDataStore()
    except Exception:
        return ("marketdata", False, 0, 0)
    return _file_state(store.root / "manifests" / "marketdata" / "confirmed.json")


def _load_configured_prices(config: AppConfig, symbols: list[str]) -> pd.DataFrame:
    if config.data_provider != "marketdata":
        return _load_prices(config.data_path)
    try:
        data = MarketDataStore().read_bars(
            symbols=symbols,
            start=config.start_date,
            end=config.end_date,
            finality="confirmed",
        )
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return data
    if not data["source"].eq("marketdata.app").all() or not data["finality"].eq("confirmed").all():
        raise ValueError("Formal quality monitoring requires confirmed marketdata.app rows")
    data = data.copy()
    data["adj_close"] = data["close"]
    return data


def _load_prices(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return pd.DataFrame()
    data = pd.read_csv(source)
    if data.empty or "symbol" not in data or "date" not in data:
        return pd.DataFrame()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data["adj_close"].isna().all():
        data["adj_close"] = data["close"]
    return data.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)


def _symbol_quality(prices: pd.DataFrame, symbols: list[str], config: AppConfig, broker: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.Index(sorted(prices["date"].dropna().unique()))
    global_latest = pd.Timestamp(prices["date"].max()).normalize()
    broker_latest = _latest_broker_close(broker)
    rows: list[dict] = []
    for symbol in symbols:
        sym = prices[prices["symbol"].eq(symbol)].sort_values("date").copy()
        if sym.empty:
            rows.append({"symbol": symbol, "status": "missing_symbol"})
            continue
        sym = sym.drop_duplicates("date", keep="last")
        sym_calendar = pd.Index(sym["date"].unique())
        missing_bar_count = int(len(calendar.difference(sym_calendar)))
        latest_date = pd.Timestamp(sym["date"].max()).normalize()
        stale_days = int((global_latest - latest_date).days)
        close = sym["adj_close"].astype(float)
        raw_close = sym["close"].astype(float)
        returns = close.pct_change()
        raw_returns = raw_close.pct_change()
        abnormal_jump_count = int(returns.abs().gt(0.35).sum())
        raw_adj_divergence_count = int((returns - raw_returns).abs().gt(0.25).sum())
        split_adjustment_flag = bool(raw_adj_divergence_count and close.notna().sum() >= 20)
        zero_volume_days = int(sym["volume"].fillna(0).le(0).sum())
        vol = sym["volume"].astype(float)
        volume_med = vol.shift(1).rolling(20, min_periods=5).median()
        volume_ratio = vol / volume_med.replace(0, np.nan)
        volume_anomaly_days = int(volume_ratio.gt(8.0).sum() + volume_ratio.lt(0.05).sum())
        latest_close = float(close.dropna().iloc[-1]) if close.notna().any() else np.nan
        broker_diff_pct = np.nan
        if symbol in broker_latest and pd.notna(latest_close) and latest_close:
            broker_diff_pct = float(broker_latest[symbol] / latest_close - 1.0)
        flags = []
        if stale_days > int(config.universe.max_stale_price_days):
            flags.append("stale_price")
        if missing_bar_count:
            flags.append("missing_bars")
        if abnormal_jump_count:
            flags.append("abnormal_jump")
        if split_adjustment_flag:
            flags.append("possible_adjustment_error")
        if zero_volume_days:
            flags.append("zero_volume")
        if volume_anomaly_days:
            flags.append("volume_anomaly")
        if pd.notna(broker_diff_pct) and abs(broker_diff_pct) > float(config.data_quality.broker_close_diff_threshold):
            flags.append("marketdata_broker_price_diff")
        blocked = _blocked_by_quality_flags(flags, config)
        rows.append(
            {
                "symbol": symbol,
                "row_count": len(sym),
                "first_date": sym["date"].min(),
                "latest_date": latest_date,
                "stale_days_vs_global_latest": stale_days,
                "missing_bar_count": missing_bar_count,
                "abnormal_jump_count": abnormal_jump_count,
                "split_adjustment_flag": split_adjustment_flag,
                "raw_adj_divergence_count": raw_adj_divergence_count,
                "zero_volume_days": zero_volume_days,
                "volume_anomaly_days": volume_anomaly_days,
                "latest_marketdata_close": latest_close,
                "broker_close_diff_pct": broker_diff_pct,
                "earnings_calendar_status": _earnings_calendar_status(config, symbol),
                "blocked_from_trading": blocked,
                "block_reason": "|".join(flags) if blocked else "",
                "status": "|".join(flags) if flags else "ok",
            }
        )
    return pd.DataFrame(rows)


def _blocked_by_quality_flags(flags: list[str], config: AppConfig) -> bool:
    blocking = set()
    if config.data_quality.block_on_stale_price:
        blocking.add("stale_price")
    if config.data_quality.block_on_abnormal_jump:
        blocking.add("abnormal_jump")
    if config.data_quality.block_on_broker_price_diff:
        blocking.add("marketdata_broker_price_diff")
    return bool(blocking.intersection(flags))


def _latest_broker_close(broker: pd.DataFrame) -> dict[str, float]:
    if broker.empty:
        return {}
    price_col = "adj_close" if "adj_close" in broker.columns else "close"
    latest = broker.sort_values("date").groupby("symbol", as_index=False).tail(1)
    return dict(zip(latest["symbol"], pd.to_numeric(latest[price_col], errors="coerce")))


def _earnings_calendar_status(config: AppConfig, symbol: str) -> str:
    source = Path(config.events_path)
    if not source.exists() or source.stat().st_size == 0:
        return "missing_events_source"
    try:
        events = pd.read_csv(source, usecols=lambda col: col in {"symbol", "date", "event_type"})
    except Exception:
        return "invalid_events_source"
    if events.empty or "symbol" not in events:
        return "missing_events_rows"
    events["symbol"] = events["symbol"].astype(str).str.upper()
    symbol_events = events[events["symbol"].eq(symbol)]
    if symbol_events.empty:
        return "missing_symbol_earnings_calendar"
    return "ok"


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty or "status" not in detail:
        return pd.DataFrame()
    exploded = detail.assign(status_item=detail["status"].fillna("unknown").astype(str).str.split("|")).explode("status_item")
    summary = exploded.groupby("status_item", as_index=False).agg(symbol_count=("symbol", "nunique"))
    total = max(detail["symbol"].nunique(), 1)
    summary["symbol_pct"] = summary["symbol_count"] / total
    return summary.sort_values("symbol_count", ascending=False).reset_index(drop=True)
