"""Data quality audit reports for research inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from quant_marketdata import MarketDataStore, normalize_bars

from quant_system.config import AppConfig
from quant_system.data.lake import QuantSystemLake
from quant_system.data.marketdata_earnings import MarketDataEarningsClient
from quant_system.data.marketdata_provider import MarketDataAPIProvider
from quant_system.fundamentals.provider import MarketDataFundamentalsRuntimeClient, earnings_to_fundamentals


def run_data_audit(config: AppConfig, out_dir: str | Path = "runs/data_audit") -> dict[str, pd.DataFrame]:
    """Audit daily prices, intraday summaries, fundamentals, and events."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    symbols = sorted(set(config.universe.custom_symbols or config.universe.symbols))
    runtime_earnings = load_marketdata_runtime_earnings(config, symbols)
    runtime_fundamentals = load_marketdata_runtime_fundamentals(config, symbols)
    reports = {
        "daily_price_audit": audit_configured_daily_prices(config, symbols),
        "intraday_summary_audit": audit_intraday_summary(config, symbols),
        "fundamental_coverage_audit": audit_table_coverage(config.fundamentals_path, symbols, "fundamentals"),
        "event_coverage_audit": audit_table_coverage(config.events_path, symbols, "events"),
        "marketdata_runtime_earnings_audit": audit_marketdata_runtime_earnings(config, symbols, runtime_earnings),
        "marketdata_runtime_fundamentals_audit": audit_marketdata_runtime_fundamentals(config, symbols, runtime_earnings, runtime_fundamentals),
    }
    for name, frame in reports.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "data_audit.html").write_text(render_data_audit_html(reports), encoding="utf-8")
    return reports


def marketdata_runtime_audit_enabled(config: AppConfig) -> bool:
    """Return whether the active config expects MarketData runtime data."""

    providers = [config.data_provider, config.fundamentals.provider, config.events.provider]
    return any("marketdata" in str(provider).lower() for provider in providers)


def load_marketdata_runtime_earnings(config: AppConfig, symbols: list[str]) -> pd.DataFrame:
    """Fetch cached MarketData earnings for runtime coverage audits."""

    if not marketdata_runtime_audit_enabled(config):
        return pd.DataFrame()
    try:
        client = MarketDataEarningsClient()
        frame = client.get_bulk_earnings(symbols, config.start_date, config.end_date)
        errors = getattr(client, "last_errors", {})
    except Exception as exc:  # pragma: no cover - real API/token failure path
        frame = pd.DataFrame()
        errors = {symbol: str(exc) for symbol in symbols}
    frame = frame.copy()
    frame.attrs["marketdata_errors"] = errors
    return frame


def load_marketdata_runtime_fundamentals(config: AppConfig, symbols: list[str]) -> pd.DataFrame:
    """Fetch configurable MarketData fundamentals endpoint for audit coverage."""

    if not marketdata_runtime_audit_enabled(config):
        return pd.DataFrame()
    try:
        client = MarketDataFundamentalsRuntimeClient()
        frame = client.get_bulk_fundamentals(symbols, config.start_date, config.end_date)
        errors = getattr(client, "last_errors", {})
    except Exception as exc:  # pragma: no cover - real API/token failure path
        frame = pd.DataFrame()
        errors = {symbol: str(exc) for symbol in symbols}
    frame = frame.copy()
    frame.attrs["marketdata_errors"] = errors
    return frame


def audit_marketdata_runtime_earnings(config: AppConfig, symbols: list[str], earnings: pd.DataFrame) -> pd.DataFrame:
    """Audit real MarketData runtime earnings coverage by symbol."""

    columns = ["symbol", "row_count", "first_report_date", "latest_report_date", "last_error", "status"]
    if not marketdata_runtime_audit_enabled(config):
        return pd.DataFrame([{"symbol": symbol, "status": "disabled_not_marketdata_provider"} for symbol in symbols], columns=columns)
    errors: dict[str, str] = earnings.attrs.get("marketdata_errors", {}) if isinstance(earnings, pd.DataFrame) else {}
    if earnings.empty:
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "row_count": 0,
                    "last_error": errors.get(symbol, ""),
                    "status": "missing_runtime_earnings" if symbol not in errors else "runtime_earnings_error",
                }
                for symbol in symbols
            ],
            columns=columns,
        )
    data = earnings.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["report_date"] = pd.to_datetime(data.get("report_date"), errors="coerce").dt.normalize()
    rows = []
    for symbol in sorted(set(symbols)):
        sym = data[data["symbol"].eq(symbol)]
        if sym.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "row_count": 0,
                    "last_error": errors.get(symbol, ""),
                    "status": "missing_runtime_earnings" if symbol not in errors else "runtime_earnings_error",
                }
            )
        else:
            rows.append(
                {
                    "symbol": symbol,
                    "row_count": len(sym),
                    "first_report_date": sym["report_date"].min(),
                    "latest_report_date": sym["report_date"].max(),
                    "last_error": errors.get(symbol, ""),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def audit_marketdata_runtime_fundamentals(
    config: AppConfig,
    symbols: list[str],
    earnings: pd.DataFrame,
    runtime_fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Audit earnings-derived MarketData fundamental factor coverage."""

    columns = [
        "symbol",
        "row_count",
        "first_known_date",
        "latest_known_date",
        "eps_yoy_growth_count",
        "earnings_surprise_count",
        "guidance_proxy_count",
        "status",
    ]
    if not marketdata_runtime_audit_enabled(config):
        return pd.DataFrame([{"symbol": symbol, "status": "disabled_not_marketdata_provider"} for symbol in symbols], columns=columns)
    frames = []
    if isinstance(runtime_fundamentals, pd.DataFrame) and not runtime_fundamentals.empty:
        frames.append(runtime_fundamentals)
    if isinstance(earnings, pd.DataFrame) and not earnings.empty:
        frames.append(earnings_to_fundamentals(earnings))
    fundamentals = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if fundamentals.empty:
        return pd.DataFrame([{"symbol": symbol, "row_count": 0, "status": "missing_runtime_fundamentals"} for symbol in symbols], columns=columns)
    fundamentals["symbol"] = fundamentals["symbol"].astype(str).str.upper()
    fundamentals["known_date"] = pd.to_datetime(fundamentals.get("known_date"), errors="coerce").dt.normalize()
    rows = []
    for symbol in sorted(set(symbols)):
        sym = fundamentals[fundamentals["symbol"].eq(symbol)]
        if sym.empty:
            rows.append({"symbol": symbol, "row_count": 0, "status": "missing_runtime_fundamentals"})
        else:
            rows.append(
                {
                    "symbol": symbol,
                    "row_count": len(sym),
                    "first_known_date": sym["known_date"].min(),
                    "latest_known_date": sym["known_date"].max(),
                    "eps_yoy_growth_count": int(sym.get("eps_yoy_growth", pd.Series(dtype=float)).notna().sum()),
                    "earnings_surprise_count": int(sym.get("earnings_surprise_pct", pd.Series(dtype=float)).notna().sum()),
                    "guidance_proxy_count": int(sym.get("guidance_up_or_down", pd.Series(dtype=float)).notna().sum()),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def audit_configured_daily_prices(config: AppConfig, symbols: list[str]) -> pd.DataFrame:
    """Audit the configured formal-price source without provider fallbacks."""

    if not _uses_shared_marketdata(config):
        return audit_daily_prices(config.data_path, symbols)

    bars = MarketDataStore().read_bars(
        symbols=symbols or None,
        start=config.start_date,
        end=config.end_date,
        finality="confirmed",
        resolution="D",
    )
    bars = normalize_bars(
        bars,
        source="marketdata.app",
        finality="confirmed",
    )
    compatible = bars.copy()
    compatible["adj_close"] = compatible["close"]
    return _audit_daily_price_frame(
        compatible,
        symbols,
        empty_status="missing_daily_source",
    )


def audit_daily_prices(path: str | Path, symbols: list[str]) -> pd.DataFrame:
    """Return per-symbol OHLCV quality diagnostics for a CSV demo source."""

    source = Path(path)
    if not source.exists():
        return _empty_daily_price_audit(symbols, "missing_daily_source")
    return _audit_daily_price_frame(pd.read_csv(source), symbols)


def _audit_daily_price_frame(
    data: pd.DataFrame,
    symbols: list[str],
    *,
    empty_status: str = "invalid_daily_source",
) -> pd.DataFrame:
    """Audit one already-loaded daily frame."""

    columns = [
        "symbol",
        "row_count",
        "first_date",
        "latest_date",
        "stale_days_vs_global_latest",
        "duplicate_date_count",
        "null_ohlcv_count",
        "zero_volume_days",
        "max_abs_adj_return",
        "suspicious_return_days",
        "status",
    ]
    if data.empty:
        return pd.DataFrame(
            [{"symbol": symbol, "status": empty_status} for symbol in symbols],
            columns=columns,
        )
    if "symbol" not in data or "date" not in data:
        return pd.DataFrame(
            [{"symbol": symbol, "status": "invalid_daily_source"} for symbol in symbols],
            columns=columns,
        )
    data = data.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["date"] = pd.to_datetime(data["date"], format="mixed", errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column not in data:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")
    wanted = set(symbols) if symbols else set(data["symbol"].dropna().unique())
    data = data[data["symbol"].isin(wanted)].copy()
    global_latest = data["date"].max()
    rows: list[dict] = []
    for symbol in sorted(wanted):
        sym = data[data["symbol"].eq(symbol)].sort_values("date")
        if sym.empty:
            rows.append({"symbol": symbol, "status": "missing_symbol"})
            continue
        returns = sym["adj_close"].pct_change()
        stale_days = int((global_latest - sym["date"].max()).days) if pd.notna(global_latest) else 0
        duplicate_count = int(sym.duplicated(["date"]).sum())
        null_ohlcv = int(sym[["open", "high", "low", "close", "adj_close", "volume"]].isna().sum().sum())
        suspicious = int(returns.abs().gt(0.50).sum())
        status = "ok"
        flags = []
        if stale_days > 5:
            flags.append("stale")
        if duplicate_count:
            flags.append("duplicate_dates")
        if null_ohlcv:
            flags.append("null_ohlcv")
        if suspicious:
            flags.append("large_adj_return")
        if int(sym["volume"].fillna(0).le(0).sum()) > 0:
            flags.append("zero_volume")
        if flags:
            status = "|".join(flags)
        rows.append(
            {
                "symbol": symbol,
                "row_count": len(sym),
                "first_date": sym["date"].min(),
                "latest_date": sym["date"].max(),
                "stale_days_vs_global_latest": stale_days,
                "duplicate_date_count": duplicate_count,
                "null_ohlcv_count": null_ohlcv,
                "zero_volume_days": int(sym["volume"].fillna(0).le(0).sum()),
                "max_abs_adj_return": float(returns.abs().max()) if returns.notna().any() else 0.0,
                "suspicious_return_days": suspicious,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _empty_daily_price_audit(symbols: list[str], status: str) -> pd.DataFrame:
    columns = [
        "symbol",
        "row_count",
        "first_date",
        "latest_date",
        "stale_days_vs_global_latest",
        "duplicate_date_count",
        "null_ohlcv_count",
        "zero_volume_days",
        "max_abs_adj_return",
        "suspicious_return_days",
        "status",
    ]
    return pd.DataFrame(
        [{"symbol": symbol, "status": status} for symbol in symbols],
        columns=columns,
    )


def _uses_shared_marketdata(config: AppConfig) -> bool:
    return str(config.data_provider).strip().lower() in {"marketdata", "marketdata.app"}


def audit_intraday_summary(config: AppConfig, symbols: list[str]) -> pd.DataFrame:
    """Audit shared provisional bars or the legacy CSV-derived summary lake."""

    columns = ["symbol", "row_count", "first_date", "latest_date", "status"]
    if _uses_shared_marketdata(config):
        resolution = MarketDataAPIProvider._resolution(config.risk.intraday_timeframe)
        data = MarketDataStore().read_bars(
            symbols=symbols or None,
            start=config.risk.intraday_start_date or config.start_date,
            end=config.end_date,
            finality="provisional",
            resolution=resolution,
        )
        data = normalize_bars(data, source="marketdata.app", finality="provisional")
        if data.empty:
            return pd.DataFrame(
                [{"symbol": symbol, "status": "missing_provisional_intraday_bars"} for symbol in symbols],
                columns=columns,
            )
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        rows = []
        for symbol in sorted(set(symbols)):
            sym = data[data["symbol"].eq(symbol)]
            rows.append(
                {"symbol": symbol, "status": "missing_provisional_intraday_bars"}
                if sym.empty
                else {
                    "symbol": symbol,
                    "row_count": len(sym),
                    "first_date": sym["date"].min(),
                    "latest_date": sym["date"].max(),
                    "status": "ok_shared_provisional_bars",
                }
            )
        return pd.DataFrame(rows, columns=columns)

    lake = QuantSystemLake()
    if not lake.has_fresh_intraday_summary(config.intraday_data_path, config.risk.intraday_timeframe):
        return pd.DataFrame([{"symbol": symbol, "status": "missing_or_stale_intraday_summary"} for symbol in symbols], columns=columns)
    data = lake.read_intraday_summary(symbols, config.risk.intraday_start_date or config.start_date, config.end_date, config.risk.intraday_timeframe)
    if data.empty:
        return pd.DataFrame([{"symbol": symbol, "status": "missing_intraday_summary_symbol"} for symbol in symbols], columns=columns)
    rows = []
    for symbol in sorted(set(symbols)):
        sym = data[data["symbol"].eq(symbol)]
        if sym.empty:
            rows.append({"symbol": symbol, "status": "missing_intraday_summary_symbol"})
        else:
            rows.append(
                {
                    "symbol": symbol,
                    "row_count": len(sym),
                    "first_date": sym["date"].min(),
                    "latest_date": sym["date"].max(),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def audit_table_coverage(path: str | Path, symbols: list[str], label: str) -> pd.DataFrame:
    """Audit point-in-time table coverage by symbol."""

    source = Path(path)
    columns = ["symbol", "row_count", "first_known_date", "latest_known_date", "status"]
    if not source.exists():
        return pd.DataFrame([{"symbol": symbol, "status": f"missing_{label}_source"} for symbol in symbols], columns=columns)
    data = pd.read_csv(source)
    if data.empty or "symbol" not in data:
        return pd.DataFrame([{"symbol": symbol, "status": f"invalid_{label}_source"} for symbol in symbols], columns=columns)
    data["symbol"] = data["symbol"].astype(str).str.upper()
    known_col = "known_date" if "known_date" in data.columns else ("date" if "date" in data.columns else None)
    if known_col:
        data[known_col] = pd.to_datetime(data[known_col], format="mixed", errors="coerce").dt.normalize()
    rows = []
    for symbol in sorted(set(symbols)):
        sym = data[data["symbol"].eq(symbol)]
        if sym.empty:
            rows.append({"symbol": symbol, "status": f"missing_{label}_rows"})
        else:
            rows.append(
                {
                    "symbol": symbol,
                    "row_count": len(sym),
                    "first_known_date": sym[known_col].min() if known_col else pd.NaT,
                    "latest_known_date": sym[known_col].max() if known_col else pd.NaT,
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def render_data_audit_html(reports: dict[str, pd.DataFrame]) -> str:
    """Render a compact audit HTML."""

    sections = []
    for name, frame in reports.items():
        status_counts = frame["status"].fillna("unknown").value_counts().reset_index() if "status" in frame else pd.DataFrame()
        summary = status_counts.to_html(index=False, border=0) if not status_counts.empty else "<p>No status column.</p>"
        detail = frame.head(100).to_html(index=False, border=0)
        sections.append(f"<h2>{name}</h2><h3>Status Summary</h3>{summary}<h3>Sample Rows</h3>{detail}")
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Quant System Data Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 28px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e4e8f0; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Quant System Data Audit</h1>
  {''.join(sections)}
</body>
</html>"""
