"""Fundamental data provider interfaces and CSV fallback."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_system.data.cache import DataCache, attach_reference_cache_manifests
from quant_system.data.marketdata_earnings import MarketDataEarningsClient
from quant_system.fundamentals.factors import FACTOR_GROUPS


class FundamentalDataProvider(ABC):
    """Point-in-time fundamental data interface."""

    @abstractmethod
    def get_fundamentals(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        """Return point-in-time fundamental rows."""

    def get_bulk_fundamentals(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        frames = [self.get_fundamentals(symbol, start, end) for symbol in symbols]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_earnings(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_estimates(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_company_profile(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_short_interest(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_float(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_sector_industry(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame()


class CSVFundamentalDataProvider(FundamentalDataProvider):
    """CSV fallback for fundamental data.

    Schema: symbol,known_date,sector,industry plus any supported factor
    columns. ``known_date`` can also be named ``as_of_date``. Legacy CSVs with
    only ``date`` still work, but in that case date is interpreted as the first
    date the data was known, not the fiscal period end.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data
        if not self.path.exists():
            self._data = pd.DataFrame(columns=["symbol", "date", "sector", "industry"])
            return self._data
        data = pd.read_csv(self.path)
        data["symbol"] = data["symbol"].astype(str).str.upper()
        if "known_date" not in data.columns:
            if "as_of_date" in data.columns:
                data["known_date"] = data["as_of_date"]
            else:
                data["known_date"] = data["date"]
        if "date" not in data.columns:
            data["date"] = data["known_date"]
        data["known_date"] = pd.to_datetime(data["known_date"]).dt.normalize()
        data["date"] = data["known_date"]
        data["pit_source_date"] = data["known_date"]
        if "fiscal_period_end" in data.columns:
            data["fiscal_period_end"] = pd.to_datetime(data["fiscal_period_end"], errors="coerce").dt.normalize()
        if "report_date" in data.columns:
            data["report_date"] = pd.to_datetime(data["report_date"], errors="coerce").dt.normalize()
        self._data = data.sort_values(["symbol", "date"])
        return self._data

    def get_fundamentals(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        data = self._load()
        mask = (data["symbol"] == symbol.upper()) & (data["date"] >= pd.Timestamp(start))
        if end:
            mask &= data["date"] <= pd.Timestamp(end)
        return data.loc[mask].copy()


class MarketDataFundamentalDataProvider(FundamentalDataProvider):
    """MarketData.app fundamentals adapter using real earnings/EPS data.

    MarketData.app currently documents stock earnings as the available
    fundamental endpoint. This provider maps point-in-time earnings rows into
    the framework's factor schema. Full financial statement factors should be
    supplied by CSV or another provider when available.
    """

    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self.earnings_client = MarketDataEarningsClient(cache_dir=cache_dir)
        self.runtime_client = MarketDataFundamentalsRuntimeClient(cache_dir=cache_dir)

    def get_fundamentals(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        runtime = self.runtime_client.get_fundamentals(symbol, start, end)
        earnings = self.get_earnings(symbol, start, end)
        earnings_fundamentals = earnings_to_fundamentals(earnings) if not earnings.empty else pd.DataFrame()
        if runtime.empty:
            return attach_reference_cache_manifests(earnings_fundamentals, runtime, earnings)
        if earnings_fundamentals.empty:
            return attach_reference_cache_manifests(runtime, runtime, earnings)
        combined = pd.concat([runtime, earnings_fundamentals], ignore_index=True, sort=False)
        out = combined.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="first")
        return attach_reference_cache_manifests(out, runtime, earnings)

    def get_bulk_fundamentals(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        frames = []
        manifest_sources = []
        for symbol in symbols:
            try:
                frame = self.get_fundamentals(symbol, start, end)
            except Exception:
                continue
            manifest_sources.append(frame)
            if not frame.empty:
                frames.append(frame)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return attach_reference_cache_manifests(out, *manifest_sources)

    def get_earnings(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        return self.earnings_client.get_earnings(symbol, start, end)


class MarketDataFundamentalsRuntimeClient:
    """Configurable MarketData.app financial-statement/fundamental endpoint.

    Set ``MARKETDATA_FUNDAMENTALS_ENDPOINT_TEMPLATE`` when your subscription
    exposes a statements/fundamentals endpoint. The parser accepts row-shaped or
    column-oriented JSON and maps common aliases into the framework factor
    schema. If no endpoint template is configured, the client returns an empty
    frame and the provider falls back to real earnings-derived factors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache",
        timeout_seconds: float = 12.0,
    ) -> None:
        if api_key is not None:
            raise ValueError("Pass MarketData credentials through MARKETDATA_TOKEN, not constructor arguments")
        self.api_key = os.getenv("MARKETDATA_TOKEN")
        self.base_url = base_url or os.getenv("MARKETDATA_BASE_URL", "https://api.marketdata.app/v1")
        self.endpoint_template = os.getenv("MARKETDATA_FUNDAMENTALS_ENDPOINT_TEMPLATE", "")
        self.cache = DataCache(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.last_errors: dict[str, str] = {}

    def get_fundamentals(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        if not self.endpoint_template:
            return pd.DataFrame()
        if not self.api_key:
            raise RuntimeError("MARKETDATA_TOKEN is not set for the fundamentals runtime endpoint.")
        symbol = symbol.upper()
        cache_cutoff = end or pd.Timestamp.now(tz="America/New_York").date().isoformat()
        cache_key = f"marketdata_runtime_fundamentals_v1_{symbol}_{start}_{cache_cutoff}"
        endpoint = self.endpoint_template.lstrip("/").format(symbol=symbol)
        request = {
            "provider": "marketdata.app",
            "dataset": "fundamentals",
            "base_url": self.base_url.rstrip("/"),
            "endpoint": f"/{endpoint}",
            "symbol": symbol,
            "start": start,
            "end": cache_cutoff,
        }
        cached = self.cache.read(cache_key, request=request)
        if cached is not None:
            return normalize_marketdata_fundamentals(symbol, cached, start, end)
        url = f"{self.base_url.rstrip('/')}/{endpoint}"
        params = {"from": start}
        if end:
            params["to"] = end
        response = requests.get(url, params=params, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout_seconds)
        response.raise_for_status()
        frame = normalize_marketdata_fundamentals(symbol, response.json(), start, end)
        self.cache.write(cache_key, frame, request=request)
        persisted = self.cache.read(cache_key, request=request)
        if persisted is None:  # pragma: no cover - defensive invariant
            raise RuntimeError(f"MarketData fundamentals cache write was not readable for {symbol}")
        return normalize_marketdata_fundamentals(symbol, persisted, start, end)

    def get_bulk_fundamentals(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        manifest_sources: list[pd.DataFrame] = []
        self.last_errors = {}
        for symbol in symbols:
            try:
                frame = self.get_fundamentals(symbol, start, end)
            except Exception as exc:  # pragma: no cover - real API failure path
                self.last_errors[str(symbol).upper()] = str(exc)
                continue
            manifest_sources.append(frame)
            if not frame.empty:
                frames.append(frame)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return attach_reference_cache_manifests(out, *manifest_sources)


def normalize_marketdata_fundamentals(symbol: str, payload: Any, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Normalize generic MarketData fundamental payloads to factor schema."""

    if isinstance(payload, pd.DataFrame):
        raw = payload.copy()
    else:
        raw = _payload_to_frame(payload)
    if raw.empty:
        return attach_reference_cache_manifests(pd.DataFrame(), raw)
    raw = raw.rename(columns={column: _canonical_fundamental_column(column) for column in raw.columns})
    out = pd.DataFrame()
    out["symbol"] = raw.get("symbol", symbol).astype(str).str.upper() if isinstance(raw.get("symbol"), pd.Series) else symbol.upper()
    date_source = next((col for col in ["known_date", "as_of_date", "filing_date", "report_date", "date", "period_end_date", "fiscal_period_end"] if col in raw.columns), None)
    if date_source is None:
        return attach_reference_cache_manifests(pd.DataFrame(), raw)
    out["known_date"] = pd.to_datetime(raw[date_source], errors="coerce").dt.normalize()
    out["date"] = out["known_date"]
    out["pit_source_date"] = out["known_date"]
    out["sector"] = raw.get("sector", "UNKNOWN")
    out["industry"] = raw.get("industry", "UNKNOWN")
    factor_names = {factor for factors in FACTOR_GROUPS.values() for factor in factors}
    for factor in factor_names:
        if factor in raw.columns:
            out[factor] = pd.to_numeric(raw[factor], errors="coerce")
    out = out.dropna(subset=["date"])
    if start:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]
    result = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return attach_reference_cache_manifests(result, raw)


def _payload_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        if payload.get("s") == "no_data":
            return pd.DataFrame()
        for key in ("data", "results", "fundamentals", "financials", "statements"):
            if key in payload:
                return _payload_to_frame(payload[key])
        arrays = {key: value for key, value in payload.items() if isinstance(value, list)}
        if arrays:
            length = max((len(value) for value in arrays.values()), default=0)
            if length:
                return pd.DataFrame({key: (value if len(value) == length else [pd.NA] * length) for key, value in arrays.items()})
        return pd.DataFrame([payload])
    return pd.DataFrame(payload)


def _canonical_fundamental_column(column: str) -> str:
    clean = str(column).strip()
    lowered = clean.replace("-", "_").replace(" ", "_").lower()
    aliases = {
        "asofdate": "known_date",
        "as_of": "known_date",
        "filingdate": "filing_date",
        "reportdate": "report_date",
        "periodenddate": "period_end_date",
        "fiscalperiodend": "fiscal_period_end",
        "revenueyoygrowth": "revenue_yoy_growth",
        "revenueqoqgrowth": "revenue_qoq_growth",
        "epsyoygrowth": "eps_yoy_growth",
        "ebitdayoygrowth": "ebitda_yoy_growth",
        "freecashflowgrowth": "free_cash_flow_growth",
        "grossmargin": "gross_margin",
        "operatingmargin": "operating_margin",
        "netmargin": "net_margin",
        "fcfmargin": "fcf_margin",
        "netdebttoebitda": "net_debt_to_ebitda",
        "interestcoverage": "interest_coverage",
        "currentratio": "current_ratio",
        "cashtodebt": "cash_to_debt",
        "cashrunwaymonths": "cash_runway_months",
        "sharecountgrowth": "share_count_growth",
        "pricetosales": "price_to_sales",
        "evtosales": "ev_to_sales",
        "evtoebitda": "ev_to_ebitda",
        "peratio": "pe_ratio",
        "fcfyield": "fcf_yield",
        "epsestimaterevision30d": "eps_estimate_revision_30d",
        "revenueestimaterevision30d": "revenue_estimate_revision_30d",
        "targetpricerevision": "target_price_revision",
        "guidanceupordown": "guidance_up_or_down",
        "analystratingchange": "analyst_rating_change",
    }
    compact = lowered.replace("_", "")
    return aliases.get(lowered, aliases.get(compact, lowered))


def _yoy_growth(values: pd.Series) -> pd.Series:
    previous = values.shift(4)
    return (values - previous) / previous.abs().where(previous.abs() > 1e-9)


def earnings_to_fundamentals(earnings: pd.DataFrame) -> pd.DataFrame:
    """Map MarketData earnings rows into point-in-time factor rows."""

    if earnings.empty:
        empty = pd.DataFrame(columns=["symbol", "date", "sector", "industry", "eps_yoy_growth", "guidance_up_or_down"])
        return attach_reference_cache_manifests(empty, earnings)
    out = earnings.sort_values(["symbol", "report_date"]).copy()
    out["known_date"] = pd.to_datetime(out["report_date"]).dt.normalize()
    out["date"] = out["known_date"]
    out["pit_source_date"] = out["known_date"]
    out["sector"] = "UNKNOWN"
    out["industry"] = "UNKNOWN"
    out["eps_yoy_growth"] = out.groupby("symbol")["reported_eps"].transform(_yoy_growth)
    # Earnings surprise is not guidance, but it is a real point-in-time
    # revision/reaction proxy. Keep the raw name too for report auditing.
    out["earnings_surprise_pct"] = out["surprise_eps_pct"]
    out["guidance_up_or_down"] = out["surprise_eps_pct"]
    keep = [
        "symbol",
        "date",
        "known_date",
        "pit_source_date",
        "sector",
        "industry",
        "eps_yoy_growth",
        "earnings_surprise_pct",
        "guidance_up_or_down",
        "reported_eps",
        "estimated_eps",
    ]
    result = out[[column for column in keep if column in out.columns]].dropna(subset=["date"])
    return attach_reference_cache_manifests(result, earnings)
