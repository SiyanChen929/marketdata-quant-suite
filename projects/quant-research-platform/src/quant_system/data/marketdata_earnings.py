"""MarketData.app earnings endpoint adapter."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_system.data.cache import DataCache, attach_reference_cache_manifests
from quant_system.data.marketdata_provider import _normalize_date


class MarketDataEarningsClient:
    """Small cached client for the MarketData.app stock earnings endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache",
        retries: int | None = None,
        pause_seconds: float = 0.25,
        timeout_seconds: float | None = None,
    ) -> None:
        if api_key is not None:
            raise ValueError("Pass MarketData credentials through MARKETDATA_TOKEN, not constructor arguments")
        self.api_key = os.getenv("MARKETDATA_TOKEN")
        self.base_url = base_url or os.getenv("MARKETDATA_BASE_URL", "https://api.marketdata.app/v1")
        self.endpoint_template = os.getenv("MARKETDATA_EARNINGS_ENDPOINT_TEMPLATE", "/stocks/earnings/{symbol}/")
        self.cache = DataCache(cache_dir)
        self.retries = retries if retries is not None else int(os.getenv("MARKETDATA_EARNINGS_RETRIES", "1"))
        self.pause_seconds = pause_seconds
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else float(os.getenv("MARKETDATA_EARNINGS_TIMEOUT", "8"))
        self.last_errors: dict[str, str] = {}

    def get_bulk_earnings(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        """Fetch earnings rows for many symbols, continuing past symbol failures."""

        frames: list[pd.DataFrame] = []
        manifest_sources: list[pd.DataFrame] = []
        self.last_errors = {}
        for symbol in symbols:
            try:
                frame = self.get_earnings(symbol, start, end)
            except Exception as exc:  # pragma: no cover - real API failure path
                self.last_errors[str(symbol).upper()] = str(exc)
                continue
            manifest_sources.append(frame)
            if not frame.empty:
                frames.append(frame)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return attach_reference_cache_manifests(out, *manifest_sources)

    def get_earnings(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        """Fetch and normalize earnings rows for one symbol."""

        if not self.api_key:
            raise RuntimeError("MARKETDATA_TOKEN is not set. Use the CSV demo or export the token in your shell.")
        symbol = symbol.upper()
        cache_cutoff = end or pd.Timestamp.now(tz="America/New_York").date().isoformat()
        cache_key = f"marketdata_earnings_{symbol}_{start}_{cache_cutoff}"
        endpoint = self.endpoint_template.lstrip("/").format(symbol=symbol)
        request = {
            "provider": "marketdata.app",
            "dataset": "stock_earnings",
            "base_url": self.base_url.rstrip("/"),
            "endpoint": f"/{endpoint}",
            "symbol": symbol,
            "start": start,
            "end": cache_cutoff,
        }
        cached = self.cache.read(cache_key, request=request)
        if cached is not None:
            return _normalize_earnings_frame(cached)
        url = f"{self.base_url.rstrip('/')}/{endpoint}"
        params: dict[str, str] = {"from": start}
        if end:
            params["to"] = end
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
                if response.status_code == 429:
                    time.sleep(self.pause_seconds * (attempt + 2))
                    continue
                response.raise_for_status()
                frame = earnings_payload_to_frame(symbol, response.json())
                self.cache.write(cache_key, frame, request=request)
                persisted = self.cache.read(cache_key, request=request)
                if persisted is None:  # pragma: no cover - defensive invariant
                    raise RuntimeError(f"MarketData earnings cache write was not readable for {symbol}")
                return _normalize_earnings_frame(persisted)
            except Exception as exc:  # pragma: no cover - real API failure path
                last_error = exc
                time.sleep(self.pause_seconds * (attempt + 1))
        raise RuntimeError(f"MarketData earnings request failed for {symbol}: {last_error}")


def earnings_payload_to_frame(symbol: str, payload: Any) -> pd.DataFrame:
    """Normalize MarketData.app earnings JSON or row-shaped mocks."""

    columns = [
        "symbol",
        "fiscal_year",
        "fiscal_quarter",
        "fiscal_period_end",
        "report_date",
        "report_time",
        "currency",
        "reported_eps",
        "estimated_eps",
        "surprise_eps",
        "surprise_eps_pct",
        "updated",
    ]
    if isinstance(payload, dict):
        status = payload.get("s")
        if status == "no_data":
            return pd.DataFrame(columns=columns)
        if status == "error":
            raise ValueError(f"MarketData.app earnings error for {symbol}: {payload.get('errmsg', payload)}")
        if {"reportDate", "reportedEPS", "estimatedEPS"}.intersection(payload):
            length = max(len(payload.get("reportDate", [])), len(payload.get("date", [])), len(payload.get("reportedEPS", [])))
            frame = pd.DataFrame(
                {
                    "symbol": _array(payload.get("symbol"), length, symbol.upper()),
                    "fiscal_year": _array(payload.get("fiscalYear"), length),
                    "fiscal_quarter": _array(payload.get("fiscalQuarter"), length),
                    "fiscal_period_end": _normalize_date(pd.Series(_array(payload.get("date"), length))),
                    "report_date": _normalize_date(pd.Series(_array(payload.get("reportDate"), length))),
                    "report_time": _array(payload.get("reportTime"), length),
                    "currency": _array(payload.get("currency"), length),
                    "reported_eps": _array(payload.get("reportedEPS"), length),
                    "estimated_eps": _array(payload.get("estimatedEPS"), length),
                    "surprise_eps": _array(payload.get("surpriseEPS"), length),
                    "surprise_eps_pct": _array(payload.get("surpriseEPSpct"), length),
                    "updated": _normalize_date(pd.Series(_array(payload.get("updated"), length))),
                }
            )
            return _normalize_earnings_frame(frame)
        for key in ("data", "results", "earnings"):
            if key in payload:
                payload = payload[key]
                break
    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rename = {
        "fiscalYear": "fiscal_year",
        "fiscalQuarter": "fiscal_quarter",
        "date": "fiscal_period_end",
        "reportDate": "report_date",
        "reportTime": "report_time",
        "reportedEPS": "reported_eps",
        "estimatedEPS": "estimated_eps",
        "surpriseEPS": "surprise_eps",
        "surpriseEPSpct": "surprise_eps_pct",
    }
    return _normalize_earnings_frame(frame.rename(columns=rename))


def _normalize_earnings_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    out["symbol"] = out.get("symbol", "").astype(str).str.upper()
    for column in ("fiscal_period_end", "report_date", "updated"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    for column in ("reported_eps", "estimated_eps", "surprise_eps", "surprise_eps_pct"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["symbol", "report_date", "fiscal_period_end"], na_position="last")


def _array(value: Any, length: int, default: Any = pd.NA) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return [default] * length
    return [value] * length
