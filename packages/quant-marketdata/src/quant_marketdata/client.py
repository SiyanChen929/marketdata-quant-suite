"""MarketData.app daily-bar client with deterministic exact-request caching."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Callable
from datetime import datetime, time as wall_time
import os
import time
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .exceptions import (
    CredentialError,
    DataContractError,
    DataUnavailableError,
    ProviderError,
)
from .options import normalize_option_filters, parse_option_chain
from .schema import Finality, empty_bars, normalize_bars, normalize_symbols, validate_finality
from .store import MarketDataStore


class HTTPSession(Protocol):
    """Small injectable HTTP surface used by the client."""

    def get(self, url: str, **kwargs: Any) -> Any: ...


class MarketDataClient:
    """MarketData.app client backed by an external :class:`MarketDataStore`.

    The API token is read only from ``MARKETDATA_TOKEN`` and only when a network
    request is required.  It is never written to a cache key or manifest.
    """

    def __init__(
        self,
        *,
        store: MarketDataStore | None = None,
        session: HTTPSession | None = None,
        base_url: str = "https://api.marketdata.app/v1",
        timeout: float = 30.0,
        retries: int = 2,
        backoff_seconds: float = 0.25,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be at least 1")
        if timeout <= 0 or backoff_seconds < 0:
            raise ValueError("timeout must be positive and backoff_seconds non-negative")
        self.store = store or MarketDataStore()
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff_seconds = float(backoff_seconds)
        self._now = now or (lambda: datetime.now(ZoneInfo("America/New_York")))

    def get_daily_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        finality: str = "confirmed",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return one symbol's inclusive daily bars in canonical long form."""

        return self.get_stock_bars(
            symbol,
            start,
            end,
            resolution="D",
            finality=finality,
            refresh=refresh,
        )

    def get_stock_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        resolution: str = "D",
        finality: str = "confirmed",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return canonical stock bars for a MarketData candle resolution.

        Daily, weekly, and monthly bars carry normalized session dates.
        Intraday resolutions retain New York-local timestamps in the canonical
        ``date`` field, allowing the shared store to support a later intraday
        adapter without changing the public schema.
        """

        selected = validate_finality(finality)
        normalized_symbol = _one_symbol(symbol)
        normalized_resolution = _resolution(resolution)
        start_date, end_date = _date_range(start, end)
        request = _request_payload(
            normalized_symbol,
            start_date,
            end_date,
            normalized_resolution,
            selected,
        )
        request_key = self.store.request_hash(request)
        if not refresh:
            cached = self.store.read_request_cache(request_key, finality=selected)
            if cached is not None:
                return cached

        payload = self._request_stock_payload(
            normalized_symbol,
            start_date,
            end_date,
            normalized_resolution,
        )
        bars = _payload_to_bars(
            payload,
            normalized_symbol,
            normalized_resolution,
            selected,
        )
        _guard_confirmed_daily_cutoff(
            bars,
            resolution=normalized_resolution,
            finality=selected,
            now=self._now(),
        )
        self.store.write_request_cache(
            request_key,
            bars,
            request=request,
            finality=selected,
        )
        if not bars.empty:
            self.store.write_bars(
                bars,
                finality=selected,
                resolution=normalized_resolution,
            )
        return bars

    def get_bulk_daily_bars(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        finality: str = "confirmed",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch multiple symbols using the same stable single-symbol contract."""

        normalized = normalize_symbols(symbols)
        if not normalized:
            return empty_bars()
        frames = [
            self.get_daily_bars(
                symbol,
                start,
                end,
                finality=finality,
                refresh=refresh,
            )
            for symbol in normalized
        ]
        nonempty = [frame for frame in frames if not frame.empty]
        if not nonempty:
            return empty_bars()
        return normalize_bars(pd.concat(nonempty, ignore_index=True))

    def get_option_chain(
        self,
        symbol: str,
        date: str | None = None,
        **filters: Any,
    ) -> pd.DataFrame:
        """Return a normalized current or historical MarketData option chain.

        Supported filters are ``expiration``, ``dte``, ``side``, ``delta``,
        ``strike_limit``, ``min_open_interest``, ``min_volume``,
        ``max_bid_ask_spread_pct``, ``monthly``, and ``weekly``.  Set
        ``refresh=True`` to bypass the exact-request cache.
        """

        normalized_symbol = _one_symbol(symbol)
        refresh = bool(filters.pop("refresh", False))
        params = normalize_option_filters(filters)
        query_date = _option_query_date(date, self._now())
        if date is not None:
            params["date"] = query_date
        request = {
            "provider": "marketdata.app",
            "dataset": "options/chain",
            "symbol": normalized_symbol,
            "query_date": query_date,
            "params": params,
            "schema_version": "1.0",
        }
        request_key = self.store.request_hash(request)
        if not refresh:
            cached = self.store.read_table_cache(request_key, namespace="options")
            if cached is not None:
                return cached

        url = f"{self.base_url}/options/chain/{normalized_symbol}/"
        payload = self._request_json(
            url,
            params=params,
            description=f"option chain for {normalized_symbol} on {query_date}",
        )
        chain = parse_option_chain(
            normalized_symbol,
            payload,
            query_date=query_date,
        )
        self.store.write_table_cache(
            request_key,
            chain,
            request=request,
            namespace="options",
        )
        return chain

    def _request_stock_payload(
        self,
        symbol: str,
        start: str,
        end: str,
        resolution: str,
    ) -> Any:
        url = f"{self.base_url}/stocks/candles/{resolution}/{symbol}/"
        params = {
            "from": start,
            "to": end,
            "adjustsplits": "true",
            "adjustdividends": "true",
        }
        return self._request_json(
            url,
            params=params,
            description=f"stock bars for {symbol} in {start}..{end}",
        )

    def _request_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        description: str,
    ) -> Any:
        token = os.getenv("MARKETDATA_TOKEN", "").strip()
        if not token:
            raise CredentialError(
                "MARKETDATA_TOKEN is not set and this exact request is not cached"
            )
        headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                status_code = int(getattr(response, "status_code", 200))
                if status_code == 404:
                    raise DataUnavailableError(f"MarketData has no data for {description}")
                if status_code == 429 or status_code >= 500:
                    raise ProviderError(f"MarketData HTTP {status_code} for {description}")
                response.raise_for_status()
                return response.json()
            except DataUnavailableError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries and self.backoff_seconds:
                    time.sleep(self.backoff_seconds * (attempt + 1))
        raise ProviderError(
            f"MarketData request failed for {description}: {type(last_error).__name__}"
        ) from last_error


def _request_payload(
    symbol: str,
    start: str,
    end: str,
    resolution: str,
    finality: Finality,
) -> dict[str, str]:
    return {
        "provider": "marketdata.app",
        "dataset": "stocks/candles",
        "resolution": resolution,
        "symbol": symbol,
        "start": start,
        "end": end,
        "adjustsplits": "true",
        "adjustdividends": "true",
        "finality": finality,
        "schema_version": "1.0",
    }


def _payload_to_bars(
    payload: Any,
    symbol: str,
    resolution: str,
    finality: Finality,
) -> pd.DataFrame:
    if isinstance(payload, Mapping):
        status = str(payload.get("s", "")).lower()
        if status == "no_data":
            return _empty_for(finality)
        if status == "error":
            message = str(payload.get("errmsg", "provider error"))
            raise ProviderError(f"MarketData error for {symbol}: {message}")
        required = {"o", "h", "l", "c", "v", "t"}
        if required.issubset(payload):
            lengths = {len(payload[field]) for field in required}
            if len(lengths) != 1:
                raise ProviderError(f"MarketData returned unequal candle arrays for {symbol}")
            frame = pd.DataFrame(
                {
                    "date": _marketdata_dates(
                        pd.Series(payload["t"]),
                        normalize_session=resolution in {"D", "W", "M"},
                    ),
                    "symbol": symbol,
                    "open": payload["o"],
                    "high": payload["h"],
                    "low": payload["l"],
                    "close": payload["c"],
                    "volume": payload["v"],
                    "source": "marketdata.app",
                    "finality": finality,
                }
            )
            return normalize_bars(frame, finality=finality)
        for key in ("data", "bars", "results", "candles"):
            if key in payload:
                payload = payload[key]
                break

    frame = pd.DataFrame(payload)
    if frame.empty:
        return _empty_for(finality)
    aliases = {
        "date": ("date", "datetime", "timestamp", "time", "t"),
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c"),
        "volume": ("volume", "v"),
    }
    canonical: dict[str, Any] = {}
    for name, choices in aliases.items():
        source = next((choice for choice in choices if choice in frame.columns), None)
        if source is None:
            raise ProviderError(
                f"MarketData response for {symbol} is missing {name}; "
                f"columns={list(frame.columns)}"
            )
        canonical[name] = frame[source]
    canonical["date"] = _marketdata_dates(
        pd.Series(canonical["date"]),
        normalize_session=resolution in {"D", "W", "M"},
    )
    canonical["symbol"] = frame["symbol"] if "symbol" in frame else symbol
    canonical["source"] = "marketdata.app"
    canonical["finality"] = finality
    return normalize_bars(pd.DataFrame(canonical), finality=finality)


def _marketdata_dates(
    values: pd.Series,
    *,
    normalize_session: bool,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not values.empty and numeric.notna().all():
        unit = "ms" if numeric.abs().max() > 10_000_000_000 else "s"
        dates = (
            pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )
    else:
        dates = (
            pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )
    return dates.dt.normalize() if normalize_session else dates


def _resolution(value: str) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "d": "D",
        "1d": "D",
        "daily": "D",
        "w": "W",
        "1w": "W",
        "weekly": "W",
        "m": "M",
        "1mo": "M",
        "monthly": "M",
        "1": "1",
        "1min": "1",
        "5": "5",
        "5min": "5",
        "15": "15",
        "15min": "15",
        "30": "30",
        "30min": "30",
        "h": "H",
        "1h": "H",
        "60min": "H",
    }
    if normalized not in mapping:
        raise DataContractError(f"unsupported MarketData candle resolution: {value!r}")
    return mapping[normalized]


def _guard_confirmed_daily_cutoff(
    bars: pd.DataFrame,
    *,
    resolution: str,
    finality: Finality,
    now: datetime,
) -> None:
    if bars.empty or resolution != "D" or finality != "confirmed":
        return
    eastern = ZoneInfo("America/New_York")
    current = now.replace(tzinfo=eastern) if now.tzinfo is None else now.astimezone(eastern)
    if current.weekday() >= 5 or current.time() >= wall_time(16, 15):
        return
    if pd.to_datetime(bars["date"]).dt.date.eq(current.date()).any():
        raise DataUnavailableError(
            "today's New York daily candle remains provisional until 16:15 ET"
        )


def _empty_for(finality: Finality) -> pd.DataFrame:
    frame = empty_bars()
    frame["finality"] = frame["finality"].astype("string")
    return frame


def _one_symbol(value: str) -> str:
    normalized = normalize_symbols(value)
    if not normalized:
        raise DataContractError("symbol cannot be empty")
    return normalized[0]


def _date_range(start: str, end: str) -> tuple[str, str]:
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise DataContractError(f"invalid date range: {start!r}..{end!r}")
    start_date = pd.Timestamp(start_ts).date()
    end_date = pd.Timestamp(end_ts).date()
    if start_date > end_date:
        raise DataContractError("start date is after end date")
    return str(start_date), str(end_date)


def _option_query_date(value: str | None, now: datetime) -> str:
    if value is not None:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise DataContractError(f"invalid option-chain date: {value!r}")
        return str(pd.Timestamp(parsed).date())
    eastern = ZoneInfo("America/New_York")
    current = now.replace(tzinfo=eastern) if now.tzinfo is None else now.astimezone(eastern)
    return str(current.date())
