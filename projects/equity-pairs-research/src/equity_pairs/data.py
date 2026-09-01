"""Market and macro data boundaries for the research pipeline.

Equity prices come from the suite's shared MarketData client and are restricted
to confirmed daily bars. FRED remains a separate macroeconomic source because
it supplies observations rather than tradable market prices. All public date
ranges use ``[start, end)`` semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from quant_marketdata import MarketDataClient, wide_close


SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_FRED_KEY_PATTERN = re.compile(r"[a-z0-9]{32}")
_FRED_SERIES_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_SENSITIVE_METADATA_PATTERN = re.compile(r"(?:api.?key|token|secret|credential)", re.I)


class DataDownloadError(RuntimeError):
    """Raised when a source cannot provide a usable response."""


class InvalidFREDApiKey(ValueError):
    """Raised when the environment does not contain a valid FRED API key."""


def normalize_market_symbol(symbol: Any) -> str:
    """Normalize a symbol without applying vendor-specific rewrites."""

    if symbol is None or symbol is pd.NA:
        raise ValueError("symbol must be a non-empty string")
    try:
        missing = pd.isna(symbol)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ValueError("symbol must be a non-empty string")
    normalized = str(symbol).strip().upper()
    if not normalized or normalized in {"NAN", "NONE"}:
        raise ValueError("symbol must be a non-empty string")
    if any(character.isspace() for character in normalized):
        raise ValueError("symbol may not contain whitespace")
    return normalized


def normalize_market_symbols(symbols: Iterable[Any] | str) -> tuple[str, ...]:
    """Normalize symbols and remove duplicates while preserving order."""

    values: Iterable[Any] = [symbols] if isinstance(symbols, str) else symbols
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = normalize_market_symbol(value)
        if symbol not in seen:
            result.append(symbol)
            seen.add(symbol)
    if not result:
        raise ValueError("at least one symbol is required")
    return tuple(result)


def _date_bounds(start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("start and end must be valid dates")
    if start_ts >= end_ts:
        raise ValueError("start must be earlier than end")
    return start_ts, end_ts


def _safe_get(
    session: requests.Session | Any,
    url: str,
    *,
    params: Mapping[str, Any] | None,
    timeout: float,
    max_retries: int,
    backoff_factor: float,
    request_name: str,
    sleep_func: Callable[[float], None] = time.sleep,
) -> Any:
    """Issue a bounded retrying GET without exposing URLs or credentials."""

    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            if attempt + 1 == attempts:
                raise DataDownloadError(f"{request_name} failed after {attempts} attempts") from exc
        else:
            status = int(getattr(response, "status_code", 200))
            if 200 <= status < 300:
                return response
            if status not in RETRYABLE_HTTP_STATUS or attempt + 1 == attempts:
                raise DataDownloadError(f"{request_name} failed with HTTP status {status}")
        if backoff_factor > 0:
            sleep_func(backoff_factor * (2**attempt))
    raise AssertionError("unreachable")


def fetch_sp500_constituents(
    *,
    session: requests.Session | Any | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    sleep_func: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Fetch the current S&P 500 membership and GICS classification snapshot.

    This is a current-universe research convenience. It is not point-in-time
    membership and therefore introduces survivorship bias in historical tests.
    """

    http = session if session is not None else requests.Session()
    response = _safe_get(
        http,
        SP500_WIKIPEDIA_URL,
        params=None,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        request_name="S&P 500 constituent snapshot",
        sleep_func=sleep_func,
    )
    html = getattr(response, "text", "")
    if not html and getattr(response, "content", b""):
        html = response.content.decode("utf-8", errors="replace")
    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, ImportError) as exc:
        raise DataDownloadError("S&P 500 constituent page did not contain a readable table") from exc
    required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "CIK"}
    table = next((frame for frame in tables if required.issubset(frame.columns)), None)
    if table is None:
        raise DataDownloadError("S&P 500 constituent table has an unexpected schema")
    result = table.loc[:, list(required)].rename(
        columns={
            "Symbol": "source_ticker",
            "Security": "security",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "subindustry",
            "CIK": "cik",
        }
    )
    result.insert(0, "ticker", result["source_ticker"].map(normalize_market_symbol))
    result["source_ticker"] = result["source_ticker"].astype(str).str.strip()
    result["cik"] = result["cik"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(10)
    result = result.drop_duplicates("ticker").reset_index(drop=True)
    if result.empty:
        raise DataDownloadError("S&P 500 constituent snapshot is empty")
    return result[["ticker", "source_ticker", "security", "sector", "subindustry", "cik"]]


def download_confirmed_close(
    symbols: Iterable[Any] | str,
    start: Any,
    end: Any,
    *,
    client: MarketDataClient | Any | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a wide close panel backed only by confirmed MarketData bars."""

    requested = normalize_market_symbols(symbols)
    start_ts, end_ts = _date_bounds(start, end)
    market_client = client if client is not None else MarketDataClient()
    bars = market_client.get_bulk_daily_bars(
        list(requested),
        start_ts.strftime("%Y-%m-%d"),
        end_ts.strftime("%Y-%m-%d"),
        finality="confirmed",
        refresh=refresh,
    )
    if not isinstance(bars, pd.DataFrame):
        raise DataDownloadError("MarketData bulk daily bars returned a non-tabular result")
    required = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "finality",
    }
    missing = required.difference(bars.columns)
    if missing:
        raise DataDownloadError(
            "MarketData bulk daily bars are missing required columns: " + ", ".join(sorted(missing))
        )
    clean = bars.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.normalize()
    clean["symbol"] = clean["symbol"].map(normalize_market_symbol)
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    finality = clean["finality"].astype(str).str.lower()
    if not finality.eq("confirmed").all():
        raise DataDownloadError("MarketData returned non-confirmed rows for a confirmed-bar request")
    clean = clean.loc[
        clean["date"].notna()
        & clean["symbol"].isin(requested)
        & clean["date"].ge(start_ts)
        & clean["date"].lt(end_ts)
    ]
    clean = clean.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    if clean.empty or clean["close"].notna().sum() == 0:
        raise DataDownloadError("MarketData returned no confirmed closes for the requested window")
    panel = wide_close(clean).sort_index().reindex(columns=list(requested))
    panel.index = pd.DatetimeIndex(panel.index, name="date")
    panel.attrs["source"] = "marketdata"
    panel.attrs["finality"] = "confirmed"
    panel.attrs["failed_tickers"] = [symbol for symbol in requested if panel[symbol].notna().sum() == 0]
    return panel


def snapshot_metadata_path(path: str | Path) -> Path:
    """Return the JSON sidecar path used for a tabular snapshot."""

    snapshot = Path(path)
    return snapshot.with_suffix(snapshot.suffix + ".metadata.json")


def _contains_sensitive_metadata(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _SENSITIVE_METADATA_PATTERN.search(str(key))
            or _contains_sensitive_metadata(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_metadata(item) for item in value)
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_timeseries_snapshot(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save a date-indexed CSV with a checksum-protected sidecar."""

    supplied = dict(metadata or {})
    if _contains_sensitive_metadata(supplied):
        raise ValueError("snapshot metadata may not contain credentials or secrets")
    snapshot = Path(path)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = snapshot.with_suffix(snapshot.suffix + ".tmp")
    sidecar = snapshot_metadata_path(snapshot)
    sidecar_temp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    prepared = frame.copy()
    prepared.index = pd.DatetimeIndex(pd.to_datetime(prepared.index), name="date")
    prepared.to_csv(csv_temp)
    csv_temp.replace(snapshot)
    document = {
        **supplied,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(prepared)),
        "columns": [str(column) for column in prepared.columns],
        "sha256": _sha256(snapshot),
    }
    sidecar_temp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar_temp.replace(sidecar)


def load_timeseries_snapshot(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and verify a date-indexed CSV snapshot and metadata sidecar."""

    snapshot = Path(path)
    sidecar = snapshot_metadata_path(snapshot)
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise DataDownloadError("snapshot metadata is missing or invalid") from exc
    expected = metadata.get("sha256") if isinstance(metadata, Mapping) else None
    if not isinstance(expected, str) or _sha256(snapshot) != expected:
        raise DataDownloadError("snapshot checksum validation failed")
    frame = pd.read_csv(snapshot, index_col=0, parse_dates=[0])
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame, dict(metadata)


def validate_fred_api_key(value: Any) -> str:
    """Validate a FRED key without including it in an exception message."""

    if not isinstance(value, str):
        raise InvalidFREDApiKey("FRED_API_KEY is missing or invalid")
    candidate = value.strip()
    if not _FRED_KEY_PATTERN.fullmatch(candidate):
        raise InvalidFREDApiKey("FRED_API_KEY is missing or invalid")
    return candidate


def _fred_api_key_from_env() -> str:
    return validate_fred_api_key(os.environ.get("FRED_API_KEY"))


def download_fred_observations(
    series: Mapping[str, str],
    start: Any,
    end: Any,
    *,
    session: requests.Session | Any | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    cache_path: str | Path | None = None,
    refresh: bool = False,
    sleep_func: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download FRED observations for ``{output_name: series_id}`` mappings.

    The API key is read only from ``FRED_API_KEY``. A matching local snapshot
    can be loaded without a key or network access.
    """

    if not isinstance(series, Mapping) or not series:
        raise ValueError("series must be a non-empty mapping of name to FRED series ID")
    start_ts, end_ts = _date_bounds(start, end)
    names: list[str] = []
    identifiers: list[str] = []
    for output_name, series_id in series.items():
        name = str(output_name).strip()
        identifier = str(series_id).strip()
        if not name:
            raise ValueError("FRED output column names must be non-empty")
        if not _FRED_SERIES_PATTERN.fullmatch(identifier):
            raise ValueError(f"invalid FRED series ID for output column {name!r}")
        names.append(name)
        identifiers.append(identifier)

    request_metadata = {
        "source": "fred_current_vintage",
        "start": start_ts.strftime("%Y-%m-%d"),
        "end_exclusive": end_ts.strftime("%Y-%m-%d"),
        "series": dict(zip(names, identifiers, strict=True)),
    }
    if cache_path is not None and not refresh:
        try:
            cached, metadata = load_timeseries_snapshot(cache_path)
        except (FileNotFoundError, OSError, DataDownloadError, ValueError):
            pass
        else:
            if all(metadata.get(key) == value for key, value in request_metadata.items()):
                return cached.reindex(columns=names).loc[
                    lambda frame: (frame.index >= start_ts) & (frame.index < end_ts)
                ]

    key = _fred_api_key_from_env()
    http = session if session is not None else requests.Session()
    end_inclusive = end_ts - pd.Timedelta(days=1)
    downloaded: list[pd.Series] = []
    for name, identifier in zip(names, identifiers, strict=True):
        response = _safe_get(
            http,
            FRED_OBSERVATIONS_URL,
            params={
                "series_id": identifier,
                "api_key": key,
                "file_type": "json",
                "observation_start": start_ts.strftime("%Y-%m-%d"),
                "observation_end": end_inclusive.strftime("%Y-%m-%d"),
            },
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            request_name=f"FRED series {identifier}",
            sleep_func=sleep_func,
        )
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            raise DataDownloadError(f"FRED series {identifier} returned invalid JSON") from exc
        observations = payload.get("observations") if isinstance(payload, Mapping) else None
        if not isinstance(observations, list):
            raise DataDownloadError(f"FRED series {identifier} response has an unexpected schema")
        dates: list[Any] = []
        values: list[Any] = []
        for observation in observations:
            if isinstance(observation, Mapping):
                dates.append(observation.get("date"))
                value = observation.get("value")
                values.append(np.nan if value in {None, ".", ""} else value)
        index = pd.to_datetime(dates, errors="coerce")
        values_series = pd.Series(
            pd.to_numeric(values, errors="coerce"), index=index, name=name, dtype=float
        )
        values_series = values_series.loc[~values_series.index.isna()]
        values_series.index = pd.DatetimeIndex(values_series.index).normalize()
        values_series = values_series.groupby(level=0, sort=True).last()
        downloaded.append(values_series.loc[
            (values_series.index >= start_ts) & (values_series.index < end_ts)
        ])

    result = pd.concat(downloaded, axis=1).sort_index().reindex(columns=names)
    result.index = pd.DatetimeIndex(result.index, name="date")
    if cache_path is not None:
        save_timeseries_snapshot(result, cache_path, metadata=request_metadata)
    return result


download_fred_series = download_fred_observations


__all__ = [
    "DataDownloadError",
    "InvalidFREDApiKey",
    "download_confirmed_close",
    "download_fred_observations",
    "download_fred_series",
    "fetch_sp500_constituents",
    "load_timeseries_snapshot",
    "normalize_market_symbol",
    "normalize_market_symbols",
    "save_timeseries_snapshot",
    "snapshot_metadata_path",
    "validate_fred_api_key",
]
