"""Event data provider interfaces and CSV fallback."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_system.data.cache import DataCache, attach_reference_cache_manifests
from quant_system.data.marketdata_earnings import MarketDataEarningsClient


class EventDataProvider(ABC):
    """Point-in-time event calendar interface."""

    @abstractmethod
    def get_earnings_calendar(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        """Return earnings calendar rows."""

    def get_major_events(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_macro_events(self, start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_corporate_actions(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame()


class CSVEventDataProvider(EventDataProvider):
    """CSV fallback for earnings and major events.

    Schema: symbol,event_date,event_type,known_date,event_risk_score,expected_move.
    known_date is mandatory for point-in-time research; if absent, event_date is
    used and a warning should be shown in reports.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data
        if not self.path.exists():
            self._data = pd.DataFrame(columns=["symbol", "event_date", "event_type", "known_date", "event_risk_score"])
            return self._data
        data = pd.read_csv(self.path)
        data["symbol"] = data["symbol"].astype(str).str.upper()
        data["event_date"] = pd.to_datetime(data["event_date"]).dt.normalize()
        if "known_date" not in data.columns:
            data["known_date"] = data["event_date"]
            data["pit_calendar_warning"] = "known_date_missing_used_event_date"
        else:
            data["pit_calendar_warning"] = ""
        data["known_date"] = pd.to_datetime(data["known_date"]).dt.normalize()
        self._data = data.sort_values(["symbol", "event_date"])
        return self._data

    def get_earnings_calendar(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        data = self._load()
        if data.empty:
            return data
        mask = data["symbol"].isin([s.upper() for s in symbols])
        mask &= data["event_type"].fillna("earnings").str.lower().eq("earnings")
        mask &= data["event_date"] >= pd.Timestamp(start)
        if end:
            mask &= data["event_date"] <= pd.Timestamp(end)
        return data.loc[mask].copy()


class MarketDataEventDataProvider(EventDataProvider):
    """MarketData.app event adapter.

    Uses the documented earnings endpoint for historical EPS reports and future
    projected earnings dates. Historical rows are treated conservatively:
    ``known_date`` defaults to ``event_date`` so the backtest does not assume a
    past calendar date was known before the source can prove it.
    """

    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self.earnings_client = MarketDataEarningsClient(cache_dir=cache_dir)
        self.runtime_client = MarketDataEventsRuntimeClient(cache_dir=cache_dir)

    def get_earnings_calendar(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        runtime = self.runtime_client.get_events(symbols, start, end)
        earnings = self.earnings_client.get_bulk_earnings(symbols, start, end)
        earnings_calendar = earnings_to_event_calendar(earnings)
        frames = [frame for frame in [runtime, earnings_calendar] if not frame.empty]
        if not frames:
            return attach_reference_cache_manifests(pd.DataFrame(), runtime, earnings_calendar)
        out = (
            pd.concat(frames, ignore_index=True, sort=False)
            .sort_values(["symbol", "event_date", "known_date"])
            .drop_duplicates(["symbol", "event_date", "event_type"], keep="first")
        )
        return attach_reference_cache_manifests(out, runtime, earnings_calendar)

    def get_major_events(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        return self.runtime_client.get_events(symbols, start, end)


class MarketDataEventsRuntimeClient:
    """Configurable MarketData.app events/calendar endpoint.

    Set ``MARKETDATA_EVENTS_ENDPOINT_TEMPLATE`` when a real corporate events
    endpoint is available. The parser accepts common event-calendar field names
    and preserves ``known_date`` when supplied; if only ``event_date`` exists,
    it marks the row with a PIT warning.
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
        self.endpoint_template = os.getenv("MARKETDATA_EVENTS_ENDPOINT_TEMPLATE", "")
        self.cache = DataCache(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.last_errors: dict[str, str] = {}

    def get_events(self, symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
        if not self.endpoint_template:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        manifest_sources: list[pd.DataFrame] = []
        self.last_errors = {}
        for symbol in symbols:
            try:
                frame = self.get_symbol_events(symbol, start, end)
            except Exception as exc:  # pragma: no cover - real API failure path
                self.last_errors[str(symbol).upper()] = str(exc)
                continue
            manifest_sources.append(frame)
            if not frame.empty:
                frames.append(frame)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return attach_reference_cache_manifests(out, *manifest_sources)

    def get_symbol_events(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError("MARKETDATA_TOKEN is not set for the events runtime endpoint.")
        symbol = symbol.upper()
        cache_cutoff = end or pd.Timestamp.now(tz="America/New_York").date().isoformat()
        cache_key = f"marketdata_runtime_events_v1_{symbol}_{start}_{cache_cutoff}"
        endpoint = self.endpoint_template.lstrip("/").format(symbol=symbol)
        request = {
            "provider": "marketdata.app",
            "dataset": "corporate_events",
            "base_url": self.base_url.rstrip("/"),
            "endpoint": f"/{endpoint}",
            "symbol": symbol,
            "start": start,
            "end": cache_cutoff,
        }
        cached = self.cache.read(cache_key, request=request)
        if cached is not None:
            return normalize_marketdata_events(symbol, cached, start, end)
        url = f"{self.base_url.rstrip('/')}/{endpoint}"
        params = {"from": start}
        if end:
            params["to"] = end
        response = requests.get(url, params=params, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout_seconds)
        response.raise_for_status()
        frame = normalize_marketdata_events(symbol, response.json(), start, end)
        self.cache.write(cache_key, frame, request=request)
        persisted = self.cache.read(cache_key, request=request)
        if persisted is None:  # pragma: no cover - defensive invariant
            raise RuntimeError(f"MarketData events cache write was not readable for {symbol}")
        return normalize_marketdata_events(symbol, persisted, start, end)


def earnings_to_event_calendar(earnings: pd.DataFrame) -> pd.DataFrame:
    """Map MarketData earnings rows into point-in-time event calendar rows."""

    if earnings.empty:
        return attach_reference_cache_manifests(pd.DataFrame(), earnings)
    frame = earnings.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["event_date"] = pd.to_datetime(frame["report_date"]).dt.normalize()
    # No lookahead: without point-in-time calendar snapshots, do not assume
    # historical earnings dates were known before report_date.
    frame["known_date"] = frame["event_date"]
    frame["event_type"] = "earnings"
    frame["event_risk_score"] = 50.0
    frame.loc[frame["surprise_eps_pct"].abs() >= 20, "event_risk_score"] = 70.0
    frame["expected_move"] = pd.NA
    keep = [
        "symbol",
        "event_date",
        "event_type",
        "known_date",
        "event_risk_score",
        "expected_move",
        "report_time",
        "reported_eps",
        "estimated_eps",
        "surprise_eps",
        "surprise_eps_pct",
    ]
    out = frame[[column for column in keep if column in frame.columns]]
    return attach_reference_cache_manifests(out, earnings)


def normalize_marketdata_events(symbol: str, payload: Any, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Normalize generic MarketData event payloads into PIT event rows."""

    raw = payload.copy() if isinstance(payload, pd.DataFrame) else _payload_to_frame(payload)
    if raw.empty:
        return attach_reference_cache_manifests(pd.DataFrame(), raw)
    raw = raw.rename(columns={column: _canonical_event_column(column) for column in raw.columns})
    out = pd.DataFrame()
    out["symbol"] = raw.get("symbol", symbol).astype(str).str.upper() if isinstance(raw.get("symbol"), pd.Series) else symbol.upper()
    date_col = next((col for col in ["event_date", "report_date", "date"] if col in raw.columns), None)
    if date_col is None:
        return attach_reference_cache_manifests(pd.DataFrame(), raw)
    out["event_date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.normalize()
    if "known_date" in raw.columns:
        out["known_date"] = pd.to_datetime(raw["known_date"], errors="coerce").dt.normalize()
        out["pit_calendar_warning"] = ""
    else:
        out["known_date"] = out["event_date"]
        out["pit_calendar_warning"] = "known_date_missing_used_event_date"
    out["event_type"] = raw.get("event_type", "major_event")
    if isinstance(out["event_type"], pd.Series):
        out["event_type"] = out["event_type"].fillna("major_event").astype(str).str.lower()
    for column in [
        "event_risk_score",
        "expected_move",
        "historical_earnings_gap_volatility",
        "recent_guidance_change",
        "dilution_event",
        "secondary_offering",
        "regulatory_event",
        "fda_event",
        "lockup_expiration",
        "debt_maturity",
        "macro_event_exposure",
    ]:
        if column in raw.columns:
            target = "FDA_event" if column == "fda_event" else column
            out[target] = raw[column]
    if "event_risk_score" not in out:
        out["event_risk_score"] = 50.0
    out = out.dropna(subset=["event_date"])
    if start:
        out = out[out["event_date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["event_date"] <= pd.Timestamp(end)]
    result = out.sort_values(["symbol", "event_date", "known_date"]).reset_index(drop=True)
    return attach_reference_cache_manifests(result, raw)


def _payload_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        if payload.get("s") == "no_data":
            return pd.DataFrame()
        for key in ("data", "results", "events", "calendar"):
            if key in payload:
                return _payload_to_frame(payload[key])
        arrays = {key: value for key, value in payload.items() if isinstance(value, list)}
        if arrays:
            length = max((len(value) for value in arrays.values()), default=0)
            if length:
                return pd.DataFrame({key: (value if len(value) == length else [pd.NA] * length) for key, value in arrays.items()})
        return pd.DataFrame([payload])
    return pd.DataFrame(payload)


def _canonical_event_column(column: str) -> str:
    lowered = str(column).strip().replace("-", "_").replace(" ", "_").lower()
    compact = lowered.replace("_", "")
    aliases = {
        "eventdate": "event_date",
        "reportdate": "report_date",
        "knowndate": "known_date",
        "asofdate": "known_date",
        "eventtype": "event_type",
        "eventriskscore": "event_risk_score",
        "expectedmove": "expected_move",
        "historicalearningsgapvolatility": "historical_earnings_gap_volatility",
        "recentguidancechange": "recent_guidance_change",
        "dilutionevent": "dilution_event",
        "secondaryoffering": "secondary_offering",
        "regulatoryevent": "regulatory_event",
        "fdaevent": "fda_event",
        "lockupexpiration": "lockup_expiration",
        "debtmaturity": "debt_maturity",
        "macroeventexposure": "macro_event_exposure",
    }
    return aliases.get(lowered, aliases.get(compact, lowered))
