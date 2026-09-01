"""MarketData option-chain normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .exceptions import DataContractError, ProviderError


OPTION_CHAIN_COLUMNS: tuple[str, ...] = (
    "option_symbol",
    "underlying",
    "expiration",
    "side",
    "strike",
    "dte",
    "bid",
    "ask",
    "mid",
    "last",
    "volume",
    "open_interest",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "updated",
    "bid_ask_spread",
    "bid_ask_spread_pct_mid",
    "query_date",
    "source",
)

OPTION_FILTERS: dict[str, str] = {
    "expiration": "expiration",
    "dte": "dte",
    "side": "side",
    "delta": "delta",
    "strike_limit": "strikeLimit",
    "min_open_interest": "minOpenInterest",
    "min_volume": "minVolume",
    "max_bid_ask_spread_pct": "maxBidAskSpreadPct",
    "monthly": "monthly",
    "weekly": "weekly",
}


def normalize_option_filters(filters: Mapping[str, Any]) -> dict[str, Any]:
    """Validate Python-facing option filters and map them to provider names."""

    unknown = sorted(set(filters).difference(OPTION_FILTERS))
    if unknown:
        raise DataContractError(f"unsupported option-chain filters: {unknown}")
    if filters.get("side") not in {None, "call", "put"}:
        raise DataContractError("option-chain side must be 'call' or 'put'")
    params: dict[str, Any] = {}
    for key, value in filters.items():
        if value is None:
            continue
        provider_key = OPTION_FILTERS[key]
        params[provider_key] = str(value).lower() if isinstance(value, bool) else value
    return params


def parse_option_chain(
    symbol: str,
    payload: Any,
    *,
    query_date: str,
) -> pd.DataFrame:
    """Normalize a MarketData column-oriented option-chain payload."""

    if not isinstance(payload, Mapping):
        raise ProviderError(f"MarketData option chain for {symbol} is not an object")
    status = str(payload.get("s", "")).lower()
    if status == "no_data":
        return empty_option_chain()
    if status != "ok":
        message = str(payload.get("errmsg", "provider error"))
        raise ProviderError(f"MarketData option-chain error for {symbol}: {message}")

    option_symbols = payload.get("optionSymbol", [])
    if not isinstance(option_symbols, list):
        raise ProviderError(f"MarketData optionSymbol is not an array for {symbol}")
    if not option_symbols:
        return empty_option_chain()
    rows = len(option_symbols)
    aliases = {
        "underlying": "underlying",
        "expiration": "expiration",
        "side": "side",
        "strike": "strike",
        "dte": "dte",
        "bid": "bid",
        "ask": "ask",
        "mid": "mid",
        "last": "last",
        "volume": "volume",
        "open_interest": "openInterest",
        "iv": "iv",
        "delta": "delta",
        "gamma": "gamma",
        "theta": "theta",
        "vega": "vega",
        "updated": "updated",
    }
    out = pd.DataFrame({"option_symbol": pd.Series(option_symbols, dtype="string")})
    for target, source in aliases.items():
        values = payload.get(source)
        if isinstance(values, list) and len(values) == rows:
            out[target] = values
        else:
            out[target] = pd.NA

    out["underlying"] = (
        out["underlying"].fillna(symbol).astype("string").str.strip().str.upper()
    )
    out["side"] = out["side"].astype("string").str.strip().str.lower()
    numeric = (
        "strike",
        "dte",
        "bid",
        "ask",
        "mid",
        "last",
        "volume",
        "open_interest",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
    )
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    if out["mid"].isna().all():
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["bid_ask_spread"] = out["ask"] - out["bid"]
    out["bid_ask_spread_pct_mid"] = out["bid_ask_spread"] / out["mid"].replace(
        0, pd.NA
    )
    out["expiration"] = _provider_time(out["expiration"], normalize=True)
    out["updated"] = _provider_time(out["updated"], normalize=False)
    parsed_query = pd.to_datetime(query_date, errors="coerce")
    if pd.isna(parsed_query):
        raise DataContractError(f"invalid option-chain query date: {query_date!r}")
    out["query_date"] = pd.Timestamp(parsed_query).normalize()
    out["source"] = "marketdata.app"
    return out.loc[:, OPTION_CHAIN_COLUMNS].sort_values(
        ["underlying", "expiration", "side", "strike"],
        kind="stable",
    ).reset_index(drop=True)


def empty_option_chain() -> pd.DataFrame:
    """Return an empty option-chain table with stable columns."""

    return pd.DataFrame(columns=OPTION_CHAIN_COLUMNS)


def _provider_time(values: pd.Series, *, normalize: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        nonnull = numeric.dropna()
        unit = "ms" if not nonnull.empty and nonnull.abs().max() > 10_000_000_000 else "s"
        result = (
            pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )
    else:
        result = (
            pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )
    return result.dt.normalize() if normalize else result
