"""Option chain provider abstractions and liquidity validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from quant_marketdata import MarketDataClient, MarketDataStore



OPTION_CHAIN_COLUMNS = [
    "option_symbol",
    "symbol",
    "quote_date",
    "chain_quote_source",
    "expiration",
    "option_type",
    "strike",
    "bid",
    "ask",
    "mid",
    "last",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
]


class OptionChainProvider(ABC):
    """Abstract option-chain provider."""

    @abstractmethod
    def get_chain(self, symbol: str, quote_date: str) -> pd.DataFrame:
        """Return one symbol's option chain for a quote date."""

    def get_bulk_chains(self, symbols: list[str], quote_date: str) -> pd.DataFrame:
        frames = [self.get_chain(symbol, quote_date) for symbol in symbols]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OPTION_CHAIN_COLUMNS)


class CSVOptionChainProvider(OptionChainProvider):
    """CSV fallback for option chains.

    Required columns: symbol, quote_date, expiration, option_type, strike, bid,
    ask. Greeks, IV, volume, and open interest are optional but recommended.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data
        if not self.path.exists():
            self._data = pd.DataFrame(columns=OPTION_CHAIN_COLUMNS)
            return self._data
        self._data = validate_option_chain(pd.read_csv(self.path))
        return self._data

    def get_chain(self, symbol: str, quote_date: str) -> pd.DataFrame:
        data = self._load()
        if data.empty:
            return data.copy()
        quote_ts = pd.Timestamp(quote_date).normalize()
        mask = data["symbol"].eq(symbol.upper()) & data["quote_date"].eq(quote_ts)
        return data.loc[mask].copy().reset_index(drop=True)


class MarketDataOptionChainProvider(OptionChainProvider):
    """MarketData.app option-chain adapter.

    Requests and exact-response caching are delegated to the suite's shared
    MarketData client before normalization into the research schema.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: str | Path = "data/cache",
        dte: int | None = None,
        side: str | None = "call",
        strike_limit: int | None = 80,
        min_open_interest: int | None = None,
        min_volume: int | None = None,
        max_bid_ask_spread_pct: float | None = None,
    ) -> None:
        if api_key is not None:
            raise ValueError("Pass MarketData credentials through MARKETDATA_TOKEN, not constructor arguments")
        store = MarketDataStore() if str(cache_dir) == "data/cache" else MarketDataStore(root=cache_dir)
        self.client = MarketDataClient(store=store)
        self.dte = dte
        self.side = side
        self.strike_limit = strike_limit
        self.min_open_interest = min_open_interest
        self.min_volume = min_volume
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct

    def get_chain(self, symbol: str, quote_date: str) -> pd.DataFrame:
        """Return one normalized MarketData option-chain snapshot."""

        symbol = str(symbol).upper()
        quote = pd.Timestamp(quote_date).normalize().date().isoformat()
        raw = _fetch_marketdata_option_chain(
            symbol,
            client=self.client,
            date=quote,
            dte=self.dte,
            side=self.side,
            strike_limit=self.strike_limit,
            min_open_interest=self.min_open_interest,
            min_volume=self.min_volume,
            max_bid_ask_spread_pct=self.max_bid_ask_spread_pct,
        )
        source = "historical_marketdata"
        normalized = normalize_marketdata_option_chain(raw, quote_date=quote)
        if not normalized.empty:
            normalized["chain_quote_source"] = source
        return normalized


def _fetch_marketdata_option_chain(
    symbol: str,
    *,
    client: MarketDataClient | None = None,
    date: str | None = None,
    dte: int | None = None,
    side: str | None = None,
    strike_limit: int | None = None,
    min_open_interest: int | None = None,
    min_volume: int | None = None,
    max_bid_ask_spread_pct: float | None = None,
) -> pd.DataFrame:
    """Fetch one option chain through the suite's shared MarketData gateway."""

    if side not in {None, "call", "put"}:
        raise ValueError("side must be 'call', 'put', or None")
    gateway = client or MarketDataClient()
    return gateway.get_option_chain(
        symbol,
        date=date,
        dte=dte,
        side=side,
        strike_limit=strike_limit,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        max_bid_ask_spread_pct=max_bid_ask_spread_pct,
    )


def normalize_marketdata_option_chain(frame: pd.DataFrame, quote_date: str | None = None) -> pd.DataFrame:
    """Normalize rows returned by the native MarketData option-chain parser."""

    if frame.empty:
        return pd.DataFrame(columns=OPTION_CHAIN_COLUMNS)
    out = frame.copy()
    rename = {
        "underlying": "symbol",
        "query_date": "quote_date",
        "side": "option_type",
        "iv": "implied_volatility",
    }
    for source, target in rename.items():
        if source in out.columns and target not in out.columns:
            out[target] = out[source]
    if quote_date is not None:
        out["quote_date"] = quote_date
    if "symbol" not in out.columns:
        out["symbol"] = pd.NA
    if "option_symbol" not in out.columns:
        out["option_symbol"] = pd.NA
    return validate_option_chain(out)


def validate_option_chain(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize option-chain rows."""

    if frame.empty:
        return pd.DataFrame(columns=OPTION_CHAIN_COLUMNS)
    frame = frame.copy()
    aliases = {
        "underlying": "symbol",
        "query_date": "quote_date",
        "side": "option_type",
        "iv": "implied_volatility",
    }
    for source, target in aliases.items():
        if source in frame.columns and target not in frame.columns:
            frame[target] = frame[source]
    required = ["symbol", "quote_date", "expiration", "option_type", "strike", "bid", "ask"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Option chain missing required columns: {missing}")
    out = frame.copy()
    if "option_symbol" not in out:
        out["option_symbol"] = pd.NA
    out["option_symbol"] = out["option_symbol"].astype("string")
    if "chain_quote_source" not in out:
        out["chain_quote_source"] = "provided"
    out["chain_quote_source"] = out["chain_quote_source"].astype(str)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["quote_date"] = pd.to_datetime(out["quote_date"], format="mixed", errors="coerce").dt.normalize()
    out["expiration"] = pd.to_datetime(out["expiration"], format="mixed", errors="coerce").dt.normalize()
    out["option_type"] = out["option_type"].astype(str).str.lower().str[0].map({"c": "call", "p": "put"}).fillna(out["option_type"])
    for column in ("strike", "bid", "ask", "mid", "last", "volume", "open_interest", "implied_volatility", "delta", "gamma", "theta", "vega"):
        if column not in out:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["mid"] = out["mid"].fillna((out["bid"] + out["ask"]) / 2.0)
    out["dte"] = (out["expiration"] - out["quote_date"]).dt.days
    out["spread_pct_mid"] = (out["ask"] - out["bid"]) / out["mid"].replace(0, np.nan)
    out = out.dropna(subset=["symbol", "quote_date", "expiration", "option_type", "strike", "bid", "ask"])
    return out[[column for column in [*OPTION_CHAIN_COLUMNS, "dte", "spread_pct_mid"] if column in out.columns]].sort_values(
        ["symbol", "quote_date", "expiration", "option_type", "strike"]
    ).reset_index(drop=True)


def validate_overlay_against_chain(
    overlay: pd.DataFrame,
    chain: pd.DataFrame,
    *,
    min_open_interest: int = 100,
    min_volume: int = 10,
    max_spread_pct_mid: float = 0.20,
) -> pd.DataFrame:
    """Attach best option-chain candidate to each overlay recommendation."""

    if overlay.empty:
        return overlay.copy()
    if chain.empty:
        out = overlay.copy()
        out["executable"] = False
        out["chain_validation_status"] = "missing_option_chain"
        return out
    chain = validate_option_chain(chain)
    rows: list[dict] = []
    for _, rec in overlay.iterrows():
        symbol = str(rec.get("symbol", "")).upper()
        signal_date = pd.Timestamp(rec.get("signal_date")).normalize()
        target_delta = float(rec.get("target_delta", 0.0) or 0.0)
        dte_min = int(rec.get("target_dte_min", 0) or 0)
        dte_max = int(rec.get("target_dte_max", 10_000) or 10_000)
        underlying_close = pd.to_numeric(pd.Series([rec.get("underlying_close", np.nan)]), errors="coerce").iloc[0]
        candidates = chain[
            chain["symbol"].eq(symbol)
            & chain["quote_date"].eq(signal_date)
            & chain["option_type"].eq("call")
            & chain["dte"].between(dte_min, dte_max)
            & chain["open_interest"].fillna(0).ge(min_open_interest)
            & chain["volume"].fillna(0).ge(min_volume)
            & chain["spread_pct_mid"].fillna(np.inf).le(max_spread_pct_mid)
        ].copy()
        if pd.notna(underlying_close) and float(underlying_close) > 0:
            lower_strike = float(underlying_close) * 0.70
            upper_strike = float(underlying_close) * 1.20
            candidates = candidates[candidates["strike"].between(lower_strike, upper_strike)]
        out_row = rec.to_dict()
        if candidates.empty:
            out_row.update({"executable": False, "chain_validation_status": "no_liquid_contract_match"})
            rows.append(out_row)
            continue
        candidates["delta_missing"] = candidates["delta"].isna()
        if pd.notna(underlying_close) and float(underlying_close) > 0:
            candidates["moneyness_distance"] = (candidates["strike"] / float(underlying_close) - 1.0).abs()
        else:
            candidates["moneyness_distance"] = 0.0
        candidates["delta_distance"] = (candidates["delta"].abs() - abs(target_delta)).abs()
        candidates["delta_distance"] = candidates["delta_distance"].fillna(9.0)
        best = candidates.sort_values(
            ["delta_missing", "delta_distance", "moneyness_distance", "spread_pct_mid", "open_interest"],
            ascending=[True, True, True, True, False],
        ).iloc[0]
        max_budget = pd.to_numeric(pd.Series([rec.get("max_premium_budget", np.nan)]), errors="coerce").iloc[0]
        contract_mid = float(best["mid"]) if pd.notna(best["mid"]) else np.nan
        recommended_contracts = 0
        estimated_premium = np.nan
        budget_utilization = np.nan
        if pd.notna(max_budget) and pd.notna(contract_mid) and contract_mid > 0:
            recommended_contracts = int(np.floor(float(max_budget) / (contract_mid * 100.0)))
            estimated_premium = recommended_contracts * contract_mid * 100.0
            budget_utilization = estimated_premium / float(max_budget) if float(max_budget) > 0 else np.nan
        if pd.notna(max_budget) and recommended_contracts < 1:
            out_row.update(
                {
                    "executable": False,
                    "chain_validation_status": "budget_too_small_for_liquid_contract",
                    "contract_option_symbol": best.get("option_symbol", pd.NA),
                    "contract_chain_quote_source": best.get("chain_quote_source", "provided"),
                    "contract_expiration": best["expiration"],
                    "contract_strike": best["strike"],
                    "contract_mid": best["mid"],
                    "contract_bid": best["bid"],
                    "contract_ask": best["ask"],
                    "contract_delta": best["delta"],
                    "contract_iv": best["implied_volatility"],
                    "contract_open_interest": best["open_interest"],
                    "contract_volume": best["volume"],
                    "contract_spread_pct_mid": best["spread_pct_mid"],
                    "recommended_contracts": recommended_contracts,
                    "estimated_option_premium": estimated_premium,
                    "premium_budget_utilization": budget_utilization,
                }
            )
            rows.append(out_row)
            continue
        out_row.update(
            {
                "executable": True,
                "chain_validation_status": "validated_liquid_contract",
                "contract_option_symbol": best.get("option_symbol", pd.NA),
                "contract_chain_quote_source": best.get("chain_quote_source", "provided"),
                "contract_expiration": best["expiration"],
                "contract_strike": best["strike"],
                "contract_mid": best["mid"],
                "contract_bid": best["bid"],
                "contract_ask": best["ask"],
                "contract_delta": best["delta"],
                "contract_iv": best["implied_volatility"],
                "contract_open_interest": best["open_interest"],
                "contract_volume": best["volume"],
                "contract_spread_pct_mid": best["spread_pct_mid"],
                "recommended_contracts": recommended_contracts,
                "estimated_option_premium": estimated_premium,
                "premium_budget_utilization": budget_utilization,
            }
        )
        rows.append(out_row)
    return pd.DataFrame(rows)
