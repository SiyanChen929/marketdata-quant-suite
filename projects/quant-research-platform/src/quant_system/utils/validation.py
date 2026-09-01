"""Data validation utilities."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
OHLCV_LINEAGE_COLUMNS = ["source", "finality"]


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    """Raise a clear error when required columns are absent."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an OHLCV table."""

    require_columns(frame, OHLCV_COLUMNS, "OHLCV")
    lineage = [column for column in OHLCV_LINEAGE_COLUMNS if column in frame.columns]
    out = frame[[*OHLCV_COLUMNS, *lineage]].copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    numeric = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "source" in out:
        out["source"] = out["source"].astype("string").str.strip().str.lower()
        if out["source"].isna().any() or out["source"].eq("").any():
            raise ValueError("OHLCV source must be non-empty when provided")
    if "finality" in out:
        out["finality"] = out["finality"].astype("string").str.strip().str.lower()
        invalid = sorted(set(out["finality"].dropna().astype(str)) - {"confirmed", "provisional"})
        if out["finality"].isna().any() or invalid:
            raise ValueError(f"OHLCV finality must be confirmed or provisional; invalid={invalid}")
    out = out.dropna(subset=["symbol", "date", "open", "high", "low", "close", "adj_close"])
    return out.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last").reset_index(drop=True)


def validate_confirmed_marketdata(frame: pd.DataFrame) -> pd.DataFrame:
    """Require MarketData lineage for a formal daily research input."""

    out = validate_ohlcv(frame)
    require_columns(out, OHLCV_LINEAGE_COLUMNS, "Confirmed MarketData OHLCV")
    if not out["source"].eq("marketdata.app").all():
        raise ValueError("formal MarketData input must carry source=marketdata.app")
    if not out["finality"].eq("confirmed").all():
        raise ValueError("formal MarketData input must contain confirmed rows only")
    return out
