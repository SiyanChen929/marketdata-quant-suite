"""Canonical long-form bar schema and transformations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

import pandas as pd

from .exceptions import DataContractError


Finality = Literal["confirmed", "provisional"]

CANONICAL_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "finality",
)
NUMERIC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
VALID_FINALITIES = frozenset({"confirmed", "provisional"})


def validate_finality(value: str) -> Finality:
    """Return a normalized finality label or raise a contract error."""

    normalized = str(value).strip().lower()
    if normalized not in VALID_FINALITIES:
        raise DataContractError(
            f"finality must be one of {sorted(VALID_FINALITIES)}, got {value!r}"
        )
    return cast(Finality, normalized)


def empty_bars() -> pd.DataFrame:
    """Return an empty frame with the canonical column order and useful dtypes."""

    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "symbol": pd.Series(dtype="string"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="string"),
            "finality": pd.Series(dtype="string"),
        }
    )


def normalize_bars(
    frame: pd.DataFrame,
    *,
    finality: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Validate and normalize a frame to the canonical long schema.

    ``finality`` and ``source`` fill missing columns.  If a supplied frame
    already carries those fields, explicit arguments must agree with every row.
    Duplicate ``(date, symbol)`` rows are accepted only when every canonical
    value is identical.
    """

    if not isinstance(frame, pd.DataFrame):
        raise DataContractError("bars must be provided as a pandas DataFrame")

    required = {"date", "symbol", *NUMERIC_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"bars are missing required columns: {missing}")

    out = frame.copy()
    if "source" not in out:
        if source is None:
            raise DataContractError("bars require a non-empty source")
        out["source"] = source
    elif source is not None and not out["source"].astype(str).eq(str(source)).all():
        raise DataContractError("explicit source disagrees with one or more rows")

    if "finality" not in out:
        if finality is None:
            raise DataContractError("bars require a finality label")
        out["finality"] = validate_finality(finality)
    elif finality is not None:
        expected = validate_finality(finality)
        actual = out["finality"].astype(str).str.strip().str.lower()
        if not actual.eq(expected).all():
            raise DataContractError(
                f"{expected} writes cannot contain rows with another finality"
            )

    out["date"] = _normalize_dates(out["date"])
    out["symbol"] = out["symbol"].astype("string").str.strip().str.upper()
    out["source"] = out["source"].astype("string").str.strip()
    out["finality"] = out["finality"].astype("string").str.strip().str.lower()
    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")

    invalid_finalities = sorted(
        set(out["finality"].dropna().astype(str)).difference(VALID_FINALITIES)
    )
    if invalid_finalities:
        raise DataContractError(f"bars contain invalid finality labels: {invalid_finalities}")

    null_columns = [
        column for column in CANONICAL_COLUMNS if out[column].isna().any()
    ]
    if null_columns:
        raise DataContractError(f"bars contain null or invalid values in: {null_columns}")
    if out["symbol"].eq("").any():
        raise DataContractError("bars contain an empty symbol")
    if out["source"].eq("").any():
        raise DataContractError("bars contain an empty source")
    if (out["volume"] < 0).any():
        raise DataContractError("volume cannot be negative")

    highest_body = out[["open", "close", "low"]].max(axis=1)
    lowest_body = out[["open", "close", "high"]].min(axis=1)
    if (out["high"] < highest_body).any():
        raise DataContractError("high must be greater than or equal to OHLC values")
    if (out["low"] > lowest_body).any():
        raise DataContractError("low must be less than or equal to OHLC values")

    canonical = out.loc[:, CANONICAL_COLUMNS].copy()
    canonical = canonical.drop_duplicates().reset_index(drop=True)
    if canonical.duplicated(["date", "symbol"], keep=False).any():
        raise DataContractError(
            "bars contain conflicting rows for the same (date, symbol) key"
        )
    return canonical.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def wide_close(frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot canonical long bars into a sorted date-by-symbol close matrix."""

    bars = normalize_bars(frame)
    if bars.empty:
        result = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        result.columns.name = "symbol"
        return result
    result = bars.pivot(index="date", columns="symbol", values="close")
    result.index = pd.DatetimeIndex(result.index, name="date")
    result.columns.name = "symbol"
    return result.sort_index().sort_index(axis=1)


def normalize_symbols(symbols: str | Iterable[str] | None) -> list[str] | None:
    """Normalize optional symbol filters while preserving caller order."""

    if symbols is None:
        return None
    values = [symbols] if isinstance(symbols, str) else list(symbols)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = str(value).strip().upper()
        if not symbol:
            raise DataContractError("symbols cannot contain empty values")
        if symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    return normalized


def _normalize_dates(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype="datetime64[ns]", index=values.index)
    dates = pd.to_datetime(values, errors="coerce", format="mixed")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        return dates.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return dates
