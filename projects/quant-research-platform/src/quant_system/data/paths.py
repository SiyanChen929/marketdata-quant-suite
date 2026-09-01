"""Shared storage paths for local quantitative research data."""

from __future__ import annotations

import os
from pathlib import Path

from quant_marketdata import MarketDataStore


def quant_data_home() -> Path:
    """Return the external data root, defaulting to the repository's data folder."""

    configured = os.getenv("QUANT_DATA_HOME")
    return Path(configured).expanduser() if configured else Path("data")


def marketdata_cache_dir() -> Path:
    """Return the strict external cache for MarketData reference responses."""

    return MarketDataStore().root / "raw" / "marketdata" / "platform-adapters"


def quant_system_lake_dir() -> Path:
    """Return the canonical DuckDB/Parquet lake root."""

    return quant_data_home() / "lake" / "confirmed" / "quant_system"


def expand_configured_path(value: str | Path) -> str:
    """Expand environment variables and the user directory in config paths."""

    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if "${QUANT_DATA_HOME}" in expanded or "$QUANT_DATA_HOME" in expanded:
        raise ValueError("QUANT_DATA_HOME must be set for this configuration")
    return expanded
