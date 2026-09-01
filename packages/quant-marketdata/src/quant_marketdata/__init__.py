"""Shared MarketData client and canonical external store."""

from .client import MarketDataClient
from .exceptions import (
    CacheIntegrityError,
    CredentialError,
    DataContractError,
    DataUnavailableError,
    ProviderError,
    QuantMarketDataError,
)
from .options import OPTION_CHAIN_COLUMNS
from .schema import CANONICAL_COLUMNS, Finality, empty_bars, normalize_bars, wide_close
from .store import MarketDataStore

__all__ = [
    "CANONICAL_COLUMNS",
    "CacheIntegrityError",
    "CredentialError",
    "DataContractError",
    "DataUnavailableError",
    "Finality",
    "MarketDataClient",
    "MarketDataStore",
    "OPTION_CHAIN_COLUMNS",
    "ProviderError",
    "QuantMarketDataError",
    "empty_bars",
    "normalize_bars",
    "wide_close",
]

__version__ = "0.1.0"
